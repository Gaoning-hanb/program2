"""coursebook-digest CLI。

一条命令闭环：
    coursebook ingest  <教材.pdf|.md|.txt> --course 课程名          # 解析+蒸馏+入库+建向量
    coursebook video   <B站链接>          --course 课程名          # 视频字幕/转写→蒸馏入库（独立成课）
    coursebook parse   <教材.pdf>          --course 课程名          # 只切章节，输出文本供人工抽查
    coursebook find    "<题目>"            --course 课程名          # 检索命中的方法卡片
    coursebook ask     "<题目>"            --course 课程名          # 检索+优先注入课本方法作答
    coursebook notes   --course 课程名                              # 思维导图 + 章节复习笔记
    coursebook courses / env                                        # 课程列表 / 环境自检
"""
from __future__ import annotations

import sys
from pathlib import Path

# 中文 Windows 控制台默认 cp936/GBK，非 BMP 字符（emoji 等）会直接崩溃。
# 统一以 UTF-8 输出，遇到无法编码的字符用替换符兜底。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import typer

from . import __version__
from .answer import ask_question
from .config import Settings, get_settings
from .denoise import denoise_chapter
from .distill import distill_chapter, flag_noise_cards, is_noise_card
from .llm import LLMClient
from .mineru_http import DEFAULT_PARSE_PATH
from .parser import filter_chapters, kill_mineru_processes, parse_source
from .retrieve import find_methods
from .store import CourseStore, VectorStore

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="教材方法库：教材 → 方法卡片 → 优先注入课本逻辑作答案",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _cloud_ready(settings: Settings) -> bool:
    return bool((settings.mineru_api_key or settings.llm_api_key).strip())


def choose_parser(settings: Settings, *, reuse: bool = False) -> str:
    """PDF 解析后端选择。``--parser auto``（未显式指定）时调用：

    - ``reuse=True``：只加载已有 md，与后端无关，直接走本地路径，不打扰；
    - 非交互（脚本/PIPE）：云端有 key 且本地不可用→云端，否则本地；
    - 交互（终端）：弹出 本地 mineru / 云端 mineru-http 菜单，回车取默认（云端优先）。
    """
    if reuse:
        return "mineru"
    from .parser import _mineru_available

    cloud_ok = _cloud_ready(settings)
    local_ok = _mineru_available()
    if not sys.stdin or not sys.stdin.isatty():
        if cloud_ok and not local_ok:
            return "mineru-http"
        if local_ok:
            return "mineru"
        return "mineru-http" if cloud_ok else "mineru"

    entries: list[tuple[str, str, str]] = []
    if cloud_ok:
        entries.append(("1", "mineru-http", "云端 mineru-http：api.llm.ustc.edu.cn（复用模型 key，无需本地 GPU）"))
    if local_ok:
        entries.append(("2", "mineru", "本地 mineru：本机 GPU / CLI 解析"))
    if not entries:
        typer.echo("（未检测到云端 key 也未检测到本地 mineru，按云端后端继续，将给出明确报错）")
        return "mineru-http"
    menu = "\n".join(f"  {num}) {desc}" for num, _val, desc in entries)
    default = entries[0][0]
    try:
        ans = typer.prompt(f"请选择 PDF 解析后端（回车默认 {default}）：\n{menu}", default=default)
    except Exception:  # noqa: BLE001  输入中断/EOF 等非交互情形
        ans = default
    ans = str(ans).strip()
    for num, val, _desc in entries:
        if ans in (num, val):
            return val
    return entries[0][1]


def _print_cards(methods, output: str | None = None) -> None:
    """打印命中方法卡片；给 --output 时同时覆写为 JSON 文件。"""
    import json
    from .answer import render_methods

    text = render_methods(methods) if methods else "（无命中）"
    if output:
        Path(output).write_text(
            json.dumps(
                [m.model_dump(mode="json") for m in methods],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        typer.echo(f"命中已写入 {output}")
    typer.echo(text)


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #
@app.command()
def ingest(
    source: str = typer.Argument(..., help="教材文件：pdf / md / txt"),
    course: str = typer.Option(..., "--course", "-c", help="课程名（命名空间，建议纯文字）"),
    parser: str = typer.Option("auto", "--parser", help="pdf 解析：auto|mineru(本地)|mineru-http(云端)"),
    chapter: str | None = typer.Option(None, "--chapter", help="只处理含此子串的章节"),
    min_cards: int = typer.Option(0, "--min-cards", help="每章至少产出多少张卡（防漏检）"),
    mineru_out: str | None = typer.Option(
        None, "--mineru-out",
        help="mineru 输出目录（默认 <pdf>同目录/mineru_out）。可分片产出/合并子目录",
    ),
    reuse_mineru: bool = typer.Option(
        False, "--reuse-mineru", is_flag=True,
        help="跳过重新运行 mineru，直接复用 --mineru-out 下已产出的 Markdown（续跑/分片用）",
    ),
    slice_pages: int = typer.Option(
        0, "--slice-pages", "-sp",
        help="每片页数（>0 启用自动分片）。本地后端把 PDF 切成多个独立 mineru 任务；"
        "云端后端作为云端分片页数，规避单任务过大/超时",
    ),
    batch_files: int = typer.Option(
        0, "--batch-files", "-bf",
        help="云端一次请求最大分片数（0=按 .env MINERU_BATCH_FILES）",
    ),
    no_strip: bool = typer.Option(
        False, "--no-strip", is_flag=True,
        help="蒸馏前不去噪（保留背景/人物/PPT模板等内容；默认去噪以提速）",
    ),
    parallel: int = typer.Option(
        0, "--parallel",
        help="蒸馏并发分块数（0=按 .env DISTILL_PARALLEL；>1 并行调用模型大幅提速）",
    ),
) -> None:
    """解析 → 蒸馏方法卡片 → 入库（JSONL）+ 建向量索引。"""
    settings = get_settings()
    settings.ensure_dirs()
    # 后端选择：--parser 未显式给（仍是 auto）时，启动时交互/启发式选择
    eff_parser = parser if parser != "auto" else choose_parser(settings, reuse=reuse_mineru)
    if eff_parser != parser:
        typer.echo(f"（PDF 解析后端：{eff_parser}）")
    src = Path(source)
    target = str(Path(mineru_out).resolve()) if mineru_out else None
    # 仅本地后端需要清孤儿进程（云端无本地进程）
    if eff_parser == "mineru":
        killed = kill_mineru_processes()
        if killed:
            typer.echo(f"（已清理上次残留的 {killed} 个 mineru 进程，释放 GPU）")
    try:
        chapters = parse_source(
            source, course, parser=eff_parser, out_dir=target,
            reuse_out=reuse_mineru, slice_size=slice_pages, settings=settings,
            batch_files=batch_files,
        )
    except KeyboardInterrupt:
        # Ctrl+C 只杀主进程，本地 mineru 服务进程会残留烧 GPU——兜底清掉
        if eff_parser == "mineru":
            kill_mineru_processes()
        typer.echo("\n已中断；已连带清理 mineru 后台进程（不会再占 GPU）。")
        raise typer.Exit(130)
    if not chapters:
        typer.echo("未解析到任何章节。", err=True)
        raise typer.Exit(1)
    chapters = filter_chapters(chapters, chapter)
    store = CourseStore(course, settings)

    llm = LLMClient(settings)
    strip_enabled = (not no_strip) and settings.strip_extraneous
    chunk_chars = settings.distill_chunk_chars
    workers = parallel if parallel and parallel > 0 else settings.distill_parallel
    if strip_enabled:
        typer.echo(
            f"（去噪开：蒸馏前剔除背景/人物/模板；分块 {chunk_chars} 字、并发 {workers}）"
        )
    all_cards = []
    total_new = 0
    with typer.progressbar(chapters, label="蒸馏中") as bar:
        for ch in bar:
            if strip_enabled:
                ch, dstat = denoise_chapter(ch)
                typer.echo(
                    f"  [去噪] {ch.chapter}: 删 {dstat.removed_lines} 行"
                    f"（背景人物 {dstat.story_removed} / 模板杂讯 {dstat.chrome_removed}）"
                    f" 约 {dstat.removed_chars} 字 / 保留 {dstat.kept_chars} 字"
                    f"（-{dstat.removed_ratio * 100:.0f}%）"
                )
            if len(ch.text) < 400:  # 节选/参考文献等噪音小章：不蒸馏
                typer.echo(f"  [跳过] {ch.chapter}: 过小（{len(ch.text)} 字，疑似噪音）")
                continue
            cards = distill_chapter(llm, ch, chunk_size=chunk_chars, parallel=workers)
            cards = flag_noise_cards(cards)  # 仅供参考类（阅读顺序/编排指引）打标，不参与考点
            if min_cards and len(cards) < min_cards:
                typer.echo(f"（警告）{ch.chapter} 仅产出 {len(cards)} 张，低于 {min_cards}")
            new = store.save_all(cards)
            total_new += new
            all_cards.extend(cards)
            typer.echo(f"  ✓ {ch.chapter}: 新 {new} / 累计 {len(all_cards)}")
    if all_cards:
        try:
            # 全量重建索引，保证向量库与 jsonl 完全一致（清掉残留旧 id）
            VectorStore(course, settings).rebuild(store.load_all())
            typer.echo("已重建向量索引")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"（向量索引跳过：{exc}）")
    typer.echo(f"完成：本课程现有 {len(store.load_all())} 张方法卡片（本次新增 {total_new}）")


@app.command()
def reindex(
    course: str = typer.Option(..., "--course", "-c", help="课程名"),
) -> None:
    """从 jsonl 全量重建该课程的向量索引（修复残留/失配后的一键重建）。"""
    settings = get_settings()
    store = CourseStore(course, settings)
    cards = store.load_all()
    if not cards:
        typer.echo("该课程没有卡片，无需重建。")
        return
    VectorStore(course, settings).rebuild(cards)
    typer.echo(f"已重建「{course}」向量索引（{len(cards)} 张卡片）")


@app.command()
def parse(
    source: str = typer.Argument(..., help="教材 pdf 文件"),
    course: str = typer.Option(..., "--course", "-c"),
    parser: str = typer.Option("auto", "--parser", help="auto|mineru(本地)|mineru-http(云端)"),
    out_dir: str | None = typer.Option(None, "--out", help="parsed 文本输出目录（默认 data/parsed/<course>）"),
) -> None:
    """只解析切章，不蒸馏：便于人工抽查解析质量。"""
    settings = get_settings()
    eff_parser = parser if parser != "auto" else choose_parser(settings)
    if eff_parser != parser:
        typer.echo(f"（PDF 解析后端：{eff_parser}）")
    try:
        chapters = parse_source(source, course, parser=eff_parser, settings=settings)
    except KeyboardInterrupt:
        if eff_parser == "mineru":
            kill_mineru_processes()
        typer.echo("\n已中断；已连带清理 mineru 后台进程。")
        raise typer.Exit(130)
    settings.ensure_dirs()
    base = Path(out_dir) if out_dir else Path(settings.data_dir) / "parsed" / course
    base.mkdir(parents=True, exist_ok=True)
    for ch in chapters:
        safe = "".join(c if c not in '\\/:*?"<>|' else "_" for c in ch.chapter)[:60]
        p = base / f"{safe}.md"
        p.write_text(ch.text, encoding="utf-8")
        typer.echo(f"{ch.chapter}：{len(ch.text)} 字 → {p}")


@app.command()
def find(
    question: str = typer.Argument(..., help="学生题目/问题"),
    course: str = typer.Option(..., "--course", "-c"),
    top_k: int = typer.Option(-1, "--top-k", "-k", help="返回几条（默认配置值）"),
    output: str | None = typer.Option(None, "--output", help="命中 JSON 另存到该文件"),
) -> None:
    """检索命中的课本方法卡片（不调用答题）。"""
    settings = get_settings()
    k = top_k if top_k > 0 else settings.top_k_default
    methods = find_methods(question, course, top_k=k, settings=settings)
    _print_cards(methods, output)


@app.command()
def ask(
    question: str = typer.Argument(..., help="学生题目"),
    course: str = typer.Option(..., "--course", "-c"),
    top_k: int = typer.Option(-1, "--top-k", "-k"),
    methods_json: str | None = typer.Option(None, "--methods-json", help="只输出命中方法卡 JSON 后退出（供 DSH 插件消费）"),
    agentic: bool = typer.Option(False, "--agentic", is_flag=True,
                                  help="Agent 模式：模型自主多轮调用检索/读卡工具后再作答（改写题、两跳题召回更好）"),
) -> None:
    """检索课本方法 → 优先注入 → LLM 作答。"""
    settings = get_settings()
    k = top_k if top_k > 0 else settings.top_k_default
    if agentic:
        from .agent_loop import AgentLoop
        import sys as _sys

        streamed = {"started": False}

        def _on_delta(text: str) -> None:
            if not streamed["started"]:
                streamed["started"] = True
                typer.echo("\n══ 作答（课本优先 · agent 流式）══")
            typer.echo(text, nl=False)
            _sys.stdout.flush()

        def _on_round(round_no: int) -> None:
            if not streamed["started"]:  # 答案开始后就不再刷状态行
                typer.echo(f"⏳ Agent 检索/思考中（第 {round_no} 轮）…")

        try:
            agent = AgentLoop(course, settings=settings, top_k=k)
            r = agent.run(question, on_delta=_on_delta, on_round=_on_round)
            if streamed["started"]:
                typer.echo("")  # 收尾换行
            else:  # 理论上不会（作答轮必有 delta），兜底整块输出
                typer.echo("\n══ 作答（课本优先 · agent 模式）══\n")
                typer.echo(r.answer)
            typer.echo("\n══ Agent 工具轨迹 ══")
            for step in r.trace:
                typer.echo(f"  · {step}")
            typer.echo(f"（{r.rounds} 轮 / {r.tool_calls} 次工具调用"
                       + ("，超轮数强制收尾）" if r.forced_final else "）"))
            if r.round_secs:
                typer.echo(f"（各轮耗时 {r.round_secs} 秒）")
            return
        except Exception as exc:  # noqa: BLE001 —— 新模式失败自动回落老管线
            typer.echo(f"agent 模式失败（{type(exc).__name__}: {exc}），已回落普通管线。", err=True)
    try:
        answer, methods = ask_question(question, course, top_k=k, settings=settings)
    except ValueError as exc:
        typer.echo(f"无法调用模型：{exc}", err=True)
        typer.echo("提示：在 .env 填好 DEEPSEEK_API_KEY 后重试；或先只跑 coursebook find 体验检索。")
        raise typer.Exit(2)
    typer.echo("══ 命中的课本方法 ══")
    _print_cards(methods, methods_json)
    typer.echo("\n══ 作答（优先采用课本方法）══\n")
    typer.echo(answer)


@app.command()
def video(
    url: str = typer.Argument(
        ..., help="B站视频链接（单P或多P系列均可；多P每个P作为一章，整个系列独立成课）"),
    course: str = typer.Option(
        ..., "--course", "-c", help="课程名（视频系列独立命名，如「操作系统-王道」）"),
    parts: str = typer.Option(
        "", "--parts", help="只处理指定分P，如 '1-3,5'（默认全部）"),
    whisper_model: str = typer.Option(
        "", "--whisper-model",
        help="无字幕分P的本地转写模型 tiny/base/small/medium（默认 .env WHISPER_MODEL）"),
    skip_whisper: bool = typer.Option(
        False, "--skip-whisper", is_flag=True, help="无字幕的分P直接跳过，不做本地转写"),
    transcript_only: bool = typer.Option(
        False, "--transcript-only", is_flag=True,
        help="只取字幕/转写并落盘 data/transcripts/，不蒸馏入库（预检字幕质量用）"),
    parallel: int = typer.Option(0, "--parallel", help="蒸馏并发分块数（0=按 .env）"),
) -> None:
    """B站视频 → 字幕/本地转写 → 蒸馏方法卡片 → 入库。

    字幕优先（B站 CC/AI 字幕秒级拿到），无字幕分P自动回退 faster-whisper
    本地转写（GPU 优先）。卡片溯源到 P号+时间点，notes 可一键跳回视频原时刻。
    """
    import json as _json

    from .distill import flag_noise_cards as _flag_noise
    from .parser import Chapter as _Chapter
    from .video import (VIDEO_SYSTEM_PROMPT, VideoPart, attach_source,
                        extract_bvid, fetch_part_subtitles, list_parts,
                        parse_json3, parse_parts_spec, segments_to_text,
                        transcribe_part)

    settings = get_settings()
    settings.ensure_dirs()
    bvid = extract_bvid(url)
    # 先粗解析 --parts：只用于提前终止逐P探测（风控兜底路径可省几分钟）
    try:
        rough = parse_parts_spec(parts, 10 ** 6)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)
    stop_after = max(rough) if rough else None
    try:
        meta = list_parts(url, settings.bili_sessdata, stop_after=stop_after)
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"无法读取视频信息：{exc}", err=True)
        raise typer.Exit(1)
    total = len(meta)
    try:
        wanted = parse_parts_spec(parts, total)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)
    sel = [m for m in meta if wanted is None or m["part_no"] in wanted]
    typer.echo(f"视频 {bvid}：共 {total} 个分P，本次处理 {len(sel)} 个")

    # ---- 1) 转写：B站字幕优先，whisper 兜底；结果落盘可复用 ---- #
    workdir = Path(settings.data_dir) / "transcripts" / course
    workdir.mkdir(parents=True, exist_ok=True)
    wm = whisper_model or settings.whisper_model
    model_cache: dict = {}
    transcripts: list[dict] = []

    def _load_or_fetch(m: dict) -> dict | None:
        pno, title = m["part_no"], m["title"]
        cache = workdir / f"p{pno}.json"
        if cache.exists():
            d = _json.loads(cache.read_text(encoding="utf-8"))
            typer.echo(f"  [复用] P{pno} 已有转写（{d['source']}，{len(d['text'])} 字）")
            return d
        try:
            sub_file, real_title = fetch_part_subtitles(
                bvid, pno, workdir, settings.bili_sessdata)
        except RuntimeError as exc:
            typer.echo(f"  [警告] P{pno} 字幕抓取失败：{exc}", err=True)
            sub_file, real_title = None, None
        if real_title:
            title = real_title  # 覆盖 flat 列表的占位标题（多P视频拿真实分P名）
        audio_dur = 0.0
        if sub_file is not None:
            segs, lang = parse_json3(sub_file)
            src = "ai" if ".ai-" in sub_file.name else "cc"
        elif skip_whisper:
            typer.echo(f"  [跳过] P{pno} 无字幕（--skip-whisper）")
            return None
        else:
            typer.echo(f"  [转写] P{pno} 无字幕，本地 whisper（{wm}）转写中…")
            try:
                segs, audio_dur = transcribe_part(bvid, pno, workdir, wm, model_cache)
            except RuntimeError as exc:
                typer.echo(f"  [跳过] P{pno}：{exc}", err=True)
                return None
            lang, src = "zh", "whisper"
            if audio_dur > 60:
                covered = segs[-1].end if segs else 0.0
                if covered / audio_dur < 0.85:
                    typer.echo(
                        f"  [提示] P{pno} 转写覆盖 {covered/60:.0f}/{audio_dur/60:.0f} 分钟："
                        "其后无语音（录播课常见：课间音乐/静音段，已自动跳过）")
        if not segs:
            typer.echo(f"  [跳过] P{pno} 转写为空")
            return None
        text = segments_to_text(segs)
        d = {"part_no": pno, "title": title, "duration": audio_dur or m["duration"],
             "source": src, "lang": lang, "text": text}
        cache.write_text(_json.dumps(d, ensure_ascii=False), encoding="utf-8")
        return d

    with typer.progressbar(sel, label="取字幕") as bar:
        for m in bar:
            d = _load_or_fetch(m)
            if d:
                transcripts.append(d)
    if not transcripts:
        typer.echo("没有任何分P取得转写（无字幕且未开 whisper？），退出。", err=True)
        raise typer.Exit(1)
    n_chars = sum(len(d["text"]) for d in transcripts)
    typer.echo(f"转写完成：{len(transcripts)}/{len(sel)} 个分P，共 {n_chars} 字"
               f"（已存 {workdir}，重跑自动复用）")
    if transcript_only:
        return

    # ---- 2) 蒸馏：每个分P一章，视频专用 prompt，溯源挂 P号+时间点 ---- #
    llm = LLMClient(settings)
    store = CourseStore(course, settings)
    chunk_chars = settings.distill_chunk_chars
    workers = parallel if parallel and parallel > 0 else settings.distill_parallel
    all_cards = []
    total_new = 0
    with typer.progressbar(transcripts, label="蒸馏中") as bar:
        for d in bar:
            pno, title = d["part_no"], d["title"]
            safe = "".join(c if c not in '\\/:*?"<>|\r\n' else " " for c in title).strip()
            ch = _Chapter(course=course, chapter=f"P{pno} {safe}"[:80], text=d["text"])
            if len(ch.text) < 400:
                typer.echo(f"  [跳过] P{pno} 过小（{len(ch.text)} 字，疑似片头/空P）")
                continue
            cards = distill_chapter(
                llm, ch, chunk_size=chunk_chars, parallel=workers,
                system_prompt=VIDEO_SYSTEM_PROMPT)
            cards = _flag_noise(cards)
            attach_source(cards, VideoPart(part_no=pno, title=safe,
                                           duration=d["duration"]), bvid)
            new = store.save_all(cards)
            total_new += new
            all_cards.extend(cards)
            typer.echo(f"  ✓ P{pno} {safe}: 新 {new} / 累计 {len(all_cards)}")
    if all_cards:
        try:
            VectorStore(course, settings).rebuild(store.load_all())
            typer.echo("已重建向量索引")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"（向量索引跳过：{exc}）")
        typer.echo(f"完成：本课程现有 {len(store.load_all())} 张卡片（本次新增 {total_new}）。"
                   f"试试：coursebook notes --course {course} 生成思维导图+笔记")
    else:
        typer.echo("未蒸馏出任何卡片（转写太短或全是闲聊？）。", err=True)


@app.command()
def noise(
    course: str = typer.Option(..., "--course", "-c"),
    apply: bool = typer.Option(False, "--apply", is_flag=True,
                               help="实际打标写入（默认仅预览命中列表）"),
    show_all: bool = typer.Option(False, "--all", is_flag=True, help="列出全部命中，不截断"),
) -> None:
    """（清扫）把“仅供参考”类噪音卡（阅读顺序/编排指引等）标记为 reference_only，
    使其不再出现在打卡/检索/考点；卡片仍保留在库中便于浏览。"""
    from .distill import flag_noise_cards, is_noise_card

    settings = get_settings()
    store = CourseStore(course, settings)
    cards = store.load_all()
    flagged = [c for c in cards if not c.reference_only and is_noise_card(c)]
    if not flagged:
        typer.echo(f"「{course}」没有新的仅供参考类卡需要打标。")
        return
    listed = flagged if show_all else flagged[:20]
    typer.echo(f"「{course}」命中仅供参考类 {len(flagged)} 张：")
    for c in listed:
        typer.echo(f"  * [{c.kind.value}] {c.topic}（{c.chapter}）")
    if not show_all and len(flagged) > 20:
        typer.echo(f"  …（其余 {len(flagged) - 20} 张，用 --all 查看全部）")
    if apply:
        store.save_all(flag_noise_cards(cards))
        typer.echo(f"已写入：{len(flagged)} 张已标记为参考，不再参与打卡/检索/考点。"
                   f"（可选执行 coursebook reindex --course {course} 以重建向量）")
    else:
        typer.echo("（预览模式：确认无误后加 --apply 真正写入）")


@app.command()
def notes(
    course: str = typer.Option(..., "--course", "-c", help="课程名"),
    out_dir: str = typer.Option("", "--out", help="输出目录（默认 notes/<课程名>/）"),
    cdn: bool = typer.Option(
        False, "--cdn", is_flag=True,
        help="思维导图改用 CDN 加载渲染库（默认随附本地脚本，离线可开）"),
    kinds: str = typer.Option(
        "", "--kinds", help="只保留指定种类，如 '方法,例题'（默认全部）"),
) -> None:
    """生成复习资料：markmap 思维导图（HTML）+ 章节笔记（Markdown）。

    教材课程与视频课程通用；视频卡在笔记里带「跳回原视频时刻」链接。
    纯代码生成、不调模型、秒出。
    """
    from .notes import write_notes

    settings = get_settings()
    cards = CourseStore(course, settings).load_all()
    if not cards:
        typer.echo(f"「{course}」没有卡片：先 coursebook ingest / video 入库。", err=True)
        raise typer.Exit(1)
    kind_list = [k.strip() for k in kinds.split("，") if k.strip()] if kinds else None
    out = Path(out_dir) if out_dir else Path(settings.data_dir).parent / "notes" / course
    try:
        written = write_notes(cards, course, out, use_cdn=cdn, kinds=kind_list)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)
    for p in written:
        typer.echo(f"  ✓ {p}")
    typer.echo(f"已生成 {len(written)} 个文件到 {out}"
               "（思维导图.html 用浏览器打开；笔记.md 支持 LaTeX 预览的编辑器看公式）")


@app.command()
def teach(
    question: str = typer.Argument(..., help="学生提问"),
    course: str = typer.Option(..., "--course", "-c"),
    user: str = typer.Option("local", "--user", "-u", help="学生 id（画像/记忆按用户隔离）"),
    top_k: int = typer.Option(-1, "--top-k", "-k", help="课本方法检索条数"),
    memory_k: int = typer.Option(2, "--memory-k", help="个人记忆检索条数"),
    no_writeback: bool = typer.Option(False, "--no-writeback", is_flag=True, help="本轮不写回记忆/画像"),
    show_preamble: bool = typer.Option(False, "--preamble", is_flag=True, help="打印注入的学生画像片段"),
) -> None:
    """（agent循环）按学生画像 + 课本方法 + 个人记忆作答，并把本轮问答写回记忆库。"""
    from .agent import TeacherAgent

    agent = TeacherAgent(user_id=user, course=course, write_back=not no_writeback)
    if show_preamble:
        typer.echo("══ 本次注入的学生画像 ══")
        typer.echo(agent.preamble)
        typer.echo()
    r = agent.turn(question, top_k=top_k if top_k > 0 else None, memory_k=memory_k)
    typer.echo(f"══ 意图：{r.intent.value} ══" + ("（离线降级）" if r.offline else ""))
    if r.methods:
        typer.echo("── 命中课本方法 ──")
        for m in r.methods[:4]:
            typer.echo(f"  {m.card.summary()}  相关度 {m.score:.2f}")
    if r.memory_hits:
        typer.echo("── 命中个人过往记忆 ──")
        for m in r.memory_hits[:3]:
            typer.echo(f"  历史·{m.card.topic}")
    typer.echo("══ 作答 ══")
    typer.echo(r.answer)
    if not r.writeback.skipped:
        typer.echo(f"\n══ 写回记忆 ══ {r.writeback.buckets_written}")
        typer.echo(
            f"画像：累计 {agent.profile.turn_count} 轮 | "
            f"薄弱主题 {len(agent.profile.weak_topics)} 个 / 常错点 {len(agent.profile.common_mistakes)} 个"
        )
    elif r.offline:
        typer.echo("（离线降级：未调用模型，未写回）")
    elif r.writeback.reason:
        typer.echo(f"（写回跳过：{r.writeback.reason}）")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址（0.0.0.0 可被同局域网访问）"),
    port: int = typer.Option(8000, "--port", help="监听端口"),
) -> None:
    """启动 R2 Web 服务（FastAPI）：GET / 演示界面，POST /api/teach 等。"""
    import uvicorn

    try:
        from .api_server import app as web_app
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"无法加载 Web 服务：{exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"启动 Web 服务：http://{host}:{port}/ （Ctrl+C 停止）")
    uvicorn.run(web_app, host=host, port=port, reload=False)


@app.command()
def courses() -> None:
    """列出已建立的课程（命名空间）。"""
    settings = get_settings()
    names = CourseStore.list_courses(settings)
    if not names:
        typer.echo("（暂无课程。用 coursebook ingest 导入第一本教材吧）")
    for n in names:
        store = CourseStore(n, settings)
        typer.echo(f"  {n}\t{len(store.load_all())} 张卡片")


@app.command()
def env() -> None:
    """环境自检：key/数据目录/存储/可选的 mineru。"""
    from .parser import _mineru_available

    settings = get_settings()
    key = "已配置" if settings.llm_api_key else "未配置（需在 .env 填 DEEPSEEK_API_KEY）"
    typer.echo(f"版本        : {__version__}")
    typer.echo(f"模型服务    : {settings.llm_base_url}  model={settings.llm_model}  key={key}")
    typer.echo(f"数据目录    : {Path(settings.data_dir).resolve()}")
    typer.echo(f"存储模式    : {settings.storage}  嵌入模式: {settings.embedding_mode}")
    typer.echo(f"mineru(本地) : {'可用' if _mineru_available() else '未安装（PDF 无法本地解析）'}")
    typer.echo(f"mineru(云端) : {'可用 ' + settings.mineru_base_url + DEFAULT_PARSE_PATH if (_cloud_ready(settings)) else '未配置 key（--parser mineru-http 需要）'}")
    typer.echo(f"云端分片     : 每片 ≤{settings.mineru_chunk_pages} 页 / 每批 ≤{settings.mineru_batch_files} 文件")
    typer.echo(f"课程        : {CourseStore.list_courses(settings)}")


if __name__ == "__main__":
    app()
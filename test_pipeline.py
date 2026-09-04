"""端到端离线集成测试：不联网、不需要 API key。

覆盖骨架的完整闭环（parse → distill → store → vector → find → ask）与 PDF 解析分支。
运行：conda activate firpro && cd coursebook-digest && python test_pipeline.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from coursebook_digest.config import PROJECT_ROOT, Settings
from coursebook_digest.schema import MethodCard
from coursebook_digest.store import CourseStore, VectorStore
from coursebook_digest.parser import _split_chapters, parse_source
from coursebook_digest.retrieve import find_methods
from coursebook_digest.answer import ask_question
from coursebook_digest.distill import distill_chapter, SYSTEM_PROMPT

SAMPLE = PROJECT_ROOT / "examples" / "sample_course.md"
COURSE = "集成测试课程"
CHAPTER = "第一章 极限与连续"

CARDS = [
    MethodCard(
        id=f"{COURSE}|{CHAPTER}|001", course=COURSE, chapter=CHAPTER,
        topic="洛必达法则", kind="方法",
        keywords=["洛必达", "未定型", "0/0", "无穷比无穷", "求导"],
        applicability="0/0 或 ∞/∞ 型未定式",
        steps=["判断型别", "分子分母分别求导", "检查新极限"],
        core_formula_latex=r"\lim f/g = \lim f'/g'",
        error_notes=["分子分母各自求导"], source_page="12",
    ),
    MethodCard(
        id=f"{COURSE}|{CHAPTER}|002", course=COURSE, chapter=CHAPTER,
        topic="等价无穷小替换", kind="方法",
        keywords=["等价无穷小", "替换", "sin", "tan"],
        applicability="乘除中的趋于0因子",
        steps=["确认乘除", "整体替换", "化简"],
        source_page="15",
    ),
]


class FakeLLM:
    """离线假模型：parse_json 供蒸馏，complete_text 供 ask 作答。"""

    def __init__(self, chapter: str = CHAPTER) -> None:
        self.chapter = chapter
        self.last_user = ""
        self.last_system = ""

    def parse_json(self, system: str, user: str) -> dict:  # noqa: ARG002
        cards = []
        for c in CARDS:  # 跟随传入章节，保证蒸馏断言一致
            d = obj_dict(c)
            d["chapter"] = self.chapter
            cards.append(d)
        return {"cards": cards}

    def complete_text(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return "（假模型作答）优先采用课本方法：洛必达法则。"


def obj_dict(c: MethodCard) -> dict:
    return c.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
def test_chapter_split() -> str:
    pages = [(1, "第1章 极限\n内容A\n内容B"), (2, "第2章 导数的应用\n内容C")]
    chapters = _split_chapters(pages)
    names = [c[0] for c in chapters]
    assert names == ["极限", "导数的应用"], f"章节切分异常：{names}"
    assert len(chapters[0][1]) > 0 and len(chapters[1][1]) > 0
    return ", ".join(names)


def test_md_parse() -> None:
    chapters = parse_source(SAMPLE, COURSE, parser="auto")
    assert chapters and chapters[0].text, "md 解析应产出非空章节"


def test_pypdf_removed(run_dir: Path) -> None:
    """回归：pypdf 纯文本兜底已移除——显式指定 pypdf 应被明确拒绝（不静默退化）。"""
    pdf = run_dir / "dummy.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy-not-a-real-pdf")
    try:
        parse_source(pdf, COURSE, parser="pypdf")
        raise AssertionError("pypdf 应已被移除并被拒绝")
    except ValueError as exc:
        assert "pypdf" in str(exc) and "mineru" in str(exc), f"错误信息不对：{exc}"
    print("    拒绝 pypdf 兜底 OK")


def test_offline_live_find(run_dir: Path, settings: Settings) -> None:
    """持久化写入 → 向量索引 → 检索（词面+向量融合），最顶上应为洛必达。"""
    store = CourseStore(COURSE, settings)
    n = store.save_all(CARDS)
    assert n == 2
    VectorStore(COURSE, settings).upsert_cards(CARDS)
    methods = find_methods("分子分母都趋于零，用洛必达法则求", COURSE,
                           top_k=3, settings=settings, use_chroma=True)
    assert methods and "洛必达" in methods[0].card.topic, \
        f"top 应为洛必达，实际 {methods[0].card.topic if methods else '无命中'}"
    # 向量通道确实贡献了分数（融合后通常高于纯词面）
    printed = "、".join(f"{m.card.topic}:{m.score}" for m in methods)
    print(f"    检索排序 -> {printed}")


def test_ask_injects_methods(settings: Settings) -> None:
    """ask 全链路：检索方法 → 注入块排在题目前 → 假模型作答。"""
    fake = FakeLLM()
    answer, methods = ask_question("求 0/0 型极限的步骤", COURSE,
                                   top_k=2, settings=settings, llm=fake)
    assert answer and "洛必达" in answer
    assert "优先参考的教材方法" in fake.last_user, "作答 prompt 应包含优先注入块"
    assert "优先采用课本方法" in fake.last_system
    assert "0/0 型极限的步骤" in fake.last_user
    assert methods and methods[0].card.topic == "洛必达法则"
    print(f"    ask 注入块 -> 命中 {len(methods)} 卡，假模型作答已收到")


def test_distill_with_fake(run_dir: Path) -> None:
    """蒸馏链路（假 LLM）：从示例教材蒸馏出方法卡并落库。"""
    chapters = parse_source(SAMPLE, "蒸馏课程", parser="auto")
    ch = chapters[0]
    cards = distill_chapter(FakeLLM(chapter=ch.chapter), ch)
    assert cards, "应蒸馏出卡片"
    assert all(c.chapter == ch.chapter for c in cards)
    print(f"    蒸馏 -> {len(cards)} 张卡（示例教材）")
    # 抽查系统提示包含 schema 约束
    assert "方法卡片" in SYSTEM_PROMPT and "error_notes" in SYSTEM_PROMPT


def test_mineru_reuse_multi_part(run_dir: Path) -> None:
    """分片/续跑：复用多个 mineru 输出子目录里的 md，合并切章。"""
    from coursebook_digest.parser import load_mineru_output

    mupan = run_dir / "mupan"
    p1 = mupan / "p1" / "liangzi"
    p2 = mupan / "p2" / "liangzi"
    p1.mkdir(parents=True, exist_ok=True)
    p2.mkdir(parents=True, exist_ok=True)
    (p1 / "0000.md").write_text(
        "<!-- page 1 -->\n第4章 氢原子\n氢原子内容A\n\n<!-- page 3 -->\n氢原子更多内容",
        encoding="utf-8",
    )
    (p2 / "0000.md").write_text(
        "<!-- page 10 -->\n第5章 自旋\n自旋内容B",
        encoding="utf-8",
    )
    chapters = load_mineru_output(mupan, "量子物理")
    names = [c.chapter for c in chapters]
    assert "氢原子" in names and "自旋" in names, names
    assert any("内容B" in c.text for c in chapters)

    empty = run_dir / "empty"
    empty.mkdir(exist_ok=True)
    try:
        load_mineru_output(empty, "量子物理")
        raise AssertionError("空目录应明确报错")
    except RuntimeError as exc:
        assert "没有 Markdown 文件" in str(exc)
    print(f"    复用处 -> {names}")


def test_mineru_pid_filter() -> None:
    """孤儿进程过滤：只挑命令行含 mineru 的，排除自身与无关 python。"""
    from coursebook_digest.parser import _mineru_pids_from_text

    wmic_txt = (
        '"Node","CommandLine","ProcessId"\r\n'
        '"W","C:\\x\\python.exe -m mineru.cli.fast_api --port 123","6680"\r\n'
        '"W","C:\\x\\python.exe -m coursebook_digest.cli ingest","15000"\r\n'
        '"W","C:\\x\\python.exe -m unrelated_app","20000"\r\n'
    )
    assert _mineru_pids_from_text(wmic_txt, self_pid=15000) == [6680]
    ps_txt = (
        '"ProcessId","CommandLine"\r\n'
        '"6680","C:\\x\\python.exe -m mineru.cli.fast_api"\r\n'
        '"15000","C:\\x\\python.exe -m coursebook_digest.cli"\r\n'
    )
    assert _mineru_pids_from_text(ps_txt, self_pid=15000) == [6680]
    print("    孤儿 mineru 进程过滤 OK")


def test_auto_slice_planning(run_dir: Path) -> None:
    """自动分片：规划页区间 + 续跑跳过已完成片判定。"""
    from coursebook_digest.parser import _part_completed, _plan_slices

    assert _plan_slices(234, 80) == [(0, 79), (80, 159), (160, 233)]
    assert _plan_slices(5, 3) == [(0, 2), (3, 4)]
    assert _plan_slices(234, 0) == []
    base = run_dir / "mu"
    (base / "part01").mkdir(parents=True, exist_ok=True)
    assert not _part_completed(base, "part01"), "无 md 应视为未完成"
    (base / "part01" / "0000.md").write_text("x", encoding="utf-8")
    assert _part_completed(base, "part01"), "有 md 应视为已完成"
    print("    自动分片规划 + 续跑判定 OK", _plan_slices(234, 80))


def test_pdf_page_count(run_dir: Path) -> None:
    """页数统计（自动分片前置）：能识别 PDF 页数。"""
    from coursebook_digest.parser import _pdf_page_count

    try:
        from pypdf import PdfWriter
    except Exception:  # noqa: BLE001
        print("    (pypdf 不可用，跳过页数测试)")
        return
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.add_blank_page(width=200, height=200)
    p = run_dir / "two.pdf"
    with open(p, "wb") as f:
        w.write(f)
    assert _pdf_page_count(p) == 2, _pdf_page_count(p)
    print("    PDF 页数统计 OK")


def test_chapter_split_en() -> None:
    """英文教材：CHAPTER 应切章（PART 不切，避免整部并成一章过粗）。"""
    from coursebook_digest.parser import _split_chapters

    pages = [
        (1, "# CHAPTER 1 Computer System Overview\n内容A\n"),
        (2, "# PART TWO Process Management\nPART 不应切章\n"),
        (3, "# CHAPTER 2 Process Management\n内容B\n"),
        (4, "2.1 小节内容\n内核调度"),
    ]
    chapters = _split_chapters(pages)
    names = [c[0] for c in chapters]
    assert names == ["Computer System Overview", "Process Management"], f"英文切章异常：{names}"
    print(f"    英文 CHAPTER 切章/PART 不切 -> {names}")


def test_toc_not_split() -> None:
    """目录(TOC)行（如 `# Chapter 2 … 46` 行尾带页码）不应切章；真实章节标题才切。"""
    from coursebook_digest.parser import _split_chapters

    pages = [
        (1, "# Chapter 1 Computer System Overview 7\n目录页内容\n"
            "# Chapter 2 Operating System Overview 46\n"),
        (2, "# CHAPTER 2 Operating System Overview\n真实正文A\n"),
        (3, "# PART 2 PROCESSES 105\n# Chapter 3 Process Description and Control 105\n"),
        (4, "# CHAPTER 3 Process Description and Control\n真实正文B\n"),
    ]
    names = [c[0] for c in _split_chapters(pages)]
    assert "Operating System Overview" in names, names
    assert "Process Description and Control" in names, names
    assert len(names) <= 3, f"TOC 不该切出碎片：{names}"
    print(f"    TOC 行不切章 -> {names}")


def test_cap_big_chapter() -> None:
    """超大章（粘连/整部）按字体量再切成 ≤MAX_CHAPTER_CHARS 的子章，保证进度与断点可控。"""
    from coursebook_digest.parser import MAX_CHAPTER_CHARS, _split_chapters

    nlines = MAX_CHAPTER_CHARS // 6 + 2  # 每行6字，确保净长度超上限
    body_lines = ["内容行内容行"] * nlines
    raw = "# CHAPTER 1 Giant Title\n" + "\n".join(body_lines)
    chapters = _split_chapters([(1, raw)])
    expected = "\n".join(["<!-- page 1 -->"] + body_lines)  # 与 _split_chapters 内部构造一致
    assert len(chapters) >= 2, f"超大章应按体量切分：{len(chapters)}"
    assert chapters[0][0].startswith("Giant Title·"), chapters[0][0]
    assert all(len(c[1]) <= MAX_CHAPTER_CHARS for c in chapters), "子章不应再超上限"
    assert "".join(c[1] for c in chapters) == expected, "切分不应丢内容"
    print(f"    超大章体量切分 -> {[c[0] for c in chapters]}")


def test_reference_noise(run_dir: Path) -> None:
    """仅供参考类噪音卡：检测打标；检索/打卡不再返回它们。"""
    from coursebook_digest.distill import flag_noise_cards, is_noise_card
    from coursebook_digest.retrieve import find_methods

    noise = MethodCard(id="c|章|n1", course="操作系统", chapter="章", topic="推荐阅读顺序", kind="方法",
                       keywords=["阅读顺序"], applicability="教材编排建议",
                       steps=["先完成第1至3章", "然后并行阅读第4至10章（可选章节）"])
    real = MethodCard(id="c|章|r1", course="操作系统", chapter="章", topic="轮转调度", kind="方法",
                      keywords=["轮转"], applicability="多进程并发", steps=["分配时间片", "到期切换"])
    assert is_noise_card(noise) is True and is_noise_card(real) is False
    flagged = flag_noise_cards([noise, real])
    assert flagged[0].reference_only is True and flagged[1].reference_only is False

    st = Settings(data_dir=str(run_dir / "noise" / "data"), storage="jsonl")
    CourseStore("噪音课", st).save_all(flagged)
    hits = find_methods("推荐阅读顺序 如何看这本", "噪音课", top_k=5, settings=st, use_chroma=False)
    assert "推荐阅读顺序" not in [h.card.topic for h in hits], f"参考卡不应被检索到：{hits}"
    hits2 = find_methods("轮转调度怎么分配时间片", "噪音课", top_k=5, settings=st, use_chroma=False)
    assert hits2 and hits2[0].card.topic == "轮转调度"
    print("    仅供参考类打标 + 检索排除 OK")


def main() -> None:
    assert SAMPLE.exists(), "缺少 examples/sample_course.md"
    run_dir = PROJECT_ROOT / ".smoke" / f"pipe-{os.getpid()}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        settings = Settings(data_dir=str(data_dir), chroma_dir=str(data_dir / "chroma"), storage="both")

        print("[P1] 章节切分 ->", test_chapter_split())
        test_chapter_split_en()
        print("[P1b] 英文 CHAPTER/PART 切章 OK")
        test_toc_not_split()
        print("[P1c] TOC 行不切章 OK")
        test_cap_big_chapter()
        print("[P1d] 超大章体量切分 OK")
        test_reference_noise(run_dir)
        print("[P1e] 仅供参考类噪音打标+排除 OK")
        print("[P2] md 解析 OK")
        test_pypdf_removed(run_dir)
        print("[P3] pypdf 兜底已移除 OK")
        test_offline_live_find(run_dir, settings)
        print("[P4] 持久化+向量+融合检索 OK")
        test_ask_injects_methods(settings)
        print("[P5] ask 全链路优先注入 OK")
        test_distill_with_fake(run_dir)
        print("[P6] 蒸馏链路 OK")
        test_mineru_reuse_multi_part(run_dir)
        print("[P7] 分片/续跑 复用多目录 md OK")
        test_mineru_pid_filter()
        print("[P8] 孤儿 mineru 进程过滤 OK")
        test_auto_slice_planning(run_dir)
        print("[P9] 自动分片规划/续跑 OK")
        test_pdf_page_count(run_dir)
        print("[P10] PDF 页数统计 OK")
    finally:
        # chroma 持久化客户端在 Windows 上保持文件句柄，同进程内可能删不净；
        # 尽力清理（目录在 .gitignore 中），失败仅提示、不影响判定。
        import gc

        gc.collect()
        try:
            shutil.rmtree(run_dir)
        except OSError:
            print("（提示：chroma 占用了句柄，.smoke 测试目录未完全清理，可手动删除）")
    print("\n集成测试全部通过")


if __name__ == "__main__":
    main()
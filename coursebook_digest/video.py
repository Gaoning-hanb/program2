# -*- coding: utf-8 -*-
"""B站视频 → 字幕 → 章节：把课程视频作为新的“教材源”接入蒸馏管线。

数据流（与教材 ingest 同构：识别 → 蒸馏 → 存储）::

    coursebook video <B站链接> --course 课程名
      1. yt-dlp 逐分P抓字幕（CC 字幕 / AI 字幕，json3 格式自带时间戳）
      2. 无字幕的分P：下载音频 → faster-whisper 本地转写（GPU 优先，CPU 兜底）
      3. 转写文本按时间窗插入 [mm:ss] 锚点 → 每个分P构造一个 Chapter
      4. 复用 distill 蒸馏（视频专用 prompt：容忍口语/同音错字、剔除三连闲聊）
      5. 卡片 source_page 记 P号+时间点，source_url 生成跳回原视频时刻的链接

依赖：``pip install yt-dlp faster-whisper``（后者仅无字幕兜底时需要；
B站自带字幕的视频完全用不到 whisper）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .parser import Chapter

# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """一句字幕：起止秒数 + 文本。"""

    start: float
    end: float
    text: str


@dataclass
class VideoPart:
    """一个分P（单P视频即 P1）的转写结果。"""

    part_no: int                 # P 号
    title: str
    duration: float              # 秒
    segments: list[Segment] = field(default_factory=list)
    source: str = ""             # cc / ai / whisper / ""
    lang: str = ""

    @property
    def chars(self) -> int:
        return sum(len(s.text) for s in self.segments)


# --------------------------------------------------------------------------- #
# yt-dlp 子进程封装
# --------------------------------------------------------------------------- #
_BVID_RE = re.compile(r"(BV[0-9A-Za-z]{10})")

# CC 字幕常见语言码 zh-Hans/zh-CN/zh；B站 AI 字幕为 ai-zh
_SUB_LANGS = "zh-Hans,zh-CN,zh,ai-zh"


def extract_bvid(url: str) -> str:
    """从任意形态的B站链接里抠出 BV 号。"""
    m = _BVID_RE.search(url)
    if not m:
        raise ValueError(f"不是有效的B站视频链接（未找到 BV 号）：{url}")
    return m.group(1)


def _run_ytdlp(args: list[str], timeout: int = 900) -> str:
    """跑 yt-dlp（同解释器环境），返回 stdout。字幕/元数据均为 UTF-8 JSON。"""
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", *args]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp 失败：{(proc.stderr or proc.stdout).strip()[:500]}")
    return proc.stdout


def _cookie_args(sessdata: str = "") -> list[str]:
    """yt-dlp 的 Cookie 头：SESSDATA（可选，AI 字幕需要）+ buvid 指纹。

    B站风控（412 Precondition Failed）的根因是请求缺 buvid3/buvid4 指纹
    cookie——裸 yt-dlp 在枚举多P列表时经常因此被拒。这里先访问首页预热
    拿 buvid3（buvid4 按B站自己的格式随机造一个），进程内缓存复用。
    """
    parts: list[str] = []
    if sessdata:
        parts.append(f"SESSDATA={sessdata}")
    warm = _warmup_cookies()
    if warm:
        parts.append(warm)
    return ["--add-headers", "Cookie: " + "; ".join(parts)] if parts else []


_warm_cache: dict = {"cookie": "", "ts": 0.0}


def _warmup_cookies() -> str:
    """首页预热 buvid 指纹 cookie；失败返回空串（网络异常时退回无 cookie）。"""
    import http.cookiejar
    import time
    import urllib.request

    now = time.time()
    if _warm_cache["cookie"] and now - _warm_cache["ts"] < 600:
        return _warm_cache["cookie"]
    try:
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        req = urllib.request.Request(
            "https://www.bilibili.com/",
            headers={"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/126.0.0.0 Safari/537.36")})
        opener.open(req, timeout=15).read(1024)
        names = {c.name: c.value for c in jar}
        if "buvid3" not in names:  # 首页没给就算了（罕见）
            return ""
        # buvid4 首页不下发，按B站格式造一个（32hex+infoc）
        import secrets
        names.setdefault("buvid4", secrets.token_hex(16) + "infoc")
        cookie = "; ".join(f"{k}={v}" for k, v in names.items())
    except Exception:  # noqa: BLE001 —— 预热失败不应阻断主流程
        return ""
    _warm_cache.update(cookie=cookie, ts=now)
    return cookie


def list_parts(url: str, sessdata: str = "", stop_after: int | None = None) -> list[dict]:
    """列出视频全部分P（单P视频返回一条）。

    优先走 yt-dlp flat-playlist（带 buvid 指纹 cookie，实测最稳）；
    B站 view API 风控较凶（412），且**失败会连带污染指纹**使随后的
    yt-dlp 请求也被拒，因此只作兜底。flat 模式标题常为空（分P名由
    抓字幕时的 ``--print`` 补齐），返回字段：part_no / title / duration / id。

    ``stop_after``：只关心前 N 个分P时（如用户指定了 --parts），逐P探测
    到 N 即止，大系列可省几分钟。
    """
    try:
        return _list_parts_ytdlp(url, sessdata)
    except Exception:  # noqa: BLE001 —— flat 被风控时回落 view API
        pass
    try:
        return _list_parts_api(url, sessdata)
    except Exception:  # noqa: BLE001 —— API 也被风控时逐P探测兜底
        return _list_parts_probe(extract_bvid(url), sessdata,
                                  stop_after=stop_after)


def _list_parts_probe(bvid: str, sessdata: str = "", max_parts: int = 300,
                      stop_after: int | None = None) -> list[dict]:
    """终极兜底：逐P单视频探测（``?p=N`` + ``--no-playlist``）。

    单视频端点对风控最宽容——实测 flat-playlist 与 view API 都被 412
    严打的时段它仍然全通。越界的 P 会报 "No video formats found" 自然
    终止；中途遇风控则带上已探测到的部分返回（总P数可能被低估）。
    每次探测带真实标题+时长（比 flat-playlist 的空标题还好）。
    """
    import time

    parts: list[dict] = []
    limit = min(stop_after, max_parts) if stop_after else max_parts
    for p in range(1, limit + 1):
        url = f"https://www.bilibili.com/video/{bvid}?p={p}"
        try:
            out = _run_ytdlp(
                ["--skip-download", "--no-playlist", "--quiet",
                 "--print", "%(title)s|%(duration)s",
                 *_cookie_args(sessdata), url], timeout=60)
        except RuntimeError as exc:
            if "No video formats found" in str(exc):
                break  # 越界：自然终点
            if p == 1:
                raise  # 连 P1 都被拒：彻底失败
            break  # 中途风控：拿到的部分仍可用
        lines = [ln.strip() for ln in out.splitlines() if "|" in ln]
        if not lines:
            if p == 1:
                raise RuntimeError("单视频探测未返回标题（P1）")
            break
        title, _, dur = lines[-1].rpartition("|")
        try:
            duration = float(dur or 0)
        except ValueError:
            duration = 0.0
        m = re.search(r"\s+p\d+\s+(.+)$", title)  # "<系列名> pNN <分P名>" → 分P名
        real = m.group(1).strip() if m else title.strip()
        parts.append({"part_no": p, "title": real or f"分P{p}",
                      "duration": duration, "id": bvid})
        if p < limit:
            time.sleep(2)  # 温柔一点，别自己触发风控
    if not parts:
        raise RuntimeError("逐P探测失败（P1 都拿不到）")
    return parts


def _list_parts_api(url: str, sessdata: str = "") -> list[dict]:
    import http.cookiejar
    import urllib.request

    bvid = extract_bvid(url)
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0.0.0 Safari/537.36"),
        "Referer": "https://www.bilibili.com/",
    }
    # 先访问首页拿 buvid3 等指纹 cookie——view 接口不带它们会 412 风控
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    req0 = urllib.request.Request("https://www.bilibili.com/", headers=headers)
    opener.open(req0, timeout=15).read(1024)
    cookies = "; ".join(f"{c.name}={c.value}" for c in jar)
    if sessdata:
        cookies = f"SESSDATA={sessdata}" + ("; " + cookies if cookies else "")
    if cookies:
        headers["Cookie"] = cookies
    req = urllib.request.Request(
        f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}", headers=headers)
    with opener.open(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != 0:
        raise RuntimeError(f"B站 API 返回错误：{data.get('message')}")
    pages = data["data"]["pages"]
    return [{
        "part_no": int(p["page"]),
        "title": str(p.get("part") or f"分P{p['page']}"),
        "duration": float(p.get("duration") or 0),
        "id": bvid,
    } for p in sorted(pages, key=lambda p: int(p["page"]))]


def _list_parts_ytdlp(url: str, sessdata: str = "") -> list[dict]:
    out = _run_ytdlp(
        ["--flat-playlist", "--dump-json", *_cookie_args(sessdata), url])
    entries: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("_type") == "playlist":
            entries.extend(data.get("entries") or [])
        else:
            entries.append(data)
    if not entries:
        raise RuntimeError("yt-dlp 未返回任何分P信息（链接可能无效或需要登录）")
    parts = []
    for i, e in enumerate(entries, 1):
        idx = e.get("playlist_index")
        parts.append({
            "part_no": int(idx) if idx else i,
            "title": str(e.get("title") or f"分P{i}"),
            "duration": float(e.get("duration") or 0),
            "id": str(e.get("id") or ""),
        })
    parts.sort(key=lambda p: p["part_no"])
    return parts


def fetch_part_subtitles(
    bvid: str, part_no: int, workdir: Path, sessdata: str = "",
) -> tuple[Path | None, str | None]:
    """抓取某个分P的字幕（CC 优先于 AI），顺带取回该分P的真实标题。

    返回 ``(字幕文件路径或 None, 分P标题或 None)``。标题靠 ``--print`` 在
    抓字幕的同一请求里拿到（yt-dlp 自带风控处理，view API 反而常 412）；
    多P系列标题形如 "<系列名> pNN <分P名>"，这里截出分P名。
    """
    part_url = f"https://www.bilibili.com/video/{bvid}?p={part_no}"
    prefix = workdir / f"p{part_no}"
    out = _run_ytdlp([
        "--skip-download", "--no-playlist", "--quiet",
        "--write-subs", "--write-auto-subs",
        "--sub-format", "json3", "--sub-langs", _SUB_LANGS,
        "--print", "%(title)s",
        "-o", str(prefix),
        *_cookie_args(sessdata),
        part_url,
    ])
    title: str | None = None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if lines:
        title = lines[-1]
        m = re.search(r"\s+p\d+\s+(.+)$", title)  # "<系列名> pNN <分P名>" → 分P名
        if m:
            title = m.group(1).strip()
    # 输出形如 p2.zh-Hans.json3 / p2.ai-zh.json3；CC(人工) 与 AI 并存时人工字幕优先
    files = sorted(
        workdir.glob(f"p{part_no}.*.json3"),
        key=lambda p: (".ai-" in p.name, p.name),  # ai 字幕排后
    )
    return (files[0] if files else None), title


def parse_json3(path: Path) -> tuple[list[Segment], str]:
    """解析 yt-dlp 的 json3 字幕 → Segment 列表（秒）+ 语言码。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    segs: list[Segment] = []
    for ev in data.get("events", []):
        text = "".join(s.get("utf8", "") for s in ev.get("segs") or [])
        text = text.replace("\n", " ").strip()
        if not text:
            continue
        t0 = ev.get("tStartMs", 0) / 1000
        t1 = t0 + ev.get("dDurationMs", 0) / 1000
        segs.append(Segment(t0, t1, text))
    # 文件名形如 p2.ai-zh.json3 → 语言码取倒数第二段
    stem_parts = path.stem.split(".")
    lang = stem_parts[1] if len(stem_parts) > 1 else ""
    return segs, lang


# --------------------------------------------------------------------------- #
# faster-whisper 本地转写兜底（无字幕视频）
# --------------------------------------------------------------------------- #
def _ensure_cuda_dlls() -> list:
    """把 pip 装的 nvidia-cublas-cu12 等 DLL 目录注册进搜索路径。

    ctranslate2 的 Windows 轮子自带 cudnn64_9.dll 但不带 cuBLAS（许可原因）；
    ``pip install nvidia-cublas-cu12`` 装到 site-packages/nvidia/cublas/bin。
    注意：ctranslate2 内部用普通 LoadLibrary 加载 cublas——它只认 PATH，
    不认 os.add_dll_directory（那仅作用于带 SEARCH 标志的加载），
    因此这里必须**同时**改写 PATH 并保留返回的句柄防止注册被回收。
    """
    handles: list = []
    if sys.platform != "win32":
        return handles
    import site

    dll_dirs: list[str] = []
    for sp in site.getsitepackages():
        nv = Path(sp) / "nvidia"
        if not nv.is_dir():
            continue
        for sub in nv.iterdir():
            bin_dir = sub / "bin"
            if bin_dir.is_dir():
                dll_dirs.append(str(bin_dir))
                handles.append(os.add_dll_directory(str(bin_dir)))
    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(dll_dirs + [os.environ.get("PATH", "")])
    return handles


# 句柄需存活到进程结束（被回收则 add_dll_directory 注册失效）
_CUDA_DLL_HANDLES: list = []


def _smoke_transcribe(model) -> None:
    """用 0.3 秒静音做功能性自检：DLL 缺失/驱动不兼容在这一步就暴露。

    注意 ``transcribe()`` 返回 ``(segments, info)`` 二元组，必须解包后再
    迭代 segments，否则生成器不执行、自检形同虚设（真实踩坑）。
    """
    import tempfile
    import wave

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "s.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 4800)
        segments, _info = model.transcribe(str(wav), language="zh", vad_filter=False)
        list(segments)


def _build_model(model_size: str) -> tuple[object, str]:
    """创建 whisper 模型：GPU 优先（含 DLL 注册与冒烟自检），失败自动回落 CPU。"""
    # 模型下载默认走国内镜像（huggingface 直连在国内常超时；可用环境变量覆盖）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from faster_whisper import WhisperModel

    try:
        global _CUDA_DLL_HANDLES
        _CUDA_DLL_HANDLES = _ensure_cuda_dlls()
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
            _smoke_transcribe(model)
            return model, "cuda"
    except Exception:  # noqa: BLE001 —— GPU 环境残缺属常态，静默降级 CPU
        pass
    return WhisperModel(model_size, device="cpu", compute_type="int8"), "cpu"


# whisper 对纯音乐段的高频"幻觉署名"（詞曲/作曲/演唱/字幕by…）；
# VAD 正常时会滤掉音乐，这里是漏网兜底：这些短语不可能出现在课程讲授里。
_MUSIC_CREDIT_RE = re.compile(
    r"(詞曲|词曲|作词|作詞|作曲|编曲|編曲|演唱|演奏|混音|和声|字幕\s*by|"
    r"subtitle|zither|harp|[♪♫])", re.IGNORECASE)


def _drop_music_hallucinations(segs: list[Segment]) -> list[Segment]:
    return [s for s in segs if not _MUSIC_CREDIT_RE.search(s.text)]


def transcribe_part(
    bvid: str, part_no: int, workdir: Path, model_size: str = "small",
    model_cache: dict | None = None,
) -> tuple[list[Segment], float]:
    """下载该分P音频并用 faster-whisper 转写，返回 ``(segments, 音频总秒数)``。

    直接取原生音频流（B站 bestaudio 本就是 m4a），不做 ffmpeg 转码，
    因此**不依赖本机安装 ffmpeg**；解码交给 faster-whisper 的 PyAV。
    ``model_cache`` 里会额外写一个 ``"__device__"`` 键供调用方报告设备。

    vad_filter=True 会自动跳过无语音段（录播课常含几十分钟课间音乐，
    实测 82 分钟视频只转写 22 分钟讲授、墙钟 59s 且零音乐幻觉）；
    返回音频总时长供调用方计算覆盖率、向用户解释"后半没内容"。
    """
    part_url = f"https://www.bilibili.com/video/{bvid}?p={part_no}"
    audio_exts = {".m4a", ".mp3", ".aac", ".ogg", ".opus", ".flac", ".wav"}
    cached = [p for p in workdir.glob(f"p{part_no}.*") if p.suffix.lower() in audio_exts]
    if cached:
        audio = cached[0]
    else:
        _run_ytdlp([
            "-f", "bestaudio", "--no-playlist",
            "-o", str(workdir / f"p{part_no}.%(ext)s"),
            *_cookie_args(),
            part_url,
        ])
        cached = [p for p in workdir.glob(f"p{part_no}.*") if p.suffix.lower() in audio_exts]
        if not cached:
            raise RuntimeError("音频下载失败（未产出音频文件）")
        audio = cached[0]
    try:
        if model_cache is not None and model_size in model_cache:
            model = model_cache[model_size]
        else:
            model, device = _build_model(model_size)
            if model_cache is not None:
                model_cache[model_size] = model
                model_cache["__device__"] = device
    except ImportError as exc:
        raise RuntimeError(
            "无字幕视频需要本地转写：pip install faster-whisper 后重试，"
            "或用 --skip-whisper 跳过") from exc

    it, info = model.transcribe(
        str(audio), language="zh", vad_filter=True, beam_size=5,
        condition_on_previous_text=False,
    )
    segs = _drop_music_hallucinations(
        [Segment(s.start, s.end, s.text.strip()) for s in it if s.text.strip()])
    audio.unlink(missing_ok=True)  # 音频用完即删（转写文本已缓存）
    return segs, float(getattr(info, "duration", 0) or 0)


# --------------------------------------------------------------------------- #
# 转写文本 → 带时间戳锚点的章节
# --------------------------------------------------------------------------- #
def _fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


_TS_RE = re.compile(r"(?:(\d{1,2}):)?(\d{1,3}):(\d{2})")


def _ts_to_sec(ts: str) -> int | None:
    """'12:35' → 755；'1:02:33' → 3753；解析失败返回 None。"""
    m = _TS_RE.search(ts)
    if not m:
        return None
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def segments_to_text(segments: list[Segment], anchor_every: float = 45.0) -> str:
    """把 Segment 流合并成带 [mm:ss] 段首锚点的转写正文（供蒸馏+溯源）。"""
    if not segments:
        return ""
    paras: list[str] = []
    buf: list[str] = []
    para_start = segments[0].start
    for seg in segments:
        if buf and seg.start - para_start >= anchor_every:
            paras.append(f"[{_fmt_ts(para_start)}] " + "".join(buf))
            buf = []
            para_start = seg.start
        buf.append(seg.text)
    if buf:
        paras.append(f"[{_fmt_ts(para_start)}] " + "".join(buf))
    return "\n".join(paras)


def part_to_chapter(part: VideoPart, course: str, min_chars: int = 400) -> Chapter | None:
    """一个分P → 一个章节；过小（片头/空P）返回 None。"""
    text = segments_to_text(part.segments)
    if len(text) < min_chars:
        return None
    title = re.sub(r'[\\/:*?"<>|\r\n]', " ", part.title).strip() or f"P{part.part_no}"
    return Chapter(course=course, chapter=f"P{part.part_no} {title}"[:80], text=text)


def attach_source(cards: list, part: VideoPart, bvid: str) -> None:
    """给该分P蒸馏出的卡片补视频溯源：source_page=P号+时间点，source_url=跳转链接。"""
    base = f"https://www.bilibili.com/video/{bvid}?p={part.part_no}"
    for c in cards:
        # 蒸馏时填的是 [mm:ss] 锚点（带方括号），溯源显示时去掉括号
        ts = (c.source_page or "").strip().strip("[]").strip()
        sec = _ts_to_sec(ts)
        c.source_page = f"P{part.part_no} {ts}".strip() if ts else f"P{part.part_no}"
        c.source_url = f"{base}&t={sec}" if sec is not None else base


# --------------------------------------------------------------------------- #
# 分P选择解析：'1-3,5' → [1,2,3,5]
# --------------------------------------------------------------------------- #
def parse_parts_spec(spec: str, total: int) -> list[int] | None:
    """解析 --parts 参数；空串返回 None（= 全部）。越界的部分自动截到范围内。"""
    spec = spec.strip()
    if not spec:
        return None
    out: set[int] = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, _, b = piece.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError as exc:
                raise ValueError(f"--parts 格式错误：{piece!r}（应形如 1-3,5）") from exc
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(piece))
            except ValueError as exc:
                raise ValueError(f"--parts 格式错误：{piece!r}（应形如 1-3,5）") from exc
    return sorted(n for n in out if 1 <= n <= total) or None


# --------------------------------------------------------------------------- #
# 视频蒸馏 prompt（与教材版同构，差异：口语转写容错 + 视频噪音剔除 + 时间戳溯源）
# --------------------------------------------------------------------------- #
VIDEO_SYSTEM_PROMPT = """你是一位严谨的大学课程学习助教。目标是【把课程视频的语音转写文本蒸馏成方法卡片】，供考试答题时优先复用课程讲授的逻辑与运算技巧。

输出必须是 JSON 对象，结构如下：
{"cards": [ {每张卡} ]}

每张卡片的字段（严格遵守）：
- "id": 形如 "{{course}}|{{chapter}}|序号"
- "course": "{{course}}"
- "chapter": "{{chapter}}"
- "topic": 主题/知识点名，一句话（用正确的书面术语）
- "kind": 取 概念/定理/方法/例题/易错点 之一
  - 概念=新名词与定义；定理=命题/公式/性质；方法=解题套路与运算技巧（重点产出）；
    例题=典型例题含关键解法；易错点=老师强调的坑
- "keywords": 检索用关键词数组，尽量多收录：术语、题型、公式别名、常见考试叫法（写正确书面术语）
- "prerequisites": 前置概念数组
- "applicability": 适用条件——什么情况下用这个方法（请写具体）
- "steps": 标准步骤数组，按顺序可执行；非方法类可给"如何理解/记忆"步骤
- "core_formula_latex": 核心公式（LaTeX）；没有就空字符串
- "technique": 常用技巧/思路点睛，一句话；没有就空字符串
- "worked_example": 代表例题及其关键解法（题干+步骤要点），没有就空字符串
- "error_notes": 常见错误/易错点数组
- "source_page": 该内容出现的视频时间戳——原样填入文本中离它最近的 [mm:ss] 或 [h:mm:ss] 锚点；没有就留空

与教材蒸馏的差异（重要）：
- 输入是语音转写字幕：口语化、可能有同音错别字（如"静太区"应为"静默区"、"进程快"应为"进程块"）。请按上下文纠正理解后抽取，topic/keywords 一律用正确的书面术语；
- **排除视频编排性内容，不要产卡**：开场问候、求三连/关注/弹幕互动、UP主与课程介绍、学习规划闲聊、下节预告、结尾感谢；这些不是知识点；
- 老师口头重复强调、板书推导、举例扩展的内容照常抽取，例题卡尽量保留老师的完整解题过程。

要求：
- 只依据给定文本抽取，不要编造视频没讲的内容；拿不准就留空或不产出
- 方法类卡片是重点，宁可多产几张小而准的方法卡，也不要糊成一大段
- 不要做总结或评价，直接输出 JSON"""

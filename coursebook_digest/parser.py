"""教材解析：PDF / Markdown / 纯文本 → 按章节切分的文本。

- ``mineru``：PDF 的唯一解析后端家族（公式/扫描件/复杂排版）。
  - ``mineru``：调用本机 mineru CLI（GPU/本地进程）；
  - ``mineru-http``：调用学校网关云端 MinerU（``POST /mineru/file_parse``，
    复用模型 API key，不依赖本机 mineru）。实现见 ``mineru_http.py``。
  **已移除 pypdf 纯文本兜底**——PDF 一律走 mineru 类后端，不静默退化。
- Markdown / 纯文本：直接读取作为一章。
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings

# 中文教材章节标题启发式：兼容 "第X章"/"## 第X章"/"# 第三章" 等 markdown 标题
# 章节标题启发式：中文 "第X章/篇/讲/节" + 英文 "CHAPTER N / CH. N / UNIT / MODULE"
# 注意：PART 不参与切分（PART 是"部"，会整部并成一章过粗，交给真实章节标题切）
_CHAPTER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:"
    r"第\s*[0-9一二三四五六七八九十百零]+(?:章|篇|讲|节)\s*[:\s]*"
    r"|(?:CHAPTER|CH\.|UNIT|MODULE)\s+[A-Z0-9]+\s*[:\s]*"
    r")(.*)$",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+[\u4e00-\u9fa5A-Za-z].*$")

# 目录(TOC)行的判别：章节标题行若以「空格+页码」结尾，多半是目录索引项而非正文标题
_TOC_PAGE_SUFFIX_RE = re.compile(r"\s[0-9]{1,4}\s*$")

PAGE_MARK_RE = re.compile(r"<!-- page (\d+) -->")


@dataclass
class Chapter:
    """一段可蒸馏的教材文本。"""

    course: str
    chapter: str
    text: str
    pages: str = ""


# 超大章体量上限：超过则按字数切成“章名·n”，保证每章可控——
# 蒸馏按章落盘，巨章(如几十万字的部/粘连章)会导致长时间无进度且断点损失大。
MAX_CHAPTER_CHARS = 50_000


def _split_chapters(pages: list[tuple[int, str]]) -> list[tuple[str, str, list[int]]]:
    """按“第X章/CHAPTER/UNIT/MODULE”标题把按页码拼接的文本切成章节；没有标题则整体算一章。

    - 目录(TOC)页会以“# Chapter 2 … 46”这种**行尾带页码**的标题列出全书章节，
      与真实章节标题（几乎不以数字结尾）区分：行尾带页码视为目录行，不切分。
    - PART 不参与切分（过粗）。
    - 超过 ``MAX_CHAPTER_CHARS`` 的超大章，按体量再切成“章名·n”的子章。
    """
    chapters: list[tuple[str, str, list[int]]] = []
    cur_name = "(未分章)"
    cur_pages: list[int] = []
    cur_buf: list[str] = []

    def flush() -> None:
        if cur_buf:
            chapters.append((cur_name, "\n".join(cur_buf), list(cur_pages)))

    for num, text in pages:
        lines = text.splitlines()
        for line in lines:
            m = _CHAPTER_RE.match(line)
            if m:
                name = m.group(1).strip()
                if not _TOC_PAGE_SUFFIX_RE.search(m.group(0)):  # 非“目录+页码” → 真章节
                    flush()
                    cur_name = name or f"第{num}页章节"
                    cur_pages = [num]
                    cur_buf = [f"<!-- page {num} -->"]
                    continue
            cur_buf.append(line)  # 目录行/普通行：留在当前章
        if not cur_pages or cur_pages[-1] != num:
            cur_pages.append(num)
    flush()

    out: list[tuple[str, str, list[int]]] = []
    for name, text, plist in chapters:
        if len(text) <= MAX_CHAPTER_CHARS:
            out.append((name, text, plist))
            continue
        n = (len(text) + MAX_CHAPTER_CHARS - 1) // MAX_CHAPTER_CHARS
        seg = (len(text) + n - 1) // n
        for i in range(n):
            out.append((f"{name}·{i + 1}", text[i * seg:(i + 1) * seg], []))
    return out


# --------------------------------------------------------------------------- #
# MinerU 后端（PDF 唯一后端）
# --------------------------------------------------------------------------- #
def _mineru_available() -> bool:
    return shutil.which("mineru") is not None or shutil.which("magic-pdf") is not None


def parse_pdf_mineru(pdf_path: str | Path, course: str, out_dir: str | Path) -> list[Chapter]:
    """调用 mineru CLI 解析 PDF，产出 Markdown 后再按我们自己的章节启发式切分。"""
    if not _mineru_available():
        raise RuntimeError(
            "未检测到 mineru。安装命令：conda activate firpro && pip install 'mineru>=2.0'"
        )
    cli = shutil.which("mineru") or shutil.which("magic-pdf")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [cli, "-p", str(pdf_path), "-o", str(out_dir)],
        check=True,
    )
    return load_mineru_output(out_dir, course)


def _pdf_page_count(pdf_path: str | Path) -> int:
    """统计 PDF 总页数（仅用于自动分片规划，不属于文本解析路径）。

    优先用 pypdf；不可用时退回正则统计（bookmark 也不可靠，故失败时明确报错）。
    """
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:  # noqa: BLE001
        data = Path(pdf_path).read_bytes()
        m = re.findall(rb"/Type\s*/Page(?!s)\b", data)
        if not m:
            raise RuntimeError(
                "无法确定 PDF 页数（可能使用了压缩对象流）。"
                "请安装 pypdf：conda activate firpro && pip install pypdf"
            )
        return len(m)


def _plan_slices(total_pages: int, slice_size: int) -> list[tuple[int, int]]:
    """把 [0, total_pages) 切成 0 起始（含端点）的若干页区间。"""
    if total_pages <= 0 or slice_size <= 0:
        return []
    return [
        (start, min(start + slice_size, total_pages) - 1)
        for start in range(0, total_pages, slice_size)
    ]


def _part_completed(out_dir: Path, part_name: str) -> bool:
    """判断某个分片目录是否已产出 Markdown（用于续跑跳过）。"""
    p = out_dir / part_name
    return p.is_dir() and any(p.rglob("*.md"))


def parse_pdf_mineru_sliced(
    pdf_path: str | Path,
    course: str,
    out_dir: str | Path,
    slice_size: int,
) -> list[Chapter]:
    """自动分片解析：把 PDF 按 ``slice_size`` 页切成多个独立 mineru 任务。

    - 每个分片独立走 ``mineru -s 起始 -e 结束``，**完成即落盘**——规避 mineru
      单任务约 1 小时总超时；某个分片失败只损失该片。
    - 天然续跑：已产出 Markdown 的分片会自动跳过，重跑同一条命令即可续。
    - 全部（剩余）分片完成后，合并输出目录下所有 Markdown 再切章蒸馏。
    """
    if not _mineru_available():
        raise RuntimeError(
            "未检测到 mineru。安装命令：conda activate firpro && pip install 'mineru>=2.0'"
        )
    cli = shutil.which("mineru") or shutil.which("magic-pdf")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import sys

    total = _pdf_page_count(pdf_path)
    plan = _plan_slices(total, slice_size)
    if not plan:
        raise RuntimeError(f"无法分片：总页数={total}，每片={slice_size} 页")
    for idx, (start, end) in enumerate(plan, 1):
        part = out_dir / f"part{idx:02d}"
        part.mkdir(parents=True, exist_ok=True)
        if _part_completed(out_dir, part.name):
            sys.stderr.write(f"[分片] 跳过已完成 {part.name}（页 {start + 1}-{end + 1}）\n")
            continue
        sys.stderr.write(
            f"[分片] 处理 {part.name}：页 {start + 1}-{end + 1} / 共 {len(plan)} 片\n"
        )
        try:
            subprocess.run(
                [cli, "-p", str(pdf_path), "-o", str(part),
                 "-s", str(start), "-e", str(end)],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"分片 {part.name}（页 {start + 1}-{end + 1}）解析失败：{exc}。\n"
                "已完成的分片已保留在输出目录；重跑同一条命令会自动跳过它们（续跑）。"
            ) from exc
    return load_mineru_output(out_dir, course)


def load_mineru_output(out_dir: str | Path, course: str) -> list[Chapter]:
    """从 mineru 输出目录(可含多个子目录/分片)递归加载所有 Markdown，合并切章。

    用途：mineru 支持 ``-s start -e end`` 按页解析，可分批产出到不同子目录；
    也支持"复用已解析结果/续跑"。本函数把目录树下所有 ``*.md`` 当作一个
    文档流（按页锚点拼回顺序），再走统一的章节切分。若目录下没有 md 则明确报错。
    """
    out_dir = Path(out_dir)
    mds = sorted(out_dir.rglob("*.md"))
    if not mds:
        raise RuntimeError(
            f"mineru 输出目录 {out_dir} 下没有 Markdown 文件。"
            "若想复用/续跑，请先用 mineru 分片产出；或改用它自动运行。"
        )
    merged: list[tuple[int, str]] = []
    for md in mds:  # 把 mineru 产出的多个 md 当作一个文档流
        merged.extend(_read_md_pages(md))
    out: list[Chapter] = []
    for name, text, plist in _split_chapters(merged):
        out.append(Chapter(course=course, chapter=name, text=text, pages=",".join(map(str, plist))))
    return out


def _read_md_pages(md: Path) -> list[tuple[int, str]]:
    text = md.read_text(encoding="utf-8")
    # 把 <!-- page N --> 标记转成分页结构（mineru 会输出这类锚点）
    parts = re.split(r"<!--\s*page\s+(\d+)\s*-->", text)
    if len(parts) == 1:
        return [(1, text)]
    out: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        body = parts[i + 1].strip()
        if body:
            out.append((num, body))
    return out or [(1, text)]


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
def _resolve_pdf_parser(parser: str, cfg: Settings) -> str:
    """把 ``auto`` 解析成具体后端：优先用配置 ``default_parser`` 覆盖；仍是
    auto 时——云端有 key 且本机无 mineru → 云端(mineru-http)，否则本地(mineru)。"""
    if parser != "auto":
        return parser
    dp = cfg.default_parser
    if dp != "auto":
        return dp
    cloud_ok = bool((cfg.mineru_api_key or cfg.llm_api_key).strip())
    if cloud_ok and not _mineru_available():
        return "mineru-http"
    return "mineru"


def parse_source(
    source: str | Path,
    course: str,
    parser: str = "auto",
    out_dir: str | Path | None = None,
    reuse_out: bool = False,
    slice_size: int = 0,
    settings: Settings | None = None,
    batch_files: int | None = None,
) -> list[Chapter]:
    """按文件类型/解析器分派。Markdown/纯文本直接读；PDF 只走 mineru 类后端。

    ``parser``：``auto``（按配置/环境择一端）/ ``mineru``（本地 CLI）/
    ``mineru-http``（学校网关云端解析，不依赖本机 mineru）。
    PDF **无纯文本兜底**；``mineru-http`` 时 ``slice_size`` 用作云端分片页数、
    ``batch_files`` 用作单请求最大分片数（留空走 .env 配置）。

    ``reuse_out=True`` 时不重新运行解析，而是直接加载 ``out_dir`` 下已产出的
    Markdown（用于复用/分片续跑，云端本地产物皆可）。
    ``slice_size>0`` 对本地后端启用自动分片；对云端后端作为每片页数。
    """
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(src)
    suffix = src.suffix.lower()
    OFFICE_SUFFIXES = {".pptx", ".ppt", ".docx", ".doc"}
    if suffix == ".pdf" or (suffix in OFFICE_SUFFIXES and parser in ("mineru-http", "auto")):
        if suffix == ".pdf":
            if parser not in ("auto", "mineru", "mineru-http"):
                raise ValueError(
                    f"不支持的 PDF 解析器：{parser!r}。已移除纯文本(pypdf)兜底，"
                    "PDF 支持 mineru（本地）/ mineru-http（云端）。"
                )
            cfg = settings or get_settings()
            effective = _resolve_pdf_parser(parser, cfg)
        else:  # PPT/DOCX：云端整文件直传；本地 mineru 分支暂只接 PDF
            if parser == "mineru":
                raise ValueError(
                    "本地 mineru 分支（--parser mineru）暂只支持 PDF；"
                    "PPT/DOCX 请用 --parser mineru-http 走云端直传。"
                )
            cfg = settings or get_settings()
            effective = "mineru-http"
        target = out_dir or src.parent / "mineru_out"
        if reuse_out:
            return load_mineru_output(target, course)
        if effective == "mineru-http":
            from .mineru_http import parse_pdf_mineru_http

            return parse_pdf_mineru_http(
                src, course, target, settings=cfg,
                chunk_pages=slice_size if slice_size and slice_size > 0 else None,
                batch_files=batch_files if batch_files and batch_files > 0 else None,
            )
        if not _mineru_available():
            raise RuntimeError(
                "未检测到 mineru：PDF 解析依赖 mineru（纯文本兜底已移除）。"
                "安装：conda activate firpro && pip install 'mineru>=2.0'；"
                "或改用云端后端：--parser mineru-http（需在 .env 配好 API key）。"
            )
        if slice_size and slice_size > 0:
            return parse_pdf_mineru_sliced(src, course, target, slice_size)
        return parse_pdf_mineru(src, course, target)
    # 文本类：md / txt 整体当作一章
    text = src.read_text(encoding="utf-8")
    chapter = _CHAPTER_RE.match(text.splitlines()[0]) if text.splitlines() else None
    name = chapter.group(1).strip() if chapter else src.stem
    return [Chapter(course=course, chapter=name, text=text, pages="")]


def filter_chapters(chapters: list[Chapter], pattern: str | None) -> list[Chapter]:
    """按章节名子串过滤（如 --chapter '第3章' 或关键词）。"""
    if not pattern:
        return chapters
    pat = re.compile(re.escape(pattern))
    return [c for c in chapters if pat.search(c.chapter)]


# --------------------------------------------------------------------------- #
# mineru 孤儿进程清理（Ctrl+C / 关窗 常留下继续烧 GPU 的服务进程）
# --------------------------------------------------------------------------- #
def _mineru_pids_from_text(text: str, self_pid: int) -> list[int]:
    """从进程列表文本（CSV）里挑出命令行含 ``mineru`` 的 python 进程 PID（排除自身）。

    兼容 ``wmic process get ProcessId,CommandLine /format:csv`` 与
    PowerShell ``Get-CimInstance ... | ConvertTo-Csv`` 两种输出。纯函数，便于测试。
    """
    pids: list[int] = []
    reader = csv.reader(text.splitlines())
    rows = list(reader)
    header = rows[0] if rows else []
    try:
        id_idx = header.index("ProcessId")
        cl_idx = header.index("CommandLine")
    except ValueError:  # 没有标准表头：宽松地按"含 mineru 的行里第一个数字"挑
        for row in rows:
            joined = " ".join(row)
            if "mineru" not in joined:
                continue
            for cell in row:
                cell = cell.strip()
                if cell.isdigit() and int(cell) != self_pid:
                    pids.append(int(cell))
                    break
        return pids
    for row in rows[1:]:
        if len(row) <= max(id_idx, cl_idx):
            continue
        if "mineru" not in row[cl_idx]:
            continue
        cell = row[id_idx].strip()
        if cell.isdigit() and int(cell) != self_pid:
            pids.append(int(cell))
    return pids


def kill_mineru_processes() -> int:
    """杀掉残留的 mineru 进程并返回杀掉的个数。

    mineru 的解析服务是独立子进程，Ctrl+C/关窗时主进程退出后它常继续占用 GPU。
    这里按"命令行含 mineru"精确匹配 python 进程，再 ``taskkill /F /T`` 连带子进程树。
    在每次 ingest 开始前调用可清掉上次中断的残留；在 KeyboardInterrupt 时兜底。
    """
    self_pid = os.getpid()
    texts: list[str] = []
    # 1) wmic（首选，Win10/11 默认自带）
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=30,
        )
        if "ProcessId" in (out.stdout or ""):
            texts.append(out.stdout or "")
    except Exception:  # noqa: BLE001
        pass
    # 2) 兜底：PowerShell CIM
    if not texts:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation"],
                capture_output=True, text=True, timeout=30,
            )
            texts.append(out.stdout or "")
        except Exception:  # noqa: BLE001
            pass
    pids: list[int] = []
    for text in texts:
        pids.extend(_mineru_pids_from_text(text, self_pid))
    killed = 0
    for pid in dict.fromkeys(pids):  # 去重
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True, timeout=20,
            )
            killed += 1
        except Exception:  # noqa: BLE001
            pass
    return killed
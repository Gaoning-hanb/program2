"""云端 MinerU 解析后端（学校网关 mineru 服务，`POST /mineru/file_parse`）。

与本地 mineru CLI 后端等效：把 PDF **按文件**拆片（云端只能传文件、不能传页区间）
→ 按批批量 multipart POST → 逐片取回 ``results.<stem>.md_content`` 落盘到
``out_dir/partNN/<stem>.md`` → 复用 ``load_mineru_output`` 统一合并/切章/入库。

特点：
- 鉴权复用模型 API key（``DEEPSEEK_API_KEY``，与 LLM 同一网关同一 Bearer），
  也可单独配 ``MINERU_API_KEY`` 覆盖；**不依赖本机 mineru 安装 / 无本地进程**。
- 单请求可带多个文件（``files`` 字段可重复），响应 ``results`` 按文件名独立返回。
- 天然续跑：已落盘 ``<stem>.md`` 的片直接跳过，中断后重跑同命令即续传。
- 云端返回的 markdown **不含 ``<!-- page N -->`` 锚点**：切章仍按“第X章”进行，
  只是 ``Chapter.pages`` 为空元数据（不影响蒸馏/检索/作答）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .config import Settings, get_settings
from .parser import Chapter, _pdf_page_count, load_mineru_output

DEFAULT_PARSE_PATH = "/mineru/file_parse"
ProgressFn = Callable[[str], None]

# MinerU 原生支持的文件类型 → 上传 Content-Type（pdf 拆片；pptx/docx/图片等整文件直传）
_CTYPES = {
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class MineruHttpError(RuntimeError):
    """云端解析失败（未配置 key / 网络 / 鉴权 / 额度 / 任务未完成）。"""


def _default_progress(msg: str) -> None:
    sys.stderr.write(f"[云端解析] {msg}\n")


# --------------------------------------------------------------------------- #
# 分片
# --------------------------------------------------------------------------- #
def _split_pdf(pdf_path: Path, chunk_pages: int, work_dir: Path) -> list[Path]:
    """用 pypdf 把 PDF 拆成 ≤chunk_pages 页的小文件，按页序命名。

    文件名带零填充起止页（``<stem>_p00000_00029.pdf``），保证按名排序 == 按页排序；
    云端 md 无页锚点时，合并顺序就靠这个排序保持一致。
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - 环境缺依赖时的友好报错
        raise MineruHttpError(
            "云端分片需要 pypdf：conda activate firpro && pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    chunks: list[Path] = []
    for start in range(0, total, chunk_pages):
        end = min(start + chunk_pages, total)
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        name = f"{pdf_path.stem}_p{start:05d}_{end - 1:05d}.pdf"
        out = work_dir / name
        with open(out, "wb") as f:
            writer.write(f)
        chunks.append(out)
    return chunks


# --------------------------------------------------------------------------- #
# HTTP：multipart POST + 结果整理
# --------------------------------------------------------------------------- #
def _build_multipart(url: str, key: str, files: list[Path],
                     path: str = DEFAULT_PARSE_PATH) -> tuple[Any, str]:
    """构造 multipart/form-data 请求体（files 可重复 + return_md / response_format_zip）。"""
    boundary = f"----dsh-mineru-{os.getpid()}-{int(time.time() * 1_000_000):x}"

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n"
        ).encode("utf-8")

    def file_part(path: Path) -> bytes:
        data = path.read_bytes()
        ctype = _CTYPES.get(path.suffix.lower(), "application/octet-stream")
        head = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"files\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode("utf-8")
        return head + data + b"\r\n"

    body = b"".join([file_part(f) for f in files])
    body += field("return_md", "true") + field("response_format_zip", "false")
    body += f"--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        url + path,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    return request, boundary


def _parse_response(raw: bytes) -> dict:
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise MineruHttpError(f"云端返回非 JSON（前 200 字节）：{raw[:200]!r}") from exc
    if not isinstance(data, dict):
        raise MineruHttpError(f"云端返回结构异常：{data!r}")
    return data


# 瞬时性 HTTP 状态（限流/上游抖动）：自动退避重试；4xx 额度/鉴权类直接失败
_TRANSIENT_HTTP = {429, 500, 502, 503, 504}
_RETRY_ATTEMPTS = 4


def _post_batch(
    base_url: str,
    key: str,
    files: list[Path],
    settings: Settings,
    urlopen: Callable[..., Any] | None = None,
    path: str = DEFAULT_PARSE_PATH,
) -> dict:
    """一次批量 POST；返回整理好的 ``{stem: md_content}``（已完成时）。

    对 429/5xx 瞬时错误自动退避重试 ``_RETRY_ATTEMPTS`` 次；4xx（鉴权/额度）之类
    直接包装为 ``MineruHttpError`` 失败。
    """
    sender = urlopen or urllib.request.urlopen
    request, _ = _build_multipart(base_url, key, files, path=path)
    last: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            with sender(request, timeout=600) as resp:
                data = _parse_response(resp.read())
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001
                pass
            if exc.code in _TRANSIENT_HTTP and attempt < _RETRY_ATTEMPTS:
                last = exc
                wait = min(2 ** attempt, 30)
                _default_progress(
                    f"请求遇 {exc.code}（瞬时错误），{wait}s 后第 {attempt} 次重试…"
                )
                time.sleep(wait)
                continue
            raise MineruHttpError(
                f"云端解析失败 HTTP {exc.code} {exc.reason}: {detail}"
            ) from exc
        except MineruHttpError:
            raise
        except Exception as exc:  # 网络层错误统一包装
            raise MineruHttpError(f"云端请求失败：{type(exc).__name__}: {exc}") from exc

        sent = {f.stem for f in files}
        status = data.get("status")
        if status == "completed":
            return _extract_results(data, sent)
        # 未立即完成（大文件可能异步）：轮询 status_url（网关内网地址不可达时明确报错）
        return _settle_task(data, settings, sent)
    raise MineruHttpError(
        f"云端解析失败（重试 {_RETRY_ATTEMPTS - 1} 次后仍为瞬时错误）：{last}"
    ) from last


def _extract_results(data: dict, sent: set[str]) -> dict:
    results = data.get("results") or {}
    out: dict[str, str] = {}
    for stem in sent:
        item = results.get(stem)
        if item is None:
            raise MineruHttpError(
                f"云端未返回文件 {stem} 的结果（results 键={list(results)}）"
            )
        md = (item or {}).get("md_content") or ""
        if not md:
            raise MineruHttpError(f"云端返回 {stem} 的 markdown 为空")
        out[stem] = md
    return out


def _settle_task(data: dict, settings: Settings, sent: set[str]) -> dict:
    task_id = data.get("task_id", "?")
    status_url = data.get("status_url") or ""
    for attempt in range(1, settings.mineru_poll_attempts + 1):
        time.sleep(settings.mineru_poll_seconds)
        try:
            req = urllib.request.Request(status_url, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                info = _parse_response(resp.read())
            if info.get("status") == "completed":
                return _extract_results(info, sent)
            if attempt >= settings.mineru_poll_attempts:
                break
        except Exception as exc:  # noqa: BLE001
            raise MineruHttpError(
                f"任务 {task_id} 未在 POST 内完成，且无法轮询 {status_url}："
                f"{type(exc).__name__}: {exc}\n"
                "请减小分片页数（--slice-pages / MINERU_CHUNK_PAGES），"
                "或用返回的 task_id 在网关侧查询。"
            ) from exc
    raise MineruHttpError(
        f"任务 {task_id} 轮询 {settings.mineru_poll_attempts} 次仍未完成"
    )


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def _part_dir(out: Path, chunk_index: int, batch_files: int) -> Path:
    """分片所属落盘目录：按“第几个分片 ÷ 每批数量”稳定编号，跨运行可续跑对齐。"""
    return out / f"part{chunk_index // batch_files + 1:02d}"


def _committed(out: Path, chunk_index: int, batch_files: int, stem: str) -> bool:
    return (_part_dir(out, chunk_index, batch_files) / f"{stem}.md").exists()


def parse_pdf_mineru_http(
    pdf_path: str | Path,
    course: str,
    out_dir: str | Path,
    settings: Settings | None = None,
    chunk_pages: int | None = None,
    batch_files: int | None = None,
    urlopen: Callable[..., Any] | None = None,
    progress: ProgressFn | None = None,
) -> list[Chapter]:
    """云端解析：拆片→批量上传→逐片落盘 md→复用 ``load_mineru_output`` 切章。"""
    cfg = settings or get_settings()
    key = (cfg.mineru_api_key or cfg.llm_api_key).strip()
    if not key:
        raise MineruHttpError(
            "未配置云端 MinerU key：.env 里填 DEEPSEEK_API_KEY，或单独设 MINERU_API_KEY"
        )
    base_url = cfg.mineru_base_url.rstrip("/")
    n_chunk = chunk_pages or cfg.mineru_chunk_pages
    n_batch = batch_files or cfg.mineru_batch_files
    say = progress or _default_progress

    src = Path(pdf_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / ".chunks"
    work.mkdir(parents=True, exist_ok=True)

    suffix = src.suffix.lower()
    if suffix == ".pdf":
        total = _pdf_page_count(src)
        chunks = _split_pdf(src, n_chunk, work)
        say(f"共 {total} 页 → 拆 {len(chunks)} 片（每片 ≤{n_chunk} 页，每批 ≤{n_batch} 文件）")
    else:
        # PPT/DOCX/图片：MinerU 原生支持，整文件直传，不分片。
        # 超大体量（如数百 MB 的巨型 pptx）若网关拒收，可先导出为 PDF 再走拆片路径。
        chunks = [src]
        say(f"文件直传（{suffix}，{src.stat().st_size // 1024} KB，无分片）")

    if chunks:
        n_batches = (len(chunks) + n_batch - 1) // n_batch
        say(f"共 {n_batches} 批；已落盘的片自动跳过（续跑）")
    for batch_idx in range(0, len(chunks), n_batch):
        batch = list(enumerate(chunks))[batch_idx:batch_idx + n_batch]
        todo = [chunk for chunk_index, chunk in batch
                if not _committed(out, chunk_index, n_batch, chunk.stem)]
        if not todo:
            say(f"批 {batch_idx // n_batch + 1}：已全部完成，跳过")
            continue
        names = ", ".join(c.stem for c in todo)
        say(f"批 {batch_idx // n_batch + 1}：上传 {len(todo)} 片（{names}）")
        md_map = _post_batch(base_url, key, todo, cfg, urlopen=urlopen)
        for chunk_index, chunk in batch:
            stem = chunk.stem
            if stem not in md_map:
                continue
            part = _part_dir(out, chunk_index, n_batch)
            part.mkdir(parents=True, exist_ok=True)
            (part / f"{stem}.md").write_text(md_map[stem], encoding="utf-8")
            if chunk.parent == work and chunk.exists():  # 只清临时切片，绝不动用户源文件
                try:
                    chunk.unlink()
                except OSError:  # pragma: no cover
                    pass
        say(f"批 {batch_idx // n_batch + 1}：完成，已写 {len([1 for _, c in batch if c.stem in md_map])} 片 md")

    return load_mineru_output(out, course)

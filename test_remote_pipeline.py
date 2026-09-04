"""云端 mineru-http 后端离线测试：mock HTTP，不联网、不花额度。

运行：conda activate firpro && cd coursebook-digest && python test_remote_pipeline.py
覆盖：分片→批量 POST→逐片落盘 md→复用 load_mineru_output 切章；续跑跳过；
     错误路径：无 key / HTTP 非 2xx / results 缺键。
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.error
from pathlib import Path

from coursebook_digest.config import PROJECT_ROOT, Settings
from coursebook_digest.mineru_http import MineruHttpError, parse_pdf_mineru_http


class FakeResp:
    """模拟 urllib 响应（支持 with：生产代码用 `with urlopen(...) as resp`）。"""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):  # noqa: ANN002
        return False


class FakeUrlOpen:
    """捕获请求、按序号返回预设响应的假 urlopen（可抛 HTTPError）。"""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, int | None]] = []

    def __call__(self, request, timeout=600):
        self.calls.append((request, timeout))
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    @property
    def last_request(self):
        return self.calls[-1][0] if self.calls else None


def _make_pdf(path: Path, pages: int = 3) -> None:
    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        w.write(f)


def _ok_response() -> bytes:
    md1 = "# 第四章 氢原子\n\n氢原子内容A\n"
    md2 = "# 第五章 自旋\n\n自旋内容B\n"
    return json.dumps({
        "task_id": "t-1",
        "status": "completed",
        "backend": "hybrid-auto-engine",
        "file_names": ["blink_p00000_00001", "blink_p00002_00002"],
        "results": {
            "blink_p00000_00001": {"md_content": md1},
            "blink_p00002_00002": {"md_content": md2},
        },
    }, ensure_ascii=False).encode("utf-8")


def test_remote_end_to_end(run_dir: Path) -> Settings:
    """3 页 PDF、每片 ≤2 页 → 2 片 1 批：POST 一次，落盘 2 个 md，切出两章。"""
    pdf = run_dir / "blink.pdf"
    _make_pdf(pdf, pages=3)
    out = run_dir / "mup"
    # _env_file=None：隔离项目 .env，防止把真实 key 带进测试（保持零联网）
    settings = Settings(mineru_api_key="sk-test-key", mineru_chunk_pages=2,
                        mineru_batch_files=2, _env_file=None)

    body_json = _ok_response()
    fake = FakeUrlOpen([FakeResp(body_json)])
    chapters = parse_pdf_mineru_http(
        pdf, "量子物理", out, settings=settings, urlopen=fake,
    )

    # 1) 恰好一次请求，且是 multipart POST 到网关
    assert len(fake.calls) == 1, f"应只 POST 一次，实际 {len(fake.calls)}"
    req = fake.last_request
    assert req.get_method() == "POST"
    assert req.full_url.endswith("/mineru/file_parse")
    assert req.get_header("Authorization") == "Bearer sk-test-key"
    # urllib 会把头部键 capitalize() 成 "Content-type"；真实发送正常
    assert "multipart/form-data; boundary=" in req.headers.get("Content-type", "")
    dat = b"".join(req.data) if isinstance(req.data, list) else req.data
    assert b"blink_p00000_00001.pdf" in dat and b"blink_p00002_00002.pdf" in dat
    assert b"return_md" in dat and b"response_format_zip" in dat and b"false" in dat

    # 2) 落盘 + 章节切分（云端 md 无页锚点，靠第X章切分）
    part = out / "part01"
    assert (part / "blink_p00000_00001.md").exists()
    assert (part / "blink_p00002_00002.md").exists()
    names = [c.chapter for c in chapters]
    assert names == ["氢原子", "自旋"], f"章节切分异常：{names}"
    assert any("自旋内容B" in c.text for c in chapters)
    print(f"    [远端] 1 批 2 片落盘+切章 -> {names}")

    # 3) 续跑：全部已落盘，重跑不再发请求
    fake2 = FakeUrlOpen([])
    chapters2 = parse_pdf_mineru_http(pdf, "量子物理", out, settings=settings, urlopen=fake2)
    assert fake2.calls == [], f"续跑不应再发请求：{len(fake2.calls)}"
    assert [c.chapter for c in chapters2] == names
    print("    [远端] 续跑跳过已落盘片 OK")
    return settings


def test_errors(run_dir: Path) -> None:
    pdf = run_dir / "err.pdf"
    _make_pdf(pdf, pages=1)
    out = run_dir / "muerr"

    # 无 key → 明确报错（_env_file=None 隔离项目 .env 的真实 key）
    try:
        parse_pdf_mineru_http(pdf, "课程", out,
                              settings=Settings(mineru_api_key="", llm_api_key="", _env_file=None))
        raise AssertionError("无 key 应报错")
    except MineruHttpError as exc:
        assert "key" in str(exc).lower() or "MINERU" in str(exc)
        print("    [远端] 无 key 报错 OK")

    # HTTP 4xx（如额度/鉴权） → 包装为 MineruHttpError 且带状态码
    http_err = urllib.error.HTTPError(
        "https://api.llm.ustc.edu.cn/mineru/file_parse", 403, "Forbidden",
        {}, None,
    )
    try:
        parse_pdf_mineru_http(
            pdf, "课程", out, settings=Settings(mineru_api_key="k", _env_file=None),
            urlopen=lambda *a, **k: (_ for _ in ()).throw(http_err),
        )
        raise AssertionError("HTTP 错误应上报")
    except MineruHttpError as exc:
        assert "403" in str(exc)
        print("    [远端] HTTP 403 错误包装 OK")

    # results 缺键 → 明确报错
    bad = FakeUrlOpen([FakeResp(json.dumps({
        "task_id": "t", "status": "completed",
        "results": {},
    }).encode("utf-8"))])
    try:
        parse_pdf_mineru_http(pdf, "课程", out,
                              settings=Settings(mineru_api_key="k", _env_file=None), urlopen=bad)
        raise AssertionError("缺结果键应报错")
    except MineruHttpError as exc:
        assert "results" in str(exc)
        print("    [远端] results 缺键报错 OK")


def test_transient_retry(run_dir: Path) -> None:
    """429/5xx 瞬时错误自动退避重试：先 503 再成功 → 共 2 次请求并落盘。"""
    pdf = run_dir / "retry.pdf"
    _make_pdf(pdf, pages=1)
    out = run_dir / "muret"
    ok = _ok_response()
    # 让成功响应包含该 1 页 PDF 的 stem
    ok_json = json.loads(ok.decode("utf-8"))
    stem = "retry_p00000_00000"
    ok_json["file_names"] = [stem]
    ok_json["results"] = {stem: {"md_content": "# 第一章 重试章节\n内容\n"}}
    fake = FakeUrlOpen([
        urllib.error.HTTPError("https://x/", 503, "Service Unavailable", {}, None),
        FakeResp(json.dumps(ok_json, ensure_ascii=False).encode("utf-8")),
    ])
    chapters = parse_pdf_mineru_http(
        pdf, "课程", out, settings=Settings(mineru_api_key="k", _env_file=None),
        urlopen=fake,
    )
    assert len(fake.calls) == 2, f"应重试 1 次，实际 {len(fake.calls)}"
    assert any(c.chapter == "重试章节" for c in chapters)
    assert (out / "part01" / f"{stem}.md").exists()
    print("    [远端] 503 瞬时错误自动重试 1 次 OK")


def test_office_passthrough(run_dir: Path) -> None:
    """非 PDF（pptx）→ 云端整文件直传：单次 POST、pptx MIME、落盘 md、不删源文件。"""
    pptx = run_dir / "讲座.pptx"
    pptx.write_bytes(b"PK\x03\x04 fake zip for test")
    out = run_dir / "muppt"
    ok = json.loads(_ok_response().decode("utf-8"))
    stem = "讲座"
    ok["file_names"] = [stem]
    ok["results"] = {stem: {"md_content": "# 第一章 引言\n内容\n"}}
    fake = FakeUrlOpen([FakeResp(json.dumps(ok, ensure_ascii=False).encode("utf-8"))])
    chapters = parse_pdf_mineru_http(
        pptx, "课程", out,
        settings=Settings(mineru_api_key="k", _env_file=None), urlopen=fake,
    )
    assert len(fake.calls) == 1, f"直传应只一次请求：{len(fake.calls)}"
    req = fake.last_request
    dat = b"".join(req.data) if isinstance(req.data, list) else req.data
    assert "讲座.pptx".encode("utf-8") in dat
    assert b"presentationml.presentation" in dat
    assert (out / "part01" / f"{stem}.md").exists()
    assert pptx.exists(), "直传不应删除用户源文件"
    assert any("引言" in c.chapter for c in chapters)
    print("    [远端] pptx 整文件直传 OK")


def main() -> None:
    run_dir = PROJECT_ROOT / ".smoke" / f"remote-{os.getpid()}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        test_remote_end_to_end(run_dir)
        test_errors(run_dir)
        test_transient_retry(run_dir)
        test_office_passthrough(run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    print("\n云端 mineru-http 离线测试全部通过")


if __name__ == "__main__":
    main()

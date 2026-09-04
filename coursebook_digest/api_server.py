"""R2 层：把 TeacherAgent 暴露成 HTTP API（本地/比赛演示用）。

- 直接使用现有 `TeacherAgent`，零改动地跑通「意图→画像→检索→作答→写回」。
- 无 key / 断网：agent 自动离线降级，接口照常返回（离线标记 + 方法卡预览）。
- 已带一个内嵌的单页对话 UI（GET /），便于快速体验与演示。
启动：python -m coursebook_digest.api_server  （或 coursebook serve --port 8000）
"""
from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import TeacherAgent
from .config import get_settings
from .schema import RetrievedMethod
from .store import CourseStore

# --------------------------------------------------------------------------- #
# 请求/响应模型
# --------------------------------------------------------------------------- #
class TeachRequest(BaseModel):
    question: str
    course: str = ""
    user_id: str = "local"
    top_k: int | None = None
    memory_k: int = 2
    writeback: bool = True
    session_id: str = ""


class DailyRequest(BaseModel):
    user_id: str = "local"
    course: str = ""
    n: int = 5
    include_memory: bool = True
    chapter: str = ""


class ReviewFeedback(BaseModel):
    user_id: str = "local"
    course: str = ""
    card_id: str
    ok: bool
    chapter: str = ""


class SessionNew(BaseModel):
    user_id: str = "local"
    course: str = ""
    title: str = ""


class SessionSwitch(BaseModel):
    user_id: str = "local"
    course: str = ""
    sid: str


# --------------------------------------------------------------------------- #
# 序列化助手
# --------------------------------------------------------------------------- #
def _method_dict(m: RetrievedMethod) -> dict:
    c = m.card
    return {
        "topic": c.topic, "kind": c.kind.value, "chapter": c.chapter,
        "score": round(m.score, 4), "keywords": c.keywords,
        "applicability": c.applicability, "steps": c.steps,
        "formula": c.core_formula_latex, "error_notes": c.error_notes,
    }


def _profile_dict(agent: TeacherAgent) -> dict:
    p = agent.profile
    return {
        "user_id": p.user_id, "turn_count": p.turn_count, "level": p.level,
        "kind_bias": p.kind_bias, "weak_topics": p.weak_topics,
        "common_mistakes": p.common_mistakes, "expression_notes": p.expression_notes,
        "general_notes": p.general_notes, "preamble": p.to_preamble(),
        "memory_buckets": _memory_counts(agent),
        "course_progress": {
            course: {
                "percent": cp.percent(), "questions": cp.questions,
                "chapters": {
                    name: {"questions": ch.questions, "depth_max": round(ch.depth_max, 2)}
                    for name, ch in cp.chapters.items()
                },
            }
            for course, cp in (p.course_progress or {}).items()
        },
        "reviews_stats": p.reviews_stats(),
        "sessions": {"current": p.current_session, "count": len(p.sessions) or 1},
        "chat_history_len": (len(p.sessions[p.current_session].get("history", []))
                             if p.sessions and p.current_session in p.sessions
                             else len(p.chat_history)),
    }


def _memory_counts(agent: TeacherAgent) -> dict[str, int]:
    from .agent import BUCKETS, GENERAL_BUCKET, bucket_course

    out = {}
    for b in BUCKETS + (GENERAL_BUCKET,):
        mode = CourseStore(bucket_course(agent.user_id, b), agent.memory_settings)
        out[b] = len(mode.load_all())
    return out


# --------------------------------------------------------------------------- #
# 异步入库任务（上传 PDF → 云端解析 → 蒸馏 → 入库，子进程跑 ingest，日志流式）
# --------------------------------------------------------------------------- #
_INGEST_JOBS: dict[str, dict] = {}
_INGEST_LOCK = threading.Lock()
_CB_ROOT = Path(__file__).resolve().parent.parent  # 项目根（coursebook-digest/）


def _upload_dir() -> Path:  # 默认上传目录；create_app(ingest_dir=...) 可覆盖（测试/部署用）
    return Path(get_settings().data_dir) / "uploads"


_UIDIR: dict = {"dir": None}


def _upload_dir_active() -> Path:
    return _UIDIR["dir"] or _upload_dir()


def _history_path() -> Path:
    return _upload_dir_active() / "_history.json"


def _load_history() -> dict:
    p = _history_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_history(history: dict) -> None:
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")


def _finish_ingest(job: dict, course: str) -> None:
    """任务结束时补全字段并落盘历史。"""
    job["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        job["cards"] = len(CourseStore(course, get_settings()).load_all())
    except Exception:  # noqa: BLE001
        job["cards"] = 0
    job["log_tail"] = job["log"][-40:]
    h = _load_history()
    h[job["jid"]] = {k: job.get(k) for k in (
        "jid", "course", "filename", "pdf", "ext", "parser",
        "slice_pages", "batch_files", "parallel",
        "status", "exit", "error", "started_at", "finished_at", "cards", "log_tail")}
    _save_history(h)


def _run_ingest(jid: str, pdf: Path, course: str, parser: str,
                slice_pages: int, batch_files: int, parallel: int) -> None:
    job = _INGEST_JOBS.get(jid)

    def log(s: str) -> None:
        with _INGEST_LOCK:
            job["log"].append(s)
            job["log"] = job["log"][-400:]

    log(f"[开始] {pdf.name} → 课程「{course}」（解析+蒸馏+入库，耗时较长，请耐心）")
    cmd = [sys.executable, "-m", "coursebook_digest.cli", "ingest", str(pdf),
           "--course", course, "--parser", parser or "mineru-http",
           "--mineru-out", str(pdf.parent / (pdf.stem + "_out"))]
    if slice_pages:
        cmd += ["--slice-pages", str(slice_pages)]
    if batch_files:
        cmd += ["--batch-files", str(batch_files)]
    if parallel:
        cmd += ["--parallel", str(parallel)]
    try:
        with _INGEST_LOCK:
            job["status"] = "running"
            job["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        proc = subprocess.Popen(cmd, cwd=str(_CB_ROOT),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)
        code = proc.wait()
        with _INGEST_LOCK:
            job["exit"] = code
            job["status"] = "done" if code == 0 else "error"
            if code != 0:
                job["error"] = f"ingest 退出码 {code}"
        log("✅ 导入完成（exit=0）" if code == 0 else f"❌ 导入失败 exit={code}")
        _finish_ingest(job, course)
    except Exception as exc:  # noqa: BLE001
        with _INGEST_LOCK:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        log(f"[异常] {exc}")
        _finish_ingest(job, course)


# --------------------------------------------------------------------------- #
# App 工厂（测试可注入 agent_factory）
# --------------------------------------------------------------------------- #
def create_app(agent_factory=None, ingest_dir: str | Path | None = None):
    factory = agent_factory or (lambda user_id, course, session="": TeacherAgent(
        user_id=user_id, course=course, session=session))
    _UIDIR["dir"] = Path(ingest_dir) if ingest_dir else None
    app = FastAPI(title="coursebook · 私人教师", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(_PAGE, headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health():
        settings = get_settings()
        return {
            "status": "ok", "courses": [c["name"] for c in _courses()],
            "model": settings.llm_model, "base_url": settings.llm_base_url,
            "api_key_configured": bool(settings.llm_api_key),
        }

    @app.get("/api/courses")
    def courses():
        return _courses()

    @app.post("/api/teach")
    def teach(req: TeachRequest):
        try:
            agent = factory(req.user_id, req.course, req.session_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"初始化学生失败：{exc}") from exc
        try:
            r = agent.turn(req.question, top_k=req.top_k, memory_k=req.memory_k,
                           write_back_async=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"作答失败：{type(exc).__name__}: {exc}") from exc
        return {
            "intent": r.intent.value,
            "offline": r.offline,
            "answer": r.answer,
            "methods": [_method_dict(m) for m in r.methods],
            "memory_hits": [_method_dict(m) for m in r.memory_hits],
            "writeback": {
                "skipped": r.writeback.skipped,
                "reason": r.writeback.reason,
                "buckets_written": r.writeback.buckets_written,
            },
            "profile": _profile_dict(agent),
        }

    @app.post("/api/teach_stream")
    def teach_stream(req: TeachRequest):
        """流式作答：SSE 逐段推送 intent → methods → delta* → done(+profile)。"""
        try:
            agent = factory(req.user_id, req.course, req.session_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"初始化学生失败：{exc}") from exc

        def gen():
            for ev in agent.turn_stream(req.question, top_k=req.top_k, memory_k=req.memory_k):
                if ev.get("type") == "done":
                    ev["profile"] = _profile_dict(agent)
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/progress")
    def progress(user_id: str = "local", course: str = ""):
        """学习进展可视化数据：整体%/覆盖章节/曲线 + 每日统计（近 14 天）。"""
        agent = factory(user_id, course)
        cp = agent.profile.course_progress.get(course)
        if not cp:
            return {"course": course, "percent": 0, "questions": 0,
                    "chapters": {}, "curve": [], "daily": [], "today_questions": 0}
        daily: dict[str, dict] = {}
        for e in cp.curve:
            day = (e.get("ts") or "")[:10]
            row = daily.setdefault(day, {"date": day, "questions": 0, "pct": 0})
            row["questions"] += 1
            row["pct"] = e.get("pct") or row["pct"]
        today = datetime.date.today().isoformat()
        return {
            "course": course,
            "percent": cp.percent(),
            "questions": cp.questions,
            "chapters": {
                name: {"questions": ch.questions, "depth_max": round(ch.depth_max, 2)}
                for name, ch in cp.chapters.items()
            },
            "curve": cp.curve[-300:],
            "daily": sorted(daily.values())[-14:],
            "today_questions": daily.get(today, {}).get("questions", 0),
        }

    @app.get("/api/profile")
    def profile(user_id: str = "local", course: str = ""):
        agent = factory(user_id, course)
        return _profile_dict(agent)

    def _daily(user_id: str, course: str, n: int, chapter: str = "") -> list[dict]:
        agent = factory(user_id, course)
        return [{"id": c.id, "topic": c.topic, "kind": c.kind.value,
                 "chapter": c.chapter, "keywords": c.keywords,
                 "steps": c.steps, "error_notes": c.error_notes}
                for c in agent.daily_cards(n=n, include_memory=True, chapter=chapter or None)]

    @app.post("/api/daily_cards")
    def daily_cards(req: DailyRequest):
        return _daily(req.user_id, req.course, req.n, req.chapter)

    @app.get("/api/daily_cards")
    def daily_cards_get(user_id: str = "local", course: str = "", n: int = 5, chapter: str = ""):
        return _daily(user_id, course, n, chapter)

    @app.post("/api/ingest")
    def ingest_upload(course: str = Form(...), file: UploadFile = File(...),
                      parser: str = Form("mineru-http"),
                      slice_pages: int = Form(0), batch_files: int = Form(0),
                      parallel: int = Form(0)):
        """上传教材 → 启动 云端解析+蒸馏+入库 后台任务，返回 job_id。"""
        ext = Path(file.filename or "").suffix.lower()
        if ext not in (".pdf", ".pptx", ".ppt", ".docx", ".doc"):
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{ext}")
        up = _upload_dir_active()
        up.mkdir(parents=True, exist_ok=True)
        jid = uuid.uuid4().hex[:12]
        dest = up / f"{jid}{ext}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        job = {"jid": jid, "course": course, "filename": file.filename, "pdf": str(dest),
               "ext": ext, "parser": parser, "slice_pages": slice_pages,
               "batch_files": batch_files, "parallel": parallel,
               "status": "queued", "log": [f"[排队] 已接收 {file.filename}"], "exit": None,
               "error": "", "started_at": None, "finished_at": None,
               "cards": None, "log_tail": []}
        _INGEST_JOBS[jid] = job
        h = _load_history()
        h[jid] = {k: job.get(k) for k in (
            "jid", "course", "filename", "pdf", "ext", "parser",
            "slice_pages", "batch_files", "parallel", "status", "error",
            "started_at", "finished_at", "cards", "log_tail")}
        _save_history(h)
        threading.Thread(
            target=_run_ingest,
            args=(jid, dest, course, parser, slice_pages, batch_files, parallel),
            daemon=True,
        ).start()
        return {"job_id": jid, "course": course}

    @app.get("/api/ingest/{jid}")
    def ingest_status(jid: str):
        job = _INGEST_JOBS.get(jid)
        if not job:
            raise HTTPException(status_code=404, detail="无此任务")
        with _INGEST_LOCK:
            return {"status": job["status"], "exit": job["exit"],
                    "error": job["error"], "log": job["log"][-120:]}

    @app.get("/api/ingest_history")
    def ingest_history():
        """导入历史（含状态/卡数/时间；运行中任务叠加实时状态）。"""
        h = _load_history()
        with _INGEST_LOCK:
            for jid, job in _INGEST_JOBS.items():
                cur = h.setdefault(jid, {})
                for k in ("status", "exit", "error", "log_tail", "started_at", "finished_at", "cards"):
                    if k in job:
                        cur[k] = job[k]
        vals = [v for v in h.values() if v.get("jid")]
        vals.sort(key=lambda x: (x.get("started_at") or x.get("finished_at") or ""), reverse=True)
        return vals

    @app.post("/api/ingest/{jid}/resume")
    def ingest_resume(jid: str):
        """续跑：复用已上传 PDF（解析产物已落盘会自动跳过只补蒸馏）。"""
        h = _load_history()
        entry = h.get(jid) or _INGEST_JOBS.get(jid)
        if not entry:
            raise HTTPException(status_code=404, detail="无此任务")
        pdf = Path(entry.get("pdf") or "")
        if not pdf.exists():
            raise HTTPException(status_code=400, detail="源文件已不存在，无法续跑")
        if _INGEST_JOBS.get(jid, {}).get("status") == "running":
            raise HTTPException(status_code=409, detail="该任务正在运行")
        job = {"jid": jid, "course": entry.get("course", ""), "filename": entry.get("filename", ""),
               "pdf": str(pdf), "ext": entry.get("ext", ".pdf"),
               "parser": entry.get("parser", "mineru-http"),
               "slice_pages": entry.get("slice_pages", 0), "batch_files": entry.get("batch_files", 0),
               "parallel": entry.get("parallel", 0),
               "status": "queued", "log": [f"[续跑] 重新导入 {entry.get('filename', '')}（已解析分片自动跳过）"],
               "exit": None, "error": "", "started_at": None, "finished_at": None,
               "cards": entry.get("cards"), "log_tail": []}
        _INGEST_JOBS[jid] = job
        h[jid] = {k: job.get(k) for k in (
            "jid", "course", "filename", "pdf", "ext", "parser",
            "slice_pages", "batch_files", "parallel", "status", "error",
            "started_at", "finished_at", "cards", "log_tail")}
        _save_history(h)
        threading.Thread(
            target=_run_ingest,
            args=(jid, pdf, job["course"], job["parser"], job["slice_pages"],
                  job["batch_files"], job["parallel"]),
            daemon=True,
        ).start()
        return {"job_id": jid}

    @app.delete("/api/ingest/{jid}")
    def ingest_delete(jid: str, delete_files: int = 0):
        """清除历史记录；delete_files=1 时连同上传的源文件与解析产物一起删除。"""
        h = _load_history()
        j = h.pop(jid, None) or _INGEST_JOBS.get(jid, {})
        _save_history(h)
        _INGEST_JOBS.pop(jid, None)
        if delete_files and j.get("pdf"):
            pdf = Path(j["pdf"])
            try:
                pdf.unlink()
            except OSError:  # noqa: BLE001
                pass
            out = Path(str(pdf)).with_suffix("")
            try:
                shutil.rmtree(str(out) + "_out")
            except OSError:  # noqa: BLE001
                pass
        return {"deleted": jid}

    @app.get("/api/sessions")
    def sessions(user_id: str = "local", course: str = ""):
        agent = factory(user_id, course)
        return {"current": agent.profile.current_session,
                "sessions": agent.profile.sessions_list()}

    @app.post("/api/sessions/new")
    def sessions_new(req: SessionNew):
        from .profile import save_profile

        agent = factory(req.user_id, req.course)
        sid = agent.profile.new_session(req.title or None)
        save_profile(agent.user_dir, agent.profile)
        return {"sid": sid, "current": agent.profile.current_session,
                "sessions": agent.profile.sessions_list()}

    @app.post("/api/sessions/switch")
    def sessions_switch(req: SessionSwitch):
        from .profile import save_profile

        agent = factory(req.user_id, req.course)
        agent.profile.set_current_session(req.sid)
        save_profile(agent.user_dir, agent.profile)
        return {"ok": True, "current": agent.profile.current_session}

    @app.post("/api/sessions/{sid}/clear")
    def sessions_clear(sid: str, req: SessionSwitch):
        from .profile import save_profile

        agent = factory(req.user_id, req.course)
        agent.profile.clear_session(sid)
        save_profile(agent.user_dir, agent.profile)
        return {"ok": True}

    @app.delete("/api/sessions/{sid}")
    def sessions_delete(sid: str, user_id: str = "local", course: str = ""):
        from .profile import save_profile

        agent = factory(user_id, course)
        agent.profile.delete_session(sid)
        save_profile(agent.user_dir, agent.profile)
        return {"ok": True, "current": agent.profile.current_session}

    @app.post("/api/review_feedback")
    def review_feedback(req: "ReviewFeedback"):
        agent = factory(req.user_id, req.course)
        box = agent.record_review(req.card_id, req.ok, req.chapter)
        return {"ok": True, "box": box, "reviews_stats": agent.profile.reviews_stats()}

    return app


def _courses() -> list[dict]:
    settings = get_settings()
    return [
        {"name": name, "cards": len(CourseStore(name, settings).load_all())}
        for name in CourseStore.list_courses(settings)
    ]


app = create_app()

_PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>coursebook · 私人教师</title>
<style>
body{font-family:system-ui,sans-serif;color:#222;background:#f1f5f9;margin:0}
.wrap{display:flex;max-width:1080px;margin:20px auto;gap:16px;padding:0 16px}
.side{width:190px;flex:0 0 auto;background:#fff;border:1px solid #e3e3e3;border-radius:14px;padding:10px;position:sticky;top:14px;align-self:flex-start;box-shadow:0 1px 2px #0001}
.brand{font-weight:800;font-size:17px;padding:10px 10px 6px}
.brand small{display:block;font-weight:400;color:#64748b;font-size:12px}
.si{display:block;width:100%;text-align:left;background:transparent;color:#334155;padding:11px 12px;margin:2px 0;border-radius:9px;border:0;font-size:14px;cursor:pointer}
.si:hover{background:#eef2ff}
.si.active{background:#2563eb;color:#fff}
.main{flex:1;min-width:0}
.row{display:flex;gap:8px;margin:8px 0}
input,select,textarea{font-size:15px;padding:8px;border:1px solid #ccc;border-radius:8px;width:100%}
button{padding:10px 18px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.meta{color:#555;font-size:13px}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:12px;margin:8px 0;white-space:pre-wrap;cursor:pointer}
.method{background:#eff6ff;border-left:3px solid #2563eb;padding:8px 12px;margin:6px 0;font-size:14px}
.hint{color:#94a3b8;font-size:12px}
.kind{font-size:12px;color:#2563eb}
.off{color:#b45309} .err{color:#b91c1c;background:#fee2e2;padding:10px;border-radius:8px;white-space:pre-wrap}
#status{color:#b45309;font-size:14px;min-height:20px}
.pcard{background:#fff;border:1px solid #e3e3e3;border-radius:12px;padding:14px;margin:8px 0}
.big{font-size:36px;font-weight:700;color:#2563eb}
.row2{display:flex;justify-content:space-between;font-size:13px;color:#475569;flex-wrap:wrap;gap:4px}
svg{width:100%;height:110px}
.smallbtn{background:#e2e8f0;color:#1e293b;font-size:12px;padding:4px 10px;margin-left:8px}
.chatwin{background:#fff;border:1px solid #e3e3e3;border-radius:14px;padding:12px;height:calc(100vh - 205px);min-height:320px;overflow-y:auto;overflow-anchor:auto}
.sys{color:#94a3b8;font-size:12px;text-align:center;margin:6px 0}
.bubble{max-width:88%;padding:10px 13px;border-radius:13px;margin:7px 0;white-space:pre-wrap;word-break:break-word;font-size:14px;line-height:1.55}
.bubble.user{background:#2563eb;color:#fff;margin-left:auto;border-bottom-right-radius:4px}
.bubble.bot{background:#f1f5f9;color:#1e293b;border-bottom-left-radius:4px}
.composer{display:flex;gap:8px;margin-top:10px;align-items:flex-end}
.composer textarea{resize:none;min-height:64px;max-height:150px}
.composer button{height:64px}
@media(max-width:720px){.wrap{flex-direction:column}.side{width:auto;position:static}}
</style></head><body>
<div class="wrap">
<aside class="side">
<div class="brand">📖 coursebook<small>私人教师 · 个性化学习</small></div>
<button class="si" id="nav-ingest" onclick="showSide('ingest')">📥 导书入库</button>
<button class="si active" id="nav-chat" onclick="showSide('chat')">💬 对话</button>
<button class="si" id="nav-progress" onclick="showSide('progress')">📈 学习进展</button>
<button class="si" id="nav-review" onclick="showSide('review')">📚 复习·打卡</button>
<button class="si" id="nav-profile" onclick="showSide('profile')">🧑 我的画像</button>
</aside>
<main class="main">
<section id="view-ingest" style="display:none">
<div class="row"><input id="icourse" value="新课程" placeholder="课程名（建议纯文字，如 数据结构）"></div>
<div class="row"><input id="ifile" type="file" accept=".pdf,.pptx,.ppt,.docx,.doc"></div>
<div class="row">
<input id="islice" placeholder="每片页数(默认自动)" style="flex:1">
<input id="ibatch" placeholder="每批文件(默认)" style="flex:1">
<input id="ipara" placeholder="并行数(默认)" style="flex:1">
</div>
<div class="row"><button onclick="startIngest()">开始导入（云端解析 → 蒸馏 → 入库）</button></div>
<div class="meta">说明：上传 PDF/PPT/DOC，走学校网关 mineru 解析并 LLM 蒸馏入库。全书耗时较长（视页数几十分钟～数小时），请在下方实时日志观察；完成后新课程会自动出现在「学习进展 / 复习 / 画像」。</div>
<div class="meta" style="margin-top:14px">—— 导入历史（查看日志 / 续跑 / 清理）——</div>
<div id="ihist"></div>
<div id="istat" class="meta"></div>
<div id="iout" style="white-space:pre-wrap;background:#0f172a;color:#e2e8f0;border-radius:10px;padding:10px;font-size:12px;height:300px;overflow:auto"></div>
</section>
<section id="view-chat">
<div class="row" style="align-items:center">
<select id="selSession" style="flex:1" onchange="switchSession()"><option>…</option></select>
<button class="smallbtn" style="background:#2563eb;color:#fff" onclick="newSession()">＋ 新会话</button>
<button class="smallbtn" onclick="clearCurSession()">清空本会话</button>
<button class="smallbtn" style="background:#b91c1c;color:#fff" onclick="delCurSession()">删除本会话</button>
</div>
<div class="chatwin" id="chatwin">
<div id="jsok" style="display:none;color:#16a34a;font-size:12px"></div>
<div class="sys" id="chathead">📖 和你的私人助教聊天吧——课程题走课本方法，常识题也能聊；连续提问会记得上文。</div>
<div id="out"></div>
</div>
<div class="row"><input id="user" value="demoSTU" placeholder="学生 id"><input id="course" value="操作系统" placeholder="课程"></div>
<div class="composer">
<textarea id="q" rows="2" placeholder="输入问题…（Enter 发送，Shift+Enter 换行）"></textarea>
<button id="send" onclick="askStream()">发送</button>
<button style="background:#e2e8f0;color:#1e293b" onclick="$('out').innerHTML='';$('chathead').style.display=''">清空</button>
</div>
<div id="status"></div>
</section>
<section id="view-progress" style="display:none"><div id="pout"></div></section>
<section id="view-review" style="display:none">
<div class="row">
<select id="rvChapter"><option value="">全部章节（随机抽卡）</option></select>
<button onclick="review()">开始复习</button><button class="smallbtn" style="background:#2563eb;color:#fff" onclick="review()">再抽 5 张</button>
</div>
<div id="rout"></div>
</section>
<section id="view-profile" style="display:none"><div id="pout2"></div></section>
</main>
</div>
<script>
const $=id=>document.getElementById(id);
$('jsok').style.display='inline';$('jsok').textContent='✓ JS 已就绪（看不到这行绿字=旧缓存，请开新标签页访问 127.0.0.1:8000）';
function line(a,b){const d=document.createElement('div');if(a)d.className=a;d.innerHTML=b;return d}
function esc(t){return String(t==null?'':t).replace(/</g,'&lt;')}
async function api(path, body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json(); if(!r.ok)throw new Error(JSON.stringify(j.detail||j)); return j;
}
function setBusy(on,streaming){$('send').disabled=on;$('status').textContent=on?(streaming?'正在生成……（流式输出中）':'思考中……（学校网关较慢，通常 1~4 分钟，请稍候）'):''}
let curSid='';
async function loadSessions(){
  try{
    const u=encodeURIComponent($('user').value),c=encodeURIComponent($('course').value);
    const r=await (await fetch('/api/sessions?user_id='+u+'&course='+c)).json();
    curSid=r.current;
    $('selSession').innerHTML=r.sessions.map(s=>'<option value="'+esc(s.sid)+'"'+(s.sid===r.current?' selected':'')+'>'+esc(s.title)+'（'+s.count+'）</option>').join('');
  }catch(e){}
}
async function switchSession(){
  const sid=$('selSession').value;if(!sid)return;
  await api('/api/sessions/switch',{user_id:$('user').value,course:$('course').value,sid:sid});
  curSid=sid;$('out').innerHTML='';$('chathead').style.display='';
  $('status').textContent='已切换到会话：'+sid;
}
async function newSession(){
  const r=await api('/api/sessions/new',{user_id:$('user').value,course:$('course').value,title:''});
  curSid=r.sid;await loadSessions();
  $('out').innerHTML='';$('chathead').style.display='';
  $('status').textContent='已新建会话（开始一段新对话，与其它会话互不影响）';
}
async function clearCurSession(){
  const sid=curSid||$('selSession').value;if(!sid)return;
  await api('/api/sessions/'+sid+'/clear',{user_id:$('user').value,course:$('course').value,sid:sid});
  $('out').innerHTML='';$('status').textContent='已清空本会话上下文';
  await loadSessions();
}
async function delCurSession(){
  const sid=curSid||$('selSession').value;if(!sid)return;
  if(sid==='default'){alert('默认会话不可删除，可直接“清空本会话”');return;}
  if(!confirm('删除该会话及其历史？'))return;
  await fetch('/api/sessions/'+sid+'?user_id='+encodeURIComponent($('user').value)+'&course='+encodeURIComponent($('course').value),{method:'DELETE'});
  $('out').innerHTML='';await loadSessions();$('status').textContent='已删除会话';
}
async function askStream(){
  const o=$('out');setBusy(true,true);
  const q=$('q').value.trim();
  if(!q){setBusy(false);return;}
  const ub=document.createElement('div');ub.className='bubble user';ub.textContent=q;
  o.append(ub);
  const sysEl=document.createElement('div');sysEl.className='sys';sysEl.textContent='…';o.append(sysEl);
  const sb=document.createElement('div');sb.className='bubble bot';sb.textContent='';o.append(sb);
  o.scrollTop=o.scrollHeight;
  $('q').value='';$('q').focus();
  try{
    const resp=await fetch('/api/teach_stream',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,user_id:$('user').value,course:$('course').value,session_id:(curSid||'')})});
    if(!resp.ok||!resp.body)throw new Error('流式接口异常 HTTP '+resp.status);
    const reader=resp.body.getReader();const dec=new TextDecoder();let buf='';
    while(true){
      const {done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf('\n\n'))>=0){
        const frame=buf.slice(0,i);buf=buf.slice(i+2);
        for(const l of frame.split('\n')){
          if(!l.startsWith('data: '))continue;
          const ev=JSON.parse(l.slice(6));
          if(ev.type==='intent'){sysEl.textContent='意图：'+ev.intent;}
          else if(ev.type==='methods'){
            if((ev.methods||[]).length){sysEl.textContent='意图：'+(ev.methods[0].chapter||'')+'　命中：'+(ev.methods.map(m=>m.topic).join(' ／ '));}
          }
          else if(ev.type==='delta'){sb.textContent+=ev.text;o.scrollTop=o.scrollHeight;}
          else if(ev.type==='done'){
            sysEl.textContent=ev.offline?'（离线降级：仅方法预览）':'';
            if(ev.profile&&ev.profile.weak_topics.length){sysEl.textContent+='　薄弱主题：'+ev.profile.weak_topics.join('、');}
          }
        }
      }
    }
  }catch(e){sysEl.textContent='出错了：'+e.message;sb.textContent='';}
  finally{setBusy(false);o.scrollTop=o.scrollHeight;}
}
function svgPoly(curve){
  if(!curve||curve.length<2)return '<div class="meta">学习记录还太少，多学几轮后即可生成进度曲线</div>';
  const W=340,H=100,P=6,step=(W-2*P)/(curve.length-1);
  const pts=curve.map((c,i)=>{const x=P+i*step;const y=H-P-Math.min((c.pct||0)/100,1)*(H-2*P);return x.toFixed(1)+','+y.toFixed(1);}).join(' ');
  return '<svg viewBox="0 0 '+W+' '+H+'"><polyline fill="none" stroke="#2563eb" stroke-width="2" points="'+pts+'"/></svg>';
}
function aggrChapters(ch){
  const m={};
  Object.keys(ch||{}).forEach(k=>{const base=k.replace(/·\d+$/,'');const d=ch[k].depth_max;
    if(!(base in m)||m[base]<d)m[base]=d;});
  return Object.keys(m).map(name=>({name:name.length>9?name.slice(0,9)+'…':name,depth:m[name]}))
    .sort((a,b)=>b.depth-a.depth).slice(0,8);
}
function radarChapter(ch){
  const items=aggrChapters(ch);
  if(!items.length)return '<div class="meta">暂无章节进度数据（先学几轮课程问题）</div>';
  if(items.length<3)return '<div class="meta">章节还太少，把更多章节学起来后再看雷达图</div>';
  const N=items.length,cx=120,cy=108,R=80;
  const ang=i=>-Math.PI/2+i*2*Math.PI/N;
  const pt=(i,r)=>{const a=ang(i);return {x:cx+r*Math.cos(a),y:cy+r*Math.sin(a)};};
  let out='<svg viewBox="0 0 240 225" style="height:225px">';
  for(let ring=1;ring<=4;ring++){const rr=R*ring/4;let p='';for(let i=0;i<N;i++){const q=pt(i,rr);p+=(i?' ':'')+q.x.toFixed(1)+','+q.y.toFixed(1);}out+='<polygon fill="none" stroke="#e2e8f0" points="'+p+'"/>';}
  let dataP='';
  for(let i=0;i<N;i++){const q=pt(i,R);out+='<line x1="'+cx+'" y1="'+cy+'" x2="'+q.x.toFixed(1)+'" y2="'+q.y.toFixed(1)+'" stroke="#e2e8f0"/>';
    const d=Math.min(Math.max(items[i].depth,0),1);const z=pt(i,R*d);dataP+=(i?' ':'')+z.x.toFixed(1)+','+z.y.toFixed(1);}
  out+='<polygon fill="rgba(37,99,235,0.22)" stroke="#2563eb" stroke-width="2" points="'+dataP+'"/>';
  for(let i=0;i<N;i++){const q=pt(i,R*1.22);const anc=q.x<cx-6?'end':(q.x>cx+6?'start':'middle');
    const lx=q.x<cx-6?q.x-2:(q.x>cx+6?q.x+2:cx);
    const lx2=lx<0?0:(lx>240?240:lx);
    out+='<text x="'+lx2.toFixed(0)+'" y="'+(q.y<8?8:(q.y>216?216:q.y)).toFixed(0)+'" text-anchor="'+anc+'" font-size="9" fill="#334155">'+esc(items[i].name)+'</text>';}
  out+='</svg>';
  return out;
}
async function fetchProgress(){
  const u=encodeURIComponent($('user').value),c=encodeURIComponent($('course').value);
  return await (await fetch('/api/progress?user_id='+u+'&course='+c)).json();
}
async function loadProgress(){
  const o=$('pout');o.innerHTML='<div class="meta">加载中……</div>';
  try{
    const r=await fetchProgress();
    o.innerHTML='';
    const chapters=r.chapters||{},nCh=Object.keys(chapters).length;
    o.append(line('pcard','<span class="big">'+r.percent+'%</span><div class="row2"><span>累计提问 '+r.questions+' 次</span><span>覆盖章节 '+nCh+' 章</span><span>今日 '+r.today_questions+' 题</span></div>'));
    o.append(line('pcard',svgPoly(r.curve)));
    if(nCh){
      o.append(line('pcard','<div class="meta">各章深入度 · 雷达图（取深度前 8 章，越大越深入）</div>'+radarChapter(chapters)));
      o.append(line('meta','已覆盖章节（点击“复习该章”跳转复习）：'));
      Object.keys(chapters).forEach(k=>{
        const row=line('method','<b>'+esc(k)+'</b>　提问 '+chapters[k].questions+' 次 · 深入度 '+chapters[k].depth_max+'<button class="smallbtn" onclick="goReview(\''+k.replace(/['\\]/g,'')+'\')">复习该章</button>');
        o.append(row);
      });
    }
    if(r.daily&&r.daily.length){
      o.append(line('meta','近 '+r.daily.length+' 天学习情况：'));
      r.daily.slice().reverse().forEach(d=>o.append(line('row2','<span>'+d.date+'</span><span>'+d.questions+' 题</span><span>当日进度 '+d.pct+'%</span>')));
    }
  }catch(e){o.append(line('err','出错了：'+e.message))}
}
async function fillChapterSelect(){
  const sel=$('rvChapter');const cur=sel.value;
  try{
    const r=await fetchProgress();
    const opts=['<option value="">全部章节（随机抽卡）</option>'].concat(Object.keys(r.chapters||{}).map(k=>'<option value="'+esc(k)+'">'+esc(k)+'</option>'));
    sel.innerHTML=opts.join('');
    if(cur&&cur!=='')sel.value=cur;
  }catch(e){}
}
function goReview(ch){
  const sel=$('rvChapter');sel.value=ch||'';
  showSide('review');review();
}
function feedback(box,c,ok){
  const b={user_id:$('user').value,course:$('course').value,card_id:c.id||'',ok:ok,chapter:c.chapter||''};
  fetch('/api/review_feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})
    .then(r=>r.json()).then(d=>{
      box.style.borderColor=ok?'#16a34a':'#b91c1c';
      const st=document.createElement('div');st.className='hint';
      st.textContent=(ok?('✅ 已记：答对（盒 '+d.box+'）'):'❌ 已记：答错，盒回到 1')+'　·　累计已评 '+(d.reviews_stats?d.reviews_stats.total:0)+' 次';
      box.appendChild(st);
      (box.querySelectorAll('button')).forEach(bn=>{bn.disabled=true;});
    }).catch(function(){});
}
function cardEl(c){
  const box=document.createElement('div');box.className='card';
  const head=document.createElement('div');
  head.innerHTML='<b>'+esc(c.kind)+'</b>　'+esc(c.topic)+'　<span class="kind">('+esc(c.chapter)+')</span><div class="hint">点击卡片查看 步骤/易错点</div>';
  const ans=document.createElement('div');ans.style.display='none';ans.className='meta';
  let t='';
  if(c.steps&&c.steps.length)t+='步骤：\n'+c.steps.map((s,i)=>(i+1)+'. '+s).join('\n')+'\n';
  if(c.error_notes&&c.error_notes.length)t+='易错点：'+c.error_notes.join('；');
  ans.textContent=t||'（暂无详细内容）';
  const foot=document.createElement('div');foot.style.marginTop='8px';
  const bY=document.createElement('button');bY.textContent='✅ 答对';bY.style.background='#16a34a';bY.className='smallbtn';
  const bN=document.createElement('button');bN.textContent='❌ 答错';bN.style.background='#b91c1c';bN.className='smallbtn';
  bY.onclick=function(ev){ev.stopPropagation();feedback(box,c,true);};
  bN.onclick=function(ev){ev.stopPropagation();feedback(box,c,false);};
  foot.appendChild(bY);foot.appendChild(bN);
  box.appendChild(head);box.appendChild(ans);box.appendChild(foot);
  box.onclick=function(){ans.style.display=(ans.style.display==='none')?'':'none';};
  return box;
}
async function review(){
  const o=$('rout');o.innerHTML='<div class="meta">加载中……</div>';
  try{
    const ch=$('rvChapter').value||'';
    const u=encodeURIComponent($('user').value),c=encodeURIComponent($('course').value);
    let url='/api/daily_cards?user_id='+u+'&course='+c+'&n=5';
    if(ch)url+='&chapter='+encodeURIComponent(ch);
    const r=await (await fetch(url)).json();
    o.innerHTML='';
    if(!r.length){o.append(line('meta',ch?('该章节还没有卡片，先学几轮再来，或换“全部章节”。'):'（暂无卡片，请先对话或换个课程）'));return;}
    o.append(line('meta',ch?('按章节复习：'+esc(ch)+'　抽 '+r.length+' 张'):('今日打卡　抽 '+r.length+' 张')));
    r.forEach(ca=>o.appendChild(cardEl(ca)));
  }catch(e){o.append(line('err','出错了：'+e.message))}
}
async function loadProfile(){
  const o=$('pout2');o.innerHTML='<div class="meta">加载中……</div>';
  try{
    const u=encodeURIComponent($('user').value),c=encodeURIComponent($('course').value);
    const r=await (await fetch('/api/profile?user_id='+u+'&course='+c)).json();
    o.innerHTML='';
    o.append(line('pcard','<b>累计对话</b> '+r.turn_count+' 轮　·　水平 '+r.level+'　·　风格偏向 '+r.kind_bias));
    o.append(line('pcard','<b>薄弱主题：</b>'+(r.weak_topics.join('、')||'暂无')+'\n<b>常错点：</b>'+(r.common_mistakes.join('、')||'暂无')+'\n<b>通识话题：</b>'+(r.general_notes.join('、')||'暂无')+'\n<b>措辞习惯：</b>'+(r.expression_notes||'暂无')));
    const mb=r.memory_buckets||{};
    if(Object.keys(mb).length)o.append(line('pcard','<b>个人记忆库：</b>'+Object.keys(mb).map(k=>k+' '+mb[k]+' 条').join('　·　')));
    const cp=r.course_progress||{};
    const cur=cp[ $('course').value ];
    if(cur){
      const pc=line('pcard','<b>本课学习进度：</b>'+cur.percent+'%　·　累计 '+cur.questions+' 题');
      const rd=document.createElement('div');rd.innerHTML=radarChapter(cur.chapters);pc.appendChild(rd);
      const tt=document.createElement('div');tt.className='meta';tt.style.margin='6px 0';
      tt.textContent='各章深入度：'+Object.keys(cur.chapters).map(k=>k.replace(/·\d+$/,'')+' '+cur.chapters[k].depth_max.toFixed(2)).slice(0,12).join(' · ');
      pc.appendChild(tt);
      o.append(pc);
    }
    const rs=r.reviews_stats||{};
    if(rs.total){o.append(line('pcard','<b>复习自评：</b>累计 '+rs.total+' 次 · 答对 '+rs.correct+' · 答错 '+rs.wrong+' · 正确率 '+(rs.accuracy*100).toFixed(0)+'% · '+rs.boxes+' 张卡已有盒号'));}
    o.append(line('card',r.preamble));
  }catch(e){o.append(line('err','出错了：'+e.message))}
}
function showSide(v){
  ['chat','ingest','progress','review','profile'].forEach(k=>{$('view-'+k).style.display=(k===v)?'':'none';});
  ['chat','ingest','progress','review','profile'].forEach(k=>{$('nav-'+k).className=(k===v)?'si active':'si';});
  if(v==='progress')loadProgress();
  if(v==='review')fillChapterSelect();
  if(v==='profile')loadProfile();
  if(v==='ingest')loadIngestHistory();
  if(v==='chat'){loadSessions();}
}
let ingestTimer=null;
async function viewLog(jid){
  let lines=[];
  try{
    const hh=await (await fetch('/api/ingest_history')).json();
    const e=hh.find(x=>x.jid===jid);
    if(e&&e.log_tail)lines=e.log_tail;
  }catch(e2){}
  try{const j=await (await fetch('/api/ingest/'+jid)).json();if(j.log)lines=j.log;}catch(e3){}
  $('iout').textContent='';lines.forEach(l=>$('iout').textContent+=l+'\n');
  $('istat').textContent='日志：任务 '+jid;
}
async function loadIngestHistory(){
  const o=$('ihist');o.innerHTML='<div class="meta">加载中……</div>';
  try{
    const r=await (await fetch('/api/ingest_history')).json();
    o.innerHTML='';
    if(!r.length){o.append(line('meta','（暂无导入历史，先上传一本教材吧）'));return;}
    r.forEach(j=>{
      const st=j.status==='done'?'✅':(j.status==='error'?'❌':(j.status==='running'?'⏳ 运行中':'🕓 排队'));
      const row=line('method','<span style="color:#334155"><b>'+esc(j.course||'?')+'</b>　'+esc(j.filename||'')+'　'+st+' '+esc(j.status)+(j.cards!=null?('　· '+j.cards+' 卡'):'')+'</span>'+
        '<button class="smallbtn" style="background:#2563eb;color:#fff" onclick="viewLog(\''+j.jid+'\')">日志</button>'+
        '<button class="smallbtn" onclick="resumeIngest(\''+j.jid+'\')">续跑</button>'+
        '<button class="smallbtn" style="background:#b91c1c;color:#fff" onclick="clearIngest(\''+j.jid+'\',0)">清除记录</button>'+
        '<button class="smallbtn" style="background:#7f1d1d;color:#fff" onclick="clearIngest(\''+j.jid+'\',1)">清除+文件</button>');
      o.append(row);
    });
  }catch(e){o.append(line('err','出错了：'+e.message))}
}
async function resumeIngest(jid){
  const r=await fetch('/api/ingest/'+jid+'/resume',{method:'POST'});
  const j=await r.json();
  if(!r.ok){alert(JSON.stringify(j.detail||j));return;}
  $('istat').textContent='已续跑任务 '+jid+'（已解析的分片会自动跳过，只补蒸馏/入库）';
  $('iout').textContent='';pollIngest(jid);loadIngestHistory();
}
async function clearIngest(jid,df){
  if(!confirm(df?'确认删除该记录及其源文件/解析产物？':'确认清除该导入记录？'))return;
  await fetch('/api/ingest/'+jid+(df?'?delete_files=1':''),{method:'DELETE'});
  $('istat').textContent='已清除任务 '+jid;
  loadIngestHistory();
}
async function startIngest(){
  const f=$('ifile').files[0];
  $('istat').textContent= f?('准备提交：'+f.name):'请先选择文件';
  if(!f)return;
  const fd=new FormData();
  fd.append('course', $('icourse').value.trim()||$('course').value);
  fd.append('file', f);
  fd.append('parser','mineru-http');
  if($('islice').value)fd.append('slice_pages',$('islice').value);
  if($('ibatch').value)fd.append('batch_files',$('ibatch').value);
  if($('ipara').value)fd.append('parallel',$('ipara').value);
  $('istat').textContent='提交中……';$('iout').textContent='';
  try{
    const r=await fetch('/api/ingest',{method:'POST',body:fd});
    const j=await r.json();
    if(!r.ok)throw new Error(JSON.stringify(j.detail||j));
    $('istat').textContent='任务 '+j.job_id+' 已启动（实时日志如下）';
    pollIngest(j.job_id);loadIngestHistory();
  }catch(e){$('istat').textContent='提交失败：'+e.message;}
}
async function pollIngest(jid){
  if(ingestTimer)clearInterval(ingestTimer);
  const seen={};
  ingestTimer=setInterval(async()=>{
    try{
      const j=await (await fetch('/api/ingest/'+jid)).json();
      const o=$('iout');
      (j.log||[]).forEach(l=>{if(!seen[l]){seen[l]=1;o.textContent+=l+'\n';}});
      o.scrollTop=o.scrollHeight;
      $('istat').textContent='状态：'+(j.status==='running'?(j.exit==null?'运行中':'结束') : j.status==='done'?'✅ 导入完成！可到「学习进展/复习」查看':'❌ 失败')+(j.error?('　'+j.error):'');
      if(j.status==='done'||j.status==='error'){
        clearInterval(ingestTimer);ingestTimer=null;loadIngestHistory();
        if(j.status==='done'){$('istat').textContent='✅ 导入完成！新课程「'+$('icourse').value.trim()+'」已入库，可切换到「学习进展 / 复习 · 打卡」使用。';}
      }
    }catch(e){}
  },2000);
}
$('q').addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();askStream();}});
</script></body></html>"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

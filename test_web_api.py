"""R2 API 壳离线测试：FastAPI TestClient + FakeLLM，零网络。

运行：conda activate firpro && cd coursebook-digest && python test_web_api.py
覆盖：health/courses、teach（在线注入+写回）、teach 离线降级、profile、daily_cards。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from coursebook_digest.agent import TeacherAgent
from coursebook_digest.api_server import create_app
from coursebook_digest.config import PROJECT_ROOT, Settings


class FakeLLM:
    def __init__(self) -> None:
        self.classify = {"intent": "答疑", "keywords": [], "wants_memory": True,
                         "stuck_topics": ["死锁"], "mistakes": ["资源顺序"], "expression_note": ""}
        self.extract = {"knowledge": [{"topic": "死锁", "content": "死锁四必要条件方法"}],
                        "thinking": [], "technique": [], "semantics": []}
        self.last_user = ""

    def parse_json(self, system: str, user: str) -> dict:  # noqa: ARG002
        if "意图路由" in system:
            return self.classify
        if "记忆提炼器" in system:
            return self.extract
        return {"cards": []}

    def complete_text(self, system: str, user: str) -> str:
        self.last_user = user
        return "（假模型）按课本方法回答死锁。"

    def complete_text_stream(self, system: str, user: str):
        self.last_user = user
        ans = "（假流式）按课本方法回答死锁。"
        for i in range(0, len(ans), 6):
            yield ans[i:i + 6]


def _make_env(run_dir: Path):
    fake = FakeLLM()
    tmp = run_dir / "users"

    def factory(user_id: str, course: str, session: str = ""):
        return TeacherAgent(user_id=user_id, course=course, llm=fake,
                            settings=Settings(_env_file=None), user_data_dir=tmp, session=session)

    app = create_app(agent_factory=factory)
    return app, fake


def test_health_and_courses(run_dir: Path) -> None:
    app, _ = _make_env(run_dir)
    with TestClient(app) as c:
        h = c.get("/api/health")
        assert h.status_code == 200 and h.json()["status"] == "ok"
        names = [x["name"] for x in c.get("/api/courses").json()]
        assert "操作系统" in names
    print("    [api] health / courses OK")


def test_teach_online_writeback(run_dir: Path) -> None:
    app, fake = _make_env(run_dir)
    with TestClient(app) as c:
        r = c.post("/api/teach", json={
            "question": "讲一下死锁的必要条件", "user_id": "stuA", "course": "操作系统"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["offline"] is False
        assert "优先参考的教材方法" in fake.last_user  # 方法卡已注入
        assert data["methods"], "应命中课本方法"
        assert data["writeback"]["reason"] and "后台" in data["writeback"]["reason"]

        # 后台写回线程稍后落盘；轮询确认记忆库出现知识卡
        import time
        from pathlib import Path

        from coursebook_digest.agent import bucket_course
        from coursebook_digest.store import CourseStore

        mem_settings = _mem_settings(run_dir)
        for _ in range(20):
            time.sleep(0.3)
            if CourseStore(bucket_course("stuA", "知识"), mem_settings).load_all():
                break
        assert len(CourseStore(bucket_course("stuA", "知识"), mem_settings).load_all()) == 1
        # 幂等：同内容再写回不新增知识卡
        c.post("/api/teach", json={
            "question": "死锁必要条件再讲一遍", "user_id": "stuA", "course": "操作系统"})
        for _ in range(20):
            time.sleep(0.3)
        assert len(CourseStore(bucket_course("stuA", "知识"), mem_settings).load_all()) == 1
        # 长对话上下文：第二轮 prompt 应带上第一轮的问题
        assert "讲一下死锁的必要条件" in fake.last_user
    print("    [api] teach 在线（注入+后台写回+幂等+上下文） OK")


def _mem_settings(run_dir: Path):
    # TeacherAgent 的存储基目录是 users/data（用户按课程名「用户·uid·版块」隔离）
    return Settings(data_dir=str(run_dir / "users" / "data"),
                    chroma_dir=str(run_dir / "users" / "data" / "chroma"),
                    _env_file=None)


def test_teach_offline_fallback(run_dir: Path) -> None:
    def factory_offline(user_id: str, course: str, session: str = ""):
        return TeacherAgent(user_id=user_id, course=course,
                            settings=Settings(llm_api_key="", _env_file=None),
                            user_data_dir=run_dir / "usersOff", session=session)
    app = create_app(agent_factory=factory_offline)
    with TestClient(app) as c:
        r = c.post("/api/teach", json={
            "question": "什么是死锁", "user_id": "off", "course": "操作系统"})
        assert r.status_code == 200
        d = r.json()
        assert d["offline"] is True and "离线降级" in d["answer"]
        assert d["writeback"]["skipped"] is True
    print("    [api] teach 离线降级 OK")


def test_teach_stream_sse(run_dir: Path) -> None:
    """SSE 流式：intent → methods → delta* → done(+profile)。"""
    app, _ = _make_env(run_dir)
    with TestClient(app) as c:
        r = c.post("/api/teach_stream", json={
            "question": "什么是死锁", "user_id": "stuS", "course": "操作系统"})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = []
        for frame in r.text.split("\n\n"):
            for line in frame.split("\n"):
                if line.startswith("data: "):
                    events.append(__import__("json").loads(line[6:]))
        kinds = [e["type"] for e in events]
        assert kinds[0] == "intent" and kinds[0] and "methods" in kinds
        assert any(k == "delta" for k in kinds), "应有逐段 delta"
        done = events[-1]
        assert done["type"] == "done" and "profile" in done
        joined = "".join(e["text"] for e in events if e["type"] == "delta")
        assert joined
    print(f"    [api] teach_stream SSE OK（事件: {kinds}）")


def test_progress_endpoint(run_dir: Path) -> None:
    """下学习进展：/api/progress 返回 percent/章节/曲线/每日统计。"""
    app, _ = _make_env(run_dir)
    with TestClient(app) as c:
        c.post("/api/teach", json={
            "question": "讲一下死锁的必要条件", "user_id": "stuA", "course": "操作系统"})
        import time
        time.sleep(0.5)
        r = c.get("/api/progress", params={"user_id": "stuA", "course": "操作系统"})
        assert r.status_code == 200, r.text
        d = r.json()
        for k in ("percent", "questions", "chapters", "curve", "daily", "today_questions"):
            assert k in d, f"缺字段 {k}"
        assert isinstance(d["percent"], int) and 0 <= d["percent"] <= 100
        assert d["questions"] >= 1, "课程问答应计入进度"
    print(f"    [api] 学习进展接口 OK（percent={d['percent']}%）")


def test_ingest_route(run_dir: Path) -> None:
    """导书入库接口：非法类型 400；未知任务 404（不触发真实解析子进程）。"""
    app, _ = _make_env(run_dir)
    with TestClient(app) as c:
        r = c.post("/api/ingest", files={"file": ("x.txt", b"hi")}, data={"course": "c"})
        assert r.status_code == 400 and "不支持" in r.text
        assert c.get("/api/ingest/nope").status_code == 404
    print("    [api] 导书入库接口（类型校验/404） OK")


def test_ingest_history_endpoints(run_dir: Path) -> None:
    """导入历史/续跑/删除接口（隔离 uploads 目录，不触发真实子进程）。"""
    from coursebook_digest.api_server import create_app

    def factory_off(u: str, c: str, session: str = ""):
        return TeacherAgent(user_id=u, course=c, settings=Settings(_env_file=None),
                            user_data_dir=run_dir / "usersH", session=session)

    app = create_app(agent_factory=factory_off, ingest_dir=run_dir / "ingest")
    with TestClient(app) as c:
        assert c.get("/api/ingest_history").json() == []
        assert c.get("/api/ingest/nope").status_code == 404
        assert c.post("/api/ingest/nope/resume").status_code == 404
        assert c.delete("/api/ingest/nope").json() == {"deleted": "nope"}
        # 手工写一条历史 → 能被读到
        hp = run_dir / "ingest" / "_history.json"
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text('{"a":{"jid":"a","course":"X","filename":"a.pdf","status":"done","cards":3,"started_at":"2026-01-01T00:00:00"}}',
                      encoding="utf-8")
        rows = c.get("/api/ingest_history").json()
        assert rows and rows[0]["jid"] == "a" and rows[0]["cards"] == 3
    print("    [api] 导入历史/续跑/删除接口 OK")


def test_session_endpoints(run_dir: Path) -> None:
    """会话记录：新建/切换/清空/删除。"""
    app, _ = _make_env(run_dir)
    with TestClient(app) as c:
        nw = c.post("/api/sessions/new", json={"user_id": "stuA", "course": "操作系统", "title": "期末冲刺"})
        assert nw.status_code == 200
        sid = nw.json()["sid"]
        ls = c.get("/api/sessions", params={"user_id": "stuA", "course": "操作系统"}).json()
        assert ls["current"] == sid and sid in [s["sid"] for s in ls["sessions"]]
        assert c.post("/api/sessions/switch",
                      json={"user_id": "stuA", "course": "操作系统", "sid": "default"}).json()["ok"]
        assert c.post("/api/sessions/" + sid + "/clear",
                      json={"user_id": "stuA", "course": "操作系统", "sid": sid}).status_code == 200
        dl = c.delete("/api/sessions/" + sid, params={"user_id": "stuA", "course": "操作系统"})
        assert dl.status_code == 200 and dl.json()["current"] == "default"
    print("    [api] 会话 新建/切换/清空/删除 OK")


def test_profile_and_daily(run_dir: Path) -> None:
    app, _ = _make_env(run_dir)
    with TestClient(app) as c:
        pr = c.get("/api/profile", params={"user_id": "stuA", "course": "操作系统"})
        assert pr.status_code == 200 and pr.json()["turn_count"] >= 1
        da = c.post("/api/daily_cards", json={"user_id": "stuA", "course": "操作系统", "n": 4})
        assert da.status_code == 200 and len(da.json()) <= 4
        # GET 变体 + 按章节筛选
        g = c.get("/api/daily_cards", params={"user_id": "stuA", "course": "操作系统", "n": 3})
        assert g.status_code == 200 and len(g.json()) <= 3
        ch = g.json()[0]["chapter"] if g.json() else "操作系统"
        gf = c.get("/api/daily_cards", params={"user_id": "stuA", "course": "操作系统", "n": 6, "chapter": ch})
        assert gf.status_code == 200 and all(ca["chapter"] == ch for ca in gf.json())
        # 打卡卡带 id 且复习自评接口生效
        card = g.json()[0]
        assert "id" in card
        fb = c.post("/api/review_feedback", json={"user_id": "stuA", "course": "操作系统",
                                                  "card_id": card["id"], "ok": True, "chapter": card["chapter"]})
        assert fb.status_code == 200 and fb.json()["ok"] and fb.json()["box"] == 2
        pr2 = c.get("/api/profile", params={"user_id": "stuA", "course": "操作系统"}).json()
        assert pr2["reviews_stats"]["total"] >= 1
    print("    [api] profile / daily_cards（GET+按章节+自评） OK")


def main() -> None:
    run_dir = PROJECT_ROOT / ".smoke" / f"webapi-{os.getpid()}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        test_health_and_courses(run_dir)
        test_teach_online_writeback(run_dir)
        test_teach_offline_fallback(run_dir)
        test_teach_stream_sse(run_dir)
        test_progress_endpoint(run_dir)
        test_ingest_route(run_dir)
        test_ingest_history_endpoints(run_dir)
        test_session_endpoints(run_dir)
        test_profile_and_daily(run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    print("\nR2 Web API 离线测试全部通过")


if __name__ == "__main__":
    main()

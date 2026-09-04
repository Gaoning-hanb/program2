"""学生画像：可持久化的个性化配置，agent 每轮读取并在问答后更新。

画像不是“用户资料表”，而是**把历史问答沉淀成可注入的教授风格偏好**：
讲解深度(level)、检索偏向(kind_bias)、讲解风格、薄弱主题、常见错误、措辞习惯。
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

MISTAKES_CAP = 12
TOPICS_CAP = 12


class ChapterProgress(BaseModel):
    """某一章节的学习进度（该生在本章的提问量/深度）。"""

    questions: int = 0
    depth_sum: float = 0.0
    depth_max: float = 0.0
    last_ts: str = ""


class CourseProgress(BaseModel):
    """某课程的实时学习进度：章节覆盖 + 深度 + 时间曲线。"""

    chapters: dict[str, ChapterProgress] = Field(default_factory=dict)
    questions: int = 0
    curve: list[dict] = Field(default_factory=list)  # [{"ts":..,"q":N,"pct":P}]

    def percent(self) -> int:
        if not self.chapters:
            return 0
        avg = sum(c.depth_max for c in self.chapters.values()) / len(self.chapters)
        return round(min(avg, 1.0) * 100)


class UserProfile(BaseModel):
    """一名学生可持久化的画像（本地 json）。"""

    user_id: str
    level: str = "初级"  # 初级 | 中级 | 高级
    kind_bias: str = "均衡"  # 概念 | 方法 | 均衡 —— 检索重排偏置
    style_flags: dict[str, bool] = Field(
        default_factory=lambda: {"直观比喻": False, "例题": False, "推导": True, "易错点": True}
    )
    weak_topics: list[str] = Field(default_factory=list)  # 反复卡壳的主题（课程）
    common_mistakes: list[str] = Field(default_factory=list)  # 该生历史常错点（课程）
    general_notes: list[str] = Field(default_factory=list)  # 通识/日常话题（与课程记忆严格分离）
    expression_notes: str = ""  # 语义表意浓缩（几十字）
    course_progress: dict[str, CourseProgress] = Field(default_factory=dict)  # 按课程的学习进度
    review_boxes: dict[str, int] = Field(default_factory=dict)  # 卡 id → Leitner 盒号（1 起）
    review_log: list[dict] = Field(default_factory=list)  # [{ts, card_id, chapter, ok}]
    chat_history: list[dict] = Field(default_factory=list)  # （旧版单会话）迁移用
    sessions: dict[str, dict] = Field(default_factory=dict)  # 多会话：sid → {title, created, updated, history[]}
    current_session: str = "default"  # 当前会话
    turn_count: int = 0
    updated_at: str = ""

    # ------------------------------------------------------------------ #
    def to_preamble(self) -> str:
        """渲染成注入到 system 的“教师须知”片段（轻量、每轮必带）。

        语义是【静默参考】：历史信息只用来调节讲解方式与深度，
        明令“不要复述/点名学生历史”，避免每轮输出都正式罗列他的易错点。
        """
        tops = "、".join(self.weak_topics[:6])
        mis = "、".join(self.common_mistakes[:6])
        parts = [f"你是面对一位{self.level}水平学生的一对一助教。"]
        if self.weak_topics or self.common_mistakes:
            seg = "该生历史表现："
            if self.weak_topics:
                seg += f"在『{tops}』上易卡壳，讲解时可放慢节奏、多给直观对照；"
            if self.common_mistakes:
                seg += f"曾犯『{mis}』类错误，讲解时注意绕开这些坑。"
            seg += "以上信息只用于调节讲解方式，**不要专门复述或点名学生的这些历史**。"
            parts.append(seg)
        flags = [k for k, v in self.style_flags.items() if v]
        if flags:
            parts.append("讲解风格：" + "、".join(flags) + "。")
        if self.expression_notes:
            parts.append(f"措辞上他习惯：{self.expression_notes}；讲解时可自然顺应，不必特意提及。")
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    def ingest_turn(self, stuck_topics: list[str] | None = None,
                    mistakes: list[str] | None = None,
                    expression_note: str = "") -> None:
        """一轮问答后更新画像（幂等、去重、封顶）。"""
        self.turn_count += 1
        for t in stuck_topics or []:
            t = (t or "").strip()
            if t and t not in self.weak_topics:
                self.weak_topics.append(t)
        for m in mistakes or []:
            m = (m or "").strip()
            if m and m not in self.common_mistakes:
                self.common_mistakes.append(m)
        if expression_note and expression_note not in self.expression_notes:
            self.expression_notes = (self.expression_notes + "；" + expression_note).strip("；")[:200]
        self.weak_topics = self.weak_topics[-TOPICS_CAP:]
        self.common_mistakes = self.common_mistakes[-MISTAKES_CAP:]
        self.updated_at = datetime.datetime.now().isoformat(timespec="seconds")

    def add_general(self, topics: list[str] | None) -> None:
        """通识/日常话题：只写 general_notes，绝不触碰课程 weak_topics/common_mistakes。"""
        for t in topics or []:
            t = (t or "").strip()[:30]
            if t and t not in self.general_notes:
                self.general_notes.append(t)
        self.general_notes = self.general_notes[-20:]
        self.turn_count += 1
        self.updated_at = datetime.datetime.now().isoformat(timespec="seconds")

    def note_course_progress(self, course: str, chapter: str, depth: float) -> None:
        """实时记录某课程某章节的一次学习进展（每轮课程问答后调用）。"""
        cp = self.course_progress.setdefault(course, CourseProgress())
        ch = cp.chapters.setdefault(chapter or "(未分章)", ChapterProgress())
        ch.questions += 1
        ch.depth_sum += depth
        ch.depth_max = max(ch.depth_max, depth)
        ch.last_ts = datetime.datetime.now().isoformat(timespec="seconds")
        cp.questions += 1
        cp.curve.append({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "q": cp.questions, "pct": cp.percent(),
        })
        cp.curve = cp.curve[-500:]
        self.updated_at = datetime.datetime.now().isoformat(timespec="seconds")

    # ------------------- 复习自评（间隔重复的铺垫） ------------------- #
    def record_review(self, card_id: str, ok: bool, chapter: str = "") -> int:
        """记录一次“答对/答错”，维护卡片的 Leitner 盒号（为将来间隔重复打底），返回新盒号。"""
        box = int(self.review_boxes.get(card_id, 1))
        box = min(box + 1, 5) if ok else 1
        self.review_boxes[card_id] = box
        self.review_log.append({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "card_id": card_id, "chapter": chapter, "ok": bool(ok), "box": box,
        })
        self.review_log = self.review_log[-1000:]
        self.updated_at = datetime.datetime.now().isoformat(timespec="seconds")
        return box

    def reviews_stats(self) -> dict:
        total = len(self.review_log)
        correct = sum(1 for r in self.review_log if r.get("ok"))
        return {
            "total": total, "correct": correct, "wrong": total - correct,
            "accuracy": round(correct / total, 2) if total else 0.0,
            "boxes": len(self.review_boxes),
        }

    # ------------------- 多会话（长对话上下文） ------------------- #
    def _ses(self, sid: str | None = None) -> str:
        """确保会话表存在并返回有效 sid（首次使用会把旧单会话历史迁入默认会话）。"""
        if not self.sessions:
            self.sessions["default"] = {
                "title": "默认会话",
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "updated": self.updated_at or datetime.datetime.now().isoformat(timespec="seconds"),
                "history": list(self.chat_history),
            }
            self.current_session = "default"
        sid = sid or self.current_session
        return sid if sid in self.sessions else "default"

    def sessions_list(self) -> list[dict]:
        """返回会话元信息（不含历史），按最近更新排序。"""
        self._ses()
        out = []
        for sid, s in self.sessions.items():
            out.append({"sid": sid, "title": s.get("title", sid),
                        "count": len(s.get("history", [])),
                        "created": s.get("created", ""), "updated": s.get("updated", "")})
        out.sort(key=lambda x: x["updated"], reverse=True)
        return out

    def new_session(self, title: str | None = None) -> str:
        self._ses()
        sid = hashlib.md5(f"{title or ''}{datetime.datetime.now().timestamp()}".encode("utf-8")).hexdigest()[:8]
        now = datetime.datetime.now().isoformat(timespec="seconds")
        self.sessions[sid] = {"title": title or f"会话 {len(self.sessions) + 1}",
                              "created": now, "updated": now, "history": []}
        self.current_session = sid
        self.updated_at = now
        return sid

    def set_current_session(self, sid: str) -> None:
        self._ses()
        if sid in self.sessions:
            self.current_session = sid

    def clear_session(self, sid: str | None = None) -> None:
        sid = self._ses(sid)
        self.sessions[sid]["history"] = []
        self.sessions[sid]["updated"] = datetime.datetime.now().isoformat(timespec="seconds")

    def delete_session(self, sid: str) -> None:
        self._ses()
        if sid in self.sessions and sid != "default":
            del self.sessions[sid]
        if self.current_session not in self.sessions:
            self.current_session = "default"

    def append_chat(self, question: str, answer: str) -> None:
        """"把一轮问答追加到【当前会话】的历史里（多会话互不串味；每会话最多保留 12 轮）。"""
        sid = self._ses()
        s = self.sessions[sid]
        # 会话标题自动生成：还是占位名（新会话/默认会话/会话N）且本会话尚未起名时，用第一问生成简短标题
        if not s.get("title") or str(s.get("title", "")).lstrip().startswith(("新会话", "会话 ", "默认会话")):
            qq = (question or "").strip().replace("\n", " ")
            s["title"] = (qq[:14] + ("…" if len(qq) > 14 else "")) if qq else "新会话"
        s.setdefault("history", []).append({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "u": (question or "")[:2000], "a": (answer or "")[:8000],
        })
        s["history"] = s["history"][-12:]
        s["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.updated_at = datetime.datetime.now().isoformat(timespec="seconds")

    def chat_context(self, n: int = 5) -> str:
        """渲染【当前会话】最近 n 轮问答作为上下文注入（仅供衔接，勿当作新提问）。"""
        sid = self._ses()
        hist = self.sessions[sid].get("history", [])
        if not hist:
            return ""
        rows = []
        for m in hist[-n:]:
            rows.append(f"学生问：{m['u'][:300]}\n助教答：{m['a'][:500]}")
        return ("【近期对话 · 仅供衔接上下文，切勿把其中内容当作本次新提问】\n"
                + "\n\n".join(rows))


def profile_path(base: Path, user_id: str) -> Path:
    return Path(base) / f"{user_id}.json"


def load_profile(base: Path, user_id: str) -> UserProfile:
    p = profile_path(base, user_id)
    if p.exists():
        try:
            return UserProfile.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— 画像损坏则重置
            pass
    return UserProfile(user_id=user_id)


def save_profile(base: Path, profile: UserProfile) -> None:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    profile_path(base, profile.user_id).write_text(
        profile.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

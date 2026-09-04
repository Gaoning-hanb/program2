"""TeacherAgent：受控版 agent 循环——让“方法库工具”变成“懂这位学生的私人教师”。

一轮 turn 的管线（可逐阶段打点/测试）：
  ① 意图分类(_classify) ② 画像注入(_compose) ③ 双库检索+偏置重排(_retrieve)
  ④ 作答(_answer，复用 answer.build_prompt/ask 语义) ⑤(可选)自检 ⑥ 写回记忆+画像(_write_back)

- 课本检索复用 find_methods；个人记忆复用 CourseStore/VectorStore（用户级 data_dir 隔离）。
- 无 API Key/模型不可用时**离线降级**：意图走规则、作答退化为方法卡预览、写回关闭，
  保证演示(或断网)现场不会失败。
- 全部不修改现有 distill/store/retrieve/answer 的默认行为（加法实现）。
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .answer import build_prompt, render_methods
from .config import Settings, get_settings
from .llm import LLMClient
from .profile import UserProfile, load_profile, save_profile
from .retrieve import find_methods
from .schema import MethodCard, MethodKind, RetrievedMethod
from .store import CourseStore

# --------------------------------------------------------------------------- #
# 意图 / 常量
# --------------------------------------------------------------------------- #
class Intent(str, Enum):
    ANSWER = "答疑"     # 解疑/计算/证明
    CONCEPT = "概念"    # 问定义/是什么
    QUIZ = "出题"       # 要求练习题/打卡
    REVIEW = "复习"     # 回顾/总结
    CHAT = "闲聊"       # 问候/道谢等非学科内容
    GENERAL = "通识"    # 常识/课纲外问题：仍正常回答，不注入课本方法，记忆独立建档

BUCKETS = ("知识", "思维习惯", "技巧选择", "语义表意")
GENERAL_BUCKET = "通识"
_BUCKET_KIND = {  # 版块 → MethodCard.kind
    "知识": MethodKind.CONCEPT,
    "思维习惯": MethodKind.METHOD,
    "技巧选择": MethodKind.METHOD,
    "语义表意": MethodKind.CONCEPT,
    GENERAL_BUCKET: MethodKind.CONCEPT,
}


def bucket_course(user_id: str, bucket: str) -> str:
    return f"用户·{user_id}·{bucket}"


class IntentResult(BaseModel):
    intent: str = "答疑"
    keywords: list[str] = Field(default_factory=list)
    wants_memory: bool = True
    stuck_topics: list[str] = Field(default_factory=list)
    mistakes: list[str] = Field(default_factory=list)
    expression_note: str = ""


@dataclass
class WritebackSummary:
    skipped: bool = False
    reason: str = ""
    buckets_written: dict[str, int] = field(default_factory=dict)


@dataclass
class TurnResult:
    question: str
    intent: Intent
    methods: list[RetrievedMethod]
    memory_hits: list[RetrievedMethod]
    answer: str
    offline: bool
    writeback: WritebackSummary


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #
INTENT_SYSTEM = """你是大学教学助教系统的“意图路由”。判断学生一句话属于哪一类：
- 答疑：解题/计算/证明/求过程（默认）
- 概念：问定义/“是什么”/概念辨析/整体认识
- 出题：要求出题/练习/模拟考/打卡题
- 复习：要求回顾/复习/总结今天内容
- 通识：与课程无关的常识/日常/实时问题（天气、新闻、生活常识等）
- 闲聊：问候/道谢等寒暄
输出 JSON（只输出 JSON）：
{"intent":"答疑|概念|出题|复习|通识|闲聊","keywords":["术语"],"wants_memory":true,
 "stuck_topics":["疑似该生薄弱主题(可空)"],"mistakes":["疑似该生上次易错点(可空)"],
 "expression_note":"观察到的用词/表述习惯(可空,高阈值)"}"""

EXTRACT_SYSTEM = """你是私人助教的“记忆提炼器”。把下面这一轮 学生问题+教师回答 提炼为四类高价值记忆，
**宁缺毋滥**：模糊/泛泛的话不记；只有明确体现“这位学生的”特征才记。

输出 JSON（只输出 JSON），每类数组可空：
{
 "knowledge": [{"topic":"知识点", "content":"这轮产生的课本事实/定义/结论要点"}],
 "thinking":   [{"topic":"特征名", "content":"该生体现的思维路径/推理习惯/卡壳点"}],
 "technique":  [{"topic":"技巧名", "content":"本轮采用或该生倾向的解题技巧"}],
 "semantics":  [{"topic":"用语名", "examples":["该生用过的原话/用词"]}]
}"""

# 方法卡被视为“确实适用”的最低相关度；个人记忆可注入的最低相关度
RELEVANCE_OK = 0.40
MEMORY_OK = 0.45

# “软模式”：整体认识/概念类（或方法相关度不足）时，先用自己的话讲清，课本仅作对照
SOFT_SYSTEM = """你是一位大学「{course}」课程助教。学生这次的诉求是【整体认识/概念理解】，请：
1) 先用你自己的话把问题讲清楚（先给总体框架与直觉，再展开要点），条理清晰、语言平实；
2) 下方【教材方法】是从课本蒸馏出的相关内容卡片——只有当其中某张与本题【确实相关】时才引用它佐证/补充课本说法；
3) 相关度不高就**不要**强行套用卡片里的步骤或公式，正常用自己的话作答；
4) 该生历史弱点仅供你调节讲解方式（放慢节奏、多给对照、绕开已犯过的坑），**不要**在回答中复述或点名这些历史，也不要写“根据你以前/根据你过往记忆”这类话。
不要机械复述卡片；目标是把“它到底是什么/用来做什么”讲得让人真正理解。"""

# 全体模式通用：历史记忆静默使用，禁止当正文复述
OUTPUT_DISCIPLINE = (
    "【输出纪律】该生的历史问答/易错点仅供你调节讲解深度与方向，"
    "切勿在回答中复述或逐条罗列，也不要用“根据你之前…”之类开头；请自然地融入讲解。"
)

# 会话标题自动总结（LLM 后台润色，≤14 字概括主题）
TITLE_SYSTEM = (
    "你是对话主题命名员。根据下面的师生问答，用不超过 14 个字的标题概括主题"
    "（例如“银行家算法与死锁避免”“虚拟内存与页面置换”）。只输出 JSON：{\"title\":\"...\"}"
)

# 通识/课纲外：不注入课本方法，正常用通用知识作答
GENERAL_SYSTEM = (
    "你是学生「{course}」学习助教背后的通用AI助手。学生在课程之外也会问常识性问题，请：\n"
    "1) 用你的一般知识正常、友好地回答即可，**不要虚构课本/课程资料**；\n"
    "2) 不需要强调“只能教这门课”或“无法搜索”；能用一般知识答就答；\n"
    "3) 只有需要实时数据（如当前天气/新闻）且你确实没有实时渠道时，才简短说明“没有实时信息渠道”，"
    "但仍尽量补充一般性背景知识；\n"
    "4) 这是日常对话，风格自然亲切，不必套用教学步骤。"
)

# 通识问答的记忆提炼（独立于课程记忆，只记录对该生的“通识话题/表达”）
GENERAL_EXTRACT = """你是私人助教的“日常对话提炼器”。把下面这轮师生对话里，属于【学生个人层面的通识话题/偏好/措辞】的要点提炼出来（课程知识、易错点一律不记）。
宁缺毋滥，输出 JSON：
{"items":[{"topic":"话题", "note":"一句话说明，如该生关心什么/怎么表达"}]}"""


# --------------------------------------------------------------------------- #
# 偏置重排（纯函数，便于单测）
# --------------------------------------------------------------------------- #
def rerank_by_bias(methods: list[RetrievedMethod], bias: str) -> list[RetrievedMethod]:
    """按画像 kind_bias 对命中结果做同分偏置（概念型 vs 方法型）。"""
    if bias == "均衡" or not methods:
        return methods
    favored = ("概念", "定理") if bias == "概念" else ("方法", "例题")
    scored = [(m, m.score + (0.25 if m.card.kind.value in favored else 0.0)) for m in methods]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [m for m, _ in scored]


def _method_brief(m: RetrievedMethod) -> dict:
    """流式事件里用的精简方法摘要。"""
    c = m.card
    return {"topic": c.topic, "kind": c.kind.value, "chapter": c.chapter, "score": round(m.score, 4)}


def _question_depth(question: str, intent: str, methods: list[RetrievedMethod]) -> float:
    """启发式“问题深度分”0~1：意图 + 长度 + 术语/公式线索 + 命中强度（无需额外模型调用）。"""
    d = {"答疑": 0.55, "概念": 0.35, "复习": 0.30, "出题": 0.50}.get(intent, 0.30)
    q = question.strip()
    if len(q) >= 20:
        d += 0.10
    if re.search(r"证明|推导|为什么|原理|机制|本质|区别|比较|公式|计算|深入|如何实现|算法", q):
        d += 0.15
    if re.search(r"[\\$_]|\b[A-Za-z]{2,}\b|_[A-Za-z]|frac|\d{2,}|方程|定理", q):
        d += 0.10
    top = max((m.score for m in methods), default=0.0)
    d += min(top, 1.0) * 0.10
    return min(d, 1.0)


def _rule_intent(question: str) -> IntentResult:
    q = question.strip()
    if re.match(r"^(你好|您好|嗨|hi|hello|谢谢|再见|哈喽)", q, re.I):
        return IntentResult(intent=Intent.CHAT.value, wants_memory=False)
    if re.search(r"天气|气温|预报|新闻|热点|热搜|股票|股市|大盘|行情|汇率|彩票|几点|星期几|几号|外卖|淘宝|京东|抖音|微博|打车|导航|票价|多少钱|哪个(好|看)|推荐.{0,6}(电影|游戏|餐厅|书|歌)|怎么样$", q):
        return IntentResult(intent=Intent.GENERAL.value, wants_memory=False)
    if re.search(r"复习|回顾|总结(一下|今天)|\breview\b", q, re.I):
        return IntentResult(intent=Intent.REVIEW.value)
    if re.search(r"(出|来|练).{0,4}(几道|一道|十道|题|打卡)|模拟题|随机出", q):
        return IntentResult(intent=Intent.QUIZ.value)
    if re.search(r"什么是|是什么|定义|概念|区别|辨析|到底是|做什么|用来(做什么|干嘛|干什么)|干嘛|干什么|整体|概述|怎么理解|通俗|简单说|大概|为什么", q):
        return IntentResult(intent=Intent.CONCEPT.value)
    return IntentResult(intent=Intent.ANSWER.value)


def _memory_course_names(user_id: str) -> list[str]:
    return [bucket_course(user_id, b) for b in BUCKETS]


# --------------------------------------------------------------------------- #
# TeacherAgent
# --------------------------------------------------------------------------- #
class TeacherAgent:
    """受控 agent 循环。每轮 = 分类 → 画像 → 检索(课本+记忆) → 作答 → 写回。"""

    def __init__(
        self,
        user_id: str = "local",
        course: str = "",
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        write_back: bool = True,
        user_data_dir: str | Path | None = None,
        study_k: int | None = None,
        memory_k: int = 2,
        session: str = "",
    ) -> None:
        self.settings = settings or get_settings()
        self.course = course or ""
        self.user_id = user_id
        user_dir = Path(user_data_dir) if user_data_dir else Path(self.settings.data_dir) / "users"
        self.user_dir = Path(user_dir)
        self.memory_settings = Settings(
            data_dir=str(self.user_dir / "data"),
            chroma_dir=str(self.user_dir / "data" / "chroma"),
            _env_file=None,
        )
        self.memory_settings.storage = "jsonl"  # 记忆库暂只落 jsonl（词面检索足够）
        self.profile: UserProfile = load_profile(self.user_dir, user_id)
        if session:  # 指定会话（多会话聊天；上下文按会话隔离）
            self.profile.set_current_session(session)
        self.study_k = study_k or self.settings.top_k_default
        self.memory_k = memory_k
        self.write_back = write_back
        try:
            self.llm = llm or LLMClient(self.settings)
        except ValueError:
            self.llm = None  # 无 key → 离线降级模式

    # ---------------- 主入口 ---------------- #
    def turn(self, question: str, *,
             top_k: int | None = None,
             memory_k: int | None = None,
             intent_hint: IntentResult | None = None,
             write_back_async: bool = False) -> TurnResult:
        store_dir = self.user_dir / "data"
        store_dir.mkdir(parents=True, exist_ok=True)
        ir = intent_hint or self._classify(question)
        try:
            intent_enum = Intent(ir.intent)
        except ValueError:  # 模型返回非法的 intent → 默认答疑
            intent_enum = Intent.ANSWER
        gen_intent = ir.intent in (Intent.GENERAL.value, Intent.CHAT.value)
        if gen_intent:  # 通识/闲聊：直接答，不检索课本，避免方法卡干扰
            methods, memory = [], []
        else:
            methods, memory = self._retrieve(question, ir, top_k=top_k, memory_k=memory_k)
        general = gen_intent or not methods
        self._note_progress(question, ir.intent, methods, general)
        answer, offline = self._answer(question, ir.intent, methods, memory)
        wb = WritebackSummary(skipped=True, reason="离线降级(无模型)" if offline else "")
        if self.write_back and not offline and self.llm is not None:
            if intent_enum == Intent.CHAT:
                wb = WritebackSummary(skipped=True, reason="闲聊不写回")
            elif write_back_async:  # 先回作答，记忆/画像是后台补写（交互更快）
                wb = WritebackSummary(skipped=False, reason="后台写回进行中")
                threading.Thread(
                    target=self._write_back,
                    args=(question, answer, ir, general),
                    daemon=True,
                ).start()
            else:
                wb = self._write_back(question, answer, ir, general=general)
        if not offline:  # 长对话：把本轮问答写进上下文历史（下一轮可衔接）
            self.profile.append_chat(question, answer)
            save_profile(self.user_dir, self.profile)
        return TurnResult(
            question=question, intent=intent_enum,
            methods=methods, memory_hits=memory, answer=answer,
            offline=offline, writeback=wb,
        )

    # ---------------- 流式入口（SSE 用）---------------- #
    def turn_stream(self, question: str, *,
                    top_k: int | None = None,
                    memory_k: int | None = None):
        """逐段产出事件 dict：intent → methods → delta* → done（供 /api/teach_stream 推 SSE）。

        意图走本地规则（免 1 次模型排队，优化“首屏”），作答逐个 delta 推送；
        全部生成完再后台写回记忆/画像。
        """
        store_dir = self.user_dir / "data"
        store_dir.mkdir(parents=True, exist_ok=True)
        ir = _rule_intent(question)
        yield {"type": "intent", "intent": ir.intent}
        # 通识/闲聊：跳检索（方法/记忆帧为空）；其余先检索再推
        gen_intent = ir.intent in (Intent.GENERAL.value, Intent.CHAT.value)
        if gen_intent:
            methods, memory = [], []
            yield {"type": "methods", "methods": [], "memory": []}
        else:
            methods, memory = self._retrieve(question, ir, top_k=top_k, memory_k=memory_k)
            yield {"type": "methods", "methods": [_method_brief(m) for m in methods[:5]],
                   "memory": [_method_brief(m) for m in memory[:3]]}
        general = gen_intent or not methods

        if self.llm is None:  # 离线降级
            body = render_methods(methods)
            answer = ("（离线降级：未检测到模型 API Key，以下为检索到的课本方法预览）\n\n" + body
                      if body else "（离线且未命中任何教材方法）")
            yield {"type": "done", "offline": True, "answer": answer}
            return

        (system, user, _mode) = self._compose_answer_prompt(question, ir.intent, methods, memory)

        parts: list[str] = []
        for delta in self.llm.complete_text_stream(system, user):
            parts.append(delta)
            yield {"type": "delta", "text": delta}
        answer = "".join(parts)
        self._note_progress(question, ir.intent, methods, general)
        self.profile.append_chat(question, answer)  # 长对话上下文（并自动起名）
        save_profile(self.user_dir, self.profile)
        # 会话≥2轮后：后台 LLM 把标题润色为主题名（不阻塞作答）
        if len(self.profile.sessions.get(self.profile.current_session, {}).get("history", [])) >= 2:
            threading.Thread(target=self._summarize_title, daemon=True).start()

        if self.write_back and self.llm is not None and ir.intent != Intent.CHAT.value:
            threading.Thread(target=self._write_back,
                             args=(question, answer, ir, general), daemon=True).start()
        yield {"type": "done", "offline": False, "answer": answer}

    # ---------------- ① 意图分类 ---------------- #
    def _classify(self, question: str) -> IntentResult:
        if self.llm is None:
            return _rule_intent(question)
        try:
            data = self.llm.parse_json(INTENT_SYSTEM, f"学生提问：{question}")
            ir = IntentResult.model_validate({
                "intent": str(data.get("intent", "答疑")),
                "keywords": data.get("keywords") or [],
                "wants_memory": bool(data.get("wants_memory", True)),
                "stuck_topics": data.get("stuck_topics") or [],
                "mistakes": data.get("mistakes") or [],
                "expression_note": str(data.get("expression_note") or ""),
            })
            return ir
        except Exception:  # noqa: BLE001 —— 分类失败退规则
            return _rule_intent(question)

    # ---------------- ③ 检索：课本 + 个人记忆，按画像重排 ---------------- #
    def _retrieve(self, question: str, ir: IntentResult, *,
                  top_k: int | None = None,
                  memory_k: int | None = None) -> tuple[list[RetrievedMethod], list[RetrievedMethod]]:
        methods = []
        if self.course and (top_k or self.study_k):
            methods = find_methods(question, self.course,
                                   top_k=top_k or self.study_k, settings=self.settings)
            methods = rerank_by_bias(methods, self.profile.kind_bias)
        memory: list[RetrievedMethod] = []
        if ir.wants_memory and self.llm is not None and self.profile.turn_count > 0:
            try:
                memory = find_methods(
                    question, bucket_course(self.user_id, "知识"),
                    top_k=memory_k or self.memory_k, settings=self.memory_settings,
                    use_chroma=False,
                )
            except Exception:  # noqa: BLE001
                memory = []
        return methods, memory

    # ---------------- 学习进度：每轮课程问答实时记 ---------------- #
    def _note_progress(self, question: str, intent: str,
                       methods: list[RetrievedMethod], general: bool) -> None:
        """把一轮“确实命中课本”的课程问答记入 course_progress（相关性+深度，实时）。"""
        if general or not self.course or not methods:
            return
        top = max(m.score for m in methods)
        if top < RELEVANCE_OK:  # 相关度不足不虚增进度
            return
        chapter = methods[0].card.chapter
        depth = _question_depth(question, intent, methods)
        self.profile.note_course_progress(self.course, chapter, depth)
        save_profile(self.user_dir, self.profile)

    def _summarize_title(self) -> None:
        """后台：把当前会话标题用 LLM 润色成主题名（≤18 字；失败静默，保留首问起名）。"""
        if self.llm is None:
            return
        sid = self.profile.current_session
        hist = self.profile.sessions.get(sid, {}).get("history", [])
        if len(hist) < 2:
            return
        ctx = "\n".join(f"问：{m['u'][:120]}\n答：{m['a'][:160]}" for m in hist[-6:])
        try:
            data = self.llm.parse_json(TITLE_SYSTEM, ctx)
            t = str(data.get("title") or "").strip()[:18]
            if t:
                self.profile.sessions[sid]["title"] = t
                save_profile(self.user_dir, self.profile)
        except Exception:  # noqa: BLE001 —— 命名失败静默，标题保持首问起名
            pass

    # ---------------- ④ 作答：意图分流 + 相关性闸门 + 记忆守卫 ---------------- #
    def _compose_answer_prompt(self, question: str, intent: str,
                               methods: list[RetrievedMethod],
                               memory: list[RetrievedMethod]) -> tuple[str, str, str]:
        """返回 (system, user, mode)。mode: hard(课本优先解题) / soft(整体认识,课本仅对照)。

        - hard：答疑/出题 且 有相关度达阈值的教材方法 → 保持“优先采用课本方法”流程；
        - soft：概念/复习/闲聊，或方法相关度不足 → 先用自己的话讲清，课本仅作对照；
        - 记忆注入守卫：仅“答疑”意图且个人记忆相关度达标才注入，且明确标注
          “仅供参考、勿当作本次提问”，避免把过往问答误当成本次要答的内容。
        """
        hard = intent in ("答疑", "出题") and any(m.score >= RELEVANCE_OK for m in methods)
        ctx = self.profile.chat_context()  # 长对话上下文（上一轮问答在 turn 末尾才入表，因此这里不会带本次）
        if ctx:
            ctx = ctx + "\n\n"
        # 通识：常识/课纲外（或检索不到任何教材方法）→ 正常用通用知识作答，不注入课本
        general = intent in (Intent.GENERAL.value, Intent.CHAT.value) or not methods
        if general:
            system = GENERAL_SYSTEM.format(course=self.course or "课程")
            return system, ctx + question, "general"
        # 记忆守卫：仅“答疑”意图且有词面命中才注入，并明确标注“背景参考、勿复述”。
        # 个人记忆是纯词面分（量级远低于教材融合分），不做阈值卡死，靠“概念类不注入+标注”治本。
        use_memory = (bool(memory) and intent == "答疑" and self.profile.turn_count > 0)
        if hard:
            sys_base, user_base = build_prompt(self.course or "课程", question, methods)
            system = (self.profile.to_preamble() + "\n\n" + sys_base
                      + "\n\n" + OUTPUT_DISCIPLINE)
            if use_memory:
                user = ("【该生过往相关问答（背景参考：不要当作本次提问来回答，也不要在回复中点名或复述这些历史）】\n"
                        + render_methods(memory[:2]) + "\n\n" + user_base)
            else:
                user = user_base
            return system, ctx + user, "hard"
        system = (self.profile.to_preamble() + "\n\n"
                  + SOFT_SYSTEM.format(course=self.course or "课程")
                  + "\n\n" + OUTPUT_DISCIPLINE)
        user = (ctx
                + "【教材方法（仅供对照，相关才引用）】\n"
                + (render_methods(methods[:4]) if methods else "（未命中相关教材方法）")
                + "\n\n【学生题目】\n" + question)
        return system, user, "soft"

    def _answer(self, question: str, intent: str,
                methods: list[RetrievedMethod],
                memory: list[RetrievedMethod]) -> tuple[str, bool]:
        if self.llm is None:  # 离线降级：只出方法卡预览
            body = render_methods(methods)
            ans = ("（离线降级：未检测到模型 API Key，以下为检索到的课本方法预览）\n\n" + body
                   if body else "（离线且未命中任何教材方法）")
            return ans, True
        system, user, _mode = self._compose_answer_prompt(question, intent, methods, memory)
        answer = self.llm.complete_text(system, user)
        return answer, False

    # ---------------- ⑥/⑦ 写回记忆 + 画像 ---------------- #
    def _write_back(self, question: str, answer: str, ir: IntentResult,
                    general: bool = False) -> WritebackSummary:
        """general=True：通识问答，只写“通识”记忆桶 + 画像 general_notes，
        绝不动课程的 weak_topics/common_mistakes（区分数据来源）。"""
        summary = WritebackSummary(skipped=False)
        try:
            if general:
                return self._write_general_back(question, answer)
            data = self.llm.parse_json(EXTRACT_SYSTEM,
                                       f"学生问题：{question}\n\n教师回答：\n{answer}")
            tips_expression = ""
            for bucket in BUCKETS:
                entries = data.get({"知识": "knowledge", "思维习惯": "thinking",
                                    "技巧选择": "technique", "语义表意": "semantics"}[bucket]) or []
                cards = [self._memory_card(bucket, e) for e in entries if isinstance(e, dict)]
                if not cards:
                    continue
                store = CourseStore(bucket_course(self.user_id, bucket), self.memory_settings)
                new = store.save_all(cards)
                summary.buckets_written[bucket] = new
                if bucket == "语义表意":
                    tips_expression = "；".join(
                        "、".join(e.get("examples") or []) for e in entries if isinstance(e, dict)
                    )[:120]
            self.profile.ingest_turn(stuck_topics=ir.stuck_topics,
                                     mistakes=ir.mistakes,
                                     expression_note=tips_expression or ir.expression_note)
            save_profile(self.user_dir, self.profile)
        except Exception as exc:  # noqa: BLE001 —— 写回失败不影响本轮作答
            summary.skipped = True
            summary.reason = f"写回异常：{type(exc).__name__}: {exc}"
        return summary

    def _write_general_back(self, question: str, answer: str) -> WritebackSummary:
        """通识问答写回：独立“通识”记忆桶 + 画像 general_notes；与课程记忆完全分离。"""
        summary = WritebackSummary(skipped=False)
        data = self.llm.parse_json(GENERAL_EXTRACT,
                                   f"学生：{question}\n\n助手：\n{answer}")
        items = [it for it in (data.get("items") or []) if isinstance(it, dict)]
        cards = [self._memory_card(GENERAL_BUCKET, {"topic": it.get("topic") or "",
                                                    "content": it.get("note") or "",
                                                    "keywords": []})
                 for it in items if (it.get("topic") or "").strip()]
        if cards:
            store = CourseStore(bucket_course(self.user_id, GENERAL_BUCKET), self.memory_settings)
            summary.buckets_written[GENERAL_BUCKET] = store.save_all(cards)
        self.profile.add_general([it.get("topic") for it in items if (it.get("topic") or "").strip()])
        save_profile(self.user_dir, self.profile)
        return summary

    def _memory_card(self, bucket: str, entry: dict) -> MethodCard:
        topic = str(entry.get("topic") or "未命名")[:80]
        content = str(entry.get("content") or "")
        examples = [str(x) for x in (entry.get("examples") or []) if str(x)]
        seed = f"{bucket}|{topic}|{content}|{'|'.join(examples)}"
        doc_id = f"记忆|{bucket}|{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:10]}"
        if bucket == "语义表意":
            return MethodCard(
                id=doc_id, course=bucket_course(self.user_id, bucket), chapter="问答",
                topic=topic, kind=_BUCKET_KIND[bucket], keywords=examples,
                applicability="该生措辞习惯", steps=[content] if content else [],
                error_notes=[], source_page="",
            )
        return MethodCard(
            id=doc_id, course=bucket_course(self.user_id, bucket), chapter="问答",
            topic=topic, kind=_BUCKET_KIND[bucket],
            keywords=entry.get("keywords") or [],
            applicability="该生相关历史问答", steps=[content] if content else [],
            core_formula_latex="", technique="", worked_example="", error_notes=[],
            source_page="",
        )

    # ---------------- 每日打卡 / 按章节复习 ---------------- #
    def daily_cards(self, n: int = 5, include_memory: bool = True,
                    chapter: str | None = None) -> list[MethodCard]:
        import random

        pool: list[MethodCard] = []
        if self.course:
            pool += [c for c in CourseStore(self.course, self.settings).load_all()
                     if not c.reference_only]  # 参考类不参与打卡
        if include_memory:
            for name in _memory_course_names(self.user_id):
                pool += CourseStore(name, self.memory_settings).load_all()
        if chapter:
            pool = [c for c in pool if c.chapter == chapter]  # 按章节精确过滤（复习该章）
        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    def record_review(self, card_id: str, ok: bool, chapter: str = "") -> int:
        """记录一次复习自评（答对/答错）→ Leitner 盒号，返回新盒号。"""
        box = self.profile.record_review(card_id, ok, chapter)
        save_profile(self.user_dir, self.profile)
        return box

    # ---------------- 画像辅助 ---------------- #
    @property
    def preamble(self) -> str:
        return self.profile.to_preamble()

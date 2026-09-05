# -*- coding: utf-8 -*-
"""AgentLoop：模型自主检索的 tool-calling 循环（agentic 模式）。

与现有「代码写死 分类→检索→注入→作答」管线的关系：
- 现有管线（ask/TeacherAgent）：检索策略写死——拿原问题一次性 find_methods，
  top-k 整卡塞进 prompt。改写题（英文/口语措辞）与两跳题（一次要两块知识）易漏检。
- AgentLoop：把 search_methods / read_card 作为工具交给模型，检索几次、用什么词、
  先看摘要还是直接读全文，由模型临场决定；循环直到模型给出最终答案。

护栏（全部内置）：
- ``max_rounds``（默认 6）防打转：超限后追加一条「立即作答」的指令且**不再带
  tools** 调用一次，保证一定有答案返回；
- 工具执行异常/参数不合法 → 把错误作为 tool 消息回喂，模型可自行调整重试；
- 模型层异常（网络/鉴权）直接抛出，由调用方决定是否回落到现有管线
  （CLI 的 --agentic 已内置该回落）。

不作答纪律上的让步：system prompt 沿用「优先采用课本方法」的作答纪律，
与 answer.build_prompt 同源（判适用条件→按步骤→引用公式→提醒易错点→交叉验证）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .llm import LLMClient
from .retrieve import find_methods
from .schema import MethodCard
from .store import CourseStore

# --------------------------------------------------------------------------- #
# 工具定义（OpenAI function calling schema）
# --------------------------------------------------------------------------- #
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_methods",
            "description": (
                "在《课程》教材蒸馏的方法卡片库中检索。返回卡片摘要列表"
                "（id/主题/种类/章节/相关度/适用条件）。"
                "同一问题可以多次调用、换不同措辞：英文术语不中时换中文教材术语，"
                "问题含多个概念时可拆开分别检索。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索词：问题片段或关键词组合（贴合教材用语效果最好）",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["概念", "定理", "方法", "例题", "易错点"],
                        "description": "限定卡片种类（可选，不填则不限）",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_card",
            "description": (
                "按检索结果里的卡片 id 读取完整内容"
                "（适用条件/标准步骤/核心公式/代表例题/易错点）。"
                "觉得某张卡与问题相关时再读，不必全读。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "string",
                        "description": "卡片 id，形如 课程|章节|序号，须来自 search_methods 的返回",
                    },
                },
                "required": ["card_id"],
            },
        },
    },
]

SYSTEM_TEMPLATE = """你是一名严谨的大学「{course}」课程助教。你手边有一部从教材蒸馏出的**方法卡片库**，可以通过工具自主检索。

检索策略：
- 系统可能已在消息里附上一次【系统预检索】的候选卡片；**若已覆盖问题所需，直接用 read_card 读相关卡，无需重复 search**；
- 仍需检索时：**问题包含多个概念时，分别检索每个概念**；
- 英文/口语措辞检索不中时，换成教材里的中文术语再试；
- 结果里哪张卡看起来相关，用 read_card 读全文确认其「适用条件」；
- 信息足够后**停止调用工具**，直接输出最终答案。

作答纪律（课本方法优先）：
1) 先判断题目落在哪些卡片的『适用条件』内，点名所用方法；
2) 严格按卡片『标准步骤』一步步推导，公式用 LaTeX 引用『核心公式』；
3) 卡片标注了『易错点』的，作答时主动提醒；
4) 最后用通用知识交叉校验结果，一句话说明教材方法与通用解法是否一致。
若检索结果均不适用，明确说明后用通用知识作答。"""

# 工具结果的渲染上限（防超长返回把上下文撑爆）
_PREVIEW_APPLICABILITY_MAX = 80
_WORKED_EXAMPLE_MAX = 500


# --------------------------------------------------------------------------- #
# 结果对象
# --------------------------------------------------------------------------- #
@dataclass
class AgenticResult:
    """一轮 agentic 问答的完整记录（评测/演示/调试都要用）。"""

    answer: str
    rounds: int = 0                      # 模型轮数（含最终作答轮）
    tool_calls: int = 0                  # 工具调用总次数
    round_secs: list[float] = field(default_factory=list)  # 每轮模型调用耗时（诊断用）
    trace: list[str] = field(default_factory=list)      # 人读轨迹："search_methods('x') → 5 卡"
    seen_topics: list[str] = field(default_factory=list)  # 所有检索返回过的卡片主题（信息集）
    read_cards: list[str] = field(default_factory=list)   # 读过全文的卡片 id
    forced_final: bool = False           # 是否因超轮数被强制收尾


# --------------------------------------------------------------------------- #
# 工具实现（包现有 find_methods / CourseStore，纯读不改库）
# --------------------------------------------------------------------------- #
def _preview(methods: list) -> str:
    """把检索命中渲染成精简摘要 JSON（模型据此决定读哪张卡）。"""
    items = []
    for m in methods:
        c = m.card
        app = (c.applicability or "")[:_PREVIEW_APPLICABILITY_MAX]
        items.append({
            "id": c.id, "topic": c.topic, "kind": c.kind.value,
            "chapter": c.chapter, "score": round(m.score, 3),
            "applicability": app,
        })
    return json.dumps({"hits": items}, ensure_ascii=False)


def _render_card(c: MethodCard) -> str:
    """整卡渲染（比 answer.render_methods 更机器友好，供模型消费）。"""
    ex = c.worked_example or ""
    if len(ex) > _WORKED_EXAMPLE_MAX:
        ex = ex[:_WORKED_EXAMPLE_MAX] + "……"
    return json.dumps({
        "id": c.id, "topic": c.topic, "kind": c.kind.value, "chapter": c.chapter,
        "keywords": c.keywords, "prerequisites": c.prerequisites,
        "applicability": c.applicability, "steps": c.steps,
        "core_formula_latex": c.core_formula_latex, "technique": c.technique,
        "worked_example": ex, "error_notes": c.error_notes,
        "source_page": c.source_page,
    }, ensure_ascii=False)


class AgentLoop:
    """两工具（search_methods / read_card）的受控 agent 循环。"""

    def __init__(
        self,
        course: str,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        max_rounds: int = 6,
        top_k: int = 5,
    ) -> None:
        self.settings = settings or get_settings()
        self.course = course
        self.llm = llm or LLMClient(self.settings)   # 无 key → ValueError，由调用方回落
        self.max_rounds = max_rounds
        self.top_k = top_k
        self._cards_by_id: dict[str, MethodCard] | None = None  # 惰性加载整卡表

    # ---------------- 工具实现 ---------------- #
    def _tool_search(self, query: str, kind: str | None = None) -> str:
        methods = find_methods(
            query, self.course,
            top_k=(self.top_k * 2 if kind else self.top_k),
            settings=self.settings,
        )
        if kind:  # 种类过滤在检索后做（词面+向量分不感知 kind）
            methods = [m for m in methods if m.card.kind.value == kind][: self.top_k]
        self._seen.update(m.card.topic for m in methods)
        return _preview(methods) if methods else json.dumps(
            {"hits": [], "hint": "无命中。建议更换为教材中文术语后重试"}, ensure_ascii=False)

    def _tool_read(self, card_id: str) -> str:
        if self._cards_by_id is None:
            store = CourseStore(self.course, self.settings)
            self._cards_by_id = {c.id: c for c in store.load_all()}
        card = self._cards_by_id.get(card_id)
        if card is None:
            return json.dumps(
                {"error": f"卡片不存在：{card_id}。请使用 search_methods 返回的 id"},
                ensure_ascii=False)
        if card_id not in self._read:
            self._read.append(card_id)
        return _render_card(card)

    # ---------------- 主循环 ---------------- #
    def run(self, question: str, on_delta=None, on_round=None) -> AgenticResult:
        """执行一轮 agent 问答。

        ``on_delta(text)``：可选回调，模型生成正文时逐段调用（打字机输出用）。
        ``on_round(round_no)``：可选回调，每轮模型调用开始时触发——工具决策轮
        无正文输出，靠它显示"思考中…"的进展状态，避免黑屏等待感。
        回调异常一律吞掉，不影响作答。
        """
        self._seen: set[str] = set()
        self._read: list[str] = []
        trace: list[str] = []

        # 投机预检索：用原问题先查一次（本地毫秒级、零 token 成本），把候选摘要
        # 直接附在首条消息里——直查题模型一睁眼就有牌，可跳过 search 轮直接读卡/作答；
        # 改写/两跳题模型仍可自行换词再搜（agency 不减，只是起点不再是白纸）。
        pre = find_methods(question, self.course,
                           top_k=self.top_k, settings=self.settings)
        self._seen.update(m.card.topic for m in pre)
        trace.append(f"pre-retrieve -> {len(pre)} 卡")
        if pre:
            first_user = (
                question
                + "\n\n【系统预检索】已用你的原问题检索过一次，候选卡片摘要如下；"
                  "相关卡片可直接 read_card 读全文，不够再自行 search_methods"
                  "（换教材中文术语 / 按概念拆分检索）：\n"
                + _preview(pre)
            )
        else:
            first_user = (
                question
                + "\n\n【系统预检索】原问题未命中任何卡片，请自行 search_methods"
                  "（建议换教材中文术语重试）。"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(course=self.course or "课程")},
            {"role": "user", "content": first_user},
        ]
        result = AgenticResult(answer="")
        n_calls = 0

        def finalize(answer: str, forced: bool = False) -> AgenticResult:
            result.answer = answer
            result.tool_calls = n_calls
            result.trace = trace
            result.seen_topics = sorted(self._seen)
            result.read_cards = list(self._read)
            result.forced_final = forced
            return result

        def one_round(msgs: list, tools: list | None) -> tuple[str, list[dict]]:
            """跑一轮流式对话。返回 (正文文本, tool_calls 列表[dict 形态，可空])。"""
            parts: list[str] = []
            calls: list[dict] = []
            for ev in self.llm.chat_with_tools_stream(msgs, tools):
                if ev["type"] == "delta":
                    parts.append(ev["text"])
                    if on_delta:
                        try:
                            on_delta(ev["text"])
                        except Exception:  # noqa: BLE001 —— 回调失败不影响作答
                            pass
                elif ev["type"] == "tool_calls":
                    calls = ev["tool_calls"]
            return "".join(parts), calls

        for round_no in range(1, self.max_rounds + 1):
            if on_round:
                try:
                    on_round(round_no)
                except Exception:  # noqa: BLE001
                    pass
            t0 = time.time()
            content, calls = one_round(messages, TOOLS)
            result.round_secs.append(round(time.time() - t0, 1))
            # 回传 assistant 消息（dict 形态，字段显式）
            turn: dict[str, Any] = {"role": "assistant", "content": content or None}
            if calls:
                turn["tool_calls"] = calls
            messages.append(turn)
            result.rounds = round_no
            if not calls:  # 不再调工具 → 正文即最终答案
                return finalize((content or "").strip())
            for tc in calls:
                n_calls += 1
                try:
                    fn = tc["function"]
                    args = json.loads(fn.get("arguments") or "{}")
                    if fn["name"] == "search_methods":
                        out = self._tool_search(
                            args.get("query", ""), args.get("kind") or None)
                        trace.append(f"search_methods({args.get('query', '')!r}"
                                     + (f", kind={args.get('kind')}" if args.get("kind") else "")
                                     + ")")
                    elif fn["name"] == "read_card":
                        out = self._tool_read(args.get("card_id", ""))
                        trace.append(f"read_card({args.get('card_id', '')!r})")
                    else:
                        out = json.dumps({"error": f"未知工具：{fn['name']}"}, ensure_ascii=False)
                        trace.append(f"??{fn['name']}")
                except Exception as exc:  # noqa: BLE001 —— 单次调用失败回喂错误，模型可自纠
                    out = json.dumps(
                        {"error": f"工具执行失败：{type(exc).__name__}: {exc}"},
                        ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})

        # 超轮数：强制收尾（不带 tools 再调一次，模型只能输出文本）
        messages.append({
            "role": "user",
            "content": "【系统】已达到工具调用轮数上限。请立即基于已获取的卡片信息作答；"
                       "信息不足的部分用通用知识补足并注明。",
        })
        content, _ = one_round(messages, None)
        return finalize((content or "(agent 未返回文本)").strip(), forced=True)

"""LLM 蒸馏：把教材章节文本 → 结构化方法卡片。

核心思路：一次性给模型"方法卡片的 schema + 抽取规范"，让它只依据给定文本
抽取出概念/定理/方法/例题/易错点，供检索与优先注入使用。
"""
from __future__ import annotations

import re

from .llm import LLMClient
from .parser import Chapter
from .schema import MethodCard, MethodKind

SYSTEM_PROMPT = """你是一位严谨的大学课程学习助教。目标是【把教材章节内容蒸馏成方法卡片】，供考试答题时优先复用课本的逻辑与运算技巧。

输出必须是 JSON 对象，结构如下：
{"cards": [ {每张卡} ]}

每张卡片的字段（严格遵守）：
- "id": 形如 "{{course}}|{{chapter}}|序号"
- "course": "{{course}}"
- "chapter": "{{chapter}}"
- "topic": 主题/知识点名，一句话
- "kind": 取 概念/定理/方法/例题/易错点 之一
  - 概念=新名词与定义；定理=命题/公式/性质；方法=解题套路与运算技巧（重点产出）；
    例题=典型例题含关键解法；易错点=章节强调的坑
- "keywords": 检索用关键词数组，尽量多收录：术语、题型、公式别名、常见考试叫法
- "prerequisites": 前置概念数组
- "applicability": 适用条件——什么情况下用这个方法（请写具体）
- "steps": 标准步骤数组，按顺序可执行；非方法类可给"如何理解/记忆"步骤
- "core_formula_latex": 核心公式（LaTeX）；没有就空字符串
- "technique": 常用技巧/思路点睛，一句话；没有就空字符串
- "worked_example": 代表例题及其关键解法（题干+步骤要点），没有就空字符串
- "error_notes": 常见错误/易错点数组
- "source_page": 该内容来源的页码锚点（文本里有 <!-- page N --> 就填 N，否则留空）

要求：
- 只依据给定文本抽取，不要编造教材没有的内容；拿不准就留空或不产出
- 方法类卡片是重点，宁可多产几张小而准的方法卡，也不要糊成一大段
- **排除教材编排性内容，不要产卡**：目录、«阅读顺序/推荐章节/如何使用本书»、编排/进度安排说明、
  致谢、参考文献清单、练习索引等；这些不是知识点，即使写成“步骤/建议”也一律不产
- 不要做总结或评价，直接输出 JSON"""


def _chunk_text(text: str, size: int = 6000, overlap: int = 600) -> list[str]:
    """带重叠的分块，避免切断一个方法/例题。"""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def distill_chapter(
    llm: LLMClient,
    chapter: Chapter,
    max_cards_per_chunk: int = 8,
    chunk_size: int = 6000,
    parallel: int = 1,
) -> list[MethodCard]:
    """对一章文本分块蒸馏，汇总、去重、赋稳定 id。

    - ``chunk_size``：分块字符数（越大调用次数越少）。
    - ``parallel>1``：并行调用模型处理各块（每块独立 → 可安全并发，墙钟大幅缩短）。
    """
    texts = _chunk_text(chapter.text, size=chunk_size)

    def run(pair: tuple[int, str]) -> list[MethodCard]:
        idx, chunk = pair
        user = (
            f"【课程】{chapter.course}\n"
            f"【章节】{chapter.chapter}\n"
            f"【文本片段 {idx}，仅依据此文本】\n{chunk}"
        )
        data = llm.parse_json(SYSTEM_PROMPT, user)
        return _cards_from_data(data, chapter.course, chapter.chapter)

    if parallel > 1 and len(texts) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            card_lists: list[list[MethodCard]] = list(ex.map(run, enumerate(texts, 1)))
    else:
        card_lists = [run(p) for p in enumerate(texts, 1)]
    cards = [c for lst in card_lists for c in lst]
    return dedupe_cards(cards)


def _cards_from_data(data: dict, course: str, chapter: str) -> list[MethodCard]:
    """从模型返回的 ``{"cards":[...]}`` 里校验出合法卡片（坏卡跳过，不中断整章）。"""
    cards: list[MethodCard] = []
    raw = (data or {}).get("cards") or []
    for item in raw:
        if not isinstance(item, dict):
            continue
        item.setdefault("course", course)
        item.setdefault("chapter", chapter)
        try:
            card = MethodCard.model_validate(item)
        except Exception:  # noqa: BLE001 —— 跳过单张坏卡，不中断整章
            continue
        if card.kind not in MethodKind:
            continue
        cards.append(card)
    return cards


# --------------------------------------------------------------------------- #
# 仅供参考类（噪音）检测：教材编排/阅读指引不是知识点，标记后不参与考点
# --------------------------------------------------------------------------- #
_REFERENCE_NOISE_RE = re.compile(
    r"阅读顺序|阅读计划|学习顺序|推荐.{0,6}(顺序|阅读|章节|内容)|如何使用(本书|这本|教材)|"
    r"建议(先|按|顺便|跳过)|任选|可选.{0,2}章|关于(本书|这本|教材)|编排|进度安排|"
    r"^参考(书目|文献)|^致谢|^目录$|导读|先.{0,4}第?\d+章",
)


def is_noise_card(card: MethodCard) -> bool:
    """判断是否为“仅供参考”类噪音卡（编排指引/阅读顺序等）。"""
    if card.reference_only:
        return True
    text = " ".join([card.topic, card.applicability, card.technique] + card.steps)
    return bool(_REFERENCE_NOISE_RE.search(text))


def flag_noise_cards(cards: list[MethodCard]) -> list[MethodCard]:
    """给命中噪音规则的卡打上 reference_only（保留但不参与考点）。"""
    out = []
    for c in cards:
        if is_noise_card(c) and not c.reference_only:
            c = c.model_copy(update={"reference_only": True})
        out.append(c)
    return out


def dedupe_cards(cards: list[MethodCard]) -> list[MethodCard]:
    """按 (chapter, topic, kind) 去重，并重写稳定 id。"""
    seen: set[tuple[str, str, str]] = set()
    out: list[MethodCard] = []
    counters: dict[str, int] = {}
    for card in cards:
        key = (card.chapter, card.topic, card.kind.value)
        if key in seen:
            continue
        seen.add(key)
        counters[card.chapter] = counters.get(card.chapter, 0) + 1
        card = card.model_copy(
            update={"id": f"{card.course}|{card.chapter}|{counters[card.chapter]:03d}"}
        )
        out.append(card)
    return out
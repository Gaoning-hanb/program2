"""作答：把检索到的课本方法放到通用知识前面，优先注入，最后交叉验证。"""
from __future__ import annotations

from .config import Settings
from .llm import LLMClient
from .retrieve import find_methods
from .schema import RetrievedMethod


def render_methods(methods: list[RetrievedMethod], example_max: int = 600) -> str:
    """把命中方法渲染成注入块的文本。"""
    lines: list[str] = []
    for i, m in enumerate(methods, 1):
        c = m.card
        lines.append(f"〔方法{i}·相关度 {m.score:.2f} · {c.chapter} / {c.topic}〕")
        lines.append(f"- 种类：{c.kind.value}")
        if c.applicability:
            lines.append(f"- 适用条件：{c.applicability}")
        if c.steps:
            steps = "\n".join(f"   {j}. {s}" for j, s in enumerate(c.steps, 1))
            lines.append(f"- 标准步骤：\n{steps}")
        if c.core_formula_latex:
            lines.append(f"- 核心公式：${c.core_formula_latex}$")
        if c.technique:
            lines.append(f"- 技巧：{c.technique}")
        if c.worked_example:
            ex = c.worked_example
            if len(ex) > example_max:
                ex = ex[:example_max] + " ……"
            lines.append(f"- 代表例题：{ex}")
        if c.error_notes:
            lines.append(f"- 易错点：{'；'.join(c.error_notes)}")
    return "\n".join(lines)


def build_prompt(course: str, question: str, methods: list[RetrievedMethod]) -> tuple[str, str]:
    system = (
        f"你是一名严谨的大学「{course}」答题助手。下面是从《{course}》教材中蒸馏出的"
        "方法卡片集（按与题目相关度降序，卡片内容源自教材原逻辑）。答用户题目时：\n"
        "1) 【优先采用课本方法】先判断题目属于哪张卡片的『适用条件』，点名方法名；\n"
        "2) 严格按卡片『标准步骤』一步步推导；代入公式计算时引用『核心公式』；\n"
        "3) 若该方法标注了『易错点』，作答时主动提醒；\n"
        "4) 步骤完整、推导清晰，公式用 LaTeX 呈现；\n"
        "5) 最后用通用知识交叉校验结果，并一句话说明教材方法与通用解法是否一致。\n"
        "若检索出的方法都不适用，请明确指出，再给出通用解法。"
    )
    if methods:
        user = (
            "【优先参考的教材方法】\n"
            + render_methods(methods)
            + "\n\n【学生题目】\n" + question
        )
    else:
        user = "【未命中任何教材方法（数据库为空或均不适用）】\n\n【学生题目】\n" + question
    return system, user


def ask_question(
    question: str,
    course: str,
    top_k: int | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> tuple[str, list[RetrievedMethod]]:
    """检索方法 → 构造优先注入 prompt → 让 LLM 作答。返回 (答案, 命中方法)。"""
    settings = settings or Settings()
    methods = find_methods(question, course, top_k=top_k, settings=settings)
    llm = llm or LLMClient(settings)
    system, user = build_prompt(course, question, methods)
    answer = llm.complete_text(system, user)
    return answer, methods
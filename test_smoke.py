"""离线冒烟测试：不联网、不需要 API key。

运行：conda activate firpro && python test_smoke.py
"""
from __future__ import annotations

from pathlib import Path

from coursebook_digest.answer import build_prompt, render_methods
from coursebook_digest.config import PROJECT_ROOT, Settings
from coursebook_digest.distill import distill_chapter
from coursebook_digest.parser import parse_source
from coursebook_digest.retrieve import find_methods
from coursebook_digest.schema import MethodCard, MethodKind, RetrievedMethod
from coursebook_digest.store import CourseStore

SAMPLE = PROJECT_ROOT / "examples" / "sample_course.md"


class FakeLLM:
    """离线假模型：distill 只需要 parse_json，无需网络。"""

    def __init__(self, cards: list[dict]) -> None:
        self._cards = cards

    def parse_json(self, system: str, user: str) -> dict:  # noqa: ARG002
        return {"cards": self._cards}


def _mkcard(course: str, kind: str, topic: str, keywords: list[str], steps: list[str],
            cid: str = "x", chapter: str = "第一章 极限与连续") -> dict:
    return {
        "id": cid, "course": course, "chapter": chapter, "topic": topic,
        "kind": kind, "keywords": keywords, "applicability": "适用条件A",
        "steps": steps, "core_formula_latex": "", "technique": "", "worked_example": "",
        "error_notes": [], "source_page": "12",
    }


def test_store_roundtrip(settings: Settings) -> str:
    store = CourseStore("高数", settings)
    cards = [
        MethodCard.model_validate(_mkcard(
            "高数", "方法", "洛必达法则", ["洛必达", "未定型", "0/0", "无穷比无穷", "求导"],
            ["判断型别", "分子分母分别求导", "检查是否仍为未定型"], cid="高数|第一章 极限与连续|001",
        )),
        MethodCard.model_validate(_mkcard(
            "高数", "方法", "等价无穷小替换", ["等价无穷小", "替换", "泰勒", "sin", "tan", "高阶"],
            ["判断趋于0", "乘除中替换", "加减谨慎"], cid="高数|第一章 极限与连续|002",
        )),
        MethodCard.model_validate(_mkcard(
            "高数", "定理", "两个重要极限", ["重要极限", "sinx/x", "e", "夹逼"],
            ["识别结构", "凑形式", "套用结果"], cid="高数|第一章 极限与连续|003",
        )),
    ]
    n = store.save_all(cards)
    assert n == 3, f"首次应新增3，实际{n}"
    again = store.save_all([cards[0]])  # 幂等：重复保存不新增
    assert again == 0, f"幂等校验失败：{again}"
    loaded = store.load_all()
    assert len(loaded) == 3
    assert all(c.id.startswith("高数|第一章 极限与连续|") for c in loaded)
    print("[1/4] store 往返 + 幂等 OK")
    return "高数"


def test_find_methods(settings: Settings, course: str) -> None:
    q = "分子分母都趋于0，用洛必达法则求极限的步骤"
    methods = find_methods(q, course, top_k=3, settings=settings, use_chroma=False)
    assert methods, "不应无命中"
    top = methods[0]
    assert "洛必达" in top.card.topic, f"top 应为洛必达，实际 {top.card.topic}"
    assert 0 <= top.score <= 1.5, f"分数异常 {top.score}"
    assert isinstance(top, RetrievedMethod)
    print(f"[2/4] find_methods 命中优先：top={top.card.topic} score={top.score}")


def test_distill(settings: Settings) -> None:
    chapters = parse_source(SAMPLE, "高数", parser="auto")
    assert chapters, "示例教材解析为空"
    ch = chapters[0]
    raw = [
        _mkcard("高数", "方法", "洛必达法则", ["洛必达"], ["判断型别", "求导"], chapter=ch.chapter),
        _mkcard("高数", "方法", "洛必达法则", ["洛必达"], ["判断型别", "求导"], chapter=ch.chapter),  # 重复→去重
        _mkcard("高数", "定理", "两个重要极限", ["重要极限"], ["凑形式"], chapter=ch.chapter),
    ]
    cards = distill_chapter(FakeLLM(raw), ch)
    assert len(cards) == 2, f"应去重为2，实际{len(cards)}"
    topics = [c.topic for c in cards]
    assert len(set(topics)) == len(topics), "topic 不应重复"
    assert all(c.chapter == ch.chapter for c in cards)
    print(f"[3/4] distill 去重 OK：{sum(1 for c in cards) if False else len(cards)} 张 {topics}")


def test_answer_surface(settings: Settings, course: str) -> None:
    methods = find_methods("求极限", course, top_k=2, settings=settings, use_chroma=False)
    rendered = render_methods(methods)
    assert "适用条件" in rendered
    system, user = build_prompt(course, "求极限过程", methods)
    assert "优先采用课本方法" in system
    assert "优先参考的教材方法" in user and "学生题目" in user
    print("[4/4] answer 注入块 OK")


def main() -> None:
    assert SAMPLE.exists(), "缺少 examples/sample_course.md"
    # 注意：本沙箱环境里 tempfile.mkdtemp 建出的目录 ACL 有缺陷（缺省模式下
    # 再建子目录会被拒），因此改用工作区普通目录，用完即删。
    import os as _os
    import shutil as _shutil

    run_dir = Path(PROJECT_ROOT) / ".smoke" / f"run-{_os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        settings = Settings(data_dir=str(data_dir), storage="jsonl")
        course = test_store_roundtrip(settings)
        test_find_methods(settings, course)
        test_distill(settings)
        test_answer_surface(settings, course)
    finally:
        _shutil.rmtree(run_dir, ignore_errors=True)
    print("\n全部通过")


if __name__ == "__main__":
    main()
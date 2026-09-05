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


def test_video_utils() -> None:
    """视频管线离线部分：时间戳工具、转写→锚点文本、分P选择、json3 解析。"""
    import json as _json

    from coursebook_digest.video import (Segment, _fmt_ts, _ts_to_sec,
                                         attach_source, extract_bvid,
                                         parse_parts_spec, parse_json3,
                                         segments_to_text)

    assert extract_bvid("https://www.bilibili.com/video/BV1AbCdEfGh2?p=3") == "BV1AbCdEfGh2"
    assert _ts_to_sec("12:35") == 755 and _ts_to_sec("1:02:33") == 3753
    assert _ts_to_sec("[03:10] P2") == 190 and _ts_to_sec("无时间") is None
    assert _fmt_ts(755) == "12:35" and _fmt_ts(3753) == "1:02:33"

    segs = [Segment(i * 20, i * 20 + 8, f"第{i}句话。") for i in range(10)]
    text = segments_to_text(segs, anchor_every=45)
    assert text.startswith("[00:00] 第0句话。") and text.count("\n") >= 2

    assert parse_parts_spec("", 9) is None
    assert parse_parts_spec("1-3,5", 9) == [1, 2, 3, 5]
    assert parse_parts_spec("2,99", 9) == [2]  # 越界截断

    # json3 解析（yt-dlp 字幕格式）
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".zh-Hans.json3", delete=False,
                                     encoding="utf-8") as f:
        _json.dump({"events": [
            {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "大家好"}]},
            {"tStartMs": 2000, "dDurationMs": 1500, "segs": [{"utf8": "\n"}, {"utf8": "今天讲"}]},
            {"tStartMs": 3500, "dDurationMs": 1000, "segs": []},  # 空→跳过
        ]}, f, ensure_ascii=False)
        path = Path(f.name)
    parsed, lang = parse_json3(path)
    assert lang == "zh-Hans" and len(parsed) == 2
    assert parsed[0].text == "大家好" and parsed[1].start == 2.0
    path.unlink()

    # attach_source：时间戳 → source_page 带P号 + 可跳转 URL
    card = MethodCard.model_validate(_mkcard(
        "测试课", "方法", "求极限", ["极限"], ["步骤"], chapter="P2 数据结构"))
    card.source_page = "12:35"
    attach_source([card], type("P", (), {"part_no": 2})(), "BV1AbCdEfGh2")
    assert card.source_page == "P2 12:35"
    assert card.source_url == "https://www.bilibili.com/video/BV1AbCdEfGh2?p=2&t=755"
    # 蒸馏实际填的是 [mm:ss] 锚点（带方括号）——溯源显示时须去掉括号
    card2 = MethodCard.model_validate(_mkcard(
        "测试课", "方法", "等价无穷小", ["无穷小"], ["步骤"], chapter="P3 某章"))
    card2.source_page = "[03:10]"
    attach_source([card2], type("P", (), {"part_no": 3})(), "BV1AbCdEfGh2")
    assert card2.source_page == "P3 03:10", card2.source_page
    assert card2.source_url == "https://www.bilibili.com/video/BV1AbCdEfGh2?p=3&t=190"
    print("[5/6] video 时间戳/分P/溯源 OK")


def test_notes_generation(settings: Settings) -> None:
    """notes 生成：导图 markdown 树 + 笔记溯源链接。"""
    from coursebook_digest.notes import (build_mindmap_markdown,
                                         build_notes_markdown)

    cards = [
        MethodCard.model_validate(_mkcard(
            "测试课", "方法", "顺序表插入", ["顺序表", "插入"], ["判位置", "后移", "放入"],
            cid="测试课|P2 线性表|001", chapter="P2 线性表")),
        MethodCard.model_validate(_mkcard(
            "测试课", "易错点", "插入位置边界", ["边界"], ["检查 i>n"],
            cid="测试课|P2 线性表|002", chapter="P2 线性表")),
    ]
    cards[0].source_page, cards[0].source_url = "P2 12:35", "https://x?p=2&t=755"
    mm = build_mindmap_markdown(cards, "测试课")
    assert mm.startswith("# 测试课") and "## P2 线性表" in mm and "顺序表插入" in mm
    assert mm.count("###") == 2, "每张卡应是一个三级节点"
    notes = build_notes_markdown(cards, "测试课")
    assert "跳回视频原时刻" in notes and "P2 12:35" in notes and "标准步骤" in notes
    assert "⚠️ 易错" in notes and "插入位置边界" in notes  # 易错点卡按徽章渲染自己的小节
    print("[6/6] notes 导图/笔记生成 OK")


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
        test_video_utils()
        test_notes_generation(settings)
    finally:
        _shutil.rmtree(run_dir, ignore_errors=True)
    print("\n全部通过")


if __name__ == "__main__":
    main()
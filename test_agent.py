"""TeacherAgent（受控 agent 循环）离线测试：FakeLLM，零网络、零 key。

运行：conda activate firpro && cd coursebook-digest && python test_agent.py
覆盖：规则意图、画像偏置重排、离线降级、写回记忆+画像、记忆幂等、记忆注入、每日打卡。
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from coursebook_digest.agent import (
    Intent, IntentResult, TeacherAgent, bucket_course, rerank_by_bias, _rule_intent,
)
from coursebook_digest.config import PROJECT_ROOT, Settings
from coursebook_digest.retrieve import find_methods
from coursebook_digest.schema import MethodCard, MethodKind, RetrievedMethod
from coursebook_digest.store import CourseStore


def _card(**kw) -> MethodCard:
    base = dict(id="x", course="操作系统", chapter="章", topic="主题",
                kind="概念", keywords=[], steps=[], source_page="")
    base.update(kw)
    return MethodCard.model_validate(base)


class FakeLLM:
    """可编程假模型：parse_json 按 system 关键词分流，complete_text 记下 prompt。"""

    def __init__(self, classify=None, extract=None, answer="（假模型作答）按课本方法：……") -> None:
        self.classify = classify or {
            "intent": "答疑", "keywords": [], "wants_memory": True,
            "stuck_topics": [], "mistakes": [], "expression_note": "",
        }
        self.extract = extract or {"knowledge": [], "thinking": [], "technique": [], "semantics": []}
        self._answer = answer
        self.last_system = ""
        self.last_user = ""

    def parse_json(self, system: str, user: str) -> dict:  # noqa: ARG002
        if "意图路由" in system:
            return self.classify
        if "记忆提炼器" in system:
            return self.extract
        if "日常对话提炼器" in system:
            return {"items": [{"topic": "天气", "note": "关心实时天气"}]}
        if "对话主题命名员" in system:
            return {"title": "死锁专题"}
        return {"cards": []}

    def complete_text(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self._answer

    def complete_text_stream(self, system: str, user: str):
        self.last_system = system
        self.last_user = user
        for i in range(0, len(self._answer), 8):
            yield self._answer[i:i + 8]


def _agent(tmp: Path, fake: FakeLLM, user: str = "stu1"):
    return TeacherAgent(
        user_id=user, course="操作系统",
        settings=Settings(_env_file=None), llm=fake,
        user_data_dir=tmp / "users",
    )


def test_rule_intent() -> None:
    assert _rule_intent("什么是进程？").intent == Intent.CONCEPT.value
    assert _rule_intent("谢谢老师").intent == Intent.CHAT.value
    assert _rule_intent("出几道调度题给我练").intent == Intent.QUIZ.value
    assert _rule_intent("复习一下今天的内存管理").intent == Intent.REVIEW.value
    assert _rule_intent("银行家算法怎么用").intent == Intent.ANSWER.value
    assert _rule_intent("今天天气怎么样").intent == Intent.GENERAL.value
    assert _rule_intent("讲讲最近的股市").intent == Intent.GENERAL.value
    print("    [agent] 规则意图识别（含通识） OK")


def test_rerank_bias() -> None:
    ms = [
        RetrievedMethod(card=_card(kind=MethodKind.CONCEPT, topic="进程概念"), score=0.5),
        RetrievedMethod(card=_card(kind=MethodKind.METHOD, topic="调度算法"), score=0.55),
        RetrievedMethod(card=_card(kind=MethodKind.EXAMPLE, topic="银行家例题"), score=0.5),
    ]
    top = rerank_by_bias(ms, "方法")[0]
    assert top.card.kind in (MethodKind.METHOD, MethodKind.EXAMPLE), top.card.kind
    topc = rerank_by_bias(ms, "概念")[0]
    assert topc.card.kind == MethodKind.CONCEPT
    assert [m.card.topic for m in rerank_by_bias(ms, "均衡")] == ["进程概念", "调度算法", "银行家例题"]
    print("    [agent] 画像 kind_bias 重排 OK")


def test_offline_fallback_and_rules(tmp: Path) -> None:
    """无 key → 离线降级：规则意图 + 方法卡预览，写回关闭。"""
    settings = Settings(llm_api_key="", _env_file=None)  # 无 key；data_dir 仍是项目里真实课程
    agent = TeacherAgent(user_id="guest", course="操作系统", settings=settings,
                         user_data_dir=tmp / "users")
    r = agent.turn("什么是死锁？")
    assert r.offline is True
    assert "离线降级" in r.answer
    assert r.methods, "离线也应能检索到课本方法"
    assert r.intent == Intent.CONCEPT
    assert r.writeback.skipped
    print("    [agent] 离线降级（规则意图+方法预览+不写回） OK")


def test_writeback_memory_and_profile(tmp: Path) -> None:
    fake = FakeLLM(
        classify={"intent": "答疑", "keywords": ["虚拟内存"], "wants_memory": True,
                  "stuck_topics": ["虚拟内存"], "mistakes": ["页表过长"], "expression_note": "喜欢问为什么"},
        extract={"knowledge": [{"topic": "虚拟内存", "content": "页面置换按需载入机制"}],
                 "thinking": [{"topic": "联想记忆", "content": "该生习惯把页表比作书本目录"}],
                 "technique": [], "semantics": [{"topic": "口语化", "examples": ["通俗点讲"]}]},
    )
    agent = _agent(tmp, fake)
    r = agent.turn("讲一下页式虚拟内存的页面置换")
    assert r.offline is False
    assert r.writeback.skipped is False
    assert r.writeback.buckets_written.get("知识") == 1
    assert r.writeback.buckets_written.get("语义表意") == 1

    # 记忆 jsonl 落盘
    mpath = CourseStore(bucket_course("stu1", "知识"), agent.memory_settings).path
    assert mpath.exists(), f"记忆库未落盘：{mpath}"
    cards = CourseStore(bucket_course("stu1", "知识"), agent.memory_settings).load_all()
    assert len(cards) == 1 and "页面置换" in cards[0].steps[0]
    assert "通俗点讲" in CourseStore(bucket_course("stu1", "语义表意"), agent.memory_settings).load_all()[0].keywords

    # 画像更新
    assert agent.profile.turn_count == 1
    assert "虚拟内存" in agent.profile.weak_topics
    assert "页表过长" in agent.profile.common_mistakes
    assert "喜欢问为什么" in agent.profile.expression_notes or "通俗点讲" in agent.profile.expression_notes
    print("    [agent] 写回：记忆入库 + 画像更新 OK")

    # 记忆守卫：仅“答疑”且强相关才注入（标记“仅供参考”）；“概念”类不再把过往当作要答的问题
    agent2 = _agent(tmp, fake, user="stu1")
    _ = agent2.turn("虚拟内存的页面置换次数怎么算？")  # 答疑意图
    assert "过往相关问答" in fake.last_user, "答疑强相关应注入个人记忆（标注仅供参考）"
    assert "优先参考的教材方法" in fake.last_user
    agent3 = _agent(tmp, fake, user="stu1")
    fake.classify = {"intent": "概念", "keywords": [], "wants_memory": True,
                     "stuck_topics": [], "mistakes": [], "expression_note": ""}
    _ = agent3.turn("什么是虚拟内存？")  # 概念意图
    assert "过往相关问答" not in fake.last_user, "概念类不应把个人记忆误当本次提问"
    # 幂等：同内容再写回不新增
    r3 = agent2.turn("换个问法再讲一次页面置换")
    assert r3.writeback.buckets_written.get("知识", 0) == 0
    assert len(CourseStore(bucket_course("stu1", "知识"), agent2.memory_settings).load_all()) == 1
    print("    [agent] 第2轮记忆注入 + 写回幂等 OK")


def test_intent_modes() -> None:
    """意图分流：概念/概述 → soft（先讲清、课本仅对照）；答疑且方法相关 → hard（课本优先）。"""
    fake = FakeLLM()
    agent = TeacherAgent(user_id="m", course="操作系统", settings=Settings(_env_file=None),
                         llm=fake, user_data_dir=PROJECT_ROOT / ".smoke" / "mode-probe")
    methods = find_methods("时间片轮转调度的平均等待时间", "操作系统", top_k=3, settings=Settings(_env_file=None))

    sys_s, _, mode_s = agent._compose_answer_prompt("操作系统到底是用来做什么的", "概念", methods, [])
    assert mode_s == "soft" and "整体认识" in sys_s and "优先采用课本方法" not in sys_s

    sys_h, user_h, mode_h = agent._compose_answer_prompt("计算轮转调度的平均等待时间", "答疑", methods, [])
    assert mode_h == "hard" and "优先采用课本方法" in sys_h and "优先参考的教材方法" in user_h

    # 记忆守卫：概念意图不注入、答疑且分不足不注入
    sys_s2, user_s2, _ = agent._compose_answer_prompt("什么是死锁", "概念",
                                                       [], [RetrievedMethod(card=_card(kind="概念", topic="缓存"), score=0.6)])
    assert "过往相关问答" not in user_s2
    print(f"    [agent] 意图分流 OK（soft:{mode_s} / hard:{mode_h}，记忆守卫生效）")


def test_general_scope(tmp: Path) -> None:
    """通识：正常作答（不注入课本），记忆写入独立“通识”桶，不碰课程易错点。"""
    fake = FakeLLM(answer="（通识）今天晴转多云……")
    agent = _agent(tmp, fake)  # stu1
    # 1) 通识意图 → system 不含课本方法/“优先采用”，user 就是原问题
    sys_g, user_g, mode = agent._compose_answer_prompt("今天天气怎么样", "通识", [], [])
    assert mode == "general" and "一般知识" in sys_g and "教材方法" not in sys_g
    assert user_g.endswith("今天天气怎么样"), "通识 prompt 应以原问题结尾（可带近期上下文前缀）"
    # 2) 通识写回：只进“通识”桶 + general_notes；课程弱项/常错不新增（用独立用户，避免串脏数据）
    fake2 = FakeLLM()
    a2 = _agent(tmp, fake2, user="stuG")
    a2._write_general_back("今天天气怎么样", "（通识回答）晴天……")
    gcards = CourseStore(bucket_course("stuG", "通识"), a2.memory_settings).load_all()
    assert gcards and any("天气" in c.topic for c in gcards), "应写入通识记忆桶"
    assert a2.profile.general_notes and not a2.profile.weak_topics, "通识不应污染课程易错点"
    print("    [agent] 通识作答+独立写回 OK（课程记忆未被污染）")


def test_progress_tracking(tmp: Path) -> None:
    """学习进度：命中课本的课程问答实时记入 course_progress；通识/无效不记账。"""
    fake = FakeLLM()
    agent = TeacherAgent(user_id="stuP", course="操作系统", settings=Settings(_env_file=None),
                         llm=fake, user_data_dir=tmp / "users")
    agent.turn("轮转调度算法的平均等待时间怎么算？")
    cp = agent.profile.course_progress.get("操作系统")
    assert cp and cp.questions >= 1, "命中课本的课程问答应记录进度"
    assert cp.percent() > 0, "应有进度百分比"
    covered = len(cp.chapters)
    before = cp.questions
    # 通识/课纲外不虚增课程进度
    agent.turn("今天天气怎么样", intent_hint=IntentResult(intent="通识", wants_memory=False))
    assert agent.profile.course_progress["操作系统"].questions == before, "通识不应计入课程进度"
    print(f"    [agent] 学习进度追踪 OK（进度 {cp.percent()}% / 覆盖 {covered} 章，通识不计入）")


def test_review_boxes(tmp: Path) -> None:
    """复习自评：答对升盒、答错回 1；写入画像并统计。"""
    from coursebook_digest.profile import load_profile, save_profile

    p = load_profile(tmp / "users", "stuR")
    assert p.record_review("卡A", True, "章X") == 2
    assert p.record_review("卡A", True, "章X") == 3
    assert p.record_review("卡A", False, "章X") == 1
    assert p.record_review("卡B", True) == 2
    st = p.reviews_stats()
    assert st["total"] == 4 and st["correct"] == 3 and st["wrong"] == 1
    assert st["accuracy"] == 0.75 and st["boxes"] == 2
    save_profile(tmp / "users", p)
    # agent 薄封装：record_review 持久化
    from coursebook_digest.config import Settings
    from coursebook_digest.agent import TeacherAgent

    a = TeacherAgent(user_id="stuR", course="操作系统", settings=Settings(_env_file=None),
                     user_data_dir=tmp / "users")
    box = a.record_review("卡A", True, "章X")
    assert box == 2
    reloaded = load_profile(tmp / "users", "stuR")
    assert reloaded.review_log[-1]["box"] == 2
    print("    [agent] 复习自评盒号 + 持久化 OK")


def test_chat_context(tmp: Path) -> None:
    """长对话：每轮问答写入画像；新实例/跨请求仍能取到上文。"""
    a1 = TeacherAgent(user_id="stuC", course="操作系统", settings=Settings(_env_file=None),
                      llm=FakeLLM(answer="（答1）死锁四条件……"), user_data_dir=tmp / "users")
    r1 = a1.turn("什么是死锁？")
    assert r1.offline is False
    hist = a1.profile.sessions[a1.profile.current_session]["history"]
    assert hist and hist[-1]["u"] == "什么是死锁？", "本轮问答应写入当前会话历史"
    # 新实例（模拟下一次 HTTP 请求）重新加载画像 → 自动带回上文
    a2 = TeacherAgent(user_id="stuC", course="操作系统", settings=Settings(_env_file=None),
                      llm=FakeLLM(answer="（答2）避免方式……"), user_data_dir=tmp / "users")
    ctx = a2.profile.chat_context()
    assert "什么是死锁" in ctx
    _sys, user_u, _mode = a2._compose_answer_prompt("那怎么避免死锁？", "概念", [], [])
    assert "什么是死锁" in user_u, "作答 prompt 应带近期对话上下文"
    print("    [agent] 长对话上下文跨请求衔接 OK")


def test_session_title_auto(tmp: Path) -> None:
    """会话标题自动生成：首问起名即见；≥2 轮后台 LLM 润色为主题名。"""
    fake = FakeLLM(answer="（答）死锁四条件……")
    a = TeacherAgent(user_id="stuT", course="操作系统", settings=Settings(_env_file=None),
                     llm=fake, user_data_dir=tmp / "users")
    list(a.turn_stream("什么是死锁？"))
    sid = a.profile.current_session
    assert "死锁" in a.profile.sessions[sid]["title"], a.profile.sessions[sid]["title"]
    list(a.turn_stream("怎么用银行家算法避免死锁？"))
    time.sleep(0.4)  # 等后台命名线程落盘
    a2 = TeacherAgent(user_id="stuT", course="操作系统", settings=Settings(_env_file=None),
                      llm=fake, user_data_dir=tmp / "users")
    assert a2.profile.sessions[a2.profile.current_session]["title"] == "死锁专题"
    print("    [agent] 会话标题自动总结（首问起名 + LLM 润色） OK")


def test_stream_turn(tmp: Path) -> None:
    """流式 turn：规则意图 → 先推方法 → delta 逐段 → done；离线也安全。"""
    agent = _agent(tmp, FakeLLM(answer="（流式）死锁四必要条件……"))
    events = list(agent.turn_stream("什么是死锁？"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "intent" and events[0]["intent"] == Intent.CONCEPT.value
    assert "methods" in kinds
    assert "delta" in kinds and "done" in kinds
    joined = "".join(e["text"] for e in events if e["type"] == "delta")
    assert joined == "（流式）死锁四必要条件……"
    done = events[-1]
    assert done["type"] == "done" and done["offline"] is False
    print(f"    [agent] 流式 turn OK（事件: {[k for k in kinds]}）")


def test_daily_cards(tmp: Path) -> None:
    fake = FakeLLM()
    agent = _agent(tmp, fake)
    # 课本库（真实课程）应能抽卡；记忆库为空也安全
    cards = agent.daily_cards(n=5)
    assert cards, "每日打卡应从课本库抽到卡"
    assert len(cards) <= 5
    print(f"    [agent] 每日打卡抽样 OK（{len(cards)} 张）")


def main() -> None:
    run_dir = PROJECT_ROOT / ".smoke" / f"agent-{os.getpid()}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        test_rule_intent()
        test_rerank_bias()
        test_offline_fallback_and_rules(run_dir)
        test_writeback_memory_and_profile(run_dir)
        test_intent_modes()
        test_general_scope(run_dir)
        test_progress_tracking(run_dir)
        test_review_boxes(run_dir)
        test_chat_context(run_dir)
        test_session_title_auto(run_dir)
        test_stream_turn(run_dir)
        test_daily_cards(run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    print("\nTeacherAgent 离线测试全部通过")


if __name__ == "__main__":
    main()

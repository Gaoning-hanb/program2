"""检索：find_methods —— 把学生题目映射到课本方法卡片并排序。

双通道融合：
- 词面分（离线必通：关键词命中加权 + 字符二元组余弦）
- 向量分（尽力而为：Chroma 已就绪则叠加；失败静默降级，不影响作答）
"""
from __future__ import annotations

from .config import Settings
from .schema import RetrievedMethod
from .store import CourseStore, VectorStore, lexical_score


def find_methods(
    question: str,
    course: str,
    top_k: int | None = None,
    settings: Settings | None = None,
    use_chroma: bool = True,
) -> list[RetrievedMethod]:
    """返回按相关度降序的方法卡片。空课程/无命中返回空列表。"""
    settings = settings or Settings()
    top_k = top_k if top_k and top_k > 0 else settings.top_k_default

    store = CourseStore(course, settings)
    cards = store.load_all()
    if not cards:
        return []
    by_id = {c.id: c for c in cards if not c.reference_only}  # 仅供参考类不进检索/考点

    scores: dict[str, float] = {}

    # 通道 1：词面分（保底，离线可跑）
    for c in cards:
        if c.reference_only:  # 仅供参考类不进考点
            continue
        s = lexical_score(question, c)
        if s > 0:
            scores[c.id] = scores.get(c.id, 0.0) + s

    # 通道 2：向量分（尽力而为）
    if use_chroma and settings.storage in ("chroma", "both"):
        try:
            vs = VectorStore(course, settings)
            for cid, score in vs.query(question, top_k=max(top_k * 3, 10)):
                if cid in by_id:  # 只采纳 jsonl 里真实存在的卡，隔离向量库残留旧 id
                    scores[cid] = scores.get(cid, 0.0) + score
        except Exception:  # noqa: BLE001 —— 离线/嵌入不可用 → 仅词面检索
            pass

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [RetrievedMethod(card=by_id[cid], score=round(s, 4)) for cid, s in ranked]
# -*- coding: utf-8 -*-
"""检索评测：对比「一次性 find_methods」基线与「agentic 工具循环」的召回率。

用法（在项目根目录运行）：
    python evals/run_eval.py --mode baseline    # 只跑基线（离线、免费、无需 API Key）
    python evals/run_eval.py --mode agentic     # 只跑 agent 模式（需 API Key）
    python evals/run_eval.py --mode all         # 两者都跑并对比
    python evals/run_eval.py --mode agentic --only 3   # 只跑前 3 题（快速抽查）

评测集：evals/questions.jsonl，每题含 expect_topics（期望命中的卡片主题）。
指标：recall —— 期望主题里有多少出现在「检索可见卡片」中。
  - baseline：find_methods(question, top_k=5) 返回的 5 张卡
  - agentic ：agent 循环中任何一次 search_methods 返回的卡片（模型的信息集）
结果落盘 evals/results/<mode>-<时间戳>.json，方便前后对比。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coursebook_digest.config import get_settings  # noqa: E402
from coursebook_digest.retrieve import find_methods  # noqa: E402

QUESTIONS = ROOT / "evals" / "questions.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"


def load_questions() -> list[dict]:
    rows = []
    with open(QUESTIONS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hit_topics(topics: list[str], expect: list[str]) -> list[str]:
    """期望主题中，被检索结果覆盖的部分（子串匹配，大小写不敏感，
    容忍卡片主题的英文大小写差异，如 Mode Switch vs mode switch）。"""
    joined = " ".join(topics).lower()
    return [e for e in expect if e.lower() in joined]


def run_baseline(rows: list[dict], top_k: int = 5) -> list[dict]:
    settings = get_settings()
    out = []
    for i, q in enumerate(rows, 1):
        methods = find_methods(q["question"], q["course"], top_k=top_k, settings=settings)
        topics = [m.card.topic for m in methods]
        hits = hit_topics(topics, q["expect_topics"])
        out.append({
            "idx": i, "course": q["course"], "tag": q["tag"], "question": q["question"],
            "expect": q["expect_topics"], "hits": hits,
            "retrieved_topics": topics,
            "recall": round(len(hits) / max(len(q["expect_topics"]), 1), 3),
        })
        print(f"  [{i:02d}] {q['tag']} recall={len(hits)}/{len(q['expect_topics'])} "
              f"{'OK ' if len(hits) == len(q['expect_topics']) else 'MISS'}  {q['question'][:30]}")
    return out


def run_agentic(rows: list[dict], only: int | None = None) -> list[dict]:
    from coursebook_digest.agent_loop import AgentLoop  # 延迟导入：基线模式不依赖 LLM

    settings = get_settings()
    out = []
    n = len(rows) if not only else min(only, len(rows))
    for i, q in enumerate(rows[:n], 1):
        t0 = time.time()
        try:
            agent = AgentLoop(q["course"], settings=settings)
            result = agent.run(q["question"])
            topics = list(result.seen_topics)
            detail = f"rounds={result.rounds} tools={result.tool_calls}"
        except Exception as exc:  # noqa: BLE001 —— 单题失败不中断整批
            topics, detail = [], f"ERROR {type(exc).__name__}: {exc}"
        hits = hit_topics(topics, q["expect_topics"])
        out.append({
            "idx": i, "course": q["course"], "tag": q["tag"], "question": q["question"],
            "expect": q["expect_topics"], "hits": hits, "seen_topics": topics,
            "detail": detail, "secs": round(time.time() - t0, 1),
            "recall": round(len(hits) / max(len(q["expect_topics"]), 1), 3),
        })
        print(f"  [{i:02d}] {q['tag']} recall={len(hits)}/{len(q['expect_topics'])} "
              f"({time.time() - t0:.0f}s, {detail})  {q['question'][:30]}")
    return out


def summarize(results: list[dict], label: str) -> None:
    if not results:
        return
    total = sum(r["recall"] for r in results)
    print(f"\n== {label} 总召回率: {total / len(results):.1%}（{len(results)} 题）==")
    for tag in ("直查", "改写", "两跳"):
        sub = [r for r in results if r["tag"] == tag]
        if sub:
            print(f"   {tag}: {sum(r['recall'] for r in sub) / len(sub):.1%}（{len(sub)} 题）")


def save(results: list[dict], mode: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "agentic", "all"], default="baseline")
    ap.add_argument("--only", type=int, default=None, help="只跑前 N 题（抽查）")
    args = ap.parse_args()

    rows = load_questions()
    if args.only:
        rows = rows[: args.only]
    print(f"评测集 {len(rows)} 题（evals/questions.jsonl）\n")

    if args.mode in ("baseline", "all"):
        print("── baseline：一次性 find_methods(top_k=5) ──")
        base = run_baseline(rows)
        summarize(base, "baseline")
        p = save(base, "baseline")
        print(f"已保存: {p}")
    base = None
    if args.mode in ("agentic", "all"):
        print("\n── agentic：tool-calling 循环 ──")
        agg = run_agentic(rows, only=args.only)
        summarize(agg, "agentic")
        p = save(agg, "agentic")
        print(f"已保存: {p}")


if __name__ == "__main__":
    main()

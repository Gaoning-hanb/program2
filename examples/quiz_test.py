"""配套习题 → coursebook 检索命中测试（离线，零 token）。

把《原子物理与量子信息综合习题》逐题转成 find_methods 查询，
评估已建库教材对配套习题的覆盖与命中质量。
用法：conda activate firpro && python examples/quiz_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coursebook_digest.config import get_settings
from coursebook_digest.retrieve import find_methods

COURSE = "量子物理"
SETTINGS = get_settings()

# (题号, 查询词)
QUIZZES = [
    # —— 简单题 ——
    ("1", "氢原子定态量子数 n l ml 取值范围 简并度 简并来源"),
    ("2", "径向概率密度 P(r)=r^2|R_nl|^2 为什么不能用 |R|^2 电子径向概率"),
    ("3", "玻尔原子模型与量子力学氢原子模型 核心区别 轨道 角动量"),
    ("4", "电偶极辐射跃迁选择定则 Delta l=±1 宇称 禁戒跃迁"),
    ("5", "原子实 碱金属原子能级 与氢原子相似的原因"),
    ("6", "斯特恩格拉赫实验 银原子 非均匀磁场 两条亮斑 电子自旋"),
    ("7", "轨道磁矩公式 负号 玻尔磁子 mue_B 表达式"),
    ("8", "自旋轨道耦合 能级分裂 l 不等于 0 分几层"),
    ("9", "正常塞曼效应 反常塞曼效应 谱线分裂条数 根源"),
    ("10", "电子自旋量子数 s=1/2 自旋角大小 S 与 z 分量 Sz 取值"),
    ("11", "l=2 轨道角动量大小 L Lz 取值 矢量与z轴夹角 cosθ"),
    ("12", "单电子总角动量 J=L+S l=2 s=1/2 可能的 j 值"),
    ("13", "朗德 g_j 因子公式 物理作用"),
    ("14", "狄拉克记号 右矢 左矢 内积 外积 矩阵类比"),
    ("15", "泡利矩阵 σ_x σ_y σ_z 本征值 自旋本征态"),
    ("16", "Bloch球 量子态表示 θ φ 物理含义"),
    ("17", "幺正变换 单量子比特 操作 Bloch球 几何旋转 Hadamard门"),
    ("18", "泡利不相容原理 完整表述 单电子量子数集合"),
    ("19", "两个电子 三重态 单重态 自旋波函数对称性 S Sz 取值"),
    ("20", "量子不可克隆定理 核心结论 量子演化线性性"),
    # —— 计算题 ——
    ("C1", "n=2 氢原子能级 En=-Z^2/n^2 hcR 简并度 l ml 组合"),
    ("C2", "基态氢径向概率 P(r)=4/a0^3 r^2 e^-2r/a0 极大值 r=a0"),
    ("C3", "l=3 轨道角动量模长 Lz 取值 最大最小夹角"),
    ("C4", "l=1 j=3/2 朗德公式 gj 有效磁矩 mujz 取值"),
    ("C5", "自旋态归一化 测量Sz概率 期望值 泡利矩阵"),
    ("C6", "磁场沿z 哈密顿 hbar omega sigma_z 自旋进动 期望值 Sx Sz"),
    ("C7", "两电子三重态单重态 自旋波函数 交换对称性"),
    ("C8", "量子不可克隆 幺正算符 线性性 矛盾 证明"),
    # —— 论述题 ——
    ("D1", "碱金属光谱双线结构 自旋轨道耦合 总角动量 朗德g"),
    ("D2", "经典粒子束与银原子束梯度磁场 量子化 斯特恩格拉赫 量子本质"),
    ("D3", "氢原子能级 由n决定 弱磁场按mj分裂 自旋轨道耦合 简并解除"),
    ("D4", "二能级系统 量子比特物理基础 Bloch球 泡利门 H门 物理旋转"),
]


def main() -> None:
    SETTINGS.ensure_dirs()
    print(f"课程《{COURSE}》：共 {len(QUIZZES)} 道题 → find_methods 命中测试（每道取 Top3）\n")
    rows = []
    for tag, q in QUIZZES:
        hits = find_methods(q, COURSE, top_k=3, settings=SETTINGS)
        if not hits:
            rows.append((tag, "无命中", "", 0.0))
            print(f"{tag:>3} | 无命中")
            continue
        tops = " | ".join(f"{h.card.chapter}/{h.card.topic[:10]} {h.score:.2f}" for h in hits)
        rows.append((tag, q[:16], hits[0].card.chapter, hits[0].score))
        print(f"{tag:>3} | {tops}")
    hit = sum(1 for r in rows if r[3] > 0)
    print(f"\n命中率：{hit}/{len(QUIZZES)}（Top1 相关度>0 的题数）")
    # 章节分布
    from collections import Counter

    chs = Counter(r[2] for r in rows if r[2])
    print("命中章节分布:", dict(chs))


if __name__ == "__main__":
    main()
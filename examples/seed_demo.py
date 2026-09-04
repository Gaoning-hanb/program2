"""离线演示种子：不联网、不需要 API key。

把示例教材（examples/sample_course.md）的人工蒸馏方法卡写入演示课程
「高等数学示例」，让你立刻能用 coursebook find 体验检索与优先注入。

用法：
    conda activate firpro && python examples/seed_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coursebook_digest.config import get_settings
from coursebook_digest.schema import MethodCard
from coursebook_digest.store import CourseStore, VectorStore

COURSE = "高等数学示例"
CHAPTER = "第一章 极限与连续"

CARDS = [
    MethodCard(
        id=f"{COURSE}|{CHAPTER}|001", course=COURSE, chapter=CHAPTER,
        topic="极限的运算法则", kind="方法",
        keywords=["极限", "四则运算", "加减乘除", "拆分", "存在"],
        applicability="当每个参与运算的极限都存在时，可拆开逐项求",
        steps=["先确认各分式/各项极限存在", "按加减乘除法则逐项求极限", "除法时先验证分母极限不为0"],
        error_notes=["拆项前必须验证每项极限都存在", "分母极限为0时不能用除法法则"],
        worked_example="lim (2x+3) 在 x→1 = 2·1+3 = 5",
        source_page="8",
    ),
    MethodCard(
        id=f"{COURSE}|{CHAPTER}|002", course=COURSE, chapter=CHAPTER,
        topic="两个重要极限", kind="定理",
        keywords=["重要极限", "sinx", "x", "e", "1/x", "夹逼"],
        applicability="形如 sin(□)/□ 且 □→0；形如 (1+1/□)^□ 或 (1+□)^(1/□)",
        steps=["识别结构是哪种重要极限", "把自变量凑成 □ 的形式", "套用 1 或 e 的结果"],
        core_formula_latex=r"\lim_{x\to0}\frac{\sin x}{x}=1,\quad \lim_{x\to\infty}\left(1+\frac{1}{x}\right)^x=e",
        error_notes=["凑形时底数与指数要联动", "sin(□)/□ 只有 □→0 时极限才为1"],
        source_page="10",
    ),
    MethodCard(
        id=f"{COURSE}|{CHAPTER}|003", course=COURSE, chapter=CHAPTER,
        topic="洛必达法则", kind="方法",
        keywords=["洛必达", "未定型", "0/0", "无穷比无穷", "求导", "极限"],
        applicability="0/0 或 ∞/∞ 型未定式，分子分母可导且分母导数不为0",
        steps=["先判断型别是否为0/0或∞/∞", "分子分母分别求导", "检查新极限是否存在，存在则等于原极限", "仍为未定型可继续用，注意最多用几次"],
        core_formula_latex=r"\lim\frac{f(x)}{g(x)}=\lim\frac{f'(x)}{g'(x)}",
        error_notes=["只能用于0/0或∞/∞，其他型先变形", "是分子分母各自求导，不是对整个分式求导", "求导后振荡则法则失效，需换方法"],
        worked_example="lim_{x→0} sin x / x 若误用洛必达会循环，本题直接用重要极限更简单——先判断型别再选法",
        source_page="12",
    ),
    MethodCard(
        id=f"{COURSE}|{CHAPTER}|004", course=COURSE, chapter=CHAPTER,
        topic="等价无穷小替换", kind="方法",
        keywords=["等价无穷小", "替换", "泰勒", "sin", "tan", "ln", "1-cosx", "高阶"],
        applicability="乘除因子中出现趋于0的常见函数可整体替换",
        steps=["确认是乘除结构", "把趋于0的因子替换成等价无穷小", "化简求极限"],
        core_formula_latex=r"x\to0:\;\sin x\sim x,\;1-\cos x\sim\frac{x^2}{2},\;e^x-1\sim x,\;\ln(1+x)\sim x",
        error_notes=["加减项不能随意替换等价无穷小", "替换要整体替换，不能只换一部分"],
        worked_example="lim_{x→0} (1-cos x)/x^2 = (x^2/2)/x^2 = 1/2",
        source_page="15",
    ),
]


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    store = CourseStore(COURSE, settings)
    n = store.save_all(CARDS)
    try:  # 尽力而为：写入向量索引；失败不影响演示
        VectorStore(COURSE, settings).upsert_cards(CARDS)
        print("向量索引已更新")
    except Exception as exc:  # noqa: BLE001
        print(f"（向量索引跳过：{exc}）")
    print(f"写入课程 {COURSE}：{len(CARDS)} 张方法卡（本次新增 {n}）")
    print(f"数据文件：{store.path}")
    print("\n试用：")
    print('  python -m coursebook_digest.cli find "分子分母趋于零怎么求极限" --course 高等数学示例')
    print('  python -m coursebook_digest.cli courses')


if __name__ == "__main__":
    main()
"""教材方法卡片的统一数据模型。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MethodKind(str, Enum):
    """方法单元的种类。"""

    CONCEPT = "概念"        # 定义 / 名词
    THEOREM = "定理"        # 命题 / 公式 / 性质
    METHOD = "方法"         # 解题套路 / 运算技巧（核心）
    EXAMPLE = "例题"        # 典型例题 + 解答
    ERROR_NOTE = "易错点"    # 常见错误 / 坑


class MethodCard(BaseModel):
    """一份可检索、可执行的方法卡片（“课本蒸馏出的逻辑与运算技巧”）。"""

    id: str = Field(description="稳定 id，形如 course|chapter|序号")
    course: str = Field(description="所属课程（命名空间隔离）")
    chapter: str = Field(description="章节名")
    topic: str = Field(description="主题/知识点名（一句话）")
    kind: MethodKind = Field(description="单元种类")
    keywords: list[str] = Field(
        default_factory=list,
        description="检索关键词：术语、题型及其常见变形叫法",
    )
    prerequisites: list[str] = Field(default_factory=list, description="前置概念")
    applicability: str = Field(default="", description="适用条件：什么时候用这个方法")
    steps: list[str] = Field(default_factory=list, description="标准步骤（按顺序可执行）")
    core_formula_latex: str = Field(default="", description="核心公式（LaTeX）")
    technique: str = Field(default="", description="常用技巧/思路点睛（可选）")
    worked_example: str = Field(default="", description="代表例题及其关键解法（可选）")
    error_notes: list[str] = Field(default_factory=list, description="易错点/常见坑（可选）")
    source_page: str = Field(default="", description="教材页码/锚点（溯源用）")
    reference_only: bool = Field(
        default=False,
        description="仅供参考类（教材编排/阅读顺序/指引等），不参与抽查/打卡/考点",
    )

    def summary(self) -> str:
        return f"[{self.kind.value}] {self.topic}（{self.chapter}）"


class RetrievedMethod(BaseModel):
    """检索命中结果，带分数与命中说明。"""

    card: MethodCard
    score: float
    note: str = Field(default="", description="命中说明/章节位置")
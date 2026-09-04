"""coursebook-digest：教材 → 结构化方法卡片 → 优先注入课本逻辑作答案。

面向大学应试场景：每学期一本书、课程之间知识不连续，
因此所有数据都按 `course`（课程）命名空间隔离，互不串味。
"""

__version__ = "0.1.0"

from .schema import MethodCard, MethodKind, RetrievedMethod

__all__ = ["MethodCard", "MethodKind", "RetrievedMethod", "__version__"]
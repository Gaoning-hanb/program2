"""蒸馏前去噪：剔除背景/人物介绍/PPT 模板等非重点内容，减少蒸馏输入与块数。

- 面向工科/应试教材与 PPT：重点保留概念/公式/方法/例题，剔除**故事性背景**、
  **人物生平**、**PPT 版式杂讯**（占位符、导航、致谢、页码、网址）。
- 保守策略：凡是疑似技术内容（含公式符 / markdown 标题 / 工程关键词 / 序号列表）
  一律不删，宁可少删不误删。
- 纯规则实现，零 LLM 开销、零网络，可离线测试。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .parser import Chapter

# --------------------------------------------------------------------------- #
# 识别规则
# --------------------------------------------------------------------------- #
_ANCHOR_RE = re.compile(r"^\s*<!--\s*page\s+\d+\s*-->\s*$")   # 页码锚点 ← 保留
_HEADING_RE = re.compile(r"^\s*(#{1,6}\b|[-=*_]{3,}\s*$)")     # markdown 标题 ← 保留
_HAS_MATH_RE = re.compile(r"[\\$=^]")  # LaTeX(`\`) / 数学符号 → 永不因故事/模板规则删除
# 疑似技术内容：含下列关键词的句子不因“故事/背景”规则删除
_TECHMARK_RE = re.compile(
    r"方程|定理|定律|效应|公式|推导|证明|求解|定义|性质|方法|步骤|例题|"
    r"实验|波函数|概率|算符|本征|能级|谱线|原子|电子|粒子|电荷|电流|电压|电阻|"
    r"能量|动量|角动量|自旋|量子|场强|磁场|电场|函数|矩阵|变换|守恒|"
    r"如图|由图|如下|表\s*\d"
)
_LIST_RE = re.compile(r"^\s*\d{1,3}\s*[、．.)]")                 # 有序列表条目 ← 保留

# PPT / 模板杂讯（命中即删，宽一点没关系）
_PLACEHOLDER_RE = re.compile(
    r"单击此处|双击此处|添加(文本|标题|正文)|在此(输入|添加)|输入文本|键入文本|"
    r"占位符|双击打开|插入图片|插入文本|替换为图片"
)
# 中文语尾（子串匹配安全，中文正文少见）；英文结束语用“整行”匹配避免误伤散文
_ENDING_RE = re.compile(
    r"谢谢(大家|聆听|观看|欣赏|配合)?|感谢(观看|聆听|大家|配合|收看)|敬请指正|"
    r"请批评指正|欢迎(指导|批评|交流|提问)|完$",
)
_ENDING_LINE_RE = re.compile(
    r"^\s*(?:thank\s*you(?:\s*very\s*much)?|thanks|the\s*end|that'?s\s*all)\s*$",
    re.IGNORECASE,
)
# 导航/目录类：中英都整行锚定，防止英文子串(contents/index/outline)误伤正文
_NAV_RE = re.compile(
    r"^\s*(?:目录|目\s*录|内容提要|主要内容|本讲目录|提纲|大纲|返回(?:目录|首页)|"
    r"contents|table\s*of\s*contents|index|outline|next|previous)\s*$",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"^\s*(https?://|www\.|[\w.\-]+@[\w.\-]+\.(com|cn|edu|org))", re.IGNORECASE)
# 英文书/教材常见版式噪音（版权页、登录码、出版社广告等——非知识内容）
_ENG_CHROME_RE = re.compile(
    r"all rights reserved|permission of the publisher|without .*written permission|"
    r"access code|scratch off|student access|premium content|redeem|"
    r"pearson|published by|printed in|isbn\s*[:0-9xX-]|"
    r"^\s*(copyright|©|\(c\))\b",
    re.IGNORECASE,
)
_PAGENO_RE = re.compile(r"^\s*[-–—]?\s*[0-9]{1,4}\s*[-–—]?\s*$")
_PAGECN_RE = re.compile(r"^\s*第\s*[0-9]{1,4}\s*页\s*$")
_DECO_RE = re.compile(r"^\s*[·•\-—=_*~]{1,4}\s*$")

# 人物生平/背景故事（保守：命中且非技术行才删）
_STORY_RE = re.compile(
    r"个人简介|人物简介|生平|简历|履历|代表作品|主要成就|教育经历|工作经历|"
    r"出生于|卒于|毕业(于|自)|任教于|留学|诺贝尔|诺奖|院士|"
    r"被誉为|被称为|著名(科学家|物理学家|数学家|工程师|发明家)?|发明家|奠基(人|者)|先驱|"
    r"先后(在|任|获|主持|领导|创立)|曾获|历任|荣获",
    re.IGNORECASE,
)
_STORY_YEAR_RE = re.compile(
    r"(出生|生于|创立|创建|毕业于|获得|荣获|发表|出任|留学|任教)\s*(于)?\s*[0-9]{3,4}\s*年?"
)


@dataclass
class DenoiseStats:
    removed_lines: int = 0
    story_removed: int = 0
    chrome_removed: int = 0
    kept_chars: int = 0
    removed_chars: int = 0

    @property
    def removed_ratio(self) -> float:
        total = self.kept_chars + self.removed_chars
        return self.removed_chars / total if total else 0.0


def _drop_story(line: str) -> bool:
    """是否命中"背景/人物"规则（附加技术内容豁免与长度门槛，降低误删）。"""
    if len(line) > 180 or _HAS_MATH_RE.search(line) or _TECHMARK_RE.search(line):
        return False
    return bool(_STORY_RE.search(line) or _STORY_YEAR_RE.search(line))


def _drop_line(line: str) -> tuple[bool, str]:
    """返回 (是否删除, 类别)。类别: story | chrome | noise | ''"""
    if _ANCHOR_RE.match(line) or _HEADING_RE.match(line) or _LIST_RE.match(line):
        return False, ""
    if _HAS_MATH_RE.search(line):  # 公式行永不因模板/故事规则删除
        return False, ""
    s = line.strip()
    if not s or _DECO_RE.match(s):
        return True, "noise"
    if _PLACEHOLDER_RE.search(s) or _ENDING_RE.search(s) or _ENDING_LINE_RE.match(s) \
            or _NAV_RE.search(s) or _ENG_CHROME_RE.search(s):
        return True, "chrome"
    if _URL_RE.match(s) or _PAGENO_RE.match(s) or _PAGECN_RE.match(s) or (
        len(s) <= 2 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", s)
    ):
        return True, "chrome"
    if _drop_story(s):
        return True, "story"
    return False, ""


def denoise_text(text: str) -> tuple[str, DenoiseStats]:
    """逐行去噪，返回 (清洗后文本, 统计)。纯函数：可能返回空串，不做回退。"""
    kept: list[str] = []
    stats = DenoiseStats()
    for raw in text.splitlines():
        drop, cat = _drop_line(raw)
        if drop:
            stats.removed_lines += 1
            stats.removed_chars += len(raw)
            if cat == "story":
                stats.story_removed += 1
            elif cat == "chrome":
                stats.chrome_removed += 1
            continue
        stats.kept_chars += len(raw)
        kept.append(raw)
    return "\n".join(kept).strip(), stats


def denoise_chapter(chapter: Chapter) -> tuple[Chapter, DenoiseStats]:
    """去噪章节；若整章被删空则**保留原文**（避免蒸馏空章节），统计置零。"""
    text, stats = denoise_text(chapter.text)
    if not text:
        return chapter, DenoiseStats(kept_chars=len(chapter.text))
    return replace(chapter, text=text), stats

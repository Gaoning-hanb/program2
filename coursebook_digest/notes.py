# -*- coding: utf-8 -*-
"""复习资料生成：方法卡片库 → 思维导图（markmap HTML）+ 章节笔记（Markdown）。

- 思维导图：课程 → 章节 → 卡片主题 → 要点，markmap 交互式可折叠；
- 笔记：按章节组织每张卡的 适用条件/标准步骤/核心公式/例题/易错点，
  视频卡带「跳回原视频时刻」链接（P2 12:35 → B站 t=755s）。

纯代码生成（不调模型、秒出）；教材课程与视频课程通用。
"""
from __future__ import annotations

import re
from pathlib import Path

from .schema import MethodCard

_ASSETS = Path(__file__).resolve().parent / "assets"
_AUTOLOADER = "markmap-autoloader.js"
_AUTOLOADER_CDN = "https://cdn.jsdelivr.net/npm/markmap-autoloader@0.18.10/dist/index.js"

_KIND_BADGE = {"方法": "🛠 方法", "例题": "📝 例题", "概念": "📘 概念",
               "定理": "📐 定理", "易错点": "⚠️ 易错"}


def _one_line(s: str, limit: int = 70) -> str:
    """压成一行并截断（导图节点/概要用）。"""
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[: limit - 1] + "…" if len(s) > limit else s


def _by_chapter(cards: list[MethodCard]) -> dict[str, list[MethodCard]]:
    """按章节分组并保持卡序（id 尾号顺序）。"""
    groups: dict[str, list[MethodCard]] = {}
    for c in sorted(cards, key=lambda c: (c.chapter, c.id)):
        groups.setdefault(c.chapter, []).append(c)
    return groups


# --------------------------------------------------------------------------- #
# 思维导图（markmap：markdown 列表即树）
# --------------------------------------------------------------------------- #
def build_mindmap_markdown(cards: list[MethodCard], course: str) -> str:
    lines: list[str] = [f"# {course}", ""]
    for chapter, ch_cards in _by_chapter(cards).items():
        lines.append(f"## {chapter}")
        for c in ch_cards:
            badge = _KIND_BADGE.get(c.kind.value, c.kind.value)
            lines.append(f"### {c.topic}｜{badge}")
            if c.applicability:
                lines.append(f"- 适用：{_one_line(c.applicability, 60)}")
            for i, step in enumerate(c.steps[:4], 1):  # 步骤太长导图会爆
                lines.append(f"- {i}. {_one_line(step, 60)}")
            if c.core_formula_latex:
                lines.append(f"- 公式：{_one_line(c.core_formula_latex, 60)}")
            for e in c.error_notes[:2]:
                lines.append(f"- ⚠ {_one_line(e, 60)}")
            lines.append("")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_markmap_html(md: str, title: str, use_cdn: bool = False) -> tuple[str, str]:
    """生成自包含思维导图 HTML；返回 (html, autoloader引用方式)。

    本地模式：把包内 assets/markmap-autoloader.js 复制到输出目录旁再引用；
    CDN 模式：直接引用 jsdelivr（首次打开稍慢，但无需随附文件）。
    两种模式的渲染库（d3/markmap-view）均由 autoloader 在浏览器端按需加载。
    """
    src = _AUTOLOADER_CDN if use_cdn else _AUTOLOADER
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 思维导图</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; }}
  .markmap {{ position: relative; width: 100%; height: 100vh; }}
  .markmap > svg {{ width: 100%; height: 100%; }}
</style>
<script src="{src}"></script>
</head>
<body>
<div class="markmap">
<script type="text/template">
---
markmap:
  maxWidth: 320
  initialExpandLevel: 2
  spacingVertical: 8
---
{md}</script>
</div>
</body>
</html>
"""
    return html, src


# --------------------------------------------------------------------------- #
# 章节复习笔记（Markdown，视频卡带跳转链接）
# --------------------------------------------------------------------------- #
def _card_section(c: MethodCard) -> str:
    badge = _KIND_BADGE.get(c.kind.value, c.kind.value)
    out: list[str] = [f"### {badge}｜{c.topic}", ""]
    if c.applicability:
        out += [f"**适用条件**：{c.applicability}", ""]
    if c.steps:
        out.append("**标准步骤**")
        out += [f"{i}. {s}" for i, s in enumerate(c.steps, 1)]
        out.append("")
    if c.core_formula_latex:
        out += [ "**核心公式**", "", f"$$ {c.core_formula_latex} $$", ""]
    if c.technique:
        out += [f"**技巧**：{c.technique}", ""]
    if c.worked_example:
        out += ["**代表例题**", "", c.worked_example, ""]
    if c.error_notes:
        out.append("**易错点**")
        out += [f"- {e}" for e in c.error_notes]
        out.append("")
    if c.source_page:
        if c.source_url:
            out.append(f"**溯源**：{c.source_page} · [▶ 跳回视频原时刻]({c.source_url})")
        else:
            out.append(f"**溯源**：{c.source_page}")
        out.append("")
    return "\n".join(out)


def build_notes_markdown(cards: list[MethodCard], course: str) -> str:
    video_cards = sum(1 for c in cards if c.source_url)
    head = [
        f"# {course} 复习笔记",
        "",
        f"> coursebook-digest 自动生成：{len(cards)} 张方法卡片"
        + (f"（含 {video_cards} 张视频卡）" if video_cards else "") + "。",
        "> 视频卡溯源格式：P2 12:35 = 第 2 分P 第 12 分 35 秒，点击可跳回视频原时刻。",
        "",
    ]
    body: list[str] = []
    for chapter, ch_cards in _by_chapter(cards).items():
        body.append(f"## {chapter}（{len(ch_cards)} 卡）")
        body.append("")
        body += ["".join(_card_section(c)) for c in ch_cards]
    return "\n".join(head) + "\n" + "\n".join(body).strip() + "\n"


# --------------------------------------------------------------------------- #
# 落盘入口（CLI 调用）
# --------------------------------------------------------------------------- #
def write_notes(
    cards: list[MethodCard],
    course: str,
    out_dir: Path,
    use_cdn: bool = False,
    kinds: list[str] | None = None,
) -> list[Path]:
    """生成全部产物并返回文件清单。"""
    if kinds:
        cards = [c for c in cards if c.kind.value in kinds]
    if not cards:
        raise ValueError("没有可用的卡片（检查 --kinds 过滤条件）")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # 1) 思维导图（html + markdown 源）
    mm_md = build_mindmap_markdown(cards, course)
    html, src = render_markmap_html(mm_md, course, use_cdn=use_cdn)
    html_path = out_dir / "思维导图.html"
    html_path.write_text(html, encoding="utf-8")
    written.append(html_path)
    md_path = out_dir / "思维导图.md"
    md_path.write_text(mm_md, encoding="utf-8")
    written.append(md_path)
    if not use_cdn:  # 本地模式：把 autoloader 复制到输出目录
        local = _ASSETS / _AUTOLOADER
        if local.exists():
            (out_dir / _AUTOLOADER).write_bytes(local.read_bytes())

    # 2) 复习笔记
    notes_path = out_dir / "复习笔记.md"
    notes_path.write_text(build_notes_markdown(cards, course), encoding="utf-8")
    written.append(notes_path)
    return written

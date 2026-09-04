"""去噪模块离线测试：零 LLM、零网络。

运行：conda activate firpro && cd coursebook-digest && python test_denoise.py
覆盖：PPT 模板/人物生平/低价值行被删；技术/公式/标题/列表/页码锚点保留；统计准确。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from coursebook_digest.config import PROJECT_ROOT
from coursebook_digest.denoise import denoise_chapter, denoise_text
from coursebook_digest.parser import Chapter


def test_chrome_removed() -> None:
    lines = [
        "谢谢观看", "单击此处添加文本", "返回目录", "https://example.com/a/b",
        "12", "·", "第 5 页", "感谢观看",
    ]
    text = "\n".join(lines)
    out, st = denoise_text(text)
    assert out == "", f"模板杂讯应全删，实际 {out!r}"
    assert st.removed_lines == len(lines)
    assert st.chrome_removed > 0
    print(f"    [去噪] 模板/噪声 {st.removed_lines} 行全删 OK")


def test_story_removed() -> None:
    lines = [
        "麦克斯韦出生于1831年，毕业于剑桥大学，被誉为物理学大师。",
        "【个人简介】他是电磁理论的奠基人之一，曾任剑桥大学教授。",
        "1997年，他荣获诺贝尔物理学奖。",
    ]
    out, st = denoise_text("\n".join(lines))
    assert out == "", f"生平/背景应删，实际 {out!r}"
    assert st.story_removed == len(lines), st
    print(f"    [去噪] 背景/人物 {st.story_removed} 行全删 OK")


def test_technical_kept() -> None:
    lines = [
        "# 第四章 氢原子", "<!-- page 3 -->",
        "1、氢原子和类氢离子：核外只有一个电子",
        "库仑定律：F = k q1 q2 / r^2",
        "$E = mc^2$",
        "麦克斯韦方程组描述了电磁场的行为",
        "爱因斯坦的光电效应解释了光的粒子性",
        "自旋轨道耦合能：$\\Delta E_{LS} = \\frac{1}{2} \\, S \\cdot L$",
    ]
    out, st = denoise_text("\n".join(lines))
    for l in lines:
        assert l in out, f"技术内容不应被删：{l!r}"
    assert st.removed_lines == 0, st
    print(f"    [去噪] 技术/公式/标题/锚点 {len(lines)} 行全保留 OK")


def test_stats() -> None:
    text = "他出生于1957年，被誉为著名物理学家。\n真·内容：公式 $x=1$ 和结论。\n谢谢\n第二行内容。"
    out, st = denoise_text(text)
    assert "谢谢" not in out and "著名物理学家" not in out
    assert "真·内容" in out and "第二行内容" in out
    assert st.removed_lines == 2  # 生平1 + 谢谢1
    assert st.kept_chars > 0 and st.removed_chars > 0
    assert 0.0 < st.removed_ratio < 1.0
    print(f"    [去噪] 统计 OK（保留 {st.kept_chars} / 删 {st.removed_chars}，降幅 {st.removed_ratio:.0%}）")


def test_all_chrome_fallback() -> None:
    # denoise_text 是纯函数：全删即返回空
    out, st = denoise_text("谢谢观看\n目录\n·")
    assert out == ""
    # 但 denoise_chapter（入库路径）防止蒸馏空章节：整章为空时保留原文
    ch = Chapter(course="量子物理", chapter="模板章", text="谢谢观看\n目录\n·", pages="1")
    new_ch, st2 = denoise_chapter(ch)
    assert new_ch.text == ch.text
    assert st2.removed_lines == 0
    print("    [去噪] 纯函数全删=空、入库路径回退原文 OK")


def test_denoise_chapter() -> None:
    ch = Chapter(course="量子物理", chapter="氢原子",
                 text="# 氢原子\n他曾获诺奖，被誉为大师。\n主内容 $p(x)$ 正常。",
                 pages="1,2")
    new_ch, st = denoise_chapter(ch)
    assert isinstance(new_ch, Chapter)
    assert new_ch.course == ch.course and new_ch.pages == ch.pages
    assert "大师" not in new_ch.text and "主内容" in new_ch.text
    assert st.story_removed == 1
    print("    [去噪] denoise_chapter 返回新 Chapter OK")


def test_english_chrome_and_keep() -> None:
    chrome = [
        "All Rights Reserved. Permission of the publisher is required to reproduce any part.",
        "PEARSON EDUCATION INC., publishing as Prentice Hall",
        "Copyright (c) 2018 Pearson Education, Inc.",
        "access code", "ISBN 978-0-13-359414-5",
    ]
    out, st = denoise_text("\n".join(chrome))
    assert out == "", f"英文版式噪音应删，实际 {out!r}"
    keep = [
        "The kernel is the core component of the operating system.",
        "CHAPTER 1 Computer System Overview",
        "A process is an instance of a program in execution.",
        "Processors and GPUs are not the end of the computational story.",
        "Thanks to multiprogramming, CPU utilization is improved.",
    ]
    out2, _ = denoise_text("\n".join(keep))
    assert keep[0] in out2 and keep[1] in out2 and keep[2] in out2
    assert keep[3] in out2 and keep[4] in out2  # 散文里的 the end / thanks 不误删
    print("    [去噪] 英文模板删、技术行（含 the end/thanks 散文）保留 OK")


def main() -> None:
    test_chrome_removed()
    test_story_removed()
    test_technical_kept()
    test_stats()
    test_all_chrome_fallback()
    test_denoise_chapter()
    test_english_chrome_and_keep()
    print("\n去噪模块测试全部通过")


if __name__ == "__main__":
    main()

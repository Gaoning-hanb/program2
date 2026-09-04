"""OpenAI 兼容的 LLM 客户端（默认 DeepSeek）：文本补全 + 结构化 JSON 输出 + 自动重试纠错。"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings

T = BaseModel


def _strip_code_fence(text: str) -> str:
    """去掉模型偶尔包裹的 ```json ... ``` 围栏。"""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    return (m.group(1) if m else text).strip()


class LLMClient:
    """一次封装：``complete_text`` 与 ``parse_json_model``。

    结构化输出采用 JSON 模式 + pydantic 校验；解析失败时把错误回喂给模型
    让它自纠，最多 ``max_fix_rounds`` 次。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        if not self.settings.llm_api_key:
            raise ValueError(
                "缺少 LLM API Key：请在 .env 中设置 DEEPSEEK_API_KEY，"
                "或设置环境变量 LLM_API_KEY"
            )
        self._client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            # 校园网关对“大文本 + JSON 长输出”较慢（实测单块可达 ~2min），
            # 给足读超时，配合下方 tenacity 对超时/断连/限流的重试。
            timeout=600.0,
            max_retries=2,
        )

    # ---- 基础调用 ----

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((
            json.JSONDecodeError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        )),
    )
    def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.llm_temperature,
            "stream": False,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def complete_text(self, system: str, user: str) -> str:
        return self._chat(system, user, json_mode=False)

    def complete_text_stream(self, system: str, user: str):
        """流式生成：逐段 yield content delta（页面实时显示用；非结构化文本）。"""
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.llm_temperature,
            "stream": True,
        }
        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ---- 结构化输出 ----

    def parse_json_model(
        self,
        model_cls: type[T],
        system: str,
        user: str,
        max_fix_rounds: int = 2,
    ) -> T:
        """让模型输出符合 ``model_cls`` 的 JSON，失败则回喂错误自纠。"""
        json_mode = hasattr(model_cls, "model_json_schema")  # 均支持
        last_err = ""
        for _ in range(max_fix_rounds + 1):
            prompt = user if not last_err else (
                user
                + "\n\n上次数模型输出未通过校验，请修正后重新输出：\n"
                + last_err
            )
            text = self._chat(system, prompt, json_mode=json_mode)
            raw = _strip_code_fence(text)
            try:
                data = json.loads(raw)
                return model_cls.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = str(exc)
        raise ValueError(
            f"模型连续 {max_fix_rounds + 1} 次未能输出合法 JSON / 结构：{last_err}"
        )

    def parse_json(self, system: str, user: str, max_fix_rounds: int = 2) -> dict[str, Any]:
        """宽松形态：只要求输出一个 JSON 对象，不做模型校验。"""
        last_err = ""
        for _ in range(max_fix_rounds + 1):
            prompt = user if not last_err else (
                user + "\n\n上次输出不是合法 JSON，请修正：\n" + last_err
            )
            text = self._chat(system, prompt, json_mode=True)
            raw = _strip_code_fence(text)
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
                raise ValueError("顶层必须是 JSON 对象")
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
        raise ValueError(f"模型连续 {max_fix_rounds + 1} 次未能输出合法 JSON：{last_err}")
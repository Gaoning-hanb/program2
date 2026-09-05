"""coursebook-digest 配置：.env 或环境变量，均可覆盖；路径以项目根为基准。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根 = 本包所在目录的上一级（即 coursebook-digest/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 模型服务（OpenAI 兼容，默认 DeepSeek）----
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "LLM_API_KEY"),
        description="模型 API Key",
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "LLM_BASE_URL"),
    )
    llm_model: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("DEEPSEEK_MODEL", "LLM_MODEL"),
        description="模型名（.env: DEEPSEEK_MODEL）",
    )
    llm_temperature: float = Field(
        default=0.2,
        validation_alias=AliasChoices("DEEPSEEK_TEMPERATURE", "LLM_TEMPERATURE"),
    )

    # ---- 数据与存储（均以项目根为相对基准）----
    data_dir: str = Field(default=str(PROJECT_ROOT / "data"))
    storage: Literal["jsonl", "chroma", "both"] = "both"
    chroma_dir: str = Field(default=str(PROJECT_ROOT / "data" / "chroma"))
    embedding_mode: Literal["default", "mini"] = Field(
        default="default",
        description="default=离线hash嵌入（零依赖，必通）；mini=下载MiniLM（需联网，语义更强）",
    )

    # ---- 解析 ----
    default_parser: Literal["auto", "mineru", "mineru-http"] = "auto"

    # ---- 云端 MinerU 解析（学校网关 mineru 服务，与模型同网关同 key）----
    # 鉴权复用 llm_api_key（同一 Bearer）；也可单独配 MINERU_API_KEY 覆盖。
    mineru_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("MINERU_API_KEY", "MINERU_KEY"),
        description="云端 MinerU API Key；留空时复用 DEEPSEEK_API_KEY（同一网关）",
    )
    mineru_base_url: str = Field(
        default="https://api.llm.ustc.edu.cn",
        validation_alias=AliasChoices("MINERU_BASE_URL", "MINERU_URL"),
        description="云端 MinerU 网关地址（不含路径）",
    )
    mineru_chunk_pages: int = Field(
        default=30, ge=1, le=500,
        description="云端按文件上传，需先把 PDF 拆成 ≤该页数 的小文件",
    )
    mineru_batch_files: int = Field(
        default=8, ge=1, le=50,
        description="一次 POST 最多携带的分片文件数（响应 results 按文件独立返回）",
    )
    mineru_poll_seconds: float = Field(default=10, gt=0)
    mineru_poll_attempts: int = Field(default=30, ge=1)

    # ---- 检索 ----
    top_k_default: int = 5

    # ---- B站视频（coursebook video）----
    bili_sessdata: str = Field(
        default="",
        validation_alias=AliasChoices("BILIBILI_SESSDATA", "BILI_SESSDATA"),
        description="B站登录 cookie（SESSDATA）；取 AI 字幕更稳，留空则匿名抓取（CC 字幕通常可用）",
    )
    whisper_model: str = Field(
        default="small",
        validation_alias=AliasChoices("WHISPER_MODEL"),
        description="无字幕视频的本地转写模型：tiny/base/small/medium（GPU 自动优先，CPU 兜底）",
    )

    # ---- 蒸馏（提速相关）----
    strip_extraneous: bool = Field(
        default=True,
        description="蒸馏前剔除背景/人物介绍/PPT模板类非重点内容（大幅提速）",
    )
    distill_chunk_chars: int = Field(
        default=8000, ge=1000, le=20000,
        description="蒸馏分块大小（字符）；越大调用次数越少、单次输入越长",
    )
    distill_parallel: int = Field(
        default=3, ge=1, le=16,
        description="蒸馏并发分块数（>1 时并行调用模型，墙钟大幅缩短）",
    )

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
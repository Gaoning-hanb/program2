"""按课程的持久化：JSONL（事实层）+ Chroma 向量库（检索层）。

课程即命名空间：data/courses/<course>.jsonl 存全部方法卡片；
data/chroma 下每个课程一个 collection。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

from .config import Settings
from .schema import MethodCard

# --------------------------------------------------------------------------- #
# 词面嵌入（零依赖、离线、确定性）：字符二元组 + ASCII 词元 → 归一化向量
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[a-zA-Z]+")


def _tokenize(text: str) -> list[str]:
    """分词：ASCII 词元 + 中文连续串 + 中文字符二元组。”"""
    text = unicodedata.normalize("NFKC", text.lower())
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(m.group(0))
    # 中文部分按连续汉字抽取二元组
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def hash_embedding(text: str, dim: int = 256) -> list[float]:
    """把词面 token 哈希进固定维度向量（余弦相似度可用，离线必通）。"""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# JSONL 课程库
# --------------------------------------------------------------------------- #
def _course_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "courses"


def _course_file(settings: Settings, course: str) -> Path:
    return _course_dir(settings) / f"{_safe(course)}.jsonl"


def _safe(name: str) -> str:
    """JSONL 文件名安全化：保留 Unicode 文字（中文课程名可读且不撞名），
    仅替换路径非法字符。"""
    return re.sub(r'[^\w.\- ]', "_", name).strip().rstrip(".") or "course"


class CourseStore:
    """单课程方法卡片的 JSONL 读写（追加 + 按 id 覆盖重写，保证幂等）。"""

    def __init__(self, course: str, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.course = course
        self.path = _course_file(self.settings, course)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save_all(self, cards: list[MethodCard]) -> int:
        existing = self.load_all()
        by_id = {c.id: c for c in existing}
        saved_new = 0
        for card in cards:
            if card.id not in by_id:
                saved_new += 1
            by_id[card.id] = card
        lines = "".join(c.model_dump_json() + "\n" for c in by_id.values())
        self.path.write_text(lines, encoding="utf-8")
        return saved_new

    def load_all(self) -> list[MethodCard]:
        if not self.path.exists():
            return []
        out: list[MethodCard] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(MethodCard.model_validate_json(line))
            except Exception:  # noqa: BLE001
                continue
        return out

    @staticmethod
    def list_courses(settings: Settings | None = None) -> list[str]:
        settings = settings or Settings()
        d = _course_dir(settings)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.jsonl"))


# --------------------------------------------------------------------------- #
# Chroma 向量库（可选检索层）
# --------------------------------------------------------------------------- #
class HashEmbeddingFunction:
    """创建集合时的嵌入函数（符合 chromadb 协议：name() 是方法）。

    hash 模式下向量由我们显式传入 ``embeddings`` / ``query_embeddings``，
    chroma 不会真正调用它，因此不需要下载模型，离线必通。
    """

    def name(self) -> str:
        return "hash-ngram-v1"

    def __call__(self, input):  # noqa: A002 —— 显式传向量时不会走到这里
        if isinstance(input, str):
            return [hash_embedding(input)]
        return [hash_embedding(t) for t in input]


class VectorStore:
    """每课程一个 Chroma collection。

    - ``default``（离线）：hash 词面嵌入，向量显式写入/查询，零下载。
    - ``mini``（联网）：用 chroma 内置 MiniLM 做语义嵌入，首次使用需联网下载模型。
    """

    SANITIZED_COLL0NAME = re.compile(r"[^0-9A-Za-z_.-]")

    def __init__(self, course: str, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.course = course
        import chromadb  # 延迟导入：chromadb 启动较慢

        self.client = chromadb.PersistentClient(path=self.settings.chroma_dir)
        # Chroma 集合名只允许 ASCII 字母开头；用 ASCII 化 + 短哈希避免中文课程撞名
        ascii_part = self.SANITIZED_COLL0NAME.sub("_", course).strip("_")[:40] or "course"
        digest = hashlib.md5(course.encode("utf-8")).hexdigest()[:8]
        name = f"cb-{ascii_part}-{digest}"
        self.hash_mode = self.settings.embedding_mode != "mini"
        ef = None if not self.hash_mode else HashEmbeddingFunction()
        self.col = self.client.get_or_create_collection(
            name=name, embedding_function=ef, metadata={"hnsw:space": "cosine"}
        )

    def upsert_cards(self, cards: list[MethodCard]) -> None:
        if not cards:
            return
        docs = [
            " ".join([c.topic, c.applicability, " ".join(c.steps), " ".join(c.keywords),
                      c.technique, c.worked_example, " ".join(c.error_notes)])
            for c in cards
        ]
        metas = [{"chapter": c.chapter, "kind": c.kind.value, "topic": c.topic} for c in cards]
        kwargs: dict = {"ids": [c.id for c in cards], "documents": docs, "metadatas": metas}
        if self.hash_mode:
            kwargs["embeddings"] = [hash_embedding(d) for d in docs]
        self.col.upsert(**kwargs)

    def query(self, question: str, top_k: int = 5) -> list[tuple[str, float]]:
        kwargs: dict = {"n_results": top_k}
        if self.hash_mode:
            kwargs["query_embeddings"] = [hash_embedding(question)]
        else:
            kwargs["query_texts"] = [question]
        res = self.col.query(**kwargs)
        ids: list = res.get("ids", [[]])[0]
        dists: list = res.get("distances", [[]])[0]
        score = lambda d: max(0.0, 1.0 - d)  # 余弦距离 → 相似度
        return [(i, score(d)) for i, d in zip(ids, dists) if i]

    def rebuild(self, cards: list[MethodCard]) -> None:
        """删除当前课程集合并用给定卡片整体重建（保证向量库与 jsonl 完全一致）。

        重跑/换版本后旧 id 会残留在向量库拖垮检索，先删集合再全量重建最稳。
        """
        try:
            self.client.delete_collection(self.col.name)
        except Exception:  # noqa: BLE001
            pass
        self.col = self.client.get_or_create_collection(
            name=self.col.name,
            embedding_function=(None if not self.hash_mode else HashEmbeddingFunction()),
            metadata={"hnsw:space": "cosine"},
        )
        self.upsert_cards(cards)


# --------------------------------------------------------------------------- #
# 词面检索（离线保底，融合关键词加权与二元组余弦）
# --------------------------------------------------------------------------- #
def lexical_score(question: str, card: MethodCard) -> float:
    """查询与卡片词面相似度：关键词命中(加权高) + 全文字符二元组余弦。"""
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return 0.0
    # 1) 关键词命中率（权重最高，因为 keywords 是“检索友好”字段）
    kw_hits = len(q_tokens & set(k.lower() for k in card.keywords))
    kw_score = kw_hits / len(q_tokens) if q_tokens else 0.0
    # 2) 主题与正文二元组余弦
    text = " ".join([card.topic, card.applicability, " ".join(card.steps),
                     " ".join(card.keywords), card.technique])
    cos = cosine(hash_embedding(question), hash_embedding(text))
    # 3) 方法类与例题加权（作答时优先方法是设计意图）
    kind_boost = 0.15 if card.kind.value in ("方法", "例题") else 0.0
    return 0.6 * kw_score + 0.35 * cos + kind_boost
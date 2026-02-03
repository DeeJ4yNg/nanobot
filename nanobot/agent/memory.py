"""Memory system for persistent agent memory."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

from nanobot.config.schema import MemoryRetrievalConfig
from nanobot.utils.helpers import ensure_dir, today_date


class MemoryStore:
    """
    Memory system for the agent.
    
    Supports daily notes (memory/YYYY-MM-DD.md) and long-term memory (MEMORY.md).
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
    
    def get_today_file(self) -> Path:
        """Get path to today's memory file."""
        return self.memory_dir / f"{today_date()}.md"
    
    def read_today(self) -> str:
        """Read today's memory notes."""
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""
    
    def append_today(self, content: str) -> None:
        """Append content to today's memory notes."""
        today_file = self.get_today_file()
        
        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            # Add header for new day
            header = f"# {today_date()}\n\n"
            content = header + content
        
        today_file.write_text(content, encoding="utf-8")
    
    def read_long_term(self) -> str:
        """Read long-term memory (MEMORY.md)."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""
    
    def write_long_term(self, content: str) -> None:
        """Write to long-term memory (MEMORY.md)."""
        self.memory_file.write_text(content, encoding="utf-8")
    
    def append_long_term(self, content: str) -> None:
        """Append content to long-term memory (MEMORY.md)."""
        existing = self.read_long_term()
        if existing:
            updated = existing.rstrip() + "\n\n" + content.strip() + "\n"
        else:
            updated = content.strip() + "\n"
        self.write_long_term(updated)
    
    def get_recent_memories(self, days: int = 7) -> str:
        """
        Get memories from the last N days.
        
        Args:
            days: Number of days to look back.
        
        Returns:
            Combined memory content.
        """
        from datetime import timedelta
        
        memories = []
        today = datetime.now().date()
        
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"{date_str}.md"
            
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                memories.append(content)
        
        return "\n\n---\n\n".join(memories)
    
    def list_memory_files(self) -> list[Path]:
        """List all memory files sorted by date (newest first)."""
        if not self.memory_dir.exists():
            return []
        
        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)
    
    def get_memory_context(self) -> str:
        """
        Get memory context for the agent.
        
        Returns:
            Formatted memory context including long-term and recent memories.
        """
        parts = []
        
        # Long-term memory
        long_term = self.read_long_term()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)
        
        # Today's notes
        today = self.read_today()
        if today:
            parts.append("## Today's Notes\n" + today)
        
        return "\n\n".join(parts) if parts else ""


@dataclass(frozen=True)
class MemoryChunk:
    """A chunk of memory content."""
    chunk_id: int
    source: str
    text: str


@dataclass(frozen=True)
class MemoryMatch:
    """A scored memory chunk match."""
    text: str
    source: str
    score: float
    bm25_score: float
    semantic_score: float


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""


class LiteLLMEmbeddingProvider:
    """Embedding provider backed by LiteLLM."""
    def __init__(self, model: str, api_key: str | None, api_base: str | None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        from litellm import aembedding
        response = await aembedding(
            model=self.model,
            input=texts,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return [item.embedding for item in response.data]


class MemoryIndex:
    """Hybrid retrieval index using BM25 and semantic embeddings."""
    def __init__(
        self,
        memory_dir: Path,
        config: MemoryRetrievalConfig,
        embedder: EmbeddingProvider | None,
    ):
        self.memory_dir = memory_dir
        self.config = config
        self.embedder = embedder
        self.db_path = memory_dir / ".memory_index.sqlite"
        self._ensure_schema()
    
    async def search(self, query: str, top_k: int | None = None) -> list[MemoryMatch]:
        """Search memory with hybrid scoring."""
        await self._ensure_index()
        chunks = self._load_chunks()
        if not chunks:
            return []
        bm25_scores = self._bm25_scores(query, chunks)
        semantic_scores = await self._semantic_scores(query, chunks, top_k)
        combined = self._combine_scores(bm25_scores, semantic_scores, chunks)
        filtered = [match for match in combined if match.score >= self.config.min_score]
        limit = top_k or self.config.top_k
        return sorted(filtered, key=lambda m: m.score, reverse=True)[:limit]
    
    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    content_hash TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id INTEGER PRIMARY KEY,
                    embedding TEXT,
                    norm REAL,
                    signature INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
    
    async def _ensure_index(self) -> None:
        if self._needs_rebuild():
            await self._rebuild_index()
    
    def _needs_rebuild(self) -> bool:
        files = self._list_memory_files()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT path, mtime, size FROM files").fetchall()
            recorded = {row[0]: (row[1], row[2]) for row in rows}
            if len(recorded) != len(files):
                return True
            for path in files:
                stat = path.stat()
                stored = recorded.get(str(path))
                if not stored or stored[0] != stat.st_mtime or stored[1] != stat.st_size:
                    return True
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            if meta.get("embedding_model") != (self.config.embedding_model or ""):
                return True
            if meta.get("embedding_api_base") != (self.config.embedding_api_base or ""):
                return True
            if meta.get("lsh_planes") != str(self.config.lsh_planes):
                return True
            if meta.get("lsh_seed") != str(self.config.lsh_seed):
                return True
        return False
    
    async def _rebuild_index(self) -> None:
        files = self._list_memory_files()
        chunks = self._build_chunks(files)
        embeddings = await self._build_embeddings([c.text for c in chunks])
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM meta")
            for path in files:
                stat = path.stat()
                conn.execute(
                    "INSERT INTO files (path, mtime, size) VALUES (?, ?, ?)",
                    (str(path), stat.st_mtime, stat.st_size),
                )
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                content_hash = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()
                conn.execute(
                    "INSERT INTO chunks (id, source, text, content_hash) VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.source, chunk.text, content_hash),
                )
                if embedding is None:
                    conn.execute(
                        "INSERT INTO embeddings (chunk_id, embedding, norm, signature) VALUES (?, ?, ?, ?)",
                        (chunk.chunk_id, None, None, None),
                    )
                else:
                    norm = self._vector_norm(embedding)
                    signature = self._lsh_signature(embedding)
                    conn.execute(
                        "INSERT INTO embeddings (chunk_id, embedding, norm, signature) VALUES (?, ?, ?, ?)",
                        (chunk.chunk_id, json.dumps(embedding), norm, signature),
                    )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_model', ?)",
                (self.config.embedding_model or "",),
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_api_base', ?)",
                (self.config.embedding_api_base or "",),
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('lsh_planes', ?)",
                (str(self.config.lsh_planes),),
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('lsh_seed', ?)",
                (str(self.config.lsh_seed),),
            )
    
    def _list_memory_files(self) -> list[Path]:
        memory_files = []
        long_term = self.memory_dir / "MEMORY.md"
        if long_term.exists():
            memory_files.append(long_term)
        daily_files = sorted(self.memory_dir.glob("????-??-??.md"), reverse=True)
        memory_files.extend(daily_files)
        return memory_files
    
    def _build_chunks(self, files: Iterable[Path]) -> list[MemoryChunk]:
        chunks: list[MemoryChunk] = []
        chunk_id = 1
        for path in files:
            text = path.read_text(encoding="utf-8")
            for piece in self._chunk_text(text):
                if len(chunks) >= self.config.max_chunks:
                    return chunks
                chunks.append(
                    MemoryChunk(
                        chunk_id=chunk_id,
                        source=path.name,
                        text=piece,
                    )
                )
                chunk_id += 1
        return chunks
    
    def _chunk_text(self, text: str) -> list[str]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        chunks = []
        current = ""
        for block in blocks:
            if not current:
                current = block
                continue
            if len(current) + len(block) + 2 <= self.config.chunk_size:
                current = current + "\n\n" + block
            else:
                chunks.append(current)
                overlap = current[-self.config.chunk_overlap :] if self.config.chunk_overlap > 0 else ""
                current = overlap + "\n\n" + block if overlap else block
        if current:
            chunks.append(current)
        return chunks
    
    async def _build_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        if not self.embedder or not self.config.embedding_enabled:
            return [None for _ in texts]
        batch_size = 32
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings.extend(await self.embedder.embed(batch))
        return embeddings
    
    def _load_chunks(self) -> list[MemoryChunk]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, source, text FROM chunks ORDER BY id").fetchall()
        return [MemoryChunk(chunk_id=row[0], source=row[1], text=row[2]) for row in rows]
    
    def _bm25_scores(self, query: str, chunks: list[MemoryChunk]) -> dict[int, float]:
        tokens = self._tokenize(query)
        if not tokens:
            return {chunk.chunk_id: 0.0 for chunk in chunks}
        doc_tokens = [self._tokenize(chunk.text) for chunk in chunks]
        doc_freq: dict[str, int] = {}
        for tokens_list in doc_tokens:
            for token in set(tokens_list):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        avgdl = sum(len(t) for t in doc_tokens) / max(len(doc_tokens), 1)
        scores: dict[int, float] = {}
        k1 = 1.5
        b = 0.75
        for chunk, tokens_list in zip(chunks, doc_tokens, strict=False):
            score = 0.0
            doc_len = len(tokens_list)
            term_counts: dict[str, int] = {}
            for token in tokens_list:
                term_counts[token] = term_counts.get(token, 0) + 1
            for term in tokens:
                if term not in term_counts:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
                tf = term_counts[term]
                denom = tf + k1 * (1 - b + b * (doc_len / max(avgdl, 1)))
                score += idf * (tf * (k1 + 1) / denom)
            scores[chunk.chunk_id] = score
        return scores
    
    async def _semantic_scores(
        self,
        query: str,
        chunks: list[MemoryChunk],
        top_k: int | None,
    ) -> dict[int, float]:
        if not self.embedder or not self.config.embedding_enabled or self.config.semantic_weight == 0:
            return {chunk.chunk_id: 0.0 for chunk in chunks}
        query_embedding = (await self.embedder.embed([query]))[0]
        query_norm = self._vector_norm(query_embedding)
        candidate_rows = self._load_candidates(query_embedding)
        if not candidate_rows:
            candidate_rows = self._load_all_embeddings()
        scores: dict[int, float] = {}
        for chunk_id, embedding, norm in candidate_rows:
            if not embedding or not norm:
                continue
            similarity = self._cosine_similarity(query_embedding, query_norm, embedding, norm)
            scores[chunk_id] = similarity
        if not scores:
            return {chunk.chunk_id: 0.0 for chunk in chunks}
        limit = top_k or self.config.top_k
        top_ids = sorted(scores, key=scores.get, reverse=True)[: max(limit * 5, limit)]
        return {chunk_id: scores[chunk_id] for chunk_id in top_ids}
    
    def _load_candidates(self, query_embedding: list[float]) -> list[tuple[int, list[float] | None, float | None]]:
        if self.config.lsh_planes <= 0:
            return []
        signature = self._lsh_signature(query_embedding)
        signatures = {signature}
        for bit in range(min(2, self.config.lsh_planes)):
            signatures.add(signature ^ (1 << bit))
        placeholders = ",".join("?" for _ in signatures)
        sql = (
            "SELECT chunk_id, embedding, norm FROM embeddings "
            f"WHERE signature IN ({placeholders})"
        )
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, tuple(signatures)).fetchall()
        results = []
        for row in rows:
            embedding = json.loads(row[1]) if row[1] else None
            results.append((row[0], embedding, row[2]))
        return results
    
    def _load_all_embeddings(self) -> list[tuple[int, list[float] | None, float | None]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT chunk_id, embedding, norm FROM embeddings").fetchall()
        results = []
        for row in rows:
            embedding = json.loads(row[1]) if row[1] else None
            results.append((row[0], embedding, row[2]))
        return results
    
    def _combine_scores(
        self,
        bm25_scores: dict[int, float],
        semantic_scores: dict[int, float],
        chunks: list[MemoryChunk],
    ) -> list[MemoryMatch]:
        bm25_norm = self._normalize_scores(bm25_scores)
        semantic_norm = self._normalize_scores(semantic_scores)
        matches = []
        for chunk in chunks:
            bm25 = bm25_norm.get(chunk.chunk_id, 0.0)
            semantic = semantic_norm.get(chunk.chunk_id, 0.0)
            score = (self.config.bm25_weight * bm25) + (self.config.semantic_weight * semantic)
            matches.append(
                MemoryMatch(
                    text=chunk.text,
                    source=chunk.source,
                    score=score,
                    bm25_score=bm25,
                    semantic_score=semantic,
                )
            )
        return matches
    
    def _normalize_scores(self, scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if math.isclose(min_score, max_score):
            return {key: 0.0 for key in scores}
        return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}
    
    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]
    
    def _vector_norm(self, vector: list[float]) -> float:
        return math.sqrt(sum(val * val for val in vector))
    
    def _cosine_similarity(
        self,
        query: list[float],
        query_norm: float,
        doc: list[float],
        doc_norm: float,
    ) -> float:
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        dot = sum(a * b for a, b in zip(query, doc, strict=False))
        return dot / (query_norm * doc_norm)
    
    def _lsh_signature(self, embedding: list[float]) -> int:
        planes = self._lsh_planes(len(embedding))
        signature = 0
        for idx, plane in enumerate(planes):
            dot = sum(a * b for a, b in zip(embedding, plane, strict=False))
            if dot >= 0:
                signature |= 1 << idx
        return signature
    
    def _lsh_planes(self, dim: int) -> list[list[float]]:
        rnd = random.Random(self.config.lsh_seed)
        planes = []
        for _ in range(self.config.lsh_planes):
            planes.append([rnd.gauss(0, 1) for _ in range(dim)])
        return planes


class MemoryRetriever:
    """Retrieve relevant memory chunks with hybrid search."""
    def __init__(
        self,
        workspace: Path,
        config: MemoryRetrievalConfig,
        api_key: str | None,
        api_base: str | None,
    ):
        self.workspace = workspace
        self.config = config
        self.store = MemoryStore(workspace)
        self.embedder = self._build_embedder(api_key, api_base)
        self.index = MemoryIndex(self.store.memory_dir, config, self.embedder)
    
    @property
    def enabled(self) -> bool:
        """Check if retrieval is enabled."""
        return self.config.enabled
    
    async def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryMatch]:
        """Retrieve relevant memory chunks."""
        if not self.enabled:
            return []
        return await self.index.search(query, top_k)
    
    async def retrieve_context(self, query: str, top_k: int | None = None) -> tuple[str, list[MemoryMatch]]:
        """Retrieve memory chunks and return formatted context."""
        results = await self.retrieve(query, top_k)
        return self.build_context(results), results
    
    def build_context(self, results: list[MemoryMatch]) -> str:
        """Build memory context for the system prompt."""
        parts = []
        for match in results:
            parts.append(f"### {match.source}\n{match.text}")
        return "\n\n".join(parts)
    
    def format_results(self, results: list[MemoryMatch]) -> str:
        """Format results for tool responses."""
        parts = []
        for match in results:
            header = f"### {match.source} (score={match.score:.3f})"
            parts.append(f"{header}\n{match.text}")
        return "\n\n".join(parts)
    
    def should_summarize(self, results: list[MemoryMatch]) -> bool:
        """Check if a new memory summary should be created."""
        if not self.config.summary_enabled:
            return False
        if not results:
            return True
        top_score = max(result.score for result in results)
        return top_score < self.config.min_score
    
    def append_summary(self, summary: str) -> None:
        """Append a memory summary to long-term storage."""
        if summary.strip():
            self.store.append_long_term(summary)
    
    def _build_embedder(self, api_key: str | None, api_base: str | None) -> EmbeddingProvider | None:
        if not self.config.embedding_enabled or self.config.semantic_weight == 0:
            return None
        resolved_key = self.config.embedding_api_key or api_key
        resolved_base = self.config.embedding_api_base or api_base
        if not resolved_key and not resolved_base:
            return None
        return LiteLLMEmbeddingProvider(self.config.embedding_model, resolved_key, resolved_base)

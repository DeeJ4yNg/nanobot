import pytest

from nanobot.agent.memory import EmbeddingProvider, MemoryIndex, MemoryMatch, MemoryRetriever
from nanobot.agent.tools.memory import MemoryRecallTool
from nanobot.config.schema import MemoryRetrievalConfig


class StubEmbedder(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            if "alpha" in text.lower():
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return embeddings


@pytest.mark.asyncio
async def test_memory_index_prefers_relevant_chunk(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    memory_file = memory_dir / "MEMORY.md"
    memory_file.write_text(
        "# Project Alpha\nalpha decision details\n\n# Project Beta\nbeta notes\n",
        encoding="utf-8",
    )

    retrieval_config = MemoryRetrievalConfig(
        min_score=0.0,
        bm25_weight=0.5,
        semantic_weight=0.5,
        chunk_size=200,
        chunk_overlap=50,
        max_chunks=100,
        lsh_planes=8,
        lsh_seed=1,
    )
    index = MemoryIndex(memory_dir, retrieval_config, StubEmbedder())

    results = await index.search("alpha decision", top_k=2)

    assert results
    assert "alpha" in results[0].text.lower()


@pytest.mark.asyncio
async def test_memory_index_bm25_fallback(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    memory_file = memory_dir / "MEMORY.md"
    memory_file.write_text(
        "# Project Alpha\nalpha decision details\n\n# Project Beta\nbeta notes\n",
        encoding="utf-8",
    )

    retrieval_config = MemoryRetrievalConfig(
        min_score=0.0,
        bm25_weight=1.0,
        semantic_weight=0.0,
        embedding_enabled=False,
        chunk_size=200,
        chunk_overlap=50,
        max_chunks=100,
    )
    index = MemoryIndex(memory_dir, retrieval_config, None)

    results = await index.search("beta notes", top_k=1)

    assert results
    assert "beta" in results[0].text.lower()


def test_memory_retriever_should_summarize(tmp_path):
    retrieval_config = MemoryRetrievalConfig(min_score=0.4, summary_enabled=True)
    retriever = MemoryRetriever(tmp_path, retrieval_config, None, None)

    assert retriever.should_summarize([])
    assert retriever.should_summarize([
        MemoryMatch(text="a", source="b", score=0.1, bm25_score=0.1, semantic_score=0.0)
    ])
    assert not retriever.should_summarize([
        MemoryMatch(text="a", source="b", score=0.8, bm25_score=0.8, semantic_score=0.0)
    ])


@pytest.mark.asyncio
async def test_memory_recall_tool_empty_results():
    class StubRetriever:
        async def retrieve(self, query: str, top_k: int | None = None):
            return []

        def format_results(self, results):
            return "unused"

    tool = MemoryRecallTool(retriever=StubRetriever())
    result = await tool.execute("anything")
    assert result == "No relevant memory found."

from typing import Any

from nanobot.agent.memory import MemoryRetriever
from nanobot.agent.tools.base import Tool


class MemoryRecallTool(Tool):
    def __init__(self, retriever: MemoryRetriever):
        self.retriever = retriever

    @property
    def name(self) -> str:
        return "memory_recall"

    @property
    def description(self) -> str:
        return "Retrieve relevant long-term memory chunks using hybrid search."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query describing what to recall",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of memory chunks to retrieve",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, top_k: int | None = None, **kwargs: Any) -> str:
        results = await self.retriever.retrieve(query, top_k)
        if not results:
            return "No relevant memory found."
        return self.retriever.format_results(results)

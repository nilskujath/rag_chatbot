"""Adapters that let RAGAS drive this project's LLM and embedder.

RAGAS expects `BaseRagasLLM` and `BaseRagasEmbeddings` instances; these two
classes forward every call to `generate` and `Embedder`, so the test set and
the scores come from the same (rate-limited) models as the chatbot.
"""

import asyncio

from langchain_core.outputs import Generation, LLMResult
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig

from rag_chatbot.llm import generate
from rag_chatbot.storage.embedder import Embedder


class GeminiRagasLLM(BaseRagasLLM):
    """Gemini behind the RAGAS LLM interface; `model` picks the judge."""

    def __init__(self, model: str | None = None):
        super().__init__()
        self.model = model

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        """One completion at the requested temperature; n, stop, callbacks unused."""
        text = generate(prompt.to_string(), model=self.model, temperature=temperature)
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(
        self, prompt, n=1, temperature=0.01, stop=None, callbacks=None
    ):
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature)

    def is_finished(self, response: LLMResult) -> bool:
        return True


class EmbedderRagasEmbeddings(BaseRagasEmbeddings):
    """The project `Embedder` behind the RAGAS embeddings interface."""

    def __init__(self, embedder: Embedder):
        super().__init__()
        # BaseRagasEmbeddings declares run_config without a default; RAGAS reads it
        self.run_config = RunConfig()
        self.embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

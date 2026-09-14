"""Exercise DualRAG's default storage stack without models or API requests."""

from __future__ import annotations

import asyncio
import importlib.util
import json

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("lightrag") is None,
    reason="Install the optional DualRAG environment with uv sync --extra dualrag",
)


def test_default_storage_and_backend_imports(tmp_path, monkeypatch):
    # Default storage and backend imports must never mutate the locked environment.
    import pipmaster

    def forbid_install(*args, **kwargs):
        pytest.fail("DualRAG attempted to install a package at runtime")

    monkeypatch.setattr(pipmaster, "install", forbid_install)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    import numpy as np
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import finalize_share_data, initialize_pipeline_status
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    assert callable(openai_complete_if_cache)
    assert callable(openai_embed)
    assert callable(ollama_model_complete)
    assert callable(ollama_embed)

    async def embed(texts):
        return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))

    async def llm(*args, **kwargs):
        pytest.fail("Storage smoke test unexpectedly called an LLM")

    async def exercise_storage():
        rag = LightRAG(
            working_dir=str(tmp_path),
            namespace_prefix="runtime_smoke",
            llm_model_func=llm,
            embedding_func=EmbeddingFunc(embedding_dim=3, max_token_size=32, func=embed),
            auto_manage_storages_states=False,
        )
        try:
            await rag.initialize_storages()
            await initialize_pipeline_status()
            await rag.full_docs.upsert({"doc-1": {"content": "remote sensing"}})
            assert await rag.full_docs.get_by_id("doc-1") == {"content": "remote sensing"}

            await rag.chunks_vdb.upsert({"chunk-1": {"content": "remote sensing"}})
            matches = await rag.chunks_vdb.query("remote sensing", top_k=1)
            assert matches[0]["id"] == "chunk-1"

            graph = rag.chunk_entity_relation_graph
            await graph.upsert_node("satellite", {"description": "remote sensing platform"})
            assert await graph.has_node("satellite")

            await rag.full_docs.index_done_callback()
            await rag.chunks_vdb.index_done_callback()
            await graph.index_done_callback()
            assert list(tmp_path.glob("*.json"))
            assert list(tmp_path.glob("*.graphml"))
        finally:
            await rag.finalize_storages()
            finalize_share_data()

    asyncio.run(exercise_storage())


def test_openai_backend_with_mock_http(monkeypatch):
    """Check real SDK serialization and Pydantic parsing without an API service."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    import httpx
    import numpy as np
    from lightrag.llm import openai as backend
    from lightrag.types import GPTKeywordExtractionFormat
    from openai import AsyncOpenAI

    keywords = {"high_level_keywords": ["remote sensing"], "low_level_keywords": ["satellite"]}
    requests = []

    def respond(request):
        requests.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "smoke",
                    "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        content = json.dumps(keywords) if "response_format" in body else "satellite"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "created": 0,
                "model": "smoke",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def exercise_sdk():
        clients = []

        def create_client(**kwargs):
            client = AsyncOpenAI(
                **kwargs,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            )
            clients.append(client)
            return client

        monkeypatch.setattr(backend, "AsyncOpenAI", create_client)
        options = {"base_url": "https://example.invalid/v1", "api_key": "test-only"}
        try:
            assert await backend.openai_complete_if_cache("smoke", "test", **options) == "satellite"
            extracted = await backend.openai_complete_if_cache(
                "smoke", "test", response_format=GPTKeywordExtractionFormat, **options
            )
            assert json.loads(extracted) == keywords
            vectors = await backend.openai_embed(["test"], model="smoke", **options)
            np.testing.assert_array_equal(vectors, [[1.0, 0.0, 0.0]])
            assert requests == ["/v1/chat/completions", "/v1/chat/completions", "/v1/embeddings"]
        finally:
            for client in clients:
                await client.close()

    asyncio.run(exercise_sdk())

"""Exercise the upgraded runtime without API keys or downloaded models."""

from pathlib import Path

import faiss
import numpy as np
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

from benchmarks.planning.score import score_single_tool_prediction
from rs_agent.controller.agent import RSAgent
from rs_agent.solution_space import builder, retriever
from rs_agent.toolkit.registry import TASK_TO_TOOL, get_stub_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_structured_agent_preserves_tool_steps():
    llm = FakeListChatModel(
        responses=[
            '["Scene_Classification"]',
            '```json\n{"action": "scene", "action_input": "sample.png"}\n```',
            '```json\n{"action": "Final Answer", "action_input": "An airport."}\n```',
        ]
    )
    agent = RSAgent(llm, get_stub_tools(), mode="task_inference_only")

    result = agent.run("What scene is shown?", "sample.png")

    assert result["predicted_task_type"] == '["Scene_Classification"]'
    assert result["output"] == "An airport."
    assert len(result["intermediate_steps"]) == 1
    assert result["intermediate_steps"][0][1] == "The scene of this image is airport."
    assert score_single_tool_prediction(
        "Scene_Classification", result["intermediate_steps"], TASK_TO_TOOL
    )


def test_task_inference_accepts_content_blocks():
    llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(content=[{"type": "text", "text": ' ["Scene_Classification"] '}])
        ]
    )
    agent = RSAgent(llm, get_stub_tools(), mode="task_inference_only")

    assert agent.infer_task_type("What scene is shown?") == '["Scene_Classification"]'


def test_solution_index_build_save_load_and_retrieve(tmp_path, monkeypatch):
    embeddings = DeterministicFakeEmbedding(size=16)
    monkeypatch.setattr(builder, "HuggingFaceEmbeddings", lambda **_kwargs: embeddings)
    monkeypatch.setattr(retriever, "HuggingFaceEmbeddings", lambda **_kwargs: embeddings)
    denoising = "Denoising: " + "Remove image noise. " * 10
    captioning = "Captioning: " + "Describe the image. " * 10
    source = tmp_path / "solutions.txt"
    source.write_text(f"{denoising.strip()}\n\n{captioning.strip()}", encoding="utf-8")
    output = tmp_path / "index"

    assert builder.build_solution_index(source, output) == 2
    assert (output / "index.faiss").is_file()
    assert (output / "index.pkl").is_file()

    solutions = retriever.SolutionRetriever(output, top_k=2)
    assert solutions.retrieve('["Denoising"]') == denoising.strip()
    assert solutions.retrieve_by_query(captioning.strip()) == captioning.strip()


@pytest.mark.parametrize("index_name", ["solution_db", "solution_db_rschatgpt"])
def test_bundled_faiss_index_loads(index_name):
    index_dir = PROJECT_ROOT / "data" / "indices" / index_name
    index = faiss.read_index(str(index_dir / "index.faiss"))
    store = FAISS.load_local(
        str(index_dir),
        DeterministicFakeEmbedding(size=index.d),
        allow_dangerous_deserialization=True,
    )

    assert store.index.ntotal > 0
    assert store.similarity_search("Scene_Classification", k=1)[0].page_content


def test_huggingface_embeddings_infers_with_local_tiny_model(tmp_path, monkeypatch):
    """Run real Torch/Transformers inference with a tiny, untrained local BERT."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "small", "airport"]
    backend = Tokenizer(WordPiece(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="[UNK]"))
    backend.normalizer = BertNormalizer()
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=32,
    )
    bert_dir = tmp_path / "tiny-bert"
    tokenizer.save_pretrained(bert_dir)
    model = BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(bert_dir)
    # Loading a plain BERT adds mean pooling, as for legacy embedding checkpoints.
    sentence_model = SentenceTransformer(str(bert_dir), device="cpu", local_files_only=True)
    sentence_dir = tmp_path / "tiny-sentence-transformer"
    sentence_model.save_pretrained(str(sentence_dir), create_model_card=False)
    embeddings = HuggingFaceEmbeddings(
        model_name=str(sentence_dir),
        model_kwargs={"device": "cpu", "local_files_only": True},
    )

    documents = np.asarray(embeddings.embed_documents(["a small airport", "airport"]))
    query = np.asarray(embeddings.embed_query("a small airport"))

    assert documents.shape == (2, 16)
    assert query.shape == (16,)
    assert np.isfinite(documents).all()
    assert np.linalg.norm(query) > 0
    np.testing.assert_allclose(documents[0], query, rtol=1e-5, atol=1e-6)

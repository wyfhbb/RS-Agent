# DualRAG: Weighted Dual-Path Retrieval-Augmented Generation

This directory contains a modified fork of [LightRAG](https://github.com/HKUDS/LightRAG) implementing the **DualRAG** method proposed in RS-Agent.

## Key Modifications

Compared to the original LightRAG baseline (`LightRAG_old` in the research workspace), DualRAG modifies:

| File | Change |
|------|--------|
| `lightrag/prompt.py` | Keyword extraction now assigns per-keyword **importance** scores (0–10) |
| `lightrag/operate.py` | **Dual-path retrieval**: global concatenated query + per-keyword weighted allocation |

## Installation

```bash
# From the RS-Agent repository root
uv sync --locked --extra dualrag
uv run --locked --extra dualrag pytest tests/test_dualrag_runtime.py
cd dualrag
```

Commands below run from `dualrag/` and explicitly select the parent project.
This installs the local modified `lightrag-hku` fork, using the root `uv.lock`.
Default JSON, NanoVectorDB, and NetworkX storage dependencies are declared up
front; default storage imports do not install packages at runtime.
The optional node2vec method needs `graspologic`, which is outside this environment
because its current release requires NumPy 1.x. The default RAG path does not use it.

## Evaluation Pipeline

```bash
# 1. Prepare unique contexts from corpus
uv run --project .. --locked --extra dualrag reproduce/Step_0.py

# 2. Index documents into knowledge graph
uv run --project .. --locked --extra dualrag reproduce/Step_1.py

# 3. Generate evaluation questions (requires OPENAI_API_KEY)
uv run --project .. --env-file ../.env --locked --extra dualrag reproduce/Step_2.py

# 4. Run queries (local / global / hybrid modes)
uv run --project .. --locked --extra dualrag reproduce/Step_3.py
uv run --project .. --locked --extra dualrag reproduce/Step_3_local.py
uv run --project .. --locked --extra dualrag reproduce/Step_3_global.py

# 5. Compare the local and hybrid outputs
uv run --project .. --env-file ../.env --locked --extra dualrag datasets/eval.py
```

Step_2 generates `mix_all_questions.txt`; the query and evaluation scripts currently
read the bundled `mix_questions.txt`. Select the intended questions before running
a newly generated evaluation set.

## Environment Variables

Copy the root `.env.example` to `.env` and fill in the API settings before Step_2
or evaluation. These commands use `--env-file ../.env` because those scripts do
not load dotenv themselves. Alternatively, export the variables and omit
`--env-file ../.env`:

```bash
export OPENAI_API_KEY=your-key
export OPENAI_API_BASE=https://api.openai.com/v1
```

## Note on Baseline

The original LightRAG baseline used for ablation in the paper is kept separately in the research workspace (`LightRAG_old/`). It is not included here to avoid duplication. For pairwise comparison, install both forks in separate virtual environments.

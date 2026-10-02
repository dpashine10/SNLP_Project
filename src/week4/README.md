# Week 4: Improved Researcher Recommendation

Week 4 recommends researchers by searching **individual papers** in a local
ChromaDB index, then ranking each paper's authors from the papers that match.
It is compared against the Week 3 baseline, which represents each researcher
as a single averaged embedding.

Both systems use the same embedding model (`all-MiniLM-L6-v2`). Any
difference in results therefore comes from the architecture: paper-level
retrieval, ChromaDB and configurable ranking strategies.

> **Status: Phase 1 (architecture only).** Every module has its final
> structure, signatures and docstrings, but all business logic is a `TODO`
> that raises `NotImplementedError`. Nothing here runs end to end yet.

## Design rules

- **Isolation.** All Week 4 code lives in `src/week4/` and all generated
  output in `Data/week4/`. No Week 2 or Week 3 file is modified.
- **Two integration points.** Only `data_adapter.py` knows Week 2's data and
  only `baseline_adapter.py` knows Week 3's code. Every other module exchanges
  the dataclasses in `schemas.py`.
- **One place for settings.** Paths, model, collection name, `top_k` values
  and ranking strategy all come from `config.py`.
- **Higher score = better match**, everywhere. Only `vector_store.py` knows
  ChromaDB returns distances; it converts them.
- **Deterministic, never silent.** No randomness; ties are broken by ID.
  Each value is validated by the component that owns it, and errors propagate
  unchanged.

## Modules

| File | Responsibility |
|---|---|
| `config.py` | Frozen `Week4Config` dataclass and the shared `CONFIG` instance. |
| `schemas.py` | Shared dataclasses: `Author`, `Paper`, `PaperHit`, `ResearcherScore`. |
| `interfaces.py` | Protocols `PaperSource` and `ResearcherRecommender`. |
| `data_adapter.py` | **Week 2 integration.** Week 2 output → `Paper` objects (`Week2PaperSource`). |
| `baseline_adapter.py` | **Week 3 integration.** Week 3 → `ResearcherRecommender` (`Week3Baseline`). |
| `embedder.py` | Text → normalized vectors. Wraps Sentence Transformers; device `auto` tries CUDA, then Apple MPS, then CPU. |
| `vector_store.py` | The only module that knows ChromaDB: persistent storage, upserts, search, model check. |
| `retriever.py` | Query → `PaperHit`s via the embedder and vector store. Orchestration only. |
| `ranking_strategy.py` | Scores one researcher from their hits: `best_paper`, `average`, `top_k_average`, `weighted` (placeholder). |
| `aggregator.py` | Groups hits by author, applies the strategy, sorts and truncates. |
| `week4_pipeline.py` | `Week4Pipeline` (retriever → aggregator) and `build_week4_pipeline(config)`, the composition root. |
| `build_chromadb.py` | Entry point: streams papers into the index in batches. |
| `evaluation.py` | Entry point: compares recommenders on labelled queries and writes reports. |

## Dependency graph

Each module depends only on the modules listed after it. Bracketed
dependencies are imported lazily on first use, so importing any Week 4 module
never loads PyTorch, ChromaDB or teammate code.

```text
config, schemas      (no Week 4 dependencies)
interfaces        -> schemas
ranking_strategy  -> config, schemas
aggregator        -> ranking_strategy, schemas
embedder          -> config                    [sentence-transformers]
vector_store      -> config, schemas           [chromadb]
retriever         -> embedder, vector_store, schemas
week4_pipeline    -> retriever, aggregator, config   (builds all of the above)
data_adapter      -> schemas                   [Week 2 files]
baseline_adapter  -> schemas                   [Week 3 code]
build_chromadb    -> data_adapter, embedder, vector_store, interfaces, config
evaluation        -> week4_pipeline, baseline_adapter, interfaces, config
```

## Data flow

```text
Indexing (build_chromadb.py)
  Week 2 output -> data_adapter -> Papers, in batches of config.batch_size
    -> embedder.embed_documents -> vector_store.add_papers -> Data/week4/chroma/

Recommendation (Week4Pipeline.recommend)
  query -> retriever: embed_query -> vector_store.query -> top paper_top_k PaperHits
        -> aggregator: group by author -> strategy.score -> sort -> top_k ResearcherScores

Evaluation (evaluation.py)
  labelled queries -> each recommender (Week 3 modes, Week 4 strategies)
    -> ranked researcher IDs -> metrics -> Data/week4/results/
```

Each Week 4 `ResearcherScore` keeps all of that researcher's paper hits as
`evidence`. Its `matched_papers` field counts the retrieved papers by that
researcher before scoring.

## How Week 2 integrates

`Week2PaperSource` reads the files set in `config.papers_path` and
`config.authorship_path` and yields `Paper` objects lazily. It decides which
fields form the embedded text. Records with no usable text are skipped, and
for duplicate IDs only the first is kept. Every skipped record is logged.
When Week 2 changes, only this file and those paths change.

## How Week 3 integrates

`Week3Baseline(mode=...)` wraps the Week 3 recommender in one of its modes:
`semantic` (the default, primary comparison), `tfidf` or `hybrid`. It
validates inputs and re-sorts and truncates Week 3's results, so Week 3
satisfies the same contract as Week 4. Week 3 is imported only on first use.

Week 3 results carry no paper evidence (`evidence=()`, `matched_papers=0`).
Their scores are never compared numerically with Week 4's.

## How ChromaDB fits in

- **Storage.** A persistent local store in `Data/week4/chroma/`, created on
  first use. The collection is named `iitb_research_papers`. No cloud service
  or API keys are involved.
- **Records.** One record per paper:
  - ID: `paper_id`
  - document: `Paper.text`
  - metadata: title, abstract and JSON-encoded authors under reserved `__`
    keys, plus Week 2's scalar metadata. Week 2 keys may not start with `__`.
- **Model check.** The collection records `__embedding_model` and
  `__distance_metric`; `__created_at` and `__schema_version` are reserved.
  Adding papers or searching with a different model is refused, with a
  message to rebuild the index.
- **Re-runs.** Writes are upserts, so re-running indexing never duplicates
  papers. It also never deletes papers that have disappeared from Week 2, and
  indexing warns when that happens.

## How evaluation works

- **Systems.** One Week 3 baseline per mode, then one Week 4 pipeline per
  ranking strategy, always in that order. Every system is asked for its top
  10 researchers (`EVALUATION_DEPTH`) on the same queries.
- **Metrics.** Precision@5, Precision@10, Recall@10, MRR, nDCG@10 and Hit@10.
  They are computed from ranked researcher IDs only, never from raw scores.
- **Outputs.** Written to `Data/week4/results/`:
  - `per_query_results.csv`: one row per system and query.
  - `summary.csv`: the mean of each metric per system, which doubles as the
    strategy comparison.
- **Exclusions and failures.** Queries with no relevant researchers are
  excluded and counted. If any system fails on a query, the run stops.
- **Open issue: pooling bias.** If the labels only cover one system's
  candidates, researchers that only Week 4 finds count as misses. This will
  be resolved in Phase 2.

## Status and next steps

**Phase 2: integration** (after pulling Week 2 and Week 3):

| Where | What to do |
|---|---|
| `config.py` | Set `papers_path`, `authorship_path` and `evaluation_labels_path`. |
| `data_adapter.py` | Map Week 2 columns to `Paper` fields and choose the embedded-text fields. Decide whether papers without authors are skipped or indexed. |
| `baseline_adapter.py` | Import the final Week 3 entry point and map its results. Make Week 2, Week 3 and label author IDs share one canonical format. |
| `evaluation.py` | Read labels through an adapter, moving `LabeledQuery` to `schemas.py` if needed. Resolve pooling bias. |
| `vector_store.py` | Confirm how the installed `chromadb` version accepts the distance metric. |
| `pyproject.toml`, `uv.lock` | Add `chromadb`, once. |
| root `.gitignore` | Add `Data/week4/chroma/`. |

**Phase 3: implementation.** Fill in every remaining `TODO(Phase 3)`:
strategies, aggregator, embedder, vector store, retriever, pipeline, indexing
and the evaluation runner. The `weighted` strategy needs its formula decided
first.

**Possible future work (not planned).**
- A shared `factories.py` for component construction.
- `--rebuild` and resumable indexing.
- Search filters (year, institution, venue, …).
- Hybrid retrieval and reranking.
- Other vector databases.
- Extra metrics and reports.
- Institution-level aggregation.

## Where to change what

| To… | Change |
|---|---|
| Try another embedding model | `config.embedding_model_name`, then rebuild the index |
| Add a ranking strategy | A class and registry entry in `ranking_strategy.py`, plus its name in `config.RankingStrategyName` |
| Retrieve more or fewer papers per query | `config.paper_top_k` |
| Adapt to Week 2 changes | `data_adapter.py` (and the paths in `config.py`) |
| Adapt to Week 3 changes | `baseline_adapter.py` |
| Replace the vector database | `vector_store.py` |
| Add a metric | A function plus an entry in `METRICS` in `evaluation.py` |

## Running (once implemented)

From the repository root, build the index first, then evaluate:

```bash
uv run python -m src.week4.build_chromadb
uv run python -m src.week4.evaluation
```

Both currently stop with `NotImplementedError`. Before they can run, they
need the Phase 2 config paths, `chromadb` installed and the Phase 3
implementations.

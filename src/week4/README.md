# Week 4: Improved Researcher Recommendation

Week 4 recommends researchers by searching **individual papers** in a local
ChromaDB index, then ranking each paper's authors from the papers that match.
It is compared against the Week 3 baseline, which represents each researcher
as a single averaged embedding.

Both systems use the same embedding model (`all-MiniLM-L6-v2`), so differences
in results come mainly from the architecture: paper-level retrieval, ChromaDB
and configurable ranking strategies. (Week 4 embeds Week 2's `paper_text`,
which is formatted slightly differently from the text Week 3 embedded.)

> **Status: working end to end.** Week 4 builds its ChromaDB index from Week
> 2's papers, recommends researchers with three ranking strategies, and
> compares them against the Week 3 baseline. Only the `weighted` strategy is
> still a placeholder (its formula is not decided), so it is not evaluated.

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
- **Deterministic, never silent.** No randomness; ties are broken by ID, and
  repeated runs on the same index give identical results. Rebuilding the index
  can shift researchers near the retrieval cut-off, because ChromaDB's HNSW
  search is approximate. Each value is validated by the component that owns
  it, and errors propagate unchanged.

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
week4_pipeline    -> retriever, aggregator, embedder, vector_store,
                     ranking_strategy, config, schemas
data_adapter      -> schemas                   [Week 2 files]
baseline_adapter  -> schemas                   [Week 3 code]
build_chromadb    -> data_adapter, embedder, vector_store, interfaces, config
evaluation        -> data_adapter, baseline_adapter, week4_pipeline, interfaces,
                     config, schemas
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

`Week2PaperSource` reads Week 2's `papers.csv` and `paper_author.csv` (the
files set in `config.papers_path` and `config.authorship_path`) and yields
`Paper` objects lazily. It embeds Week 2's own `paper_text` (title, abstract,
topics and keywords) rather than composing its own. Records with no usable
text and papers without authors are skipped, and for duplicate IDs only the
first is kept. Every skipped record is logged.

`load_labeled_queries` in the same module reads Week 2's labelled benchmark
(`evaluation_candidates_labeled.csv`). When Week 2 changes, only this file and
those paths change.

## How Week 3 integrates

`Week3Baseline(mode=...)` wraps the Week 3 recommender,
`ResearchDiscoveryPipeline` in `src/week2/pipeline.py` with the embeddings
from `src/week3/build_embeddings.py`, in one of its modes: `semantic` (the
default, primary comparison), `tfidf` or `hybrid`. It validates inputs and
re-sorts and truncates Week 3's results, so Week 3 satisfies the same
contract as Week 4. Week 3 is imported only on first use, and one loaded
instance is shared by every mode.

Week 3 results carry no paper evidence (`evidence=()`, `matched_papers=0`).
Their scores are never compared numerically with Week 4's.

## How ChromaDB fits in

- **Storage.** A persistent local store in `Data/week4/chroma/`, created on
  first use. The collection is named `iitb_research_papers` and uses cosine
  distance through its HNSW configuration. No cloud service or API keys are
  involved, and ChromaDB telemetry is turned off.
- **Records.** One record per paper:
  - ID: `paper_id`
  - document: `Paper.text`
  - metadata: title, abstract and JSON-encoded authors under reserved `__`
    keys, plus Week 2's scalar metadata. Week 2 keys may not start with `__`.
- **Model check.** The collection records `__embedding_model` and
  `__distance_metric`; `__created_at` and `__schema_version` are reserved.
  Adding papers or searching with a different model or metric is refused,
  with a message to rebuild the index.
- **Re-runs.** Writes are upserts, so re-running indexing never duplicates
  papers. It also never deletes papers that have disappeared from Week 2, and
  indexing warns when that happens.

## How evaluation works

- **Systems.** The Week 3 baseline in its `semantic`, `tfidf` and `hybrid`
  modes, then one Week 4 pipeline per strategy (`best_paper`, `average`,
  `top_k_average`), always in that order. Every system is asked for its top
  10 researchers (`EVALUATION_DEPTH`) on the same queries. Each Week 4
  pipeline has its own embedder, so the model loads once per strategy during
  setup; queries reuse the loaded model.
- **Metrics.** Precision@5, Precision@10, Recall@10, MRR, nDCG@10 and Hit@10.
  They are computed from ranked researcher IDs only, never from raw scores.
- **Outputs.** Written to `Data/week4/results/`:
  - `per_query_results.csv`: one row per system and query.
  - `summary.csv`: the mean of each metric per system, which doubles as the
    strategy comparison.
- **Exclusions and failures.** Queries with no relevant researchers are
  excluded and counted. If any system fails on a query, the run stops.
- **Known limitation: pooling bias.** Week 2's benchmark labels exactly the
  TF-IDF top 10 per query, all as relevant, so researchers that only other
  systems find count as misses and TF-IDF scores perfectly by construction.
  The evaluation logs this assumption on every run; the scores measure
  agreement with TF-IDF's candidates rather than true relevance.

## Status and next steps

**Still open:**

| Where | What to do |
|---|---|
| `evaluation.py` | Decide how to handle pooling bias in Week 2's benchmark. |
| root `.gitignore` | Add `Data/week4/chroma/` so the built index is not committed. |
| `ranking_strategy.py` | Define the `weighted` formula, or remove the strategy. |

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
| Try another embedding model | `config.embedding_model_name`, then delete `Data/week4/chroma/` and rebuild the index |
| Add a ranking strategy | A class and registry entry in `ranking_strategy.py`, its name in `config.RankingStrategyName`, and in `evaluation.WEEK4_STRATEGIES` to evaluate it |
| Retrieve more or fewer papers per query | `config.paper_top_k` |
| Adapt to Week 2 changes | `data_adapter.py` (and the paths in `config.py`) |
| Adapt to Week 3 changes | `baseline_adapter.py` |
| Replace the vector database | `vector_store.py` |
| Add a metric | A function plus an entry in `METRICS` in `evaluation.py` |

## Running

From the repository root, build the index first, then evaluate:

```bash
uv run python -m src.week4.build_chromadb
uv run python -m src.week4.evaluation
```

Indexing all Week 2 papers takes a few minutes on an Apple Silicon GPU.
Re-running it updates the index in place. The evaluation prints the summary
table and writes both CSV files to `Data/week4/results/`.

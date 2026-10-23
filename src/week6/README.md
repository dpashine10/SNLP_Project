# Week 6: Reusable NLP Asset Engineering

**Deliverable:** Working Reusable NLP Asset for Researcher Recommendation  
**Target:** 16 October  
**Architecture:** Paper-Level Semantic Retrieval (Weeks 4 & 5) + Calibrated Confidence Scoring + Production REST API

---

## 📌 Executive Summary

Week 6 packages the research discovery engine into a self-contained, enterprise-grade **Reusable NLP Software Asset**.

The asset converts the NLP pipeline into:
1. **Python SDK Module (`NLPAsset`):** Clean, object-oriented API that can be imported into any application.
2. **Structured Contracts:** Typed input (`RecommendRequest`) and output (`RecommendResponse`) schemas with strict boundary validation.
3. **Confidence Engine:** Bounded ranking-confidence scores ($[0.05, 0.98]$) categorized into `HIGH`, `MEDIUM`, and `LOW` tiers with transparent rationales. They are review-prioritization aids, not calibrated probabilities.
4. **Flask REST API Microservice:** Standardized HTTP API with endpoints for single recommendation, batch queries, health monitoring, and schema discovery.
5. **Interactive CLI:** Command-line tool supporting human-readable ASCII tables and piping JSON.

---

## 🏗️ Architecture & Data Flow

```text
               ┌────────────────────────────────────────────────────────┐
               │                  Incoming Request                      │
               │   (Python SDK / Flask REST API / Command Line CLI)     │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │    Input Schema & Validation         │
                      │  RecommendRequest (query, top_k, s)  │
                      └───────────────────┬──────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │       Query Vector Embedding         │
                      │  Sentence-BERT (all-MiniLM-L6-v2)    │
                      └───────────────────┬──────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │       Paper-Level Vector Store       │
                      │  (ChromaDB or Fast In-Memory Store)  │
                      │    Retrieves Top-K PaperHits         │
                      └───────────────────┬──────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │    Author Aggregation & Strategy     │
                      │   top_k_average_author_discount      │
                      │    (Week 5 Co-Author Fix Applied)    │
                      └───────────────────┬──────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │     Confidence Calibrator Engine     │
                      │  Logistic Scaling + Margin + Depth   │
                      │       HIGH / MEDIUM / LOW Bands      │
                      └───────────────────┬──────────────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────────────┐
                      │       Structured JSON Response       │
                      │ RecommendResponse (Rankings + Evid)  │
                      └──────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Python SDK

```python
from src.week6 import NLPAsset, RecommendRequest

# Initialize asset (automatically uses optimal backend)
asset = NLPAsset()

# Execute a query
response = asset.recommend(
    RecommendRequest(
        query="Transformer architectures and deep neural networks",
        top_k=5,
        strategy="top_k_average_author_discount",
        include_evidence=True
    )
)

print(f"Latency: {response.execution_time_ms:.2f} ms")
for rec in response.results:
    print(f"[{rec.rank}] {rec.name} | Conf: {rec.confidence.level} ({rec.confidence.score:.2f})")
    for paper in rec.evidence[:1]:
        print(f"      Top Paper ({paper.year}): {paper.title} [sim={paper.similarity:.2f}]")
```

### 2. Launch the Flask REST API

```bash
# Start server on default port 8080 (or PORT=5000)
uv run python -m src.week6.api
```

Send a recommendation request via `curl`:

```bash
curl -X POST http://127.0.0.1:8080/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "natural language processing and deep learning",
    "top_k": 3,
    "strategy": "top_k_average_author_discount"
  }'
```

Sample JSON Response:

```json
{
  "query": "natural language processing and deep learning",
  "execution_time_ms": 34.12,
  "total_results": 3,
  "model_info": {
    "name": "all-MiniLM-L6-v2",
    "strategy": "top_k_average_author_discount",
    "backend": "fast_npz_paper_level",
    "embedding_dim": 384,
    "version": "1.0.0"
  },
  "results": [
    {
      "rank": 1,
      "author_id": "https://openalex.org/A5034638760",
      "name": "Srinivas V.",
      "score": 0.5825,
      "confidence": {
        "score": 0.9712,
        "level": "HIGH",
        "raw_score": 0.5825,
        "margin": 0.0123,
        "explanation": "High confidence: strong contextual and semantic match; based on 1 relevant publication."
      },
      "matched_papers": 1,
      "evidence": [
        {
          "paper_id": "https://openalex.org/W4408104760",
          "title": "NATURAL LANGUAGE PROCESSING: RECENT DEVELOPMENTS, TRENDS, AND DIFFICULTIES",
          "year": "2026",
          "similarity": 0.5825,
          "authors_count": 1
        }
      ]
    }
  ]
}
```

### 3. Command-Line Interface (CLI)

```bash
# Formatted ASCII table search
uv run python -m src.week6.cli --query "graph neural networks" --top-k 5

# Machine-readable JSON output
uv run python -m src.week6.cli --query "biomedical text summarization" --json

# Inspect asset health and metadata
uv run python -m src.week6.cli --health
uv run python -m src.week6.cli --info
```

---

## 🎯 Calibrated Confidence Scores

Raw cosine similarity values from Sentence-BERT models generally occupy a narrow band ($\approx 0.30 - 0.70$) and do not represent true probabilities. The Week 6 asset features an **empirical calibration engine**:

1. **Temperature-Scaled Logistic Transformation:**
   $$\text{Base Confidence} = \frac{1}{1 + e^{-(s - \mu) / \tau}}$$
   where $\mu$ is the empirical strategy midpoint and $\tau$ is the scaling temperature.
2. **Score Margin Adjustment:**
   Candidates with clear separation from the next rank receive a margin bonus.
3. **Evidence Depth Bonus:**
   Candidates supported by multiple distinct high-scoring papers receive an evidence bonus.
4. **Confidence Categorization:**
   - **`HIGH` ($\ge 0.75$):** Core topical authority with high-confidence publication evidence.
   - **`MEDIUM` ($0.50 - 0.74$):** Sub-discipline overlap or secondary relevance.
   - **`LOW` ($< 0.50$):** Peripheral overlap or broad keyword association.

---

## 🧪 Verification & Test Suite

The asset includes a comprehensive unit and integration test suite covering input validation, boundary constraints, confidence scoring, REST API contracts, and end-to-end execution:

```bash
uv run python -m unittest discover -s src/week6/tests -t .
```

The current source suite contains 31 test cases covering schemas, confidence,
API routes, live asset behavior, and the Group 6 HTTP contract. Run the command
above in the project's Python 3.12 environment to verify the local result.

---

## 📂 Module Structure

| File | Purpose |
|---|---|
| [`src/week6/__init__.py`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/__init__.py) | Package public API exports (`NLPAsset`, schemas, calibrator). |
| [`src/week6/asset.py`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/asset.py) | Core reusable component with dual ChromaDB / in-memory vector store support. |
| [`src/week6/schemas.py`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/schemas.py) | Input/output contracts, validation rules, and JSON serialization. |
| [`src/week6/confidence.py`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/confidence.py) | Mathematical confidence calibration and qualitative tier assignments. |
| [`src/week6/api.py`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/api.py) | Flask REST API microservice with `/health`, `/info`, `/recommend`, `/batch`, and `/schema`. |
| [`src/week6/cli.py`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/cli.py) | Standalone CLI utility for scripting and terminal interaction. |
| [`src/week6/tests/`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/tests/) | 31 automated tests covering schemas, confidence calibration, API routes, and live asset integration. |

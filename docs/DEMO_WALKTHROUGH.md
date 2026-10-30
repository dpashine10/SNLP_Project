# Faculty Evaluation & Demo Walkthrough Guide

**Project:** AI University Research Discovery & Reusable NLP Asset  
**Live Production URL:** [https://research.dpashine.tech](https://research.dpashine.tech)  
**Local Development:** `http://127.0.0.1:8000` (Main App) | `http://127.0.0.1:8080` (Standalone API)  
**Repository:** [dpashine10/SNLP_Project](https://github.com/dpashine10/SNLP_Project.git)

---

## Executive Summary & Architecture

This implementation presents a production-grade, 3-stage NLP system designed for academic researcher discovery and university advisor triage:

```
[User Query / Student Topic]
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Dense Bi-Encoder Semantic Retrieval           │
│ Model: sentence-transformers/all-MiniLM-L6-v2 (R^384)   │
│ Scoring: Top-3 paper average discounted by 1/√n authors│
└────────────────────────────────────────────────────────┘
             │ Top retrieved candidates & papers
             ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Laya System-1 Calibrated Decision Model       │
│ Model: convaiinnovations/laya (RLCD typed-decisions)   │
│ Primitives: choice (methodology), score (0-3 rubric),   │
│             noul (substantive publication filter)      │
└────────────────────────────────────────────────────────┘
             │ Classified publications & verified evidence
             ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 3: System-2 Dataset-Grounded RAG Reasoning       │
│ Models: Groq Llama-3.3-70B ↔ Google Gemini 2.5 Flash   │
│ Fallback: Deterministic Extractive Synthesis (offline) │
│ Output: Inline [Paper #k · DOI] citation synthesis     │
└────────────────────────────────────────────────────────┘
```

---

## Direct Mapping to Faculty Evaluation Criteria

### Criterion 1: Effectiveness of the Demo

| Demonstration Action | What to Show | URL / Command |
|---|---|---|
| **Ambiguous Query Disambiguation** | Search `"transformer"`. Show dynamic sub-field chips (`NLP & Language Models` vs. `Electrical Power & Energy Systems`). Click a chip to filter results in one click without losing state. | `/?q=transformer` |
| **Analytical Query & Laya Classification** | Search `"How do survey papers evaluate neural machine translation?"`. Observe the **Laya Query Bar** (`Laya Domain: NLP & Language Models · Analytical Question`) and researcher cards displaying **`Laya · Survey / Review`** and **`Direct Match (3/3)`** badges. | `/?q=How+do+survey+papers+evaluate+neural+machine+translation%3F` |
| **Dataset AI Reasoning (Gemini & Groq)** | In the **Dataset AI Reasoning** card on Search, select provider (`Auto`, `Groq`, or `Gemini`) and click **"Generate Grounded Answer"**. Highlight the sub-second latency badge and inline citation pills (`[Paper #1]`, `[Paper #2]`) linking to verified OpenAlex DOIs. | `/?q=transformer&reason=1` |
| **Collaboration & Outreach Workflow** | Use collaboration filters (Institution autocomplete e.g. `IIT Bombay`, `Indexed papers ≥ 2`, `2024+`). Click **"+ Shortlist"** on top candidates, open the **Shortlist Drawer**, export Markdown/CSV briefs, and click **"Copy Outreach Draft"**. | `/?q=nlp&institution=IIT+Bombay&min_papers=2` |
| **University Bridge Simulator** | On `/integration`, trigger the **At-Risk `CS 521`** scenario. Show the live HTTP call to `/api/v1/recommend`, inspect the zero-PII request contract, and toggle the **"Inspect Raw JSON"** inspector. Switch to **Routine Email** to show the early-warning guardrail blocking outbound HTTP calls. | `/integration` |

---

### Criterion 2: Completeness of Implementation in the Form of a Standalone Asset

The NLP functionality is packaged as an independent, fully decoupled Python asset located in [`src/week6/`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/):

1. **Self-Contained Python Library:**
   ```python
   from src.week6.asset import NLPAsset, AssetConfig
   from src.week6.schemas import RecommendRequest

   asset = NLPAsset(AssetConfig())
   response = asset.recommend(RecommendRequest(
       query="graph neural networks for drug discovery",
       top_k=3,
       classify_papers=True,
       include_reasoning=True,
   ))

   print(response.results[0].name)
   print(response.results[0].confidence.score)
   print(response.reasoning.answer)
   ```

2. **Standalone Command-Line Interface (`src/week6/cli.py`):**
   ```bash
   # Basic recommendation
   python -m src.week6.cli -q "neural machine translation" -k 3

   # With Laya paper classification
   python -m src.week6.cli -q "transformer models" --classify

   # With Groq / Gemini grounded AI reasoning
   python -m src.week6.cli -q "multimodal retrieval" --reason --provider auto

   # Raw structured JSON export
   python -m src.week6.cli -q "reinforcement learning" --json
   ```

3. **Immutable Typed Data Contracts ([`src/week6/schemas.py`](file:///c:/Users/nitaa/Downloads/SNLP_Project/src/week6/schemas.py)):**
   - Built with `@dataclass(frozen=True, slots=True)` with strict validation (`ValidationError`), bounds checking, and bidirectional `to_dict()` / `from_dict()`.

---

### Criterion 3: Availability Through API Calls for Integration with Other Assets

The asset exposes a versioned REST API documented via OpenAPI-style schemas at `/api/v1/schema` and discovery metadata at `/api/v1/info`:

| Endpoint | Method | Purpose | Input Schema |
|---|---|---|---|
| `/api/v1/health` | `GET` | Health, readiness, and backend diagnostics | None |
| `/api/v1/info` | `GET` | Metadata, supported strategies, Laya status, limits | None |
| `/api/v1/schema` | `GET` | Self-documenting JSON schemas for all request/response objects | None |
| `/api/v1/recommend` | `POST` | Structured researcher recommendation query | `RecommendRequest` |
| `/api/v1/batch` | `POST` | Multi-query batch execution (up to 50 queries) | `BatchRequest` |
| `/api/v1/classify` | `POST` | Standalone Laya paper methodology and query classification | `ClassifyRequest` |
| `/api/v1/reason` | `POST` | Grounded dataset reasoning using Gemini / Groq | `ReasoningRequest` |

#### University Integration Contract (Group 6 → Group 7 Bridge)
- **Data Minimization:** Zero student PII (no names, student IDs, email bodies, or triage sentiment) crosses the network. Only sanitized course codes mapped to research topics (`natural language processing`) are transmitted.
- **Advisory Only:** All returned recommendations carry an explicit `advisory_only: true` disclaimer; no automated advisor assignment or grade-altering actions occur without human review.

---

### Criterion 4: Identification of Areas of Improvement

*(Explicitly avoiding the common cliché that "access to real data was limited")*

Our team identified **5 concrete architectural and algorithmic improvements** to elevate the state of the art in academic researcher recommendation:

1. **Domain-Adapted Citation-Graph Bi-Encoders (`allenai/specter2`)**:
   - *Limitation:* Generic web-text models (`all-MiniLM-L6-v2`) embed lexical and syntactic surface forms rather than academic relationships.
   - *Improvement:* Transition to transformers trained on academic citation networks (e.g., `SPECTER2`), where positive training pairs are papers that cite or co-cite one another. This structures the embedding manifold according to true scholarly lineage.

2. **Exponential Temporal Expertise Decay**:
   - *Limitation:* A researcher who published a seminal paper in 2011 currently receives identical similarity credit to an active researcher who published three papers in 2025.
   - *Improvement:* Formulate a half-life recency weighting $\lambda(t) = \exp(-\gamma \cdot [Y_{\text{current}} - Y_{\text{pub}}])$ with $\gamma = \frac{\ln(2)}{5}$ (5-year academic half-life). This prioritizes contemporary subject-matter specialists while retaining seminal works with appropriate discount.

3. **Authorship-Position Credit Assignment Priors**:
   - *Limitation:* The current $1/\sqrt{N_{\text{authors}}}$ discount assumes equal contribution across all co-authors on a paper.
   - *Improvement:* Adopt domain-standard fractional credit allocation based on authorship conventions: first author ($w_1 \approx 0.50$), corresponding author ($w_c \approx 0.35$), and distributed middle authors ($w_m \approx 0.15 / [N-2]$). This prevents postdocs and lab directors from being under-credited on high-impact publications.

4. **Distribution-Free Split-Conformal Prediction Sets**:
   - *Limitation:* Confidence bands (`HIGH`, `MEDIUM`, `LOW`) are based on heuristic score margins and score thresholds.
   - *Improvement:* Calibrate the system using finite-sample split-conformal prediction. Given a target significance level $\alpha$ (e.g., $\alpha = 0.10$), the model outputs a dynamic prediction set $\mathcal{C}_{1-\alpha}(q)$ with guaranteed coverage $P(\text{Relevant Researcher} \in \mathcal{C}_{1-\alpha}(q)) \ge 1 - \alpha$.

5. **Automated Cross-Encoder NLI Faithfulness Verification**:
   - *Limitation:* Generative System-2 RAG models (even with low temperature $T=0.2$) risk subtle semantic drift or unfaithful paraphrasing.
   - *Improvement:* Introduce a secondary Natural Language Inference (NLI) model (`deberta-v3-small-nli`) as an automated verification guardrail. Every generated sentence in the reasoning output is treated as a hypothesis checked against the cited paper abstract premise. Any claim failing entailment ($P(\text{entailment}) < 0.85$) is automatically pruned prior to presentation.

---

### Criterion 5: Extent of Understanding of Demonstrated NLP Concepts

The project demonstrates deep conceptual mastery across multiple core NLP domains:

1. **Dense Vector Representations & Retrieval:**
   - Query and document encoding into a continuous dense metric space $\mathbf{x} \in \mathbb{R}^{384}$.
   - Exact and approximate nearest neighbor retrieval via cosine similarity $\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$.
   - Asymmetric bi-encoder indexing scaling as $O(N)$ dot products vs. $O(N \cdot L)$ cross-encoders.

2. **Non-Autoregressive Decision Models (RLCD / Laya):**
   - Implementation of non-autoregressive decision classification trained via Reinforcement Learning from Calibrated Decisions (RLCD).
   - Leveraging native decision primitives:
     - `choice`: Multi-class methodology categorization without generation overhead.
     - `score`: Calibrated 0–3 relevance matching directly reflecting graded evaluation rubrics.
     - `noul`: Non-scholarly artifact detection to screen retractions, editorials, and errata.

3. **Retrieval-Augmented Generation (RAG) & Grounding:**
   - Multi-provider LLM orchestration with automatic failover (Groq OpenAI-compatible API ↔ Google Gemini REST API).
   - Strict hallucination control through prompt engineering, temperature calibration ($T=0.2$), and deterministic extractive synthesis fallbacks.
   - Explicit citation binding requiring every assertion to cite an indexed paper identifier.

4. **Rigorous IR Evaluation Methodology:**
   - Multi-system pooling evaluation following TREC standards (214 judged pairs across 6 systems on 8 benchmark queries).
   - Diagnosis and audit of single-system pooling bias (demonstrating why single-system ground truth artificially rewarded the baseline and penalized superior dense models).
   - Evaluation metrics: Graded nDCG@10, Mean Reciprocal Rank (MRR), Precision@K, and Jaccard Candidate Set Stability under query rephrasings.


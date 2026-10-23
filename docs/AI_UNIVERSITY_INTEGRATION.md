# AI University integration: Group 6 → Group 7

This document defines the cross-project contract used by the class demonstration.
`SNLP_Project` is Group 7's research-discovery service. The Group 6 S3T bridge
calls it over HTTP after triaging a student-support message.

## Architecture

```text
student-support message
        |
        v
Group 6 S3T triage
  category, urgency, courses, human-review flag
        |
        v
Group 6 bridge
  transparent early-warning signals
  course code/name -> topic catalog lookup
        |
        |  only the topic crosses this boundary
        v
Group 7 Week 6 API (this project)
  paper retrieval -> author aggregation -> confidence + evidence
        |
        v
staff review / University support workflow
```

The repositories remain separate processes and environments. Group 6 supports
Python 3.9+, while this project requires Python 3.12+. HTTP keeps their
dependencies isolated and makes the component reusable by other University
applications.

## Start the demo

### Terminal 1: Group 7 service

For the all-in-one Group 7 dashboard on port `8000`, run:

```bash
cd SNLP_Project
uv run python app.py
```

The main dashboard also exposes the Week 6 compatibility API at
`/api/v1/health` and `/api/v1/recommend`, so Group 6 can target port `8000`.
Alternatively, run the standalone API on port `8080`:

```bash
cd SNLP_Project
uv sync
uv run python -m src.week6.api
```

The standalone API listens on `http://127.0.0.1:8080` by default. To use another port:

```bash
PORT=8081 uv run python -m src.week6.api
```

Warm-up and health check:

```bash
curl http://127.0.0.1:8080/api/v1/health
```

The response status is `unloaded` before the first recommendation and `ready`
after the asset has loaded. Both are valid integration states; the first query
may take longer while the embedding model and vector store initialize.

Use the matching Group 7 URL in the Group 6 command: `8000` for `app.py`, or
`8080` for the standalone Week 6 API.

### Terminal 2: Group 6 integration UI

From the adjacent Group 6 repository:

```bash
cd group6_for_group7/S3T_Week6_Asset
python3 -m venv .venv
source .venv/bin/activate
pip install -e "[all]"
cd ../integration
PYTHONPATH=. python ui_server.py \
  --group7 http://127.0.0.1:8080 \
  --port 5000 \
  --timeout 180
```

Open `http://127.0.0.1:5000`. The integration UI shows Group 6 status, the
early-warning decision, the Group 7 response, evidence papers, raw JSON, and
failure/guardrail states.

The Group 7 dashboard also polls Group 6 at `GROUP6_URL` (default
`http://127.0.0.1:5000`) and displays whether the Group 6 integration UI is
reachable, unhealthy, incompatible, or unavailable.

## HTTP contract

### Health

```http
GET /api/v1/health
```

The bridge UI accepts `ready` and `unloaded` as healthy service states. A
failed load returns HTTP 503 with an explanatory status payload.

### Recommendation

```http
POST /api/v1/recommend
Content-Type: application/json
```

Minimum bridge payload:

```json
{
  "query": "natural language processing",
  "top_k": 5,
  "include_evidence": true
}
```

The response contains `results`. Each result includes:

```json
{
  "rank": 1,
  "name": "Researcher name",
  "author_id": "OpenAlex author id",
  "score": 0.58,
  "confidence": {
    "score": 0.81,
    "level": "HIGH"
  },
  "evidence": [
    {
      "title": "Supporting publication",
      "year": "2024",
      "similarity": 0.72
    }
  ]
}
```

Additional fields such as `matched_papers`, `topics`, `model_info`, and
execution timing are available to richer consumers. The bridge ignores fields
it does not need.

The default Week 6 ranking strategy is
`top_k_average_author_discount`: it averages a researcher's three strongest
paper matches and discounts papers with many co-authors. It is a publication
relevance ranking. It does **not** rank by most recent publication, most
published researcher, unique expertise, teaching assignment, or mentor
availability. The API returns publication years and supporting-paper counts so
staff can inspect recency and evidence depth themselves.

### Data minimization rule

Group 7 should receive only a course/research topic. Do **not** send:

- student email or message text;
- student name, number, or identifier;
- crisis details or triage probabilities;
- the complete S3T result.

This boundary is checked by the Group 6 bridge tests and is also documented by
`GET /api/v1/info`. The Group 7 API remains reusable for ordinary research
queries from other consumers; topic-only handling is the Group 6 bridge
contract, not a content classifier enforced for every caller.

## Demo sequence

1. Show both status indicators in the Group 6 UI.
2. Analyze the at-risk `CS 521` sample. The catalog resolves it to `natural language processing`.
3. Explain the three stages: S3T triage, transparent early-warning signals, and Group 7 publication matches.
4. Show the ranked researchers, evidence depth, confidence explanation, and up to two supporting papers per result.
5. Run the routine transcript query. It should return `no_risk` and make no Group 7 request.
6. Run the unknown-course example. It should return `needs_topic` and make no Group 7 request until a topic override is supplied.
7. Expand Raw JSON and explain that a staff member makes the final decision.

The in-project `/integration` page mirrors this runbook so the Group 7 UI can
be used as a presenter aid even before the heavier search pipeline has loaded.

## Verification

Run the API contract and schema tests from `SNLP_Project`:

```bash
uv run python -m unittest discover -s src/week6/tests -t .
```

Run the browser UI tests:

```bash
uv run python -m unittest discover -s tests -t .
```

Run the Group 6 bridge tests from its integration directory with the S3T
environment active:

```bash
PYTHONPATH=.:../S3T_Week6_Asset pytest -q tests
```

## Scope and terminology

The Group 7 API returns researchers whose publications match a topic. It does
not verify that a person teaches a course, belongs to the presenting
university, or is available to mentor a student. During the demo, use
“researcher matches” rather than “confirmed faculty mentors.”

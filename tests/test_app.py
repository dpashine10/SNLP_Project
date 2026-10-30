"""Tests for the web interface (``app.py``, ``services.py`` and the templates).

Run from the repository root::

    uv run python -m unittest discover -s tests -t .

A fake pipeline stands in for ``ResearchDiscoveryPipeline``, so these tests
need neither the dataset nor the embedding model.
"""

import math
import threading
import unittest
import zipfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from markupsafe import escape
from sklearn.feature_extraction.text import TfidfVectorizer

import services
from app import create_app


def _raw_result(rank: int, mode: str, **overrides: Any) -> dict[str, Any]:
    result = {
        "rank": rank,
        "author_id": f"https://openalex.org/A{rank}",
        "author_name": f"Researcher {rank}",
        "similarity_score": 0.5 - rank / 100,
        "n_papers": rank + 2,
        "relevant_paper": {"title": f"Paper {rank}", "year": "2024", "work_id": "W1", "paper_score": 0.4321},
        "related_topics": ["Topic A", "Topic B"],
        "mode": mode,
    }
    result.update(overrides)
    return result


class FakePipeline:
    """Records every search call and returns ``top_k`` canned results."""

    model_name = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self.profiles_df = pd.DataFrame({"author_id": range(52736)})
        self.papers_df = pd.DataFrame({"work_id": range(62637)})
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, top_k: int = 10, mode: str = "semantic", alpha: float = 0.7) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "top_k": top_k, "mode": mode, "alpha": alpha})
        if not query.strip():
            return []
        return [_raw_result(rank, mode) for rank in range(1, top_k + 1)]


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = FakePipeline()
        self.app = create_app(lambda: self.pipeline, background=False)
        self.client = self.app.test_client()

    def get(self, url: str, status: int = 200) -> str:
        response = self.client.get(url)
        self.assertEqual(response.status_code, status, url)
        return response.get_data(as_text=True)

    def test_search_page_without_query_does_not_search(self) -> None:
        html = self.get("/")
        self.assertIn("Search settings", html)
        self.assertIn('id="config-panel"', html)
        self.assertIn('id="q"', html)
        self.assertNotIn("Found ", html)
        self.assertEqual(self.pipeline.calls, [])
        for label, text in services.QUERY_PRESETS:
            self.assertIn(f'data-preset="{escape(text)}">{escape(label)}</button>', html)

    def test_search_passes_settings_and_renders_cards(self) -> None:
        html = self.get("/?q=quantum+computing&mode=hybrid&top_k=15&alpha=0.45")
        self.assertEqual(self.pipeline.calls, [{"query": "quantum computing", "top_k": 15, "mode": "hybrid", "alpha": 0.45}])
        self.assertIn("Found 15 researchers matching query using", html)
        self.assertIn("<strong>Hybrid (Semantic + TF-IDF)</strong>", html)
        self.assertEqual(html.count('class="card researcher-card"'), 15)
        self.assertIn("Similarity: 0.4900", html)
        self.assertIn("Paper Match Score: 0.4321", html)
        self.assertIn('href="https://openalex.org/A1"', html)
        self.assertIn('<li class="badge">Topic A</li>', html)
        self.assertIn(">quantum computing</textarea>", html)

    def test_alpha_is_ignored_outside_hybrid_mode(self) -> None:
        self.get("/?q=nlp&mode=tfidf&top_k=5&alpha=0.1")
        self.assertEqual(self.pipeline.calls[0]["alpha"], services.DEFAULT_ALPHA)

    def test_query_is_stripped_and_empty_query_warns(self) -> None:
        self.get("/?q=++nlp++")
        self.assertEqual(self.pipeline.calls[0]["query"], "nlp")
        html = self.get("/?q=+++")
        self.assertIn("Enter a research topic or choose a query preset first.", html)
        self.assertEqual(len(self.pipeline.calls), 1)

    def test_invalid_settings_are_rejected(self) -> None:
        for query in ("mode=bm25", "top_k=7", "top_k=30", "top_k=abc", "alpha=1.5", "alpha=x"):
            html = self.get(f"/?q=nlp&{query}", status=400)
            self.assertIn('class="alert alert-error"', html, query)
        self.assertEqual(self.pipeline.calls, [])

    def test_output_is_html_escaped(self) -> None:
        self.pipeline.search = lambda **kwargs: [  # type: ignore[method-assign]
            _raw_result(1, "semantic", author_name="<script>alert(1)</script>", author_id="javascript:alert(1)")
        ]
        html = self.get("/?q=x")
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn('href="javascript:', html)

    def test_missing_paper_fields_use_previous_defaults(self) -> None:
        self.pipeline.search = lambda **kwargs: [  # type: ignore[method-assign]
            _raw_result(1, "semantic", relevant_paper={"title": "N/A", "score": 0.0}, related_topics=[])
        ]
        html = self.get("/?q=x")
        self.assertIn("(N/A)", html)
        self.assertIn("Paper Match Score: 0.0000", html)
        self.assertIn("None listed", html)

    def test_default_search_uses_paper_discount_and_renders_confidence_badges(self) -> None:
        html = self.get("/?q=transformers")
        self.assertEqual(
            self.pipeline.calls,
            [{"query": "transformers", "top_k": 10, "mode": "paper_discount", "alpha": services.DEFAULT_ALPHA}],
        )
        self.assertIn("<strong>Paper-Level + Co-Author Discount (Week 5/6)</strong>", html)
        self.assertIn("badge-confidence-high", html)
        self.assertIn("class=\"confidence-note\"", html)
        self.assertIn("<dt>Matched Papers</dt>", html)

    def test_compare_runs_semantic_and_tfidf_top_5(self) -> None:
        html = self.get("/compare")
        self.assertIn(f'value="{services.COMPARISON_DEFAULT_QUERY}"', html)
        self.assertEqual(
            [(c["mode"], c["top_k"], c["query"]) for c in self.pipeline.calls],
            [
                ("paper_discount", 5, services.COMPARISON_DEFAULT_QUERY),
                ("semantic", 5, services.COMPARISON_DEFAULT_QUERY),
                ("tfidf", 5, services.COMPARISON_DEFAULT_QUERY),
            ],
        )
        self.assertEqual(html.count('class="compare-item"'), 15)

        self.pipeline.calls.clear()
        html = self.get("/compare?q=knowledge+graphs")
        self.assertEqual(
            [(c["mode"], c["top_k"], c["query"]) for c in self.pipeline.calls],
            [
                ("paper_discount", 5, "knowledge graphs"),
                ("semantic", 5, "knowledge graphs"),
                ("tfidf", 5, "knowledge graphs"),
            ],
        )
        self.assertIn("Paper-Level + Author Discount (Week 5/6)", html)
        self.assertIn("Sentence Transformers (Dense Semantic)", html)
        self.assertIn("TF-IDF Baseline (Sparse Keyword)", html)
        self.assertIn("5/5 distinct papers", html)
        self.assertIn('class="badge badge-delta"', html)
        self.assertEqual(html.count('class="compare-item"'), 15)

    def test_compare_computes_distinct_papers_and_rank_deltas(self) -> None:
        def custom_search(query: str, top_k: int = 5, mode: str = "paper_discount", alpha: float = 0.7) -> list[dict[str, Any]]:
            if mode == "tfidf":
                return [
                    _raw_result(1, mode, author_id="A1", relevant_paper={"title": "Paper 1", "year": "2024", "paper_score": 0.4}),
                    _raw_result(2, mode, author_id="A2", relevant_paper={"title": "Paper 2", "year": "2024", "paper_score": 0.3}),
                    _raw_result(3, mode, author_id="A3", relevant_paper={"title": "Paper 3", "year": "2024", "paper_score": 0.2}),
                ]
            if mode == "semantic":
                # Simulate co-author crowding: 3 authors sharing the exact same multi-author paper
                return [
                    _raw_result(1, mode, author_id="A3", relevant_paper={"title": "Shared Team Paper", "year": "2024", "paper_score": 0.8}),
                    _raw_result(2, mode, author_id="A4", relevant_paper={"title": "Shared Team Paper", "year": "2024", "paper_score": 0.8}),
                    _raw_result(3, mode, author_id="A1", relevant_paper={"title": "Shared Team Paper", "year": "2024", "paper_score": 0.8}),
                ]
            return [
                _raw_result(1, mode, author_id="A3", relevant_paper={"title": "Solo Paper A", "year": "2024", "paper_score": 0.7}),
                _raw_result(2, mode, author_id="A5", relevant_paper={"title": "Solo Paper B", "year": "2024", "paper_score": 0.6}),
                _raw_result(3, mode, author_id="A1", relevant_paper={"title": "Solo Paper C", "year": "2024", "paper_score": 0.5}),
            ]

        self.pipeline.search = custom_search  # type: ignore[method-assign]
        html = self.get("/compare?q=transformers")
        self.assertIn("3/3 distinct papers", html)
        self.assertIn("1/3 distinct papers", html)
        self.assertIn("↑2 vs TF-IDF", html)
        self.assertIn("↓2 vs TF-IDF", html)
        self.assertIn("New in Top 5", html)

    def test_compare_with_empty_query_shows_no_results(self) -> None:
        html = self.get("/compare?q=")
        self.assertEqual(html.count("No researchers returned for this query."), 3)

    def test_university_integration_page_is_available_while_pipeline_is_ready(self) -> None:
        html = self.get("/integration")
        self.assertIn("University integration", html)
        self.assertIn("/api/v1/recommend", html)
        self.assertIn("Topic only crosses the bridge", html)
        self.assertIn("Publication relevance, with inspectable evidence", html)
        self.assertIn("researcher publication matches", html)
        self.assertIn('href="/integration" aria-current="page"', html)

    def test_integration_page_renders_bridge_simulator_and_collapsible_runbook(self) -> None:
        html = self.get("/integration")
        self.assertIn('id="bridge-simulator"', html)
        self.assertIn("Interactive Group 6 → Group 7 Bridge Simulator", html)
        self.assertIn('data-scenario="at_risk"', html)
        self.assertIn('data-scenario="no_risk"', html)
        self.assertIn('data-scenario="needs_topic"', html)
        self.assertIn("data-sim-override-form", html)
        self.assertIn('id="sim-topic-input"', html)
        self.assertIn("data-sim-raw-drawer", html)
        self.assertIn('<details class="collapsible runbook-collapsible">', html)
        self.assertIn("Presenter CLI Runbook", html)
        self.assertIn('/static/js/integration.js', html)

    def test_search_settings_shortcut_preserves_current_search_context(self) -> None:
        html = self.get("/?q=quantum+computing&mode=hybrid&top_k=15&alpha=0.45")
        self.assertIn('href="#config-heading"', html)
        self.assertIn(">Search settings</span>", html)
        self.assertIn('data-context-target="config-heading"', html)
        self.assertIn("These controls apply to the next search submission", html)

    def test_non_search_pages_link_to_search_settings(self) -> None:
        html = self.get("/evaluation")
        self.assertIn('href="/#config-heading"', html)
        self.assertIn(">Search settings</span>", html)
        self.assertIn(">Dataset stats</span>", html)
        self.assertIn('data-context-target="stats-heading"', html)

    def test_group6_status_route_reports_configured_health(self) -> None:
        health = {
            "state": "ready",
            "status": "ready",
            "reachable": True,
            "healthy": True,
            "url": "http://127.0.0.1:5000",
        }
        with patch("services.group6_health", return_value=health):
            response = self.client.get("/api/integration-status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), health)

    def test_group6_health_probe_accepts_integration_ui_health_contract(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b'{"status":"ready","service":"group6-integration-ui"}'
        with patch("services.urllib.request.urlopen", return_value=response):
            health = services.group6_health("http://127.0.0.1:5000")
        self.assertTrue(health["healthy"])
        self.assertTrue(health["reachable"])
        self.assertEqual(health["service"], "group6-integration-ui")

    def test_group6_health_probe_rejects_another_ready_json_service(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b'{"status":"ready","service":"another-service"}'
        with patch("services.urllib.request.urlopen", return_value=response):
            health = services.group6_health("http://127.0.0.1:5000")
        self.assertEqual(health["state"], "incompatible")
        self.assertFalse(health["healthy"])

    def test_main_dashboard_exposes_week6_health_contract(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()["status"], ("unloaded", "ready"))

    def test_seo_and_social_metadata_rendered_without_duplicate_tags(self) -> None:
        search_html = self.get("/")
        self.assertIn('<meta name="description" content="Semantic and hybrid academic researcher discovery', search_html)
        self.assertIn('<meta property="og:title" content="Research Search · Research Discovery &amp; Collaboration Engine">', search_html)
        self.assertIn('<meta property="og:type" content="website">', search_html)
        self.assertIn('<meta name="twitter:card" content="summary">', search_html)
        self.assertIn('data-integration-page="false"', search_html)
        self.assertIn("Bridge: standalone", search_html)

        integration_html = self.get("/integration")
        self.assertIn('<meta property="og:title" content="University integration · Research Discovery &amp; Collaboration Engine">', integration_html)
        self.assertIn('data-integration-page="true"', integration_html)

    def test_pipeline_loader_on_ready_triggers_asset_warmup(self) -> None:
        warmed = threading.Event()
        received_model: list[Any] = []

        def _on_ready(pipeline: Any) -> None:
            received_model.append(getattr(pipeline, "model_name", None))
            warmed.set()

        loader = services.PipelineLoader(FakePipeline, on_ready=_on_ready)
        loader.start(background=True)
        self.assertTrue(warmed.wait(3))
        self.assertEqual(received_model, ["all-MiniLM-L6-v2"])

    def test_dataset_statistics_in_sidebar(self) -> None:
        html = self.get("/evaluation")
        for text in ("Indexed OpenAlex Researchers", "52,736", "Indexed Academic Papers", "62,637", "all-MiniLM-L6-v2"):
            self.assertIn(text, html)

    def test_evaluation_tables_match_csv_files(self) -> None:
        html = self.get("/evaluation")
        self.assertIn("214", html)
        self.assertIn("0.882", html)
        self.assertIn("5.5 → 10.0", html)
        self.assertIn("Benchmark Audit Case Study (Why Single-System Pooling Failed)", html)
        self.assertIn('<details class="card collapsible audit-case-study">', html)
        for table in services.EVALUATION_TABLES:
            if not table.path.exists():
                continue
            frame = pd.read_csv(table.path, dtype=str, keep_default_na=False)
            self.assertIn(table.title, html)
            self.assertIn(f'id="table-{table.key}"', html)
            self.assertIn(f"{len(frame)} rows", html)
            for column in frame.columns:
                self.assertIn(f'<th scope="col">{column}</th>', html)

    def test_rich_researcher_metadata_links_top3_papers_and_topic_dedup(self) -> None:
        self.pipeline.search = lambda **kwargs: [  # type: ignore[method-assign]
            _raw_result(
                1,
                "paper_discount",
                institution_name="Indian Institute of Technology Bombay; Indian Institute of Technology Bombay",
                matched_papers=4,
                relevant_paper={
                    "title": "Neural Machine Translation for Indian Languages",
                    "year": "2025",
                    "work_id": "https://openalex.org/W101",
                    "doi": "https://doi.org/10.1145/3580305.3599999",
                    "abstract": "We study multilingual transformer translation across 12 languages.",
                    "authors_count": 2,
                    "paper_score": 0.8123,
                },
                matching_papers=[
                    {
                        "title": "Neural Machine Translation for Indian Languages",
                        "year": "2025",
                        "work_id": "https://openalex.org/W101",
                        "doi": "https://doi.org/10.1145/3580305.3599999",
                        "abstract": "We study multilingual transformer translation across 12 languages.",
                        "authors_count": 2,
                        "paper_score": 0.8123,
                    },
                    {
                        "title": "Cross-Lingual Pretraining at Scale",
                        "year": "2024",
                        "work_id": "https://openalex.org/W102",
                        "doi": "10.18653/v1/2024.acl-long.42",
                        "paper_score": 0.745,
                    },
                    {
                        "title": "Low-Resource Subword Tokenization",
                        "year": "2023",
                        "work_id": "https://openalex.org/W103",
                        "doi": "javascript:alert(1)",
                        "paper_score": 0.698,
                    },
                    {
                        "title": "Fourth Paper Beyond Cap",
                        "year": "2022",
                        "work_id": "https://openalex.org/W104",
                        "paper_score": 0.610,
                    },
                ],
                related_topics=[
                    "Natural Language Processing,  Machine Translation|Multilingual NLP",
                    "natural language processing",
                    "Subword Modeling, Cross-Lingual Transfer, Low-Resource Languages, Extra Seventh Tag",
                ],
            )
        ]
        html = self.get("/?q=machine+translation")
        self.assertIn('<p class="researcher-affiliation">Indian Institute of Technology Bombay</p>', html)
        self.assertIn('href="https://openalex.org/A1"', html)
        self.assertIn(">OpenAlex Profile ↗</a>", html)
        self.assertIn('href="https://doi.org/10.1145/3580305.3599999"', html)
        self.assertIn("DOI: 10.1145/3580305.3599999 ↗", html)
        self.assertIn('href="https://openalex.org/W101"', html)
        self.assertIn('href="https://doi.org/10.18653/v1/2024.acl-long.42"', html)
        self.assertNotIn('href="javascript:', html)
        self.assertIn("Additional Matching Papers", html)
        self.assertIn("Cross-Lingual Pretraining at Scale", html)
        self.assertIn("Low-Resource Subword Tokenization", html)
        self.assertNotIn("Fourth Paper Beyond Cap", html)
        self.assertIn("Abstract &amp; Match Rationale", html)
        self.assertIn("We study multilingual transformer translation across 12 languages.", html)
        self.assertIn("1/√co-authors", html)
        # Topic badges split on ',' and '|', deduplicate case-insensitively, and cap at 6
        self.assertIn('<li class="badge">Natural Language Processing</li>', html)
        self.assertIn('<li class="badge">Machine Translation</li>', html)
        self.assertIn('<li class="badge">Multilingual NLP</li>', html)
        self.assertNotIn("Extra Seventh Tag", html)
        self.assertEqual(html.count('<li class="badge">'), 6)

    def test_collaboration_filter_controls_and_shortlist_drawer_rendered(self) -> None:
        self.pipeline.profiles_df = pd.DataFrame(
            {
                "author_id": [1, 2],
                "institution_name": [
                    "Indian Institute of Technology Bombay; IISc Bangalore",
                    "Indian Institute of Technology Bombay",
                ],
            }
        )
        html = self.get("/")
        self.assertIn("Collaboration Filters", html)
        self.assertIn('id="institution"', html)
        self.assertIn('id="institution-list"', html)
        self.assertIn('<option value="Indian Institute of Technology Bombay"></option>', html)
        self.assertIn('id="min_papers"', html)
        self.assertIn('id="since_year"', html)
        self.assertIn('id="shortlist-drawer"', html)
        self.assertIn('data-shortlist-export="md"', html)
        self.assertIn('data-shortlist-export="csv"', html)
        self.assertIn("data-shortlist-copy-brief", html)

        results_html = self.get("/?q=nlp&top_k=5")
        self.assertEqual(results_html.count("data-card-actions"), 5)
        self.assertIn("data-shortlist-toggle", results_html)
        self.assertIn("data-copy-outreach", results_html)

    def test_invalid_collaboration_filters_are_rejected(self) -> None:
        for query in ("min_papers=0", "min_papers=4", "min_papers=abc", "since_year=2019", "since_year=xyz"):
            html = self.get(f"/?q=nlp&{query}", status=400)
            self.assertIn('class="alert alert-error"', html, query)
        self.assertEqual(self.pipeline.calls, [])

    def test_active_collaboration_filters_expand_pool_renumber_and_handle_empty_match(self) -> None:
        recorded_calls: list[dict[str, Any]] = []

        def filtered_pool(query: str, top_k: int = 10, mode: str = "paper_discount", alpha: float = 0.7) -> list[dict[str, Any]]:
            recorded_calls.append({"query": query, "top_k": top_k, "mode": mode})
            return [
                _raw_result(
                    1,
                    mode,
                    author_name="Junior Researcher",
                    institution_name="IIT Bombay",
                    n_papers=2,
                    relevant_paper={"title": "Recent NLP Paper", "year": "2024", "paper_score": 0.72},
                ),
                _raw_result(
                    2,
                    mode,
                    author_name="Senior Older Paper",
                    institution_name="IIT Bombay",
                    n_papers=8,
                    relevant_paper={"title": "Foundational NLP Paper", "year": "2021", "paper_score": 0.70},
                ),
                _raw_result(
                    3,
                    mode,
                    author_name="Established Active Faculty",
                    institution_name="IIT Bombay",
                    n_papers=7,
                    relevant_paper={"title": "Modern LLM Extraction", "year": "2025", "paper_score": 0.68},
                ),
                _raw_result(
                    4,
                    mode,
                    author_name="Other Institute Faculty",
                    institution_name="IISc Bangalore",
                    n_papers=9,
                    relevant_paper={"title": "Modern Speech Modeling", "year": "2024", "paper_score": 0.66},
                ),
            ]

        self.pipeline.search = filtered_pool  # type: ignore[method-assign]
        html = self.get("/?q=nlp&institution=IIT+Bombay&min_papers=5&since_year=2024&top_k=5")
        self.assertEqual(recorded_calls[-1]["top_k"], 120)
        self.assertIn("Found 1 researchers matching query", html)
        self.assertIn('<span class="rank">#1</span> Established Active Faculty', html)
        self.assertNotIn("Junior Researcher", html)
        self.assertNotIn("Senior Older Paper", html)
        self.assertNotIn("Other Institute Faculty", html)
        self.assertIn("Institution: <strong>IIT Bombay</strong>", html)
        self.assertIn("Indexed papers ≥ <strong>5</strong>", html)
        self.assertIn("Published <strong>2024+</strong>", html)
        self.assertIn(">Reset filters</a>", html)

        empty_html = self.get("/?q=nlp&institution=Nonexistent+University")
        self.assertIn("No researchers matched your current filter combination.", empty_html)
        self.assertIn(">Clear collaboration filters</a>", empty_html)

    def test_dynamic_topic_cluster_chips_disambiguate_ambiguous_queries(self) -> None:
        def multi_domain_pool(query: str, top_k: int = 10, mode: str = "paper_discount", alpha: float = 0.7) -> list[dict[str, Any]]:
            return [
                _raw_result(
                    1,
                    mode,
                    author_name="NLP Researcher A",
                    relevant_paper={"title": "Transformer Language Models for Translation", "year": "2024", "paper_score": 0.78},
                    related_topics=["Natural Language Processing", "Machine Translation"],
                ),
                _raw_result(
                    2,
                    mode,
                    author_name="Power Grid Researcher",
                    relevant_paper={"title": "Power Transformer Insulation Diagnostics", "year": "2024", "paper_score": 0.76},
                    related_topics=["Power Transformer", "High Voltage Engineering"],
                ),
                _raw_result(
                    3,
                    mode,
                    author_name="NLP Researcher B",
                    relevant_paper={"title": "Multilingual Question Answering", "year": "2025", "paper_score": 0.73},
                    related_topics=["Natural Language Processing", "Question Answering"],
                ),
            ]

        self.pipeline.search = multi_domain_pool  # type: ignore[method-assign]
        html = self.get("/?q=transformer")
        self.assertIn("Disambiguate by Sub-field / Topic", html)
        self.assertIn("<span>All sub-fields</span>", html)
        self.assertIn("<span>NLP &amp; Language Models</span>", html)
        self.assertIn("<span>Electrical Power &amp; Energy Systems</span>", html)

        disambiguated_html = self.get("/?q=transformer&topic=NLP+%26+Language+Models")
        self.assertIn("Found 2 researchers matching query", disambiguated_html)
        self.assertIn("NLP Researcher A", disambiguated_html)
        self.assertIn("NLP Researcher B", disambiguated_html)
        self.assertNotIn("Power Grid Researcher", disambiguated_html)
        # Sibling sub-field chip stays visible so users can switch clusters in one click
        self.assertIn("<span>Electrical Power &amp; Energy Systems</span>", disambiguated_html)
        self.assertIn("Sub-field: <strong>NLP &amp; Language Models</strong>", disambiguated_html)

    def test_csv_download_returns_file_unchanged(self) -> None:
        for table in services.EVALUATION_TABLES:
            if not table.path.exists():
                continue
            response = self.client.get(f"/evaluation/{table.key}.csv")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "text/csv")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertEqual(response.data, table.path.read_bytes())
            response.close()

    def test_laya_paper_and_query_classifications_render_on_search_and_compare(self) -> None:
        def laya_pool(query: str, top_k: int = 10, mode: str = "paper_discount", alpha: float = 0.7) -> list[dict[str, Any]]:
            return [
                _raw_result(
                    1,
                    mode,
                    author_name="Survey Specialist",
                    relevant_paper={
                        "title": "A Comprehensive Survey of Neural Machine Translation",
                        "year": "2024",
                        "paper_score": 0.76,
                    },
                    matching_papers=[
                        {
                            "work_id": "https://openalex.org/W_SURVEY",
                            "title": "A Comprehensive Survey of Neural Machine Translation",
                            "year": "2024",
                            "paper_score": 0.76,
                            "abstract": "Systematic literature review of transformer translation architectures.",
                        },
                        {
                            "work_id": "https://openalex.org/W_BENCH",
                            "title": "A Multilingual Translation Evaluation Benchmark and Dataset",
                            "year": "2023",
                            "paper_score": 0.64,
                            "abstract": "We release a curated benchmark corpus across 20 languages.",
                        },
                    ],
                    related_topics=["Natural Language Processing", "Machine Translation"],
                )
            ]

        self.pipeline.search = laya_pool  # type: ignore[method-assign]
        html = self.get("/?q=How+do+survey+papers+evaluate+neural+machine+translation%3F")
        self.assertIn("Laya Domain: NLP &amp; Language Models", html)
        self.assertIn("Analytical Question", html)
        self.assertIn("Laya · Survey / Review", html)
        self.assertIn("Direct Match (3/3)", html)
        self.assertIn("Dataset / Benchmark", html)

        compare_html = self.get("/compare?q=neural+machine+translation")
        self.assertIn('class="badge badge-laya-subtle"', compare_html)
        self.assertIn("Survey / Review", compare_html)

    def test_dataset_ai_reasoning_renders_in_search_and_sidebar(self) -> None:
        html = self.get("/?q=nlp&reason=1&reason_provider=groq")
        self.assertIn("AI Dataset Reasoning", html)
        self.assertIn("Dataset AI Reasoning (Gemini &amp; Groq)", html)
        self.assertIn('id="reasoning-panel"', html)
        self.assertIn('id="reason_provider"', html)
        self.assertIn('name="reason"', html)
        self.assertIn("Grounded Citations", html)

    def test_evaluation_page_renders_nlp_architecture_and_roadmap(self) -> None:
        html = self.get("/evaluation")
        self.assertIn("NLP Concepts &amp; Architecture Showcase", html)
        self.assertIn("Dense Bi-Encoder (MiniLM)", html)
        self.assertIn("System-1 RLCD (Laya Model)", html)
        self.assertIn("System-2 Grounded RAG (Groq / Gemini)", html)
        self.assertIn("Engineering Roadmap: 5 Identified Areas for Improvement", html)
        self.assertIn("Domain-Adapted Citation-Graph Bi-Encoders (SPECTER2)", html)
        self.assertIn("Exponential Temporal Expertise Decay", html)
        self.assertIn("Authorship-Position Credit Assignment Priors", html)
        self.assertIn("Distribution-Free Split-Conformal Prediction Sets", html)
        self.assertIn("Automated Cross-Encoder NLI Faithfulness Verification", html)

    def test_unknown_pages_and_tables_return_404(self) -> None:
        self.get("/evaluation/papers.csv", status=404)
        self.get("/evaluation/../app.py", status=404)
        html = self.get("/nowhere", status=404)
        self.assertIn("This page does not exist.", html)
        self.assertIn("Error 404: Not Found", html)
        self.assertIn('class="skip-link"', html)

    def test_wrong_method_uses_site_error_page(self) -> None:
        response = self.client.post("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 405)
        self.assertIn("GET", response.headers["Allow"])
        self.assertIn("Error 405: Method Not Allowed", html)
        self.assertIn('class="skip-link"', html)

    def test_unhandled_exception_shows_generic_500_page(self) -> None:
        app = create_app(FakePipeline, background=False)

        @app.get("/boom")
        def boom() -> str:
            raise RuntimeError("secret internal detail")

        with self.assertLogs(app.logger, "ERROR"):
            response = app.test_client().get("/boom")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 500)
        self.assertIn("Error 500: Internal Server Error", html)
        self.assertIn("An unexpected error occurred while handling this request.", html)
        self.assertIn('class="skip-link"', html)
        self.assertNotIn("secret internal detail", html)
        self.assertNotIn("Traceback", html)

    def test_navigation_marks_current_page(self) -> None:
        html = self.get("/compare")
        self.assertIn('href="/compare" aria-current="page"', html)
        self.assertNotIn('href="/" aria-current="page"', html)

    def test_static_assets_are_served(self) -> None:
        for path in (
            "css/main.css",
            "css/integration.css",
            "css/responsive.css",
            "js/main.js",
            "js/ui.js",
            "js/search.js",
            "js/tables.js",
            "js/integration.js",
            "images/favicon.svg",
        ):
            response = self.client.get(f"/static/{path}")
            self.assertEqual(response.status_code, 200, path)
            response.close()


class PipelineStateTests(unittest.TestCase):
    def test_loading_page_until_pipeline_is_ready(self) -> None:
        release = threading.Event()

        def slow_factory() -> FakePipeline:
            release.wait(5)
            return FakePipeline()

        app = create_app(slow_factory)
        client = app.test_client()
        response = client.get("/?q=nlp")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Retry-After"], "3")
        self.assertIn('http-equiv="refresh"', response.get_data(as_text=True))
        self.assertNotIn(">Search settings</span>", response.get_data(as_text=True))
        self.assertEqual(client.get("/evaluation").status_code, 200)
        release.set()
        loader = app.extensions["pipeline_loader"]
        loader._ready.wait(5)
        self.assertEqual(client.get("/?q=nlp").status_code, 200)

    def test_error_page_when_pipeline_fails(self) -> None:
        def broken_factory() -> FakePipeline:
            raise FileNotFoundError("researcher_profiles.csv not found")

        client = create_app(broken_factory, background=False).test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 500)
        self.assertIn("The NLP pipeline could not be loaded: researcher_profiles.csv not found", response.get_data(as_text=True))
        self.assertIn("Unavailable", client.get("/evaluation").get_data(as_text=True))

    def test_failure_outside_common_exception_families_still_shows_error_page(self) -> None:
        # A corrupt .npz cache makes np.load raise zipfile.BadZipFile, which derives
        # directly from Exception; this guards the deliberately broad except clause.
        def corrupt_cache() -> FakePipeline:
            raise zipfile.BadZipFile("File is not a zip file")

        app = create_app(corrupt_cache)
        loader = app.extensions["pipeline_loader"]
        self.assertTrue(loader._ready.wait(5))
        self.assertEqual((loader.status, loader.error), ("failed", "File is not a zip file"))
        client = app.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 500)
        self.assertIn("The NLP pipeline could not be loaded: File is not a zip file", response.get_data(as_text=True))
        self.assertEqual(client.get("/evaluation").status_code, 200)


class SettingsTests(unittest.TestCase):
    def test_dataset_counts_are_raw_integers(self) -> None:
        stats = {stat.label: stat.value for stat in services.dataset_stats(FakePipeline())}
        self.assertEqual(stats["Indexed OpenAlex Researchers"], 52736)
        self.assertEqual(stats["Indexed Academic Papers"], 62637)

    def test_defaults(self) -> None:
        settings, errors = services.parse_search_settings({})
        self.assertEqual((settings.mode, settings.top_k, settings.alpha, errors), ("paper_discount", 10, 0.7, []))

    def test_all_slider_positions_are_accepted(self) -> None:
        for top_k in range(5, 30, 5):
            self.assertEqual(services.parse_search_settings({"top_k": str(top_k)})[1], [])
        for step in range(21):
            alpha = f"{step * 0.05:.2f}"
            self.assertEqual(services.parse_search_settings({"alpha": alpha})[1], [], alpha)

    def test_institution_suggestions_extracts_and_caches_top_institutions(self) -> None:
        pipeline = FakePipeline()
        pipeline.profiles_df = pd.DataFrame(
            {
                "author_id": [1, 2, 3, 4, 5],
                "institution_name": [
                    "IIT Bombay; IISc Bangalore",
                    "IIT Bombay",
                    "IIT Bombay; BITS Pilani",
                    "IISc Bangalore",
                    "nan",
                ],
            }
        )
        suggestions = services.institution_suggestions(pipeline, limit=2)
        self.assertEqual(suggestions, ("IIT Bombay", "IISc Bangalore"))
        # Mutating profiles_df afterwards confirms cached tuple is returned
        pipeline.profiles_df = pd.DataFrame({"author_id": [1]})
        self.assertEqual(services.institution_suggestions(pipeline, limit=2), ("IIT Bombay", "IISc Bangalore"))


class PipelinePaperDiscountAndFilterTests(unittest.TestCase):
    """Verify paper-level co-author discounting and non-research artifact filtering."""

    def _build_synthetic_pipeline(self) -> Any:
        from src.week2.pipeline import ResearchDiscoveryPipeline

        pipeline = ResearchDiscoveryPipeline(auto_load=False)
        papers = [
            {
                "work_id": "W_SOLO",
                "title": "Code-Switching and Neural Language Modeling",
                "abstract": "Transformer language models for multilingual text.",
                "publication_year": "2024",
                "authors": "Solo Specialist",
                "author_ids": "https://openalex.org/A_SOLO",
                "topics": "Natural Language Processing|Multilingual Modeling",
                "keywords": "code-switching|transformers",
            },
            {
                "work_id": "W_TEAM",
                "title": "Large Consortium Survey of Neural Models",
                "abstract": "Overview of neural models across nine co-authors.",
                "publication_year": "2023",
                "authors": "; ".join(f"Team Member {i}" for i in range(1, 10)),
                "author_ids": "|".join(f"https://openalex.org/A_TEAM_{i}" for i in range(1, 10)),
                "topics": "Deep Learning",
                "keywords": "survey",
            },
            {
                "work_id": "W_RETRACT",
                "title": "Retraction Note to: Neural Language Modeling",
                "abstract": "This article has been retracted.",
                "publication_year": "2024",
                "authors": "Retracted Author",
                "author_ids": "https://openalex.org/A_RETRACT",
                "topics": "NLP",
                "keywords": "retraction",
            },
            {
                "work_id": "W_CODE",
                "title": "Code for the ML models",
                "abstract": "Repository dump.",
                "publication_year": "2024",
                "authors": "Artifact Author",
                "author_ids": "https://openalex.org/A_ARTIFACT",
                "topics": "Software",
                "keywords": "code",
            },
        ]
        pipeline.papers_df = pd.DataFrame(papers)
        pipeline.papers_by_id = {p["work_id"]: p for p in papers}
        pipeline.paper_work_ids = np.array([p["work_id"] for p in papers])
        # Query embedding will be [1.0, 0.0]:
        # W_SOLO has raw cosine 0.75 -> discounted 0.75 / sqrt(1) = 0.75
        # W_TEAM has higher raw cosine 0.90 -> discounted 0.90 / sqrt(9) = 0.30
        # W_RETRACT & W_CODE have perfect 1.0 cosine -> must be completely excluded!
        pipeline.paper_embeddings = np.array(
            [
                [0.75, math.sqrt(1 - 0.75**2)],
                [0.90, math.sqrt(1 - 0.90**2)],
                [1.00, 0.0],
                [1.00, 0.0],
            ],
            dtype=np.float32,
        )
        pipeline.work_to_emb_idx = {p["work_id"]: idx for idx, p in enumerate(papers)}

        profiles = [
            {
                "author_id": "https://openalex.org/A_SOLO",
                "author_name": "Solo Specialist",
                "n_papers": 5,
                "paper_ids": "W_SOLO",
                "profile_text": "code switching neural language modeling transformers",
            },
            {
                "author_id": "https://openalex.org/A_TEAM_1",
                "author_name": "Team Member 1",
                "n_papers": 2,
                "paper_ids": "W_TEAM",
                "profile_text": "large consortium survey neural models",
            },
            {
                "author_id": "https://openalex.org/A_RETRACT",
                "author_name": "Retracted Author",
                "n_papers": 1,
                "paper_ids": "W_RETRACT",
                "profile_text": "retraction note neural language modeling",
            },
        ]
        pipeline.profiles_df = pd.DataFrame(profiles)
        pipeline.researcher_author_ids = np.array([p["author_id"] for p in profiles])
        pipeline.researcher_embeddings = np.array(
            [
                [0.75, math.sqrt(1 - 0.75**2)],
                [0.85, math.sqrt(1 - 0.85**2)],
                [1.00, 0.0],
            ],
            dtype=np.float32,
        )
        pipeline.author_to_emb_idx = {p["author_id"]: idx for idx, p in enumerate(profiles)}
        pipeline.vectorizer = TfidfVectorizer()
        pipeline.tfidf_matrix = pipeline.vectorizer.fit_transform(pipeline.profiles_df["profile_text"])

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([1.0, 0.0], dtype=np.float32)
        pipeline.model = fake_model
        return pipeline

    def test_paper_discount_penalizes_large_coauthor_teams_and_drops_artifacts(self) -> None:
        pipeline = self._build_synthetic_pipeline()
        results = pipeline.search("neural language modeling", top_k=5, mode="paper_discount")
        returned_ids = [r["author_id"] for r in results]

        self.assertNotIn("https://openalex.org/A_RETRACT", returned_ids)
        self.assertNotIn("https://openalex.org/A_ARTIFACT", returned_ids)
        self.assertEqual(results[0]["author_id"], "https://openalex.org/A_SOLO")
        self.assertAlmostEqual(results[0]["similarity_score"], 0.75, places=3)
        self.assertEqual(results[0]["relevant_paper"]["title"], "Code-Switching and Neural Language Modeling")
        # 9-author paper with 0.90 raw similarity is discounted to 0.90 / sqrt(9) = 0.30
        team_member = next(r for r in results if r["author_id"] == "https://openalex.org/A_TEAM_1")
        self.assertAlmostEqual(team_member["similarity_score"], 0.30, places=3)

    def test_profile_modes_exclude_retraction_only_authors(self) -> None:
        pipeline = self._build_synthetic_pipeline()
        for mode in ("semantic", "tfidf", "hybrid"):
            results = pipeline.search("neural language modeling", top_k=5, mode=mode)
            returned_ids = [r["author_id"] for r in results]
            self.assertNotIn("https://openalex.org/A_RETRACT", returned_ids, mode)

    def test_extract_clean_topics_prioritizes_matched_evidence_without_off_topic_pollution(self) -> None:
        from src.week2.pipeline import _extract_clean_topics

        matched = [
            {
                "topics": "Natural Language Processing,  Information Extraction|Syntax",
                "keywords": "transformers, natural language processing, named entity recognition, parsing, extra keyword",
            }
        ]
        unrelated_profile_papers = [
            {
                "topics": "Marine Biology, Coral Reef Ecology",
                "keywords": "oceanography",
            }
        ]
        cleaned = _extract_clean_topics(matched, fallback_papers=unrelated_profile_papers, limit=6)
        self.assertEqual(
            cleaned,
            [
                "Natural Language Processing",
                "Information Extraction",
                "Syntax",
                "transformers",
                "named entity recognition",
                "parsing",
            ],
        )
        self.assertNotIn("Marine Biology", cleaned)


if __name__ == "__main__":
    unittest.main()

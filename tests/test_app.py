"""Tests for the web interface (``app.py``, ``services.py`` and the templates).

Run from the repository root::

    uv run python -m unittest discover -s tests -t .

A fake pipeline stands in for ``ResearchDiscoveryPipeline``, so these tests
need neither the dataset nor the embedding model.
"""

import threading
import unittest
import zipfile
from typing import Any

import pandas as pd
from markupsafe import escape

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
        self.assertIn("Search Configuration", html)
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

    def test_compare_runs_semantic_and_tfidf_top_5(self) -> None:
        html = self.get("/compare")
        self.assertIn(f'value="{services.COMPARISON_DEFAULT_QUERY}"', html)
        self.assertEqual(self.pipeline.calls, [])
        html = self.get("/compare?q=knowledge+graphs")
        self.assertEqual(
            [(c["mode"], c["top_k"], c["query"]) for c in self.pipeline.calls],
            [("semantic", 5, "knowledge graphs"), ("tfidf", 5, "knowledge graphs")],
        )
        self.assertIn("Sentence Transformers (Dense Semantic)", html)
        self.assertIn("TF-IDF Baseline (Sparse Keyword)", html)
        self.assertEqual(html.count('class="compare-item"'), 10)

    def test_compare_with_empty_query_shows_no_results(self) -> None:
        html = self.get("/compare?q=")
        self.assertEqual(html.count("No researchers returned for this query."), 2)

    def test_dataset_statistics_in_sidebar(self) -> None:
        html = self.get("/evaluation")
        for text in ("Faculty &amp; Researchers", "52,736", "Indexed Academic Papers", "62,637", "all-MiniLM-L6-v2"):
            self.assertIn(text, html)

    def test_evaluation_tables_match_csv_files(self) -> None:
        html = self.get("/evaluation")
        for table in services.EVALUATION_TABLES:
            if not table.path.exists():
                continue
            frame = pd.read_csv(table.path, dtype=str, keep_default_na=False)
            self.assertIn(table.title, html)
            self.assertIn(f'id="table-{table.key}"', html)
            self.assertIn(f"{len(frame)} rows", html)
            for column in frame.columns:
                self.assertIn(f'<th scope="col">{column}</th>', html)

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
        for path in ("css/main.css", "css/responsive.css", "js/main.js", "js/search.js", "js/tables.js", "images/favicon.svg"):
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
        self.assertEqual(stats["Faculty & Researchers"], 52736)
        self.assertEqual(stats["Indexed Academic Papers"], 62637)

    def test_defaults(self) -> None:
        settings, errors = services.parse_search_settings({})
        self.assertEqual((settings.mode, settings.top_k, settings.alpha, errors), ("semantic", 10, 0.7, []))

    def test_all_slider_positions_are_accepted(self) -> None:
        for top_k in range(5, 30, 5):
            self.assertEqual(services.parse_search_settings({"top_k": str(top_k)})[1], [])
        for step in range(21):
            alpha = f"{step * 0.05:.2f}"
            self.assertEqual(services.parse_search_settings({"alpha": alpha})[1], [], alpha)


if __name__ == "__main__":
    unittest.main()

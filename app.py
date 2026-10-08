"""
Web application: Research Discovery & Collaboration Engine.
AI University NLP Course Project (Deliverable Week 3: Version 1 Core NLP Pipeline).

Run from the repository root:

    uv run python app.py            # http://127.0.0.1:8000
    uv run flask --app app run      # same app, Flask's default port

Routes handle requests only. Data is prepared in ``services`` and all HTML is
rendered by the Jinja templates in ``templates/``.
"""

import os
from collections.abc import Callable
from typing import Any

from flask import Flask, abort, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Response

import services

NAV_ITEMS = (
    ("search", "Research Search"),
    ("compare", "Model Comparison"),
    ("evaluation", "Benchmark & Metrics"),
)
EMPTY_QUERY_WARNING = "Enter a research topic or choose a query preset first."
# Error-page messages that replace Werkzeug's default descriptions.
ERROR_MESSAGES = {
    404: "This page does not exist.",
    500: "An unexpected error occurred while handling this request.",
}


def _default_pipeline_factory() -> Any:
    from src.week2.pipeline import ResearchDiscoveryPipeline

    return ResearchDiscoveryPipeline()


def create_app(
    pipeline_factory: Callable[[], Any] = _default_pipeline_factory, *, background: bool = True
) -> Flask:
    """Build the app and start loading the pipeline (in a background thread by default)."""
    app = Flask(__name__)
    loader = services.PipelineLoader(pipeline_factory)
    app.extensions["pipeline_loader"] = loader
    loader.start(background=background)

    @app.context_processor
    def layout_context() -> dict[str, Any]:
        ready = loader.status == "ready"
        return {
            "nav_items": NAV_ITEMS,
            "pipeline_status": loader.status,
            "stats": services.dataset_stats(loader.pipeline) if ready else [],
        }

    def unavailable() -> tuple[str, int, dict[str, str]] | None:
        """The loading or error page while the pipeline cannot answer queries."""
        if loader.status == "loading":
            return render_template("loading.html"), 503, {"Retry-After": "3"}
        if loader.status == "failed":
            message = f"The NLP pipeline could not be loaded: {loader.error}"
            page = render_template("error.html", code=500, name="Internal Server Error", message=message)
            return page, 500, {}
        return None

    @app.get("/")
    def search() -> Any:
        if (response := unavailable()) is not None:
            return response
        settings, errors = services.parse_search_settings(request.args)
        submitted = "q" in request.args
        query = request.args.get("q", "")
        outcome = None
        warning = None
        if submitted and not errors:
            if query.strip():
                with loader.search_lock:
                    outcome = services.run_search(loader.pipeline, query, settings)
            else:
                warning = EMPTY_QUERY_WARNING
        page = render_template(
            "search.html",
            settings=settings,
            query=query,
            outcome=outcome,
            errors=errors,
            warning=warning,
            mode_labels=services.MODE_LABELS,
            presets=services.QUERY_PRESETS,
            top_k_range=(services.TOP_K_MIN, services.TOP_K_MAX, services.TOP_K_STEP),
            alpha_range=(services.ALPHA_MIN, services.ALPHA_MAX, services.ALPHA_STEP),
        )
        return page, 400 if errors else 200

    @app.get("/compare")
    def compare() -> Any:
        if (response := unavailable()) is not None:
            return response
        submitted = "q" in request.args
        query = request.args.get("q", services.COMPARISON_DEFAULT_QUERY)
        columns = None
        if submitted:
            with loader.search_lock:
                columns = services.run_comparison(loader.pipeline, query)
        return render_template("compare.html", query=query, columns=columns, top_k=services.COMPARISON_TOP_K)

    @app.get("/evaluation")
    def evaluation() -> str:
        tables = {table.key: services.load_table(table) for table in services.EVALUATION_TABLES}
        return render_template("evaluation.html", tables=tables)

    @app.get("/evaluation/<key>.csv")
    def download_table(key: str) -> Any:
        table = services.evaluation_table(key)
        if table is None or not table.path.exists():
            abort(404)
        return send_file(table.path, mimetype="text/csv", as_attachment=True, download_name=f"{key}.csv")

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> Response:
        """Render every HTTP error (404, 405, and 500 for unhandled exceptions) with the site layout.

        Flask logs unhandled exceptions before calling this handler; the page
        itself shows only a generic message, never the exception.
        """
        code = error.code or 500
        response = error.get_response()  # Keeps headers such as Allow on a 405.
        response.set_data(
            render_template(
                "error.html", code=code, name=error.name, message=ERROR_MESSAGES.get(code, error.description)
            )
        )
        response.content_type = "text/html; charset=utf-8"
        return response

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))

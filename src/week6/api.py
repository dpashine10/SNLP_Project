"""Flask REST API service for the Reusable NLP Asset.

Exposes structured JSON endpoints for health, metadata, single recommendations,
batch queries, and schema discovery.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from flask import Blueprint, Flask, current_app, jsonify, request

from src.week6.asset import AssetConfig, NLPAsset
from src.week6.schemas import (
    DEFAULT_PAPER_TOP_K,
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    MIN_TOP_K,
    SUPPORTED_STRATEGIES,
    RecommendRequest,
    ValidationError,
)

logger = logging.getLogger(__name__)

api_v1 = Blueprint("asset_api_v1", __name__, url_prefix="/api/v1")


def _get_asset() -> NLPAsset:
    return current_app.extensions["nlp_asset"]


@api_v1.get("/health")
def health() -> Any:
    """Health check and backend status."""
    active_asset = _get_asset()
    health_info = active_asset.health_check()
    status_code = 200 if health_info["status"] in ("ready", "unloaded") else 503
    return jsonify(health_info), status_code


@api_v1.get("/info")
def info() -> Any:
    """Metadata regarding available models, strategies, and limits."""
    active_asset = _get_asset()
    return jsonify({
        "asset_version": "1.0.0",
        "model_metadata": active_asset.get_metadata().to_dict(),
        "supported_strategies": list(SUPPORTED_STRATEGIES),
        "default_strategy": DEFAULT_STRATEGY,
        "limits": {
            "min_top_k": MIN_TOP_K,
            "max_top_k": MAX_TOP_K,
            "default_top_k": DEFAULT_TOP_K,
            "default_paper_top_k": DEFAULT_PAPER_TOP_K,
            "max_query_length": MAX_QUERY_LENGTH,
        },
    }), 200


@api_v1.post("/recommend")
def recommend() -> Any:
    """Execute a structured researcher recommendation search."""
    active_asset = _get_asset()

    if not request.is_json:
        return jsonify({
            "error": "Bad Request",
            "message": "Request body must be valid application/json.",
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "error": "Bad Request",
            "message": "Malformed JSON object in request body.",
        }), 400

    try:
        req = RecommendRequest.from_dict(data)
    except (ValueError, TypeError) as exc:
        return jsonify({
            "error": "Validation Error",
            "message": str(exc),
        }), 400

    response = active_asset.recommend(req)
    return jsonify(response.to_dict()), 200


@api_v1.post("/batch")
def batch_recommend() -> Any:
    """Execute multiple recommendation queries in a single request."""
    active_asset = _get_asset()

    if not request.is_json:
        return jsonify({
            "error": "Bad Request",
            "message": "Request body must be valid application/json.",
        }), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "queries" not in data:
        return jsonify({
            "error": "Bad Request",
            "message": "Batch payload must contain a 'queries' array.",
        }), 400

    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        return jsonify({
            "error": "Validation Error",
            "message": "'queries' must be a non-empty list of search queries or request objects.",
        }), 400

    top_k = int(data.get("top_k", DEFAULT_TOP_K))
    strategy = str(data.get("strategy", DEFAULT_STRATEGY))

    requests_list = []
    for q in queries:
        if isinstance(q, str):
            requests_list.append(RecommendRequest(query=q, top_k=top_k, strategy=strategy))
        elif isinstance(q, dict):
            requests_list.append(RecommendRequest.from_dict(q))
        else:
            return jsonify({
                "error": "Validation Error",
                "message": "Each item in 'queries' must be a string or request object.",
            }), 400

    start_time = time.perf_counter()
    responses = active_asset.batch_recommend(requests_list)
    total_time_ms = (time.perf_counter() - start_time) * 1000.0

    return jsonify({
        "batch_size": len(responses),
        "total_execution_time_ms": round(total_time_ms, 2),
        "results": [r.to_dict() for r in responses],
    }), 200


@api_v1.get("/schema")
def schema() -> Any:
    """Self-documenting input/output JSON schemas."""
    return jsonify({
        "RecommendRequest": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": MAX_QUERY_LENGTH, "description": "Research synopsis"},
                "top_k": {"type": "integer", "minimum": MIN_TOP_K, "maximum": MAX_TOP_K, "default": DEFAULT_TOP_K},
                "strategy": {"type": "string", "enum": list(SUPPORTED_STRATEGIES), "default": DEFAULT_STRATEGY},
                "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
                "include_evidence": {"type": "boolean", "default": True},
                "paper_top_k": {"type": "integer", "default": DEFAULT_PAPER_TOP_K},
            },
            "required": ["query"],
        },
        "RecommendResponse": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "execution_time_ms": {"type": "number"},
                "total_results": {"type": "integer"},
                "model_info": {"type": "object"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer"},
                            "author_id": {"type": "string"},
                            "name": {"type": "string"},
                            "score": {"type": "number"},
                            "confidence": {
                                "type": "object",
                                "properties": {
                                    "score": {"type": "number"},
                                    "level": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                    "margin": {"type": "number"},
                                    "explanation": {"type": "string"},
                                },
                            },
                            "evidence": {"type": "array"},
                        },
                    },
                },
            },
        },
    }), 200


def create_asset_app(asset: NLPAsset | None = None, *, eager_warmup: bool = False) -> Flask:
    """Application factory for the NLP Asset REST service."""
    app = Flask(__name__)

    # Inject or construct NLP Asset singleton
    if asset is None:
        asset = NLPAsset(AssetConfig(eager_load=eager_warmup))
    app.extensions["nlp_asset"] = asset

    @app.after_request
    def add_cors_headers(response: Any) -> Any:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError) -> tuple[Any, int]:
        return jsonify(error.to_dict()), 400

    @app.errorhandler(400)
    def handle_bad_request(error: Any) -> tuple[Any, int]:
        return jsonify({"error": "Bad Request", "message": str(error)}), 400

    @app.errorhandler(404)
    def handle_not_found(error: Any) -> tuple[Any, int]:
        return jsonify({"error": "Not Found", "message": "The requested endpoint does not exist."}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(error: Any) -> tuple[Any, int]:
        return jsonify({"error": "Method Not Allowed", "message": "HTTP method is not supported."}), 405

    @app.errorhandler(500)
    def handle_internal_error(error: Any) -> tuple[Any, int]:
        return jsonify({"error": "Internal Server Error", "message": str(error)}), 500

    app.register_blueprint(api_v1)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app = create_asset_app()
    print(f"Starting NLP Asset REST API on http://127.0.0.1:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

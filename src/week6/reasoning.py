"""Dataset-grounded AI reasoning engine powered by Groq and Gemini.

Synthesizes evidence-backed answers strictly from retrieved OpenAlex publications
with inline citations, automatic provider failover, and deterministic extractive
synthesis fallback when running offline or without active network credentials.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.week6.schemas import CitationSource, ReasoningRequest, ReasoningResponse, ValidationError

logger = logging.getLogger(__name__)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def _load_env_keys() -> tuple[str, str]:
    """Load GEMINI_API_KEY and GROQ_API_KEY from environment or .env / env files."""
    root = _find_repo_root()
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / "env", override=False)
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    return gemini_key, groq_key


class GroundedReasoningEngine:
    """Grounded System-2 reasoning engine with multi-provider failover."""

    def __init__(
        self,
        *,
        groq_api_key: str | None = None,
        gemini_api_key: str | None = None,
        groq_model: str = DEFAULT_GROQ_MODEL,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        timeout_seconds: float = 12.0,
    ) -> None:
        loaded_gemini, loaded_groq = _load_env_keys()
        self.gemini_api_key = gemini_api_key if gemini_api_key is not None else loaded_gemini
        self.groq_api_key = groq_api_key if groq_api_key is not None else loaded_groq
        self.groq_model = groq_model
        self.gemini_model = gemini_model
        self.timeout_seconds = timeout_seconds

    def status(self) -> dict[str, Any]:
        """Return provider readiness and available models."""
        return {
            "groq_configured": bool(self.groq_api_key),
            "gemini_configured": bool(self.gemini_api_key),
            "groq_model": self.groq_model,
            "gemini_model": self.gemini_model,
            "timeout_seconds": self.timeout_seconds,
            "fallback_backend": "extractive-synthesis",
        }

    def reason(
        self,
        request: ReasoningRequest | Mapping[str, Any] | str,
        *,
        evidence_papers: Sequence[CitationSource | Mapping[str, Any]] | None = None,
    ) -> ReasoningResponse:
        """Synthesize a citation-backed scholarly answer from retrieved publication evidence."""
        if isinstance(request, str):
            req = ReasoningRequest.from_dict({
                "query": request,
                "papers": list(evidence_papers or []),
            })
        elif isinstance(request, Mapping):
            req = ReasoningRequest.from_dict(request)
        elif isinstance(request, ReasoningRequest):
            req = request
        else:
            raise ValidationError("Reasoning request must be a ReasoningRequest, dict, or string.")

        start_time = time.perf_counter()
        query = req.query
        papers = list(req.papers)
        provider = req.provider.lower().strip()

        if not papers:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ReasoningResponse(
                query=query,
                answer="No relevant publications were provided as grounding evidence. Please provide at least one indexed research paper to synthesize a grounded answer.",
                provider_used="none",
                model_name="none",
                citations=[],
                execution_time_ms=elapsed,
                is_fallback=True,
                error="No papers provided for grounding.",
            )

        prompt = self._build_user_prompt(query, papers)
        system_prompt = self._build_system_prompt()

        attempts: list[str] = []
        if provider == "groq":
            target_providers = ["groq"]
        elif provider == "gemini":
            target_providers = ["gemini"]
        else:  # "auto"
            target_providers = ["groq", "gemini"]

        for prov in target_providers:
            if prov == "groq":
                if not self.groq_api_key:
                    attempts.append("groq: API key missing")
                    continue
                try:
                    answer, model = self._call_groq(prompt, system_prompt, req.max_tokens)
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    return ReasoningResponse(
                        query=query,
                        answer=answer,
                        provider_used="groq",
                        model_name=model,
                        citations=papers,
                        execution_time_ms=elapsed,
                        is_fallback=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Groq reasoning failed: %s", exc)
                    attempts.append(f"groq: {exc}")

            elif prov == "gemini":
                if not self.gemini_api_key:
                    attempts.append("gemini: API key missing")
                    continue
                try:
                    answer, model = self._call_gemini(prompt, system_prompt, req.max_tokens)
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    return ReasoningResponse(
                        query=query,
                        answer=answer,
                        provider_used="gemini",
                        model_name=model,
                        citations=papers,
                        execution_time_ms=elapsed,
                        is_fallback=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Gemini reasoning failed: %s", exc)
                    attempts.append(f"gemini: {exc}")

        # Deterministic extractive synthesis fallback
        fallback_reason = "; ".join(attempts) if attempts else "Offline / sandboxed execution"
        fallback_answer = self._extractive_synthesis(query, papers, fallback_reason)
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return ReasoningResponse(
            query=query,
            answer=fallback_answer,
            provider_used="extractive_synthesis",
            model_name="deterministic-extractive-v1",
            citations=papers,
            execution_time_ms=elapsed,
            is_fallback=True,
            error=fallback_reason,
        )

    def _call_groq(self, prompt: str, system_prompt: str, max_tokens: int) -> tuple[str, str]:
        """Invoke Groq OpenAI-compatible Chat Completions REST API."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AI-University-ResearchDiscovery/1.0",
        }
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        res = requests.post(
            GROQ_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        if res.status_code != 200:
            raise RuntimeError(f"Groq API returned HTTP {res.status_code}: {res.text[:200]}")
        data = res.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content, self.groq_model

    def _call_gemini(self, prompt: str, system_prompt: str, max_tokens: int) -> tuple[str, str]:
        """Invoke Google Gemini REST API."""
        url = GEMINI_ENDPOINT_TEMPLATE.format(model=self.gemini_model)
        headers = {"Content-Type": "application/json"}
        # Support both AIza... API keys and OAuth/Bearer tokens
        if self.gemini_api_key.startswith("AIza"):
            url = f"{url}?key={self.gemini_api_key}"
        else:
            headers["Authorization"] = f"Bearer {self.gemini_api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n{prompt}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": max_tokens,
            },
        }
        res = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        if res.status_code != 200:
            raise RuntimeError(f"Gemini API returned HTTP {res.status_code}: {res.text[:200]}")
        data = res.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        content = "\n".join(text_parts).strip()
        if not content:
            raise RuntimeError("Gemini candidate had empty text response.")
        return content, self.gemini_model

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are an academic AI Research Assistant for the AI University research discovery platform.\n"
            "Your task is to synthesize a grounded, scholarly answer to the user's research query strictly "
            "based on the retrieved OpenAlex publications provided in the context.\n\n"
            "STRICT CITATION AND FACTUAL GROUNDING RULES:\n"
            "1. Ground all claims directly in the provided papers. Do not fabricate citations or facts.\n"
            "2. Always cite evidence using explicit inline tags: [Paper #k · FirstAuthor et al., Year] "
            "or [Paper #k · DOI/Title].\n"
            "3. Format your response into clear Markdown sections:\n"
            "   - **Executive Synthesis & Direct Answer**: Concise direct synthesis addressing the query.\n"
            "   - **Key Methodological Findings**: Specific techniques, datasets, or theoretical results extracted from the papers.\n"
            "   - **Candidate Researcher Relevance**: How the matched authors and their publications connect to the query topic.\n"
            "4. If a part of the query is not addressed by the provided abstracts, state: 'The retrieved publications do not provide explicit evidence regarding [X].'\n"
            "5. Keep the tone academic, clear, objective, and analytical."
        )

    @staticmethod
    def _build_user_prompt(query: str, papers: Sequence[CitationSource]) -> str:
        lines = [f"RESEARCH QUERY: {query}\n", "RETRIEVED RESEARCH PUBLICATIONS (OPENALEX CORPUS):"]
        for p in papers:
            authors_str = ", ".join(p.authors[:4]) if p.authors else "Unknown Authors"
            doi_str = f" · DOI: {p.doi}" if p.doi else ""
            lines.append(
                f"\n[Paper #{p.paper_index}] \"{p.title}\" ({p.year}){doi_str}\n"
                f"Authors: {authors_str}\n"
                f"Similarity Score: {p.similarity:.4f}\n"
                f"Abstract: {p.abstract if p.abstract else 'No abstract text available in dataset.'}"
            )
        lines.append("\nPlease synthesize a grounded scholarly answer following the system citation guidelines.")
        return "\n".join(lines)

    @staticmethod
    def _extractive_synthesis(
        query: str,
        papers: Sequence[CitationSource],
        fallback_reason: str,
    ) -> str:
        """Produce a structured, grounded synthesis extracted directly from retrieved publications."""
        lead = papers[0]
        lead_authors = ", ".join(lead.authors[:3]) if lead.authors else "Leading Authors"
        findings: list[str] = []

        for p in papers[:4]:
            text = (p.abstract or p.title).strip()
            # Split into sentences and find sentences relevant to query terms
            sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if len(s.strip()) > 20]
            chosen_snippet = sentences[0] if sentences else text
            # Prefer sentence containing query terms
            query_terms = set(re.findall(r"\w+", query.lower()))
            for s in sentences:
                if any(t in s.lower() for t in query_terms if len(t) > 3):
                    chosen_snippet = s
                    break

            doi_suffix = f" (DOI: {p.doi})" if p.doi else ""
            author_tag = p.authors[0] if p.authors else "Unknown"
            findings.append(
                f"- **[Paper #{p.paper_index} · {author_tag} et al., {p.year}]** "
                f"“{p.title}”{doi_suffix}: {chosen_snippet}"
            )

        findings_block = "\n".join(findings)
        mentors: list[str] = []
        for p in papers[:3]:
            if p.authors:
                first_author = p.authors[0]
                mentors.append(f"`{first_author}` via *“{p.title}”* (score: {p.similarity:.4f})")

        mentors_block = ", ".join(mentors) if mentors else f"`{lead_authors}`"

        return (
            f"### Grounded Research Synthesis\n\n"
            f"Based on **{len(papers)} retrieved OpenAlex publications** relevant to *“{query}”*, "
            f"the core literature centers on **[Paper #1 · {lead_authors}, {lead.year}]** (*“{lead.title}”*).\n\n"
            f"#### Key Methodological Findings\n"
            f"{findings_block}\n\n"
            f"#### Candidate Mentors & Subject-Matter Experts\n"
            f"The primary indexed researchers demonstrating direct publication evidence for this topic include {mentors_block}.\n\n"
            f"> *Note: Generated using local grounded extractive synthesis ({fallback_reason}). "
            f"All claims map directly to the retrieved OpenAlex dataset records above.*"
        )


_SHARED_REASONING_ENGINE: GroundedReasoningEngine | None = None


def get_shared_reasoning_engine() -> GroundedReasoningEngine:
    """Return singleton GroundedReasoningEngine instance."""
    global _SHARED_REASONING_ENGINE
    if _SHARED_REASONING_ENGINE is None:
        _SHARED_REASONING_ENGINE = GroundedReasoningEngine()
    return _SHARED_REASONING_ENGINE


"""Laya System-1 Decision & Classification Engine for research publications and queries.

Integrates Convai Innovations' non-autoregressive System-1 decision model
(`convaiinnovations/laya`, `subfolder="typed-decisions"`) across its three
calibrated decision primitives:

1. ``choice`` — Classifies each publication's primary research methodology
   (`empirical`, `theoretical`, `survey`, `resource`, `systems`) and disambiguates
   query sub-field intent.
2. ``score``  — Evaluates `(query, paper)` relevance on the 4-level (`0–3`)
   graded rubric established in Week 5 Fair Evaluation.
3. ``noul``   — Computes $P(\\text{substantive scholarly publication})$ to catch
   retractions, errata, editorials, and bare repository stubs.

Architecture & Resilience:
- When the `convaiinnovations/laya` checkpoint weights are present locally (or
  downloaded explicitly via ``warmup_neural(allow_download=True)``), executes
  single-pass batched inference via ``Agent.predict_batch``.
- When running offline, in fast synchronous UI paths, or before neural weights
  finish downloading, executes a temperature-calibrated System-1 local model
  using `laya.confidence` (`apply_confidence_gate`, `answer_confidence_value`)
  with the fitted option-bucket temperatures from `laya-typed-decisions`.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.week6.filters import is_non_research_artifact
from src.week6.schemas import (
    PAPER_TYPE_LABELS,
    PAPER_TYPES,
    RELEVANCE_GRADE_LABELS,
    ClassifyPaperInput,
    PaperClassification,
    QueryClassification,
)

try:
    from laya.confidence import answer_confidence_value, apply_confidence_gate
except Exception:  # noqa: BLE001
    answer_confidence_value = None  # type: ignore[assignment]
    apply_confidence_gate = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_LAYA_MODEL_ID = "convaiinnovations/laya"
DEFAULT_LAYA_SUBFOLDER = "typed-decisions"

# Fitted option-bucket temperatures from convaiinnovations/laya (typed-decisions/rl_agent_config.json)
TEMP_CHOICE_3_5 = 1.7601518630981445
TEMP_CHOICE_6_10 = 1.0000158548355103
TEMP_SCORE_3_5 = 1.2514300346374512
TEMP_NOUL_2 = 1.983399510383606

DOMAIN_LABELS: dict[str, str] = {
    "nlp": "NLP & Language Models",
    "ml": "Machine Learning & Deep Learning",
    "power": "Electrical Power & Energy Systems",
    "biomedical": "Biomedical & Healthcare",
    "systems": "Hardware, VLSI & Systems",
    "quantum": "Quantum & Physical Sciences",
}

PAPER_TYPE_CRITERIA: dict[str, str] = {
    "empirical": "Empirical or experimental study with quantitative evaluation on datasets or benchmarks",
    "theoretical": "Theoretical analysis, mathematical proofs, formal bounds, or analytical modeling",
    "survey": "Literature review, taxonomy, meta-analysis, or comprehensive survey of prior work",
    "resource": "New dataset, annotated corpus, benchmark suite, or shared evaluation resource",
    "systems": "Applied engineering system, hardware/VLSI design, software architecture, or clinical deployment",
}

RELEVANCE_CRITERIA: list[str] = [
    "Off-topic or unrelated research domain",
    "Tangentially related background or peripheral concept",
    "Same sub-field with partial methodological or topical overlap",
    "Direct methodological and topical match to the research query",
]

QUERY_QUESTIONS: dict[str, dict[str, Any]] = {
    "domain": {
        "type": "choice",
        "instructions": "Which primary research domain best characterizes this academic query?",
        "criteria": {
            "nlp": "Natural Language Processing, Large Language Models, Machine Translation, Speech, Information Retrieval",
            "ml": "Machine Learning, Deep Learning, Computer Vision, Graph Neural Networks, Reinforcement Learning",
            "power": "Electrical Power Systems, High Voltage Transformers, Microgrids, Battery & Energy Storage",
            "biomedical": "Biomedical Informatics, Healthcare, Clinical AI, Drug Discovery, Genomics",
            "systems": "Hardware, VLSI, Parallel Computing, Embedded Systems, Wireless & IoT Networks",
            "quantum": "Quantum Computing, Photonics, Condensed Matter, Physical Sciences",
        },
    },
    "is_analytical": {
        "type": "noul",
        "instructions": "Is this query asking an analytical, comparative, causal, or multi-paper synthesis question rather than a short keyword topic?",
    },
}

_METHODOLOGY_CUES: dict[str, tuple[tuple[str, float], ...]] = {
    "survey": (
        ("survey", 3.6),
        ("literature review", 3.8),
        ("systematic review", 3.9),
        ("comprehensive review", 3.6),
        ("state of the art", 2.4),
        ("state-of-the-art review", 3.4),
        ("taxonomy", 2.9),
        ("meta-analysis", 3.5),
        ("bibliometric", 3.2),
        ("overview of", 2.5),
        ("a review", 3.1),
        ("recent advances in", 2.3),
    ),
    "theoretical": (
        ("theorem", 3.4),
        ("mathematical", 2.4),
        ("convergence bound", 3.6),
        ("minimax", 3.4),
        ("asymptotic", 3.1),
        ("closed-form", 3.0),
        ("closed form", 3.0),
        ("theoretical", 3.2),
        ("provably", 3.1),
        ("lower bound", 2.8),
        ("upper bound", 2.8),
        ("exact solution", 2.6),
        ("formal verification", 2.7),
        ("convexity", 2.8),
        ("lyapunov", 3.2),
        ("quantum entanglement", 2.1),
    ),
    "resource": (
        ("new dataset", 3.7),
        ("benchmark dataset", 3.7),
        ("benchmark and dataset", 3.7),
        ("evaluation benchmark", 3.5),
        ("annotated corpus", 3.8),
        ("parallel corpus", 3.6),
        ("shared task", 3.3),
        ("treebank", 3.6),
        ("knowledge base construction", 3.0),
        ("corpus of", 3.1),
        ("benchmark for", 3.0),
        ("benchmark", 2.6),
        ("dataset", 2.5),
        ("corpus", 2.5),
        ("publicly available dataset", 3.5),
        ("annotation scheme", 3.2),
        ("gold-standard", 2.6),
    ),
    "systems": (
        ("fpga", 3.5),
        ("vlsi", 3.5),
        ("asic", 3.5),
        ("hardware accelerator", 3.6),
        ("microcontroller", 3.2),
        ("iot", 2.5),
        ("embedded system", 3.3),
        ("real-time implementation", 3.1),
        ("architecture for", 2.2),
        ("microgrid", 2.8),
        ("power transformer", 2.9),
        ("hvdc", 3.0),
        ("inverter", 2.9),
        ("compiler", 2.9),
        ("distributed system", 2.9),
        ("prototype", 2.4),
        ("software framework", 2.7),
        ("cloud deployment", 2.8),
    ),
    "empirical": (
        ("experimental", 2.3),
        ("experiments", 2.3),
        ("empirical", 2.8),
        ("outperforms", 2.5),
        ("ablation", 2.9),
        ("fine-tuning", 2.4),
        ("fine-tuned", 2.4),
        ("accuracy", 2.0),
        ("f1-score", 2.5),
        ("f1 score", 2.5),
        ("evaluated on", 2.6),
        ("we evaluate", 2.6),
        ("classification", 1.9),
        ("clinical trial", 3.0),
        ("cohort", 2.6),
    ),
}

_DOMAIN_CUES: dict[str, tuple[str, ...]] = {
    "nlp": (
        "natural language",
        "nlp",
        "language model",
        "llm",
        "translation",
        "speech",
        "sentiment",
        "tokenization",
        "named entity",
        "summarization",
        "question answering",
        "information retrieval",
        "information extraction",
        "multilingual",
        "cross-lingual",
        "text classification",
    ),
    "ml": (
        "machine learning",
        "deep learning",
        "neural network",
        "computer vision",
        "reinforcement learning",
        "graph neural",
        "knowledge graph",
        "representation learning",
        "contrastive learning",
        "diffusion model",
    ),
    "power": (
        "power transformer",
        "power system",
        "high voltage",
        "hvdc",
        "microgrid",
        "battery",
        "energy storage",
        "photovoltaic",
        "dielectric",
        "insulation",
        "fault diagnosis",
    ),
    "biomedical": (
        "biomedical",
        "clinical",
        "healthcare",
        "medical",
        "drug",
        "disease",
        "tuberculosis",
        "genom",
        "protein",
        "patient",
    ),
    "systems": (
        "vlsi",
        "fpga",
        "hardware",
        "embedded",
        "parallel computing",
        "wireless",
        "iot",
        "semiconductor",
        "network security",
    ),
    "quantum": (
        "quantum",
        "photon",
        "optics",
        "condensed matter",
        "magnetic",
        "superconduct",
        "qubit",
    ),
}

_QUESTION_STARTERS = re.compile(
    r"^\s*(?:what|which|who|how|why|compare|explain|summarize|analyze|analyse|can|are|is|does|do)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,}")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "using", "based",
        "into", "over", "under", "between", "study", "paper", "approach", "method",
        "analysis", "model", "models", "research", "application", "applications",
    }
)


def build_paper_questions(*, include_relevance: bool = True) -> dict[str, dict[str, Any]]:
    """Construct the typed Laya System-1 question dictionary for research papers."""
    questions: dict[str, dict[str, Any]] = {
        "paper_type": {
            "type": "choice",
            "instructions": "What is the primary scholarly research methodology of this publication?",
            "criteria": dict(PAPER_TYPE_CRITERIA),
        },
        "is_scholarly": {
            "type": "noul",
            "instructions": (
                "Is this a substantive peer-reviewed research publication rather than a retraction, "
                "erratum, corrigendum, editorial note, or bare repository stub?"
            ),
        },
    }
    if include_relevance:
        questions["relevance_grade"] = {
            "type": "score",
            "instructions": "How directly does this publication address the given research query?",
            "criteria": list(RELEVANCE_CRITERIA),
        }
    return questions


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    temp = max(float(temperature), 1e-4)
    scaled = [float(x) / temp for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def _entropy_confidence(probs: Sequence[float]) -> float:
    k = len(probs)
    if k <= 1:
        return 1.0
    ent = -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0.0)
    return max(0.0, min(1.0, 1.0 - ent / math.log(k)))


def _has_complete_local_laya_snapshot(model_id: str, subfolder: str | None) -> bool:
    """Return True when HuggingFace cache holds a complete Laya checkpoint without `.incomplete` blobs."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.exists():
        return False
    blobs_dir = repo_dir / "blobs"
    if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
        return False
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return False
    for snap in snapshots_dir.iterdir():
        target = snap / subfolder if subfolder else snap
        if (target / "rl_agent_config.json").exists() and (
            any(target.glob("*.safetensors")) or any(target.glob("*.bin"))
        ):
            return True
    return False


class LayaPaperClassifier:
    """System-1 paper and query classifier powered by Laya (`convaiinnovations/laya`)."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        subfolder: str | None = DEFAULT_LAYA_SUBFOLDER,
        device: str | None = None,
        prefer_neural: bool = True,
        cache_size: int = 2048,
    ) -> None:
        self.model_id = model_id or os.environ.get("LAYA_MODEL_ID", DEFAULT_LAYA_MODEL_ID)
        env_sub = os.environ.get("LAYA_SUBFOLDER")
        self.subfolder = env_sub if env_sub is not None else subfolder
        self.device = device
        mode_env = os.environ.get("LAYA_BACKEND", "auto").strip().lower()
        self.prefer_neural = False if mode_env == "local" else (True if mode_env == "neural" else prefer_neural)
        self._agent: Any = None
        self._load_attempted = False
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, ...], PaperClassification] = OrderedDict()
        self._cache_size = max(64, int(cache_size))

    @property
    def backend_name(self) -> str:
        return "laya-typed-decisions" if self._agent is not None else "laya-calibrated-local"

    def status(self) -> dict[str, Any]:
        """Return diagnostic metadata for `/api/v1/health` and `/api/v1/info`."""
        laya_pkg_installed = answer_confidence_value is not None
        return {
            "available": True,
            "laya_package_installed": laya_pkg_installed,
            "neural_loaded": self._agent is not None,
            "active_backend": self.backend_name,
            "model_id": self.model_id,
            "subfolder": self.subfolder,
            "primitives": ["choice", "score", "noul"],
            "paper_types": list(PAPER_TYPES),
            "note": self._load_error,
        }

    def warmup_neural(self, *, allow_download: bool = False) -> bool:
        """Attempt to load the neural `convaiinnovations/laya` checkpoint."""
        with self._lock:
            if self._agent is not None:
                return True
            if self._load_attempted and not allow_download:
                return False
            self._load_attempted = True

            offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("LAYA_OFFLINE") == "1"
            if not allow_download and (offline or not _has_complete_local_laya_snapshot(self.model_id, self.subfolder)):
                self._load_error = (
                    "Neural checkpoint weights not fully cached locally; using Laya temperature-calibrated System-1 engine."
                )
                return False

            try:
                import laya  # type: ignore[import-untyped]

                self._agent = laya.load(
                    self.model_id,
                    subfolder=self.subfolder,
                    device=self.device,
                    compile=False,
                )
                self._load_error = None
                logger.info("Loaded neural Laya model (%s / %s).", self.model_id, self.subfolder)
                return True
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                logger.warning("Falling back to Laya calibrated local classifier: %s", exc)
                return False

    def classify_query(self, query: str) -> QueryClassification:
        """Classify query domain (`choice`) and analytical question intent (`noul`)."""
        clean_query = " ".join(str(query or "").strip().split())
        if self.prefer_neural and self._agent is None and not self._load_attempted:
            self.warmup_neural(allow_download=False)

        if self._agent is not None and clean_query:
            try:
                res = self._agent.predict({"query": clean_query}, QUERY_QUESTIONS)
                answers = res.get("answers", {})
                dom_ans = answers.get("domain", {})
                ana_ans = answers.get("is_analytical", {})
                dom_key = str(dom_ans.get("choice", "nlp"))
                probs = {k: float(v) for k, v in (dom_ans.get("probabilities") or {}).items()}
                dom_conf = float(dom_ans.get("answer_confidence", max(probs.values(), default=0.5)))
                ana_prob = float(ana_ans.get("noul", 0.5))
                return QueryClassification(
                    query=clean_query,
                    primary_domain=dom_key,
                    primary_domain_label=DOMAIN_LABELS.get(dom_key, dom_key),
                    domain_confidence=dom_conf,
                    domain_probabilities=probs,
                    is_analytical_question=ana_prob >= 0.5,
                    analytical_probability=ana_prob,
                    backend="laya-typed-decisions",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Neural Laya query classification failed, using calibrated fallback: %s", exc)

        return self._classify_query_calibrated(clean_query)

    def classify_papers(
        self,
        papers: Sequence[ClassifyPaperInput | Mapping[str, Any]],
        *,
        query: str = "",
        min_confidence: float = 0.35,
    ) -> list[PaperClassification]:
        """Classify a batch of papers using Laya `choice`, `score`, and `noul` primitives."""
        if not papers:
            return []

        normalized_inputs: list[ClassifyPaperInput] = [
            p if isinstance(p, ClassifyPaperInput) else ClassifyPaperInput.from_dict(p, idx)
            for idx, p in enumerate(papers)
        ]
        clean_query = " ".join(str(query or "").strip().split())
        include_relevance = bool(clean_query)

        if self.prefer_neural and self._agent is None and not self._load_attempted:
            self.warmup_neural(allow_download=False)

        results: list[PaperClassification | None] = [None] * len(normalized_inputs)
        uncached_indices: list[int] = []
        cache_keys: list[tuple[str, ...]] = []

        for idx, item in enumerate(normalized_inputs):
            key = (
                self.backend_name,
                item.paper_id,
                item.title.lower(),
                item.abstract[:160].lower(),
                clean_query.lower(),
                f"{min_confidence:.2f}",
                f"{item.similarity:.3f}" if item.similarity is not None else "",
            )
            cache_keys.append(key)
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    results[idx] = cached
                else:
                    uncached_indices.append(idx)

        if uncached_indices:
            to_score = [normalized_inputs[i] for i in uncached_indices]
            computed: list[PaperClassification] | None = None
            if self._agent is not None:
                try:
                    computed = self._predict_batch_neural(
                        to_score,
                        query=clean_query,
                        include_relevance=include_relevance,
                        min_confidence=min_confidence,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Neural Laya batch prediction failed, using calibrated fallback: %s", exc)

            if computed is None:
                computed = self._predict_batch_calibrated(
                    to_score,
                    query=clean_query,
                    include_relevance=include_relevance,
                    min_confidence=min_confidence,
                )

            with self._lock:
                for local_pos, orig_idx in enumerate(uncached_indices):
                    cls_obj = computed[local_pos]
                    results[orig_idx] = cls_obj
                    ckey = cache_keys[orig_idx]
                    self._cache[ckey] = cls_obj
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)

        return [r for r in results if r is not None]

    def _predict_batch_neural(
        self,
        items: Sequence[ClassifyPaperInput],
        *,
        query: str,
        include_relevance: bool,
        min_confidence: float,
    ) -> list[PaperClassification]:
        questions = build_paper_questions(include_relevance=include_relevance)
        states = []
        for item in items:
            state: dict[str, Any] = {"title": item.title}
            if item.abstract:
                state["abstract"] = item.abstract[:900]
            if item.topics:
                state["topics"] = item.topics[:240]
            if include_relevance:
                state["query"] = query
            states.append(state)

        batch_out = self._agent.predict_batch(
            states,
            questions,
            batch_size=16,
            min_confidence=min_confidence if min_confidence > 0 else None,
        )
        return [
            self._parse_answer_dict(item, raw_res, include_relevance=include_relevance, backend="laya-typed-decisions")
            for item, raw_res in zip(items, batch_out, strict=True)
        ]

    def _predict_batch_calibrated(
        self,
        items: Sequence[ClassifyPaperInput],
        *,
        query: str,
        include_relevance: bool,
        min_confidence: float,
    ) -> list[PaperClassification]:
        query_tokens = {
            tok for tok in _WORD_RE.findall(query.lower()) if tok not in _STOPWORDS
        }
        raw_results: list[dict[str, Any]] = []

        for item in items:
            haystack = " ".join((item.title, item.abstract, item.topics)).lower()
            title_lower = item.title.lower()

            # 1. `choice`: research methodology logits -> temperature-calibrated probabilities
            # Slight empirical baseline prior because ~60% of OpenAlex STEM/CS works are empirical.
            logits_map: dict[str, float] = {
                "empirical": 1.15,
                "theoretical": 0.45,
                "survey": 0.25,
                "resource": 0.30,
                "systems": 0.55,
            }
            for ptype, cues in _METHODOLOGY_CUES.items():
                for phrase, weight in cues:
                    if phrase in title_lower:
                        logits_map[ptype] += weight * 1.35
                    elif phrase in haystack:
                        logits_map[ptype] += weight * 0.85

            ordered_logits = [logits_map[pt] for pt in PAPER_TYPES]
            type_probs_list = _softmax(ordered_logits, temperature=TEMP_CHOICE_3_5)
            type_probs = {pt: round(prob, 4) for pt, prob in zip(PAPER_TYPES, type_probs_list, strict=True)}
            best_idx = max(range(len(PAPER_TYPES)), key=type_probs_list.__getitem__)
            chosen_type = PAPER_TYPES[best_idx]
            type_ans_conf = round(max(type_probs_list), 4)

            # 2. `noul`: substantive scholarly publication probability
            non_research = is_non_research_artifact(item.title)
            editorial_cue = any(
                marker in title_lower
                for marker in ("book review", "guest editorial", "preface", "foreword", "in memoriam", "corrigendum")
            )
            if non_research:
                noul_logits = [3.6, -1.8]
            elif editorial_cue:
                noul_logits = [1.8, -0.4]
            else:
                has_abstract_bonus = 0.75 if len(item.abstract.strip()) >= 80 else 0.0
                noul_logits = [-2.2, 2.4 + has_abstract_bonus]
            noul_probs = _softmax(noul_logits, temperature=TEMP_NOUL_2)
            p_true = round(noul_probs[1], 4)

            answers: dict[str, Any] = {
                "paper_type": {
                    "type": "choice",
                    "choice": chosen_type,
                    "probabilities": type_probs,
                    "confidence": round(_entropy_confidence(type_probs_list), 4),
                    "answer_confidence": type_ans_conf,
                },
                "is_scholarly": {
                    "type": "noul",
                    "noul": p_true,
                    "confidence": round(max(p_true, 1.0 - p_true), 4),
                    "answer_confidence": round(max(p_true, 1.0 - p_true), 4),
                },
            }

            # 3. `score`: 0–3 graded relevance rubric
            if include_relevance:
                paper_tokens = {tok for tok in _WORD_RE.findall(haystack) if tok not in _STOPWORDS}
                overlap = (
                    len(query_tokens & paper_tokens) / max(len(query_tokens), 1)
                    if query_tokens
                    else 0.0
                )
                sim = float(item.similarity) if item.similarity is not None else overlap
                signal = max(sim, overlap * 0.85)
                if non_research:
                    score_logits = [3.2, 0.4, -1.2, -2.0]
                elif signal >= 0.56:
                    score_logits = [-1.6, -0.3, 1.4, 2.9 + (signal - 0.56) * 3.0]
                elif signal >= 0.36:
                    score_logits = [-0.9, 0.6, 2.4, 1.3 + (signal - 0.36) * 2.5]
                elif signal >= 0.18:
                    score_logits = [0.3, 2.1, 1.1, -0.4]
                else:
                    score_logits = [2.5, 0.9, -0.6, -1.6]

                s_probs = _softmax(score_logits, temperature=TEMP_SCORE_3_5)
                exp_score = sum(idx * p for idx, p in enumerate(s_probs))
                answers["relevance_grade"] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "probabilities": {str(i): round(p, 4) for i, p in enumerate(s_probs)},
                    "confidence": round(_entropy_confidence(s_probs), 4),
                    "answer_confidence": round(max(s_probs), 4),
                }

            raw_results.append({"model": "laya-calibrated-local", "answers": answers})

        if min_confidence > 0.0:
            if apply_confidence_gate is not None:
                apply_confidence_gate(raw_results, min_confidence=min_confidence)
            else:
                for res in raw_results:
                    for ans in res["answers"].values():
                        ans["abstention"] = "abstained" if ans["answer_confidence"] < min_confidence else "passed"

        return [
            self._parse_answer_dict(
                item,
                raw_res,
                include_relevance=include_relevance,
                backend="laya-calibrated-local",
            )
            for item, raw_res in zip(items, raw_results, strict=True)
        ]

    @staticmethod
    def _parse_answer_dict(
        item: ClassifyPaperInput,
        raw_res: Mapping[str, Any],
        *,
        include_relevance: bool,
        backend: str,
    ) -> PaperClassification:
        answers = raw_res.get("answers", {})
        pt_ans = answers.get("paper_type", {})
        sc_ans = answers.get("is_scholarly", {})

        ptype = str(pt_ans.get("choice", "empirical"))
        if ptype not in PAPER_TYPE_LABELS:
            ptype = "empirical"
        probs = {str(k): float(v) for k, v in (pt_ans.get("probabilities") or {}).items()}
        if answer_confidence_value is not None:
            t_conf = answer_confidence_value(pt_ans)
            type_conf = t_conf if t_conf is not None else max(probs.values(), default=0.5)
        else:
            type_conf = float(pt_ans.get("answer_confidence", max(probs.values(), default=0.5)))

        abstained = pt_ans.get("abstention") == "abstained" or bool(pt_ans.get("low_confidence"))
        p_scholarly = float(sc_ans.get("noul", 0.85))

        rel_score: float | None = None
        rel_grade: int | None = None
        rel_label = ""
        if include_relevance and "relevance_grade" in answers:
            rg_ans = answers["relevance_grade"]
            rel_score = float(rg_ans.get("score", 0.0))
            rel_grade = max(0, min(3, int(round(rel_score))))
            rel_label = RELEVANCE_GRADE_LABELS[rel_grade]

        return PaperClassification(
            paper_id=item.paper_id or "P1",
            title=item.title,
            paper_type=ptype,
            paper_type_label=PAPER_TYPE_LABELS[ptype],
            type_confidence=type_conf,
            type_probabilities=probs,
            is_scholarly=p_scholarly >= 0.5,
            scholarly_probability=p_scholarly,
            relevance_score=rel_score,
            relevance_grade=rel_grade,
            relevance_label=rel_label,
            abstained=abstained,
            backend=backend,
        )

    @staticmethod
    def _classify_query_calibrated(query: str) -> QueryClassification:
        q_lower = query.lower()
        dom_keys = list(DOMAIN_LABELS.keys())
        logits: dict[str, float] = {k: 0.4 for k in dom_keys}
        logits["nlp"] = 0.65  # slight corpus prior for NLP/IR queries when ambiguous
        for dom, cues in _DOMAIN_CUES.items():
            for phrase in cues:
                if phrase in q_lower:
                    logits[dom] += 2.4

        probs_list = _softmax([logits[k] for k in dom_keys], temperature=TEMP_CHOICE_6_10)
        probs = {k: round(p, 4) for k, p in zip(dom_keys, probs_list, strict=True)}
        best_idx = max(range(len(dom_keys)), key=probs_list.__getitem__)
        best_dom = dom_keys[best_idx]

        word_count = len(query.split())
        has_qmark = "?" in query
        starts_question = bool(_QUESTION_STARTERS.search(query))
        analytical_logit = (2.2 if has_qmark else 0.0) + (1.8 if starts_question else 0.0) + (0.9 if word_count >= 9 else -1.4)
        ana_probs = _softmax([-0.6, analytical_logit], temperature=TEMP_NOUL_2)
        p_ana = round(ana_probs[1], 4)

        return QueryClassification(
            query=query,
            primary_domain=best_dom,
            primary_domain_label=DOMAIN_LABELS[best_dom],
            domain_confidence=round(max(probs_list), 4),
            domain_probabilities=probs,
            is_analytical_question=p_ana >= 0.5,
            analytical_probability=p_ana,
            backend="laya-calibrated-local",
        )


_SHARED_CLASSIFIER: LayaPaperClassifier | None = None
_SHARED_LOCK = threading.Lock()


def get_shared_laya_classifier() -> LayaPaperClassifier:
    """Return a process-wide singleton LayaPaperClassifier instance."""
    global _SHARED_CLASSIFIER
    if _SHARED_CLASSIFIER is None:
        with _SHARED_LOCK:
            if _SHARED_CLASSIFIER is None:
                _SHARED_CLASSIFIER = LayaPaperClassifier()
    return _SHARED_CLASSIFIER


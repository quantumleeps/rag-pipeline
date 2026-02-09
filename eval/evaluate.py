"""RAGAS evaluation harness for the 3×3 chunking × embedding matrix."""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, cast

from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from ragas import EvaluationDataset, SingleTurnSample
from ragas.dataset_schema import SingleTurnSampleOrMultiTurnSample
from ragas.integrations.llama_index import evaluate as ragas_evaluate
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

from rag_pipeline.chunkers import ChunkStrategy
from rag_pipeline.embed import EmbedModelName
from rag_pipeline.query import get_query_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RESULTS_FILE = Path(__file__).parent / "results.json"

EVAL_QUESTIONS = [
    {
        "user_input": ("What is the maximum contaminant level (MCL) for bromate?"),
        "reference": "The MCL for bromate is 0.010 mg/L.",
    },
    {
        "user_input": (
            "What are the best available technologies"
            " (BAT) for PFAS removal in drinking water?"
        ),
        "reference": (
            "BAT for PFAS includes granular activated"
            " carbon (GAC), anion exchange resins, and"
            " high-pressure membranes such as"
            " nanofiltration and reverse osmosis."
        ),
    },
    {
        "user_input": (
            "What CT value is required for 3-log"
            " inactivation of Giardia using ozone"
            " at 10\u00b0C?"
        ),
        "reference": (
            "The CT value for 3-log Giardia"
            " inactivation with ozone at 10\u00b0C is"
            " approximately 0.72 mg\u00b7min/L."
        ),
    },
    {
        "user_input": (
            "What is the Safe Drinking Water Act (SDWA) and what does it regulate?"
        ),
        "reference": (
            "The SDWA is the federal law that protects"
            " public drinking water supplies by"
            " authorizing EPA to set national"
            " health-based standards for contaminants"
            " in drinking water."
        ),
    },
    {
        "user_input": (
            "What disinfection byproducts are regulated under the Stage 1 DBPR?"
        ),
        "reference": (
            "The Stage 1 DBPR regulates total"
            " trihalomethanes (TTHM), haloacetic acids"
            " (HAA5), bromate, and chlorite."
        ),
    },
    {
        "user_input": ("How does a sequencing batch reactor (SBR) treat wastewater?"),
        "reference": (
            "An SBR treats wastewater in a single tank"
            " through sequential phases: fill, react"
            " (aeration), settle, decant, and idle."
        ),
    },
    {
        "user_input": (
            "What is the purpose of disinfection"
            " profiling and benchmarking under"
            " the LT1ESWTR?"
        ),
        "reference": (
            "Disinfection profiling characterizes a"
            " system's existing disinfection practice"
            " to ensure that any changes maintain"
            " adequate microbial inactivation."
        ),
    },
    {
        "user_input": (
            "What are the primary mechanisms by which ozone disinfects water?"
        ),
        "reference": (
            "Ozone disinfects through direct oxidation"
            " by molecular ozone and indirect oxidation"
            " by hydroxyl radicals produced during"
            " ozone decomposition."
        ),
    },
]


def build_eval_dataset() -> EvaluationDataset:
    samples: list[SingleTurnSampleOrMultiTurnSample] = [
        SingleTurnSample(
            user_input=q["user_input"],
            reference=q["reference"],
        )
        for q in EVAL_QUESTIONS
    ]
    return EvaluationDataset(samples=samples)


def _sanitize_for_json(val: Any) -> float | None:
    """Convert numpy floats and NaN to JSON-safe types."""
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _load_existing_results() -> dict[str, Any]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def _save_results(results: dict[str, Any]) -> None:
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")
    log.info("Results written to %s", RESULTS_FILE)


def run_evaluation(*, resume: bool = True) -> dict[str, Any]:
    existing = _load_existing_results() if resume else {}

    evaluator_llm = Anthropic(model="claude-haiku-4-5-20251001")
    eval_embeddings = VoyageEmbedding(model_name="voyage-3.5")
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        LLMContextRecall(),
    ]

    run_config = RunConfig(max_workers=2, timeout=300, max_retries=15)

    results: dict[str, Any] = dict(existing)
    for strategy in ChunkStrategy:
        for model in EmbedModelName:
            variant = f"{strategy.value}_{model.value}"

            if variant in results:
                log.info("Skipping %s (already in results file)", variant)
                continue

            log.info("Evaluating %s", variant)

            dataset = build_eval_dataset()
            query_engine = get_query_engine(strategy, model)
            result = ragas_evaluate(
                query_engine=query_engine,
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=eval_embeddings,
                run_config=run_config,
            )

            per_sample = {}
            scores = {}
            for m in metrics:
                raw = [_sanitize_for_json(v) for v in result[m.name]]
                per_sample[m.name] = raw
                valid = [v for v in raw if v is not None]
                scores[m.name] = sum(valid) / len(valid) if valid else None

            samples = []
            for sample in dataset.samples:
                s = cast(SingleTurnSample, sample)
                samples.append(
                    {
                        "user_input": s.user_input,
                        "reference": s.reference,
                        "response": s.response,
                        "retrieved_contexts": s.retrieved_contexts,
                    }
                )

            results[variant] = {
                "scores": scores,
                "per_sample": per_sample,
                "samples": samples,
            }
            log.info("  %s", scores)

            _save_results(results)

    return results


def print_results(results: dict[str, Any]) -> None:
    header = [
        "variant",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "llm_context_recall",
    ]
    print("\t".join(header))
    for variant, data in results.items():
        scores = data.get("scores", data)
        row = [variant] + [
            f"{scores.get(h, 0):.3f}" if scores.get(h) is not None else "NaN"
            for h in header[1:]
        ]
        print("\t".join(row))


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    results = run_evaluation(resume=not fresh)
    print("\n--- Results ---")
    print_results(results)

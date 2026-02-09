"""RAGAS evaluation harness for the 3×3 chunking × embedding matrix."""

import argparse
import json
import logging
import math
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
            " approximately 1.43 mg\u00b7min/L."
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


def build_eval_dataset(
    question_indices: list[int] | None = None,
) -> EvaluationDataset:
    questions = EVAL_QUESTIONS
    if question_indices is not None:
        questions = [EVAL_QUESTIONS[i] for i in question_indices]
    samples: list[SingleTurnSampleOrMultiTurnSample] = [
        SingleTurnSample(
            user_input=q["user_input"],
            reference=q["reference"],
        )
        for q in questions
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


def _recalculate_scores(
    per_sample: dict[str, list[float | None]],
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for metric, values in per_sample.items():
        valid = [v for v in values if v is not None]
        scores[metric] = sum(valid) / len(valid) if valid else None
    return scores


def run_evaluation(
    *,
    resume: bool = True,
    question_indices: list[int] | None = None,
) -> dict[str, Any]:
    existing = _load_existing_results() if resume or question_indices else {}

    if question_indices is not None and not existing:
        log.error(
            "Cannot use --questions without existing results.json to patch into. "
            "Run a full evaluation first."
        )
        raise SystemExit(1)

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

            if question_indices is None and variant in results:
                log.info("Skipping %s (already in results file)", variant)
                continue

            if question_indices is not None and variant not in results:
                log.warning("Skipping %s (not in results file)", variant)
                continue

            qi_label = f" (questions {question_indices})" if question_indices else ""
            log.info("Evaluating %s%s", variant, qi_label)

            dataset = build_eval_dataset(question_indices)
            query_engine = get_query_engine(strategy, model)
            result = ragas_evaluate(
                query_engine=query_engine,
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=eval_embeddings,
                run_config=run_config,
            )

            if question_indices is not None:
                # Patch mode: merge new scores into existing results
                variant_data = results[variant]
                for m in metrics:
                    new_scores = [_sanitize_for_json(v) for v in result[m.name]]
                    for k, qi in enumerate(question_indices):
                        variant_data["per_sample"][m.name][qi] = new_scores[k]
                    variant_data["scores"] = _recalculate_scores(
                        variant_data["per_sample"]
                    )

                for k, qi in enumerate(question_indices):
                    s = cast(SingleTurnSample, dataset.samples[k])
                    variant_data["samples"][qi] = {
                        "user_input": s.user_input,
                        "reference": s.reference,
                        "response": s.response,
                        "retrieved_contexts": s.retrieved_contexts,
                    }
            else:
                # Full mode: write all scores
                per_sample = {}
                for m in metrics:
                    raw = [_sanitize_for_json(v) for v in result[m.name]]
                    per_sample[m.name] = raw

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
                    "scores": _recalculate_scores(per_sample),
                    "per_sample": per_sample,
                    "samples": samples,
                }

            log.info("  %s", results[variant]["scores"])
            _save_results(results)

    return results


def print_results(results: dict[str, Any]) -> None:
    header = [
        "variant",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
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
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fresh",
        action="store_true",
        help="Re-run all variants from scratch",
    )
    group.add_argument(
        "--questions",
        type=int,
        nargs="+",
        metavar="IDX",
        help="Re-evaluate specific questions (0-indexed) across all variants",
    )
    args = parser.parse_args()

    results = run_evaluation(
        resume=not args.fresh,
        question_indices=args.questions,
    )
    print("\n--- Results ---")
    print_results(results)

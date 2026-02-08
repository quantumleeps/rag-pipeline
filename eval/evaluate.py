"""RAGAS evaluation harness for the 3×3 chunking × embedding matrix."""

import json
import logging
import sys

from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from ragas import EvaluationDataset, SingleTurnSample
from ragas.dataset_schema import SingleTurnSampleOrMultiTurnSample
from ragas.integrations.llama_index import evaluate as ragas_evaluate
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness

from rag_pipeline.chunkers import ChunkStrategy
from rag_pipeline.embed import EmbedModelName
from rag_pipeline.query import get_query_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

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


def run_evaluation() -> dict[str, dict[str, float]]:
    dataset = build_eval_dataset()
    evaluator_llm = Anthropic(model="claude-sonnet-4-5-20250929")
    eval_embeddings = VoyageEmbedding(model_name="voyage-3.5")
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        LLMContextRecall(),
    ]

    results = {}
    for strategy in ChunkStrategy:
        for model in EmbedModelName:
            variant = f"{strategy.value}_{model.value}"
            log.info("Evaluating %s", variant)

            query_engine = get_query_engine(strategy, model)
            result = ragas_evaluate(
                query_engine=query_engine,
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=eval_embeddings,
            )

            scores = {}
            for m in metrics:
                per_sample = result[m.name]
                scores[m.name] = sum(per_sample) / len(per_sample)
            results[variant] = scores
            log.info("  %s", scores)

    return results


def print_results(results: dict[str, dict[str, float]]) -> None:
    header = [
        "variant",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "llm_context_recall",
    ]
    print("\t".join(header))
    for variant, scores in results.items():
        row = [variant] + [f"{scores.get(h, 0):.3f}" for h in header[1:]]
        print("\t".join(row))


if __name__ == "__main__":
    results = run_evaluation()
    print("\n--- Results ---")
    print_results(results)

    if "--json" in sys.argv:
        print(json.dumps(results, indent=2))

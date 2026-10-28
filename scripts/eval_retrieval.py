"""Run the agent on the frozen test set and score it with RAGAS.

Outputs are named after the configuration (answer model, k, reranker candidates):
`data/eval/answers_<config>.jsonl` is the resumable checkpoint of agent answers,
`data/eval/results_<config>.csv` the per-question scores. Rows that carry a
`channel` field (hand-written questions) also yield a routing accuracy."""

import argparse
import json
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (  # the legacy classes: the only ones that take BaseRagasLLM
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from rag_chatbot.agent.workflow import DATABASE, RULEBOOK
from rag_chatbot.evaluate import GeminiRagasLLM
from rag_chatbot.settings import settings
from scripts._common import agent_or_exit, configure_logging, new_thread, read_jsonl

SAMPLE_KEYS = ("user_input", "reference", "response", "retrieved_contexts")
METRICS = [
    Faithfulness(),
    FactualCorrectness(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]
METRIC_COLUMNS = [
    "faithfulness",
    "factual_correctness(mode=f1)",
    "llm_context_precision_with_reference",
    "context_recall",
]
CHANNELS = {(RULEBOOK,): "text", (DATABASE,): "sql", (RULEBOOK, DATABASE): "both"}


def observed_channel(sources: list[str]) -> str:
    """The route the agent took, in the test set's vocabulary (text, sql, both)."""
    return CHANNELS.get(tuple(sources), "none")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, default="data/eval/testset.jsonl")
    parser.add_argument("--vector-db", default=settings.vector_db_path)
    parser.add_argument("-k", type=int, default=settings.vector_k)
    args = parser.parse_args()
    configure_logging()

    agent = agent_or_exit(args.vector_db, args.k)
    testset = read_jsonl(args.testset)
    config = f"{settings.llm_model}_k{args.k}_rerank{settings.rerank_candidates}"
    checkpoint_path = args.testset.with_name(f"answers_{config}.jsonl")
    output_path = args.testset.with_name(f"results_{config}.csv")
    answers = {row["user_input"]: row for row in read_jsonl(checkpoint_path)}

    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for i, row in enumerate(testset, 1):
            question = row["user_input"]
            if question in answers:
                continue
            print(f"[{i}/{len(testset)}] {question[:70]}")
            state = agent.invoke({"messages": [("user", question)]}, new_thread())
            answers[question] = {
                "user_input": question,
                "reference": row["reference"],
                "response": state["messages"][-1].content,
                "retrieved_contexts": state["contexts"],
                "sources": state["sources"],
            }
            checkpoint.write(json.dumps(answers[question], ensure_ascii=False) + "\n")
            checkpoint.flush()

    records = [answers[row["user_input"]] for row in testset]
    scores = evaluate(
        EvaluationDataset.from_list(
            [{key: r[key] for key in SAMPLE_KEYS} for r in records]
        ),
        metrics=METRICS,
        llm=GeminiRagasLLM(settings.judge_model),
        run_config=RunConfig(max_workers=1, timeout=300),
    ).to_pandas()

    table = scores[["user_input", *METRIC_COLUMNS]].copy()
    table.insert(1, "channel", [row.get("channel", "") for row in testset])
    table.insert(2, "route", [observed_channel(r["sources"]) for r in records])
    table["response"] = [r["response"] for r in records]
    table.to_csv(output_path, index=False)

    print(f"\nMean scores over {len(table)} questions (saved to {output_path}):")
    print(table[METRIC_COLUMNS].mean().round(3).to_string())
    print("\nBy route taken:")
    print(table.groupby("route")[METRIC_COLUMNS].mean().round(3).to_string())
    labelled = table[table["channel"] != ""]
    if len(labelled):
        correct = int((labelled["channel"] == labelled["route"]).sum())
        print(f"\nRouting accuracy on labelled questions: {correct}/{len(labelled)}")


if __name__ == "__main__":
    main()

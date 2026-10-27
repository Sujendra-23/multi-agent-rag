"""Score collected results with RAGAS: faithfulness, answer relevancy, context precision, context recall.

    source .venv-eval/bin/activate
    python eval/ragas_eval.py --results eval/results_default.json --out eval/ragas_scores.json

The judge LLM is OpenAI (JUDGE_MODEL, default OPENAI_MODEL); answer-relevancy embeddings use text-embedding-3-small.
"""
import argparse
import json
import os
import re
import sys
import types

from dotenv import load_dotenv

load_dotenv()

# ragas 0.4.3 imports a module that recent langchain-community releases removed; stub it.
for _name, _attr in [("langchain_community.chat_models.vertexai", "ChatVertexAI"), ("langchain_community.llms", "VertexAI")]:
    try:
        __import__(_name)
    except Exception:  # noqa: BLE001
        _m = types.ModuleType(_name)
        setattr(_m, _attr, type(_attr, (), {}))
        sys.modules[_name] = _m

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from ragas import EvaluationDataset, RunConfig, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)



CITE = re.compile(r"\s*\[[CWS]\d+\]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.results)) if r.get("answer") and r.get("contexts")]
    print(f"scoring {len(rows)} questions")
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "retrieved_contexts": r["contexts"],
                "response": CITE.sub("", r["answer"]).strip(),
                "reference": r["reference"],
            }
            for r in rows
        ]
    )
    # Recent models reject the non-default temperature RAGAS sets, so bypass it.
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=os.environ.get("JUDGE_MODEL", os.environ["OPENAI_MODEL"]), api_key=os.environ["OPENAI_API_KEY"]),
        bypass_temperature=True,
    )
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"]))
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=4, timeout=240),
    )
    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in rows])
    num = df.select_dtypes("number")
    summary = {c: round(float(num[c].mean()), 3) for c in num.columns}
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "per_question": json.loads(df.to_json(orient="records"))}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

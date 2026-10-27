"""Retrieval-only benchmark: does the expected paper show up in the top-k chunks? (embeddings only, no LLM cost)

    python eval/retrieval_bench.py [--k 5] [--questions heldout_questions.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=app.config.RAG_TOP_K)
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    qs = json.load(open(os.path.join(os.path.dirname(__file__), args.questions)))
    hit = prec = mrr = rec = 0.0
    n_ref = 0
    for q in qs:
        res = app.rag.retrieve(q["question"], args.k)
        srcs = [r["source"] for r in res]
        n = srcs.count(q["source"])
        hit += n > 0
        prec += n / max(len(srcs), 1)
        mrr += 1 / (srcs.index(q["source"]) + 1) if n else 0
        if q.get("reference"):  # lexical proxy for context recall: share of the reference's key terms found in the chunks
            ref = set(app._tokens(q["reference"]))
            ctx = set(app._tokens(" ".join(r["text"] for r in res)))
            rec += len(ref & ctx) / max(len(ref), 1)
            n_ref += 1
        if args.verbose or n == 0:
            print(f"{q['id']} {'HIT ' if n else 'MISS'} {n}/{len(srcs)}  want {q['source']}  got {[s[:10] for s in srcs]}")
    t = len(qs)
    print(f"k={args.k}  hit@k={hit / t:.2f}  chunk-precision={prec / t:.2f}  MRR={mrr / t:.2f}"
          + (f"  ref-term-recall={rec / n_ref:.2f}" if n_ref else "") + f"  (n={t})")


if __name__ == "__main__":
    main()

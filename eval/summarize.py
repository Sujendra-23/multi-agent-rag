"""Print markdown latency / cost / provider tables from collected result files.

    python eval/summarize.py eval/results_default.json
    python eval/summarize.py --providers eval/results_default.json eval/results_force_anthropic.json eval/results_force_openai.json
"""
import json
import statistics as st
import sys


PRICE = {"anthropic": (2.00, 10.00), "openai": (4.00, 20.00), "nvidia": (0.0, 0.0)}  # USD per 1M tokens (in, out)


def cost_of(m: dict) -> float:
    return sum(
        (v["in_tokens"] * PRICE[p][0] + v["out_tokens"] * PRICE[p][1]) / 1e6 for p, v in m["per_provider"].items()
    )


def metrics(r: dict) -> dict:
    return r.get("done", {}).get("metrics", {})


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def per_query(path: str) -> None:
    rows = [r for r in json.load(open(path)) if metrics(r)]
    print("| id | total ms | TTFT ms | retrieval ms | LLM ms | tokens in/out | cost USD | providers |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = metrics(r)
        prov = "+".join(sorted({x["provider"] for x in r["routing"]}))
        print(f"| {r['id']} | {m['total_ms']} | {m['ttft_ms']} | {m['retrieval_ms']} | {m['llm_ms']} | {m['in_tokens']}/{m['out_tokens']} | {cost_of(m):.5f} | {prov} |")
    tot = [metrics(r)["total_ms"] for r in rows]
    ttft = [metrics(r)["ttft_ms"] for r in rows if metrics(r)["ttft_ms"] is not None]
    cost = [cost_of(metrics(r)) for r in rows]
    print(f"\nn={len(rows)}  total ms: median {st.median(tot):.0f}, p90 {pct(tot, .9):.0f}  |  TTFT ms: median {st.median(ttft):.0f}, p90 {pct(ttft, .9):.0f}  |  mean cost/query ${st.mean(cost):.5f}")
    n_by = {}
    for r in rows:
        for p, v in metrics(r)["per_provider"].items():
            d = n_by.setdefault(p, {"calls": 0, "in": 0, "out": 0, "ms": 0.0, "cost": 0.0})
            d["calls"] += v["calls"]; d["in"] += v["in_tokens"]; d["out"] += v["out_tokens"]; d["ms"] += v["ms"]; d["cost"] += (v["in_tokens"] * PRICE[p][0] + v["out_tokens"] * PRICE[p][1]) / 1e6
    print("\n| provider | LLM calls | tokens in | tokens out | mean ms/call | est. cost USD |")
    print("|---|---|---|---|---|---|")
    for p, d in n_by.items():
        print(f"| {p} | {d['calls']} | {d['in']} | {d['out']} | {d['ms'] / d['calls']:.0f} | {d['cost']:.4f} |")


def providers(paths: list[str]) -> None:
    print("| final-answer provider | n | median total ms | median TTFT ms | mean cost/query USD | mean answer chars |")
    print("|---|---|---|---|---|---|")
    for path in paths:
        rows = [r for r in json.load(open(path)) if metrics(r)]
        synth = {x["provider"] for r in rows for x in r["routing"] if x.get("stage") == "synthesize"}
        label = "/".join(sorted(synth))
        tot = [metrics(r)["total_ms"] for r in rows]
        ttft = [metrics(r)["ttft_ms"] for r in rows if metrics(r)["ttft_ms"] is not None]
        print(f"| {label} | {len(rows)} | {st.median(tot):.0f} | {st.median(ttft):.0f} | {st.mean(cost_of(metrics(r)) for r in rows):.5f} | {st.mean(len(r['answer']) for r in rows):.0f} |")


if __name__ == "__main__":
    if sys.argv[1] == "--providers":
        providers(sys.argv[2:])
    else:
        per_query(sys.argv[1])

"""Run the question set against a running instance of app.py and save answers, contexts and metrics.

    python eval/collect.py --base http://127.0.0.1:8080 --out eval/results_default.json
    python eval/collect.py --base https://<cloud-run-url> --queries eval/routing_queries.json --out eval/results_live.json

Talks to the app over HTTP only, so it works against localhost or the deployed service. To compare
providers, start the server with FORCE_SYNTH_PROVIDER=<provider> and collect once per provider.
"""
import argparse
import json
import time

import httpx


def ask(client: httpx.Client, base: str, query: str) -> dict:
    out = {"answer": "", "citations": [], "routing": [], "trace": [], "done": {}, "error": None}
    t0 = time.perf_counter()
    first_byte = None
    with client.stream("POST", f"{base}/chat", json={"query": query, "history": []}, timeout=300) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
            if first_byte is None:
                first_byte = time.perf_counter() - t0
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                data = json.loads(line[6:])
                if event == "token":
                    out["answer"] += data["text"]
                elif event == "reset":
                    out["answer"] = ""
                elif event == "citations":
                    out["citations"] = data["items"]
                elif event == "route":
                    out["routing"].append(data)
                elif event == "trace":
                    out["trace"].append(data)
                elif event == "error":
                    out["error"] = data["message"]
                elif event == "done":
                    out["done"] = data
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    out["first_byte_s"] = round(first_byte or 0, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--queries", default="eval/questions.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0, help="seconds to pause between queries (the NVIDIA free tier rate-limits bursts)")
    args = ap.parse_args()

    items = json.load(open(args.queries))
    if args.limit:
        items = items[: args.limit]
    results = []
    with httpx.Client() as client:
        for n, it in enumerate(items):
            if n and args.sleep:
                time.sleep(args.sleep)
            try:
                res = ask(client, args.base, it["question"])
            except Exception as e:  # noqa: BLE001
                res = {"error": f"{type(e).__name__}: {e}", "answer": "", "citations": [], "routing": [], "done": {}}
            contexts = []
            for c in res["citations"]:
                if c.get("kind") == "corpus" and c.get("chunk_id"):
                    try:
                        ch = client.get(f"{args.base}/api/chunk/{c['chunk_id']}", timeout=30).json()
                        contexts.append(ch.get("text", ""))
                    except Exception:  # noqa: BLE001
                        pass
            m = res.get("done", {}).get("metrics", {})
            providers = sorted({r["provider"] for r in res["routing"]})
            print(
                f"{it['id']}: {res['wall_s'] if 'wall_s' in res else '-'}s ttft={m.get('ttft_ms')} "
                f"ctx={len(contexts)} providers={providers} err={res['error']}"
            )
            results.append({**it, **res, "contexts": contexts})
    json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"saved {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()

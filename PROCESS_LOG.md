# Process Log

## AI tools used
Claude Code (Claude Sonnet 5.5) wrote and debugged most of `app.py`, the evaluation scripts and the docs. NVIDIA Nemotron 3.5 Lightning (cheap tier) and OpenAI GPT-5.6 Sol (strong tier, embeddings) run inside the system; RAGAS 0.4.3 with a GPT-5.1 judge scored it.

## Development narrative
I built a supervisor with research, summary and data agents in one FastAPI and LangGraph file. Providers kept failing for non-code reasons: Gemini's free tier throttled and hung requests, OpenAI and then Anthropic ran out of credit, and NVIDIA's reasoning model took 50 seconds per question until I disabled thinking. I dropped Gemini and Anthropic, so the final pair is NVIDIA plus OpenAI. Cloud Run deployed on the first build.

## Verification narrative
I checked every key and model ID with a real call, then injected faults: bad Tavily and NVIDIA keys, a missing database, and unavailable embeddings. These found real bugs: Tavily error text was cited as evidence, workers had no failover, and retrieval died without embeddings. I also scanned streams and logs for key patterns. RAGAS showed retrieval returning the wrong paper, so I added titles, hybrid keyword search and k=8, measuring each step: context recall rose from 0.64 to 0.84, while precision and faithfulness stayed flat.

## What I learned
The tools could not get keys, add billing or credit, or decide that a free model was usable; I had to do that. I learned that SDK timeouts cannot be trusted, that a tool saying "ok" is not proof of success, and that evaluation exposes weaknesses that demos hide.

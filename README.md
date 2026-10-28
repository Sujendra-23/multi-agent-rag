# Multi-Agent RAG (CSCI 599 Assignment 2)

A LangGraph **supervisor** with three specialised agents (research, summary, data), a RAG pipeline over 40 arXiv papers (Chroma), live web grounding through **Tavily Remote MCP** shared by two agents, and LLM routing across **NVIDIA NIM** (cheap tier) and **OpenAI** (strong tier). FastAPI streams Server-Sent Events to a single vanilla HTML/JS page that shows routing decisions, an orchestration trace, and clickable citations. Deployed on Google Cloud Run.

**Live demo:** https://multi-agent-rag-135663485077.us-west1.run.app (scales to zero, so the first answer after an idle period takes longer)

- Backend, orchestration, RAG and frontend: one file, [app.py](app.py)
- Design details, state schema, routing table, prompts: [docs/architecture.md](docs/architecture.md)
- RAGAS scores, retrieval benchmark, latency, cost: [docs/evaluation.md](docs/evaluation.md)
- Frontend framework: none (vanilla HTML/CSS/JS, no HTMX, no build step)

## Provider requirements and current cost/access (verified Oct 2026)

You need keys for **at least two different LLM providers** (here NVIDIA and OpenAI) plus Tavily; OpenAI also provides the embeddings.

| Service | Used for | Model ID pinned in `.env.example` | Notes |
|---|---|---|---|
| NVIDIA NIM (build.nvidia.com) | cheap tier: classification, research and summary steps, simple answers | `nvidia/nemotron-3.5-lightning-30b-a3b` | free/trial tier; cost counted as $0; which Nemotron models are reachable depends on the account |
| OpenAI | strong tier (data agent's SQL, grounded and complex final answers, failover), embeddings (`text-embedding-3-small`, about $0.02 per million tokens) | `gpt-5.6-sol` | $4 / $20 per million input / output tokens on OpenAI's model page (promotional rate, listed through at least 21 November 2026; $5 / $30 before); needs prepaid credit, roughly $0.01 per query. Embedding the 40-paper corpus costs about $0.02 |
| Tavily | live web search/extract over MCP | n/a | free/student plan is enough |

The app enforces the assignment's two-provider rule: it refuses to start with fewer than two providers configured or with the same provider on both tiers (`ALLOW_SINGLE_PROVIDER=1` is for offline measurements only). NVIDIA's free tier rate-limits (HTTP 429) intermittently; the app then redoes that step on the other provider, which is slower and costs a little. Any pair of providers works: set `CHEAP_PROVIDER` / `STRONG_PROVIDER` to `openai`, `nvidia` or `anthropic` (Anthropic `claude-sonnet-5-5` is supported as an optional provider: set `ANTHROPIC_API_KEY`) (two models from one provider do not count). Local embeddings (`EMBEDDING_BACKEND=local`, needs `pip install sentence-transformers`) remove the OpenAI dependency but make the image much larger.

## Local setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt           # every dependency is pinned with ==

cp .env.example .env                      # then fill in keys and CORPUS_DESCRIPTION; never commit .env

bash scripts/fetch_corpus.sh              # downloads the 40 papers, pinned versions, into corpus/ (about 130 MB, ~3 min)
python app.py ingest --corpus ./corpus    # chunk (1000/150), embed with titles, write ./chroma_index and ./data/sample.db
python app.py serve                       # http://127.0.0.1:8080
```

Check `http://127.0.0.1:8080/api/config`: it should list your providers, `tavily_enabled: true` and `corpus_chunks` of about 6,114.

### Corpus version and rebuilding it

The corpus is 40 arXiv papers (about 130 MB of PDFs), downloaded on **25 October 2026**. [scripts/corpus_manifest.md](scripts/corpus_manifest.md) lists each arXiv ID with its exact version (e.g. `2210.03629v3`), and `scripts/fetch_corpus.sh` downloads those versions, so a later revision on arXiv does not change the corpus. The PDFs are not in the zip (the manifest and `scripts/fetch_corpus.sh` re-download them). The **prebuilt Chroma index (`chroma_index/`, about 88 MB) is in the zip**, because the Dockerfile copies it into the image and rebuilding it would cost embedding credits; it is the same index the deployed service uses, and it matches `data/chunks.jsonl.gz` exactly (6,114 chunks, identical ids and text, 40 sources). With the zip you can deploy without running `ingest`; the steps below rebuild everything from the PDFs if you prefer. The GitHub repository does not contain the index (it is gitignored); the zip does.

Rebuilding from scratch, measured on a laptop: the download takes about 3 minutes (3 s pause between papers), and `ingest` took 106 s and produced the same 40 documents and 6,114 chunks as the deployed index. Embedding costs about $0.02.

The chunked text is included without rebuilding: `data/chunks.jsonl.gz` has all 6,114 chunks (one JSON object per line: `id`, `text`, `source`, `title`, `page`, `chunk_index`), and `data/sample.db` has the per-document metadata table the data agent queries.

`corpus/titles.json` maps each file name to its paper title (used for citations and retrieval); files not listed fall back to the PDF metadata title, then the file name. Retrieval is hybrid (dense + BM25, fused with reciprocal rank fusion).

Rebuild the index whenever the corpus changes or after upgrading the code (the index format includes titles): delete `chroma_index/` and `data/sample.db`, then run `ingest` again. The index is baked into the Docker image, so **run `ingest` before `gcloud builds submit`**.

Try: a corpus question ("What is the difference between Self-RAG and Corrective RAG?"), a live question ("What is the latest news about AI agents this week?"), a data question ("How many chunks does each document have?"), and a mixed one ("Summarize the latest news about AI agents and compare it with what the corpus says about ReAct"). Watch the *Routing* and *Trace* panels.

Behaviour worth trying (details in [docs/architecture.md](docs/architecture.md)):

- **Follow-ups:** ask "Tell me about ReAct", then "What tasks did it evaluate on?", then "Why?". Short follow-ups are rewritten with the earlier turns before retrieval, and the Trace shows `follow-up resolved: ...`.
- **Read-only:** "Delete all documents." or "Drop table documents;" is refused with no model call (the Trace shows a `guardrail` line); SQL runs only through a read-only connection.
- **Honesty and dates:** "Who won the 2030 World Cup?" says the event has not happened; answers use your browser's local date for "today" and "this week".
- **No source, no memory answer:** a question the corpus does not cover (for example "Explain what a transformer is") is answered from a Tavily web search with `[W#]` citations, not from the model's memory.

## Deploy to Cloud Run

Prerequisites: `gcloud` installed and logged in (`gcloud auth login`), a project with billing enabled, and `ingest` already run locally.

```bash
export GCP_PROJECT_ID=<your-project-id>
export GCP_REGION=us-west1
gcloud config set project $GCP_PROJECT_ID

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

# 1. Keys go into Secret Manager, never into code or --set-env-vars.
for pair in openai-key:OPENAI_API_KEY nvidia-key:NVIDIA_API_KEY tavily-key:TAVILY_API_KEY; do
  secret=${pair%%:*}; var=${pair##*:}
  printf '%s' "${!var}" | gcloud secrets create "$secret" --data-file=-   # export the three variables first
done
NUM=$(gcloud projects describe $GCP_PROJECT_ID --format='value(projectNumber)')
for s in openai-key nvidia-key tavily-key; do
  gcloud secrets add-iam-policy-binding $s \
    --member="serviceAccount:${NUM}-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
done

# 2. Non-secret settings (a file, because CORPUS_DESCRIPTION contains commas).
cat > cloudrun-env.yaml <<'EOF'
OPENAI_MODEL: "gpt-5.6-sol"
NVIDIA_MODEL: "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE_URL: "https://integrate.api.nvidia.com/v1"
CHEAP_PROVIDER: "nvidia"
STRONG_PROVIDER: "openai"
EMBEDDING_BACKEND: "openai"
OPENAI_EMBEDDING_MODEL: "text-embedding-3-small"
RAG_TOP_K: "8"
RATE_LIMIT_PER_MIN: "60"
CORPUS_DESCRIPTION: "40 arXiv research papers on retrieval-augmented generation (RAG), LLM-based agents, tool use, multi-agent systems, and RAG evaluation."
EOF

# 3. Build and deploy (flags as in the assignment's section 4.5).
gcloud builds submit --tag gcr.io/${GCP_PROJECT_ID}/multi-agent-rag
gcloud run deploy multi-agent-rag \
  --image gcr.io/${GCP_PROJECT_ID}/multi-agent-rag \
  --platform managed --region ${GCP_REGION} --allow-unauthenticated \
  --memory 1Gi --cpu 1 --concurrency 10 --timeout 300 \
  --min-instances 0 --max-instances 3 \
  --env-vars-file cloudrun-env.yaml \
  --set-secrets OPENAI_API_KEY=openai-key:latest,NVIDIA_API_KEY=nvidia-key:latest,TAVILY_API_KEY=tavily-key:latest

gcloud run services describe multi-agent-rag --region ${GCP_REGION} --format='value(status.url)'
```

`.gcloudignore` and `.dockerignore` keep `.env`, virtualenvs and the PDF corpus out of the build upload and the image. `RATE_LIMIT_PER_MIN` is 12 per client IP by default and is raised to 60 here because Cloud Run traffic can share IPs.

Verify: open the URL after the service has been idle (cold start: the page loads at once, the first answer is slower), then run the four example queries above and check `/api/config` for `mcp_call_log` entries from two agents.

## Security notes

- Keys live only in `.env` (gitignored) locally and in Secret Manager on Cloud Run (`secretKeyRef`, never plain values).
- `scrub()` in `app.py` redacts secret values and key-shaped strings from everything streamed to the browser and from startup logs; the Tavily key travels in the MCP URL as the assignment requires, so this matters.
- Rotate every key after grading and set provider spend caps.

## Evaluation

```bash
python3.13 -m venv .venv-eval && source .venv-eval/bin/activate
pip install -r requirements-dev.txt
python app.py serve &                                    # or use --base with the Cloud Run URL
python eval/collect.py --sleep 9 --out eval/results_default.json   # 25 questions: answers, contexts, metrics
JUDGE_MODEL=gpt-5.1 python eval/ragas_eval.py --results eval/results_default.json --out eval/ragas_scores.json   # pin the judge so scores stay comparable
python eval/summarize.py eval/results_default.json       # latency / cost tables
python eval/retrieval_bench.py                           # retrieval-only check (embeddings only, no LLM cost)
```

## Layout

```
app.py                  backend + orchestration + RAG + frontend
requirements.txt        pinned runtime dependencies
requirements-dev.txt    pinned evaluation dependencies (RAGAS)
.env.example            placeholders and verified model IDs
Dockerfile              multi-stage, python:3.13-slim, non-root
corpus/                 PDFs (not committed; scripts/fetch_corpus.sh)
chroma_index/, data/    generated by `ingest`; shipped prebuilt in the zip (not in git), baked into the image
eval/                   question set, collector, RAGAS scorer, summarizer, result files
scripts/                fetch_corpus.sh (re-download the pinned papers), corpus_manifest.md (arXiv ids, versions, titles)
cloudrun-env.yaml       non-secret Cloud Run settings used by the deploy command (keys come from Secret Manager)
docs/                   architecture.md, evaluation.md (submitted as separate files, not inside the code zip)
PROCESS_LOG.md          process log (submitted as a separate file)
```

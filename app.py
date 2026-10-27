"""Multi-Agent RAG in ONE file: FastAPI + SSE, LangGraph supervisor, 3 agents, Chroma, Tavily MCP.

Run:
    pip install -r requirements.txt
    cp .env.example .env                  # fill in keys + model names
    python app.py ingest --corpus ./corpus
    python app.py serve                   # open http://127.0.0.1:8080

Sections (search for the "=====" banners):
    1 config   2 state   3 events   4 embeddings   5 LLM calls   6 RAG   7 router
    8 Tavily MCP   9 supervisor   10 agents   11 final answer + graph   12 web server
    13 the web page (HTML)   14 command line
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import functools
import json
import math
import operator
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# 1. CONFIG: every setting, read from environment variables
# ============================================================================

load_dotenv()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# All settings live on one object called `config` (read from environment variables / .env).
config = SimpleNamespace(
    env=env,
    # provider -> model id + api key
    PROVIDERS={
        "anthropic": {"model": env("ANTHROPIC_MODEL"), "key": env("ANTHROPIC_API_KEY")},
        "openai": {"model": env("OPENAI_MODEL"), "key": env("OPENAI_API_KEY")},
        "nvidia": {"model": env("NVIDIA_MODEL"), "key": env("NVIDIA_API_KEY")},
    },
    # Rough USD per 1M tokens (input, output). Only used for the cost estimate shown in the UI.
    # Anthropic/OpenAI per third-party aggregators, Oct 2026; NVIDIA NIM trial/free-tier is billed $0. Re-check before relying on them.
    PRICE_PER_MTOK={
        "anthropic": (2.00, 10.00),
        "openai": (4.00, 20.00),
        "nvidia": (0.00, 0.00),
    },
    CHEAP_PROVIDER=env("CHEAP_PROVIDER", "nvidia"),
    STRONG_PROVIDER=env("STRONG_PROVIDER", "openai"),
    CHROMA_PERSIST_DIR=env("CHROMA_PERSIST_DIR", "./chroma_index"),
    CHROMA_COLLECTION=env("CHROMA_COLLECTION", "corpus"),
    RAG_TOP_K=int(env("RAG_TOP_K", "8")),
    RAG_HYBRID=env("RAG_HYBRID", "1") not in ("0", "false", "off"),
    RAG_PAPER_FOCUS=env("RAG_PAPER_FOCUS", "1") not in ("0", "false", "off"),
    EMBEDDING_BACKEND=env("EMBEDDING_BACKEND", "openai"),
    OPENAI_EMBEDDING_MODEL=env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    LOCAL_EMBEDDING_MODEL=env("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    TAVILY_API_KEY=env("TAVILY_API_KEY"),
    # The Tavily key goes in the URL query string, as the assignment says.
    TAVILY_MCP_URL=f"https://mcp.tavily.com/mcp/?tavilyApiKey={env('TAVILY_API_KEY')}",
    SQLITE_PATH=env("SQLITE_PATH", "./data/sample.db"),
    # One sentence describing your corpus. The planner uses it to decide when documents can help.
    CORPUS_DESCRIPTION=env("CORPUS_DESCRIPTION", "A collection of technical documents and research papers."),
    # Evaluation only: force ALL final answers to one provider (leave empty normally).
    FORCE_SYNTH_PROVIDER=env("FORCE_SYNTH_PROVIDER", ""),
    # If the evidence is longer than this many characters, the final answer goes to the strong model.
    COMPLEX_CONTEXT_CHARS=int(env("COMPLEX_CONTEXT_CHARS", "6000")),
)
app_config = config  # alias: graph nodes have a parameter called `config` that hides the name


def _available_providers() -> list[str]:
    """Providers that have both a key and a model id configured."""
    return [p for p, c in config.PROVIDERS.items() if c["key"] and c["model"]]


config.available_providers = _available_providers


def provider_rule() -> tuple[bool, str]:
    """Assignment 2.3: routing across at least two different LLM providers (two models of one provider do not count)."""
    avail = config.available_providers()
    if len(avail) < 2:
        return False, f"only {len(avail)} LLM provider(s) configured ({', '.join(avail) or 'none'}); at least two are required"
    if config.CHEAP_PROVIDER == config.STRONG_PROVIDER:
        return False, f"CHEAP_PROVIDER and STRONG_PROVIDER are both '{config.CHEAP_PROVIDER}'; they must be different providers"
    missing = [p for p in (config.CHEAP_PROVIDER, config.STRONG_PROVIDER) if p not in avail]
    if missing:
        return False, f"routing tier provider(s) {missing} have no API key or model id"
    return True, f"routing across {config.CHEAP_PROVIDER} (cheap) and {config.STRONG_PROVIDER} (strong)"


# ---- secret scrubbing: nothing shaped like a key ever reaches the browser or the logs.
_SECRET_VALUES = sorted(
    (v for k, v in os.environ.items() if re.search(r"KEY|TOKEN|SECRET|PASSWORD", k, re.I) and v),
    key=len,
    reverse=True,  # longest first, so a short value can't mask part of a longer one
)
_SECRET_PATTERNS = [
    re.compile(p)
    for p in (
        r"sk-[A-Za-z0-9_-]{8,}",
        r"AIza[A-Za-z0-9_-]{10,}",
        r"tvly-[A-Za-z0-9_-]{8,}",
        r"nvapi-[A-Za-z0-9_-]{8,}",
        r"(?i)(tavilyApiKey=)[^&\s]+",
        r"(?i)(bearer\s+)[A-Za-z0-9._-]{8,}",
    )
]


def scrub(text: str) -> str:
    """Redact known secret values and key-shaped strings before they leave the server."""
    if not text:
        return text
    for secret in _SECRET_VALUES:
        text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
    return text


def scrub_obj(obj: Any) -> Any:
    """Recursively apply scrub() to every string inside a dict/list/str."""
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, dict):
        return {k: scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v) for v in obj]
    return obj


# ============================================================================
# 2. STATE: the shared notebook all agents read and write
# ============================================================================

def merge_dicts(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class State(TypedDict, total=False):
    # input
    query: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}], sent by the client

    # supervisor planning (inspectable routing fields)
    planned: bool
    plan: list[str]  # ordered worker names the supervisor chose
    needs_corpus: bool
    needs_live: bool
    complexity: str  # simple | complex
    plan_reason: str
    next: list[str]  # workers to run now, in parallel: research | summary | data, or ["synthesize"]
    slots: dict[str, int]  # worker -> position in the current parallel batch (its own citation-label range)
    steps: int

    # accumulated by workers (reducers append/merge so workers never overwrite each other)
    completed: Annotated[list[str], operator.add]
    agent_outputs: Annotated[dict[str, Any], merge_dicts]
    citations: Annotated[list[dict], operator.add]
    routing: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]

    # output
    answer: str


# ============================================================================
# 3. EVENTS: per-request queue (streaming), timing, token counts, trace lines
# ============================================================================

@dataclass
class Run:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    t0: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0
    per_provider: dict[str, dict[str, float]] = field(default_factory=dict)

    def now_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    async def emit(self, type_: str, **data: Any) -> None:
        await self.queue.put({"type": type_, "t_ms": self.now_ms(), **data})

    def mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def add_llm(self, provider: str, in_tok: int, out_tok: int, ms: float, cost: float) -> None:
        self.llm_ms += ms
        self.in_tokens += in_tok
        self.out_tokens += out_tok
        self.cost_usd += cost
        p = self.per_provider.setdefault(
            provider, {"calls": 0, "in_tokens": 0, "out_tokens": 0, "ms": 0.0, "cost_usd": 0.0}
        )
        p["calls"] += 1
        p["in_tokens"] += in_tok
        p["out_tokens"] += out_tok
        p["ms"] += ms
        p["cost_usd"] += cost

    def metrics(self) -> dict[str, Any]:
        total_ms = self.now_ms()
        ttft = (
            int((self.first_token_at - self.t0) * 1000) if self.first_token_at is not None else None
        )
        return {
            "total_ms": total_ms,
            "ttft_ms": ttft,
            "retrieval_ms": int(self.retrieval_ms),
            "llm_ms": int(self.llm_ms),
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "per_provider": self.per_provider,
        }


class Tracer:
    """Collects trace entries for one node invocation and streams them live."""

    def __init__(self, run: Run, node: str):
        self.run = run
        self.node = node
        self.entries: list[dict] = []

    async def __call__(self, event: str, detail: str = "", **extra: Any) -> None:
        entry = {"t_ms": self.run.now_ms(), "node": self.node, "event": event, "detail": detail, **extra}
        self.entries.append(entry)
        await self.run.emit("trace", **{k: v for k, v in entry.items() if k != "t_ms"})


# ============================================================================
# 4. EMBEDDINGS: text -> vectors for search
# ============================================================================

class Embedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder(Embedder):
    def __init__(self):
        from langchain_openai import OpenAIEmbeddings

        self._e = OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL, api_key=config.env("OPENAI_API_KEY")
        )
        # Query-time client: fail fast so a slow embedding call falls back to keyword-only
        # retrieval instead of stalling the whole answer.
        self._q = OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL, api_key=config.env("OPENAI_API_KEY"), timeout=15, max_retries=1
        )

    def embed_documents(self, texts):
        return self._e.embed_documents(texts)

    def embed_query(self, text):
        return self._q.embed_query(text)


class LocalEmbedder(Embedder):
    def __init__(self):
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

        self._m = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)

    def embed_documents(self, texts):
        return self._m.encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if config.EMBEDDING_BACKEND == "local":
        return LocalEmbedder()
    return OpenAIEmbedder()


# ============================================================================
# 5. LLM CALLS: build each provider's model; stream a completion and record usage
# ============================================================================

# Tests can inject fake chat models here: {"nvidia": FakeModel, "anthropic": FakeModel}
LLM_TIMEOUT_SECONDS = float(config.env("LLM_TIMEOUT_SECONDS", "25"))

_OVERRIDES: dict[str, BaseChatModel] = {}


def set_overrides(models: dict[str, BaseChatModel]) -> None:
    _OVERRIDES.clear()
    _OVERRIDES.update(models)
    get_llm.cache_clear()


@lru_cache(maxsize=None)
def get_llm(provider: str) -> BaseChatModel:
    if provider in _OVERRIDES:
        return _OVERRIDES[provider]

    cfg = config.PROVIDERS.get(provider)
    if not cfg or not cfg["key"] or not cfg["model"]:
        raise RuntimeError(f"Provider '{provider}' is not configured (missing key or model id).")

    # A hung backend must raise, not block forever, so the existing failover/retry logic
    # (see synthesizer_node's "except Exception" -> other_provider) actually gets a chance to run.
    timeout = LLM_TIMEOUT_SECONDS

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=cfg["model"], api_key=cfg["key"], max_tokens=2048, timeout=timeout
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg["model"], api_key=cfg["key"], timeout=timeout)
    if provider == "nvidia":
        from langchain_openai import ChatOpenAI

        # NVIDIA NIM exposes an OpenAI-compatible endpoint.
        return ChatOpenAI(
            model=cfg["model"],
            api_key=cfg["key"],
            base_url=config.env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            temperature=0.2,
            timeout=timeout,
            # Nemotron 3.5 "thinks" by default (10s vs 0.8s here); the cheap tier doesn't need it.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    raise ValueError(f"Unknown provider: {provider}")


def estimate_cost(provider: str, in_tokens: int, out_tokens: int) -> float:
    pin, pout = config.PRICE_PER_MTOK.get(provider, (0.0, 0.0))
    return (in_tokens * pin + out_tokens * pout) / 1_000_000


def _text_of(content) -> str:
    """Chunk content is a str for some providers and a list of blocks for others."""
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict) and block.get("type") in (None, "text"):
            out.append(block.get("text", ""))
    return "".join(out)


async def call_llm(
    run: Run,
    provider: str,
    messages: list[BaseMessage],
    *,
    stream_tokens: bool = False,
    max_tokens: int | None = None,
    total_timeout: float | None = None,
) -> str:
    """Stream a completion, optionally forwarding tokens to the browser. Records usage.

    max_tokens caps the output of NVIDIA calls only (thinking is off there; a cap on a reasoning
    model could be spent on hidden reasoning and leave an empty answer). total_timeout bounds the
    whole call, so a model that trickles tokens slowly still fails over instead of running on.
    """
    llm = get_llm(provider)
    if max_tokens and provider == "nvidia":
        llm = llm.bind(max_tokens=max_tokens)
    t = time.perf_counter()
    text = ""
    agg = None
    # The provider SDKs' own `timeout` kwargs are not reliably honored (some wrappers hang
    # past it), so enforce a deadline per chunk here; a stall raises and triggers failover.
    stream = llm.astream(messages).__aiter__()
    try:
        while True:
            wait = LLM_TIMEOUT_SECONDS
            if total_timeout:
                remaining = total_timeout - (time.perf_counter() - t)
                if remaining <= 0:
                    raise TimeoutError(f"{provider} did not finish within {total_timeout:.0f}s")
                wait = min(wait, remaining)
            try:
                chunk = await asyncio.wait_for(stream.__anext__(), timeout=wait)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                if wait < LLM_TIMEOUT_SECONDS:  # cut short by the whole-call limit
                    raise TimeoutError(f"{provider} did not finish within {total_timeout:.0f}s") from None
                raise TimeoutError(
                    f"{provider} produced no output for {LLM_TIMEOUT_SECONDS:.0f}s"
                ) from None
            piece = _text_of(chunk.content)
            agg = chunk if agg is None else agg + chunk
            if piece:
                text += piece
                if stream_tokens:
                    run.mark_first_token()
                    await run.emit("token", text=piece)
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose:
            try:
                await asyncio.wait_for(aclose(), timeout=2)
            except Exception:  # noqa: BLE001 - best-effort cleanup of a stalled stream
                pass
    ms = (time.perf_counter() - t) * 1000

    usage = getattr(agg, "usage_metadata", None) or {}
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    if not in_tok:  # provider didn't report usage: rough 4-chars-per-token estimate
        in_tok = sum(len(_text_of(m.content)) for m in messages) // 4
    if not out_tok:
        out_tok = len(text) // 4
    run.add_llm(provider, in_tok, out_tok, ms, estimate_cost(provider, in_tok, out_tok))
    return text


def llm_error_text(provider: str, e: Exception) -> str:
    """Readable provider error. NVIDIA is called through the OpenAI SDK, so its exceptions carry
    OpenAI class names ("OpenAIRateLimitError") that would blame the wrong provider."""
    if getattr(e, "status_code", None) == 429 or "RateLimit" in type(e).__name__:
        return f"{provider} rate limit (HTTP 429)"
    return f"{provider} error ({type(e).__name__.removeprefix('OpenAI')}): {e}"


WORKER_TOTAL_S = float(config.env("LLM_WORKER_TOTAL_SECONDS", "60"))  # whole-call limit for a worker step
WORKER_NOTES_MAX_TOKENS = int(config.env("WORKER_NOTES_MAX_TOKENS", "1200"))  # research/summary notes on NVIDIA


async def _heartbeat(tr, label: str, every: float = 10.0) -> None:
    """Worker LLM calls are not streamed to the browser; say so every few seconds instead of
    leaving the trace silent while the model works."""
    waited = 0.0
    while True:
        await asyncio.sleep(every)
        waited += every
        await tr("progress", f"{label} still working ({waited:.0f}s)")


async def call_llm_fo(
    run: Run, tr, decision, messages: list[BaseMessage], *, max_tokens: int | None = None
) -> str:
    """call_llm for worker steps: if the routed provider fails, try each other provider in turn."""
    agent = decision.stage.split(":")[-1] if ":" in decision.stage else "supervisor"
    beat = asyncio.create_task(_heartbeat(tr, agent))
    try:
        try:
            return await call_llm(
                run, decision.provider, messages, max_tokens=max_tokens, total_timeout=WORKER_TOTAL_S
            )
        except Exception as e:  # noqa: BLE001
            failed, err = decision.provider, e
        for alt in fallback_providers(decision.provider):
            await tr("warn", f"{llm_error_text(failed, err)}; failing over to {alt}")
            await run.emit(
                "route", stage=decision.stage, provider=alt, model=config.PROVIDERS[alt]["model"],
                dimension="fallback", reason=f"{failed} errored; failover to {alt}", agent=agent,
            )
            try:
                return await call_llm(run, alt, messages, max_tokens=max_tokens, total_timeout=WORKER_TOTAL_S)
            except Exception as e2:  # noqa: BLE001
                failed, err = alt, e2
        raise err
    finally:
        beat.cancel()


# ============================================================================
# 6. RAG: chunk + embed + store documents (ingest), and search them (retrieve)
# ============================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH = 64
SUPPORTED = {".pdf", ".md", ".txt"}


# ---------------------------------------------------------------- storage

@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def reset_cache() -> None:
    get_collection.cache_clear()
    get_bm25.cache_clear()
    _title_map.cache_clear()


# ---------------------------------------------------------------- ingestion

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(p) > size:  # oversized paragraph: flush, then hard-split with overlap
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(size - overlap, 1)
            for i in range(0, len(p), step):
                piece = p[i : i + size].strip()
                if piece:
                    chunks.append(piece)
            continue
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


MIN_CHUNK = 200  # a shorter tail is merged into the previous chunk instead of becoming a fragment
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"“])")


def normalize_pdf_text(text: str) -> str:
    """PDF extraction leaves one newline per visual line and hyphenated line breaks: rebuild running text."""
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def chunk_document(pages: list[tuple[int, str]], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, str]]:
    """Sentence-aware chunks (<= size chars, ~overlap chars repeated) over the whole document, so a passage
    that crosses a page break stays together. Returns [(page where the chunk starts, text)]."""
    text, starts = "", []
    for page, raw in pages:
        raw = normalize_pdf_text(raw) if page != -1 else raw.strip()
        if raw:
            starts.append((len(text), page))
            text += raw + "\n\n" if page == -1 else raw + " "
    if pages and pages[0][0] == -1:  # .md/.txt: keep the paragraph-based splitter
        return [(-1, c) for c in chunk_text(text, size, overlap)]
    # sentence spans (offsets into text); over-long "sentences" are cut at word boundaries
    spans, pos = [], 0
    for m in list(_SENT_END.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        while end - pos > size:
            cut = text.rfind(" ", pos + size // 2, pos + size)
            cut = cut if cut > pos else pos + size
            spans.append((pos, cut)); pos = cut + 1
        if end > pos:
            spans.append((pos, end))
        pos = (m.end() if m else len(text))
    import bisect

    def page_at(offset: int) -> int:
        return starts[max(bisect.bisect_right([a for a, _ in starts], offset) - 1, 0)][1]

    chunks: list[tuple[int, str]] = []
    i = 0
    while i < len(spans):
        j, first = i, spans[i][0]
        while j + 1 < len(spans) and spans[j + 1][1] - first <= size:
            j += 1
        piece = text[first : spans[j][1]].strip()
        if piece:
            chunks.append((page_at(first), piece))
        if j + 1 >= len(spans):
            break
        k = j  # next chunk starts ~overlap chars back, always advancing
        while k > i and spans[j][1] - spans[k][0] < overlap:
            k -= 1
        while k < j and spans[j][1] - spans[k][0] > 2 * overlap:
            k += 1
        i = max(k, i + 1)
    merged: list[tuple[int, str]] = []  # no fragments: fold a too-short chunk into its neighbour
    for pg, c in chunks:
        if merged and len(c) < MIN_CHUNK:
            merged[-1] = (merged[-1][0], f"{merged[-1][1]} {c}")
        elif merged and len(merged[-1][1]) < MIN_CHUNK:
            merged[-1] = (merged[-1][0], f"{merged[-1][1]} {c}")
        else:
            merged.append((pg, c))
    return merged


def load_document(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number or -1, text)]."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(-1, path.read_text(encoding="utf-8", errors="ignore"))]


def resolve_title(path: Path, titles: dict[str, str]) -> str:
    """Human title for a document: corpus/titles.json override, else PDF metadata, else the file name."""
    if path.name in titles:
        return titles[path.name]
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            meta = (PdfReader(str(path)).metadata or {}).title or ""
            if len(meta.strip()) > 8 and "untitled" not in meta.lower():
                return meta.strip()
        except Exception:  # noqa: BLE001
            pass
    return path.stem.replace("_", " ").replace("-", " ")


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def ingest_directory(corpus_dir: str, log=print) -> dict:
    """Ingest every supported file under corpus_dir. Returns per-document stats."""
    root = Path(corpus_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No .pdf/.md/.txt files found under {corpus_dir}")

    col = get_collection()
    embedder = get_embedder()
    stats: dict[str, dict] = {}
    t0 = time.time()
    titles_file = root / "titles.json"
    titles = json.loads(titles_file.read_text(encoding="utf-8")) if titles_file.exists() else {}

    for f in files:
        rel = str(f.relative_to(root))
        title = resolve_title(f, titles)
        ids, docs, metas = [], [], []
        per_page: dict[int, int] = defaultdict(int)
        for page, chunk in chunk_document(load_document(f)):
            ci = per_page[page]
            per_page[page] += 1
            ids.append(f"{_slug(rel)}::p{page}::c{ci}")
            docs.append(chunk)
            metas.append({"source": rel, "title": title, "page": page, "chunk_index": ci})
        for i in range(0, len(ids), EMBED_BATCH):
            b_ids, b_docs, b_metas = (
                ids[i : i + EMBED_BATCH],
                docs[i : i + EMBED_BATCH],
                metas[i : i + EMBED_BATCH],
            )
            col.upsert(
                ids=b_ids,
                documents=b_docs,
                metadatas=b_metas,
                # Contextual header: the paper title is embedded with each chunk (not stored in the
                # text) so a query that names a paper matches its chunks even when the chunk doesn't.
                embeddings=embedder.embed_documents([f"{m['title']}\n\n{d}" for m, d in zip(b_metas, b_docs)]),
            )
        stats[rel] = {"chunks": len(ids), "bytes": f.stat().st_size, "title": title}
        log(f"  {rel}: {len(ids)} chunks")

    log(f"Ingested {len(files)} files, {sum(s['chunks'] for s in stats.values())} chunks "
        f"in {time.time() - t0:.1f}s. Collection size: {col.count()}")
    return stats


# ---------------------------------------------------------------- retrieval

_STOP = frozenset(
    "a an and are as at be by can do does for from has have how in is it its of on or that the this to "
    "was what when where which who why with their they them than then these those into about over also "
    "each between both not no more most other such only same so very".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


class _BM25:
    """In-memory BM25 over chunk text + title, built once from the Chroma collection."""

    K1, B = 1.5, 0.75

    def __init__(self) -> None:
        data = get_collection().get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
            toks = _tokens(f"{meta.get('title', '')} {doc}")
            self.lengths.append(len(toks))
            counts: dict[str, int] = defaultdict(int)
            for t in toks:
                counts[t] += 1
            for t, c in counts.items():
                self.postings[t].append((i, c))
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0

    def top(self, query: str, n: int) -> list[str]:
        scores: dict[int, float] = defaultdict(float)
        n_docs = len(self.ids)
        for t in set(_tokens(query)):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = tf + self.K1 * (1 - self.B + self.B * self.lengths[i] / self.avgdl)
                scores[i] += idf * tf * (self.K1 + 1) / norm
        return [self.ids[i] for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]


@lru_cache(maxsize=1)
def get_bm25() -> _BM25:
    return _BM25()


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """Hybrid retrieval: dense (cosine) and BM25 candidate lists fused with reciprocal rank fusion."""
    k = k or config.RAG_TOP_K
    col = get_collection()
    if col.count() == 0:
        return []
    pool = min(max(k * 6, 30), col.count())
    rows: dict[str, tuple] = {}
    dense_ids: list[str] = []
    qv: list[float] | None = None
    try:
        qv = get_embedder().embed_query(query)
        res = col.query(query_embeddings=[qv], n_results=pool)
        dense_ids = res["ids"][0]
        rows = {
            cid: (doc, meta, 1 - float(dist))  # cosine distance -> similarity
            for cid, doc, meta, dist in zip(dense_ids, res["documents"][0], res["metadatas"][0], res["distances"][0])
        }
    except Exception:  # noqa: BLE001
        if not config.RAG_HYBRID:
            raise
        # Embedding service down (no credit, outage): keyword-only retrieval still works.
    ranked = dense_ids
    focus_ids: list[str] = []
    named = named_sources(query) if config.RAG_PAPER_FOCUS else []
    if named and qv is not None:
        # The query names a paper: that paper's best chunks become a third candidate list.
        where = {"source": named[0]} if len(named) == 1 else {"source": {"$in": named}}
        res = col.query(query_embeddings=[qv], n_results=12, where=where)
        focus_ids = res["ids"][0]
        for cid, doc, meta, dist in zip(focus_ids, res["documents"][0], res["metadatas"][0], res["distances"][0]):
            rows.setdefault(cid, (doc, meta, 1 - float(dist)))
    if config.RAG_HYBRID:
        lexical_ids = get_bm25().top(query, pool)
        fused: dict[str, float] = defaultdict(float)
        for ranking in (dense_ids, lexical_ids, focus_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] += 1.0 / (60 + rank)
        ranked = sorted(fused, key=lambda c: -fused[c])
    ranked = ranked[:k]
    missing = [c for c in ranked if c not in rows]  # lexical-only hits: fetch text (+ similarity if we have a query vector)
    if missing:
        got = col.get(ids=missing, include=["documents", "metadatas", "embeddings"])
        qn = math.sqrt(sum(x * x for x in qv)) if qv else 1.0
        for cid, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]):
            sim = 0.0
            if qv:
                en = math.sqrt(sum(float(x) * float(x) for x in emb)) or 1.0
                sim = sum(float(a) * float(b) for a, b in zip(qv, emb)) / ((qn or 1.0) * en)
            rows[cid] = (doc, meta, sim)
    return [
        {
            "chunk_id": cid,
            "text": rows[cid][0],
            "source": rows[cid][1].get("source"),
            "title": rows[cid][1].get("title"),
            "page": rows[cid][1].get("page", -1),
            "score": round(rows[cid][2], 4),
        }
        for cid in ranked
    ]


def get_chunk(chunk_id: str) -> dict | None:
    res = get_collection().get(ids=[chunk_id])
    if not res["ids"]:
        return None
    meta = res["metadatas"][0]
    return {"chunk_id": chunk_id, "text": res["documents"][0], **meta}


def list_sources() -> list[str]:
    col = get_collection()
    if col.count() == 0:
        return []
    metas = col.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


@lru_cache(maxsize=1)
def _title_map() -> tuple[tuple[str, str], ...]:
    return tuple(list_titles().items())


_RAG_PHRASE = re.compile(r"retrieval[- ]augmented generation", re.I)


def title_names(title: str) -> set[str]:
    """Names a user may call a paper by: the title before ':', a '(Acronym)', a leading acronym
    before ' - ', and each of those with "Retrieval-Augmented Generation" shortened to "RAG"
    (so "Corrective RAG" finds "Corrective Retrieval Augmented Generation")."""
    names = {title.split(":")[0], *re.findall(r"\(([^)]+)\)", title)}
    if " - " in title:
        names.add(title.split(" - ")[0])
    names |= {_RAG_PHRASE.sub("RAG", n) for n in names}
    return names


def named_sources(query: str) -> list[str]:
    """Corpus papers the query names (e.g. "Tree of Thoughts", "the Ragas framework"), at most 2."""
    q = _norm(query)
    hits = []
    for s, title in _title_map():
        names = title_names(title)
        if any(
            len(n := _norm(name)) >= 4 and re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", q)
            for name in names
        ):
            hits.append(s)
    return hits[:2]


def list_titles() -> dict[str, str]:
    col = get_collection()
    if col.count() == 0:
        return {}
    return {m["source"]: m.get("title", "") for m in col.get(include=["metadatas"])["metadatas"]}


def get_document_chunks(source: str, max_chunks: int = 24) -> list[dict]:
    """All chunks for one document in reading order, evenly down-sampled to max_chunks."""
    res = get_collection().get(where={"source": source}, include=["documents", "metadatas"])
    rows = [
        {"chunk_id": i, "text": d, **m}
        for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
    ]
    rows.sort(key=lambda r: (r.get("page", -1), r.get("chunk_index", 0)))
    if len(rows) > max_chunks:
        step = len(rows) / max_chunks
        rows = [rows[int(i * step)] for i in range(max_chunks)]
    return rows


# `rag.retrieve(...)` etc. are used by the agents below.
rag = SimpleNamespace(
    get_collection=get_collection, reset_cache=reset_cache, chunk_text=chunk_text,
    load_document=load_document, ingest_directory=ingest_directory, retrieve=retrieve,
    get_chunk=get_chunk, list_sources=list_sources, list_titles=list_titles, get_document_chunks=get_document_chunks,
)


# ============================================================================
# 7. ROUTER: which LLM provider handles which step, and why
# ============================================================================

@dataclass
class Decision:
    stage: str  # classify | worker:<name> | synthesize
    provider: str
    model: str
    dimension: str  # cost | capability | fallback | forced
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve(preferred: str, alternate: str) -> tuple[str, str | None]:
    """Return (provider, fallback_note). Raises if nothing is configured."""
    avail = config.available_providers()
    if not avail:
        raise RuntimeError("No LLM provider is configured. Set at least two API keys + model ids.")
    if preferred in avail:
        return preferred, None
    if alternate in avail:
        return alternate, f"'{preferred}' not configured, fell back to '{alternate}'"
    return avail[0], f"'{preferred}' not configured, fell back to '{avail[0]}'"


def _model(provider: str) -> str:
    return config.PROVIDERS[provider]["model"]


def route_cheap(stage: str, why: str) -> Decision:
    provider, note = _resolve(config.CHEAP_PROVIDER, config.STRONG_PROVIDER)
    reason = f"{why} -> low-cost provider"
    if note:
        reason += f" ({note})"
    return Decision(stage, provider, _model(provider), "fallback" if note else "cost", reason)


def route_strong(stage: str, why: str) -> Decision:
    """Capability routing for a worker step that needs the stronger model (e.g. exact SQL)."""
    provider, note = _resolve(config.STRONG_PROVIDER, config.CHEAP_PROVIDER)
    reason = f"{why} -> high-capability provider"
    if note:
        reason += f" ({note})"
    return Decision(stage, provider, _model(provider), "fallback" if note else "capability", reason)


SQL_ROWS_STRONG = 10  # a longer SQL result must be copied row for row; the cheap model shortens it


def route_synthesis(complexity: str, n_agents: int, context_chars: int, sql_rows: int = 0) -> Decision:
    """Capability routing for the final answer."""
    forced = config.FORCE_SYNTH_PROVIDER
    if forced and forced in config.available_providers():
        return Decision(
            "synthesize", forced, _model(forced), "forced",
            "FORCE_SYNTH_PROVIDER is set (evaluation mode): routing bypassed",
        )
    reasons = []
    if complexity == "complex":
        reasons.append("classifier marked the query complex")
    if n_agents >= 2:
        reasons.append(f"{n_agents} agents contributed evidence to merge")
    if context_chars > config.COMPLEX_CONTEXT_CHARS:
        reasons.append(f"grounded answer over a substantial evidence set ({context_chars} chars)")
    if sql_rows > SQL_ROWS_STRONG:
        reasons.append(f"the answer must reproduce a {sql_rows}-row SQL result in full")

    if reasons:
        provider, note = _resolve(config.STRONG_PROVIDER, config.CHEAP_PROVIDER)
        why = "; ".join(reasons) + " -> high-capability provider"
        dim = "capability"
    else:
        provider, note = _resolve(config.CHEAP_PROVIDER, config.STRONG_PROVIDER)
        why = (
            f"simple query, {n_agents} agent, {context_chars} chars of evidence "
            "-> low-cost provider is sufficient"
        )
        dim = "cost"
    if note:
        why += f" ({note})"
        dim = "fallback"
    return Decision("synthesize", provider, _model(provider), dim, why)


def fallback_providers(current: str) -> list[str]:
    """Every other configured provider, in the order failover should try them."""
    return [p for p in config.available_providers() if p != current]


def other_provider(current: str) -> str | None:
    """A different configured provider, for mid-request failover."""
    return next(iter(fallback_providers(current)), None)


# ============================================================================
# 8. TAVILY MCP: one shared web-search connection used by two agents
# ============================================================================

CALL_TIMEOUT_S = 25
RETRIEVE_TIMEOUT_S = 45


async def retrieve_chunks(query: str) -> list[dict]:
    """rag.retrieve in a worker thread with a hard deadline (a thread cannot be cancelled, so a
    stalled retrieval would otherwise hang the whole run with nothing in the trace)."""
    return await asyncio.wait_for(asyncio.to_thread(rag.retrieve, query, config.RAG_TOP_K), RETRIEVE_TIMEOUT_S)


def _flatten(result: Any) -> str:
    """MCP tool results come back as a str or a list of content blocks."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for b in result:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(b.get("text", ""))
            else:
                parts.append(getattr(b, "text", str(b)))
        return "\n".join(p for p in parts if p)
    return str(result)


def today_line() -> str:
    """The model has no clock: without this it guesses the date from its training data."""
    return f"Today's date is {time.strftime('%A, %d %B %Y', time.gmtime())} (UTC)."


def dated(system: str) -> str:
    return f"{system}\n\n{today_line()}"


_NEWS_WINDOWS = (
    (r"\b(today|right now|breaking)\b", "day"),
    (r"\b(this week|past week|last week)\b", "week"),
    (r"\b(this month|past month|last month)\b", "month"),
    (r"\b(latest|recent|recently|news|released)\b", "month"),
)


def search_args(query: str) -> dict:
    """Tavily arguments: time-sensitive questions use its news index over a matching window
    (a plain search for 'news this week' returns evergreen help pages)."""
    q = query.lower()
    for pattern, window in _NEWS_WINDOWS:
        if re.search(pattern, q):
            return {"query": query, "topic": "news", "time_range": window}
    return {"query": query}


async def web_search(agent: str, query: str, tr) -> str:
    args = search_args(query)
    try:
        raw = await hub.call(agent, "search", args, tr)
    except RuntimeError as e:  # Tavily's per-minute limit (HTTP 429) clears within seconds: retry once
        if "429" not in str(e):
            raise
        if tr:
            await tr("warn", "Tavily rate limit (HTTP 429); retrying in 3s")
        await asyncio.sleep(3)
        raw = await hub.call(agent, "search", args, tr)
    if "topic" in args and not parse_search_results(raw):  # nothing in the news window: widen it
        raw = await hub.call(agent, "search", {"query": query}, tr)
    return raw


def parse_search_results(raw: str, limit: int = 5) -> list[dict]:
    """Best-effort parse of Tavily's text output into [{title, url, content}].

    Tavily's MCP server returns blocks like 'Title: ...\\nURL: ...\\nContent: ...'. If the
    format differs, fall back to returning the whole text as one result so nothing is lost.
    """
    try:  # current Tavily MCP output: {"results": [{"url", "title", "content"}, ...]}
        data = json.loads(raw)
        items = data.get("results") if isinstance(data, dict) else None
    except ValueError:
        items = None
    if items:
        parsed = [
            {
                "title": str(it.get("title") or it.get("url") or "Web result").strip(),
                "url": str(it.get("url") or "").strip(),
                "content": str(it.get("content") or it.get("raw_content") or "").strip()[:1500],
            }
            for it in items
            if isinstance(it, dict) and (it.get("url") or it.get("content"))
        ]
        if parsed:
            return parsed[:limit]
    blocks = re.split(r"(?=^\s*Title:\s)", raw, flags=re.M)
    results = []
    for b in blocks:
        t = re.search(r"Title:\s*(.+)", b)
        u = re.search(r"URL:\s*(\S+)", b)
        c = re.search(r"Content:\s*(.+?)(?=\n\s*(?:Raw Content|Title|URL):|\Z)", b, flags=re.S)
        if u:
            results.append(
                {
                    "title": (t.group(1).strip() if t else u.group(1)),
                    "url": u.group(1).strip(),
                    "content": (c.group(1).strip() if c else b.strip())[:1500],
                }
            )
    if not results and raw.strip():
        results = [{"title": "Web search result", "url": "", "content": raw.strip()[:3000]}]
    return results[:limit]


def parse_extract_result(raw: str) -> tuple[str, str]:
    """(title, page text) from Tavily extract output ({"results": [{"url", "raw_content"}]});
    plain text is returned unchanged."""
    try:
        data = json.loads(raw)
        item = (data.get("results") or [None])[0] if isinstance(data, dict) else None
    except ValueError:
        item = None
    if isinstance(item, dict) and (item.get("raw_content") or item.get("content")):
        text = str(item.get("raw_content") or item.get("content"))
        return str(item.get("title") or item.get("url") or "Web page"), text
    return "", raw


class TavilyHub:
    """ONE MCP connection shared by the research and summary agents. Every call is logged
    with the calling agent's name (this is your proof for the 'MCP tool sharing' rubric item)."""

    def __init__(self) -> None:
        self._client: MultiServerMCPClient | None = None
        self._tools: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self.call_log: list[dict] = []  # {agent, tool, ms, ok}

    @property
    def enabled(self) -> bool:
        return bool(config.TAVILY_API_KEY)

    async def _load(self) -> dict[str, Any]:
        async with self._lock:
            if self._tools is None:
                self._client = MultiServerMCPClient(
                    {"tavily": {"transport": "streamable_http", "url": config.TAVILY_MCP_URL}}
                )
                tools = await asyncio.wait_for(self._client.get_tools(), CALL_TIMEOUT_S)
                self._tools = {t.name: t for t in tools}
            return self._tools

    async def tool_names(self) -> list[str]:
        return sorted((await self._load()).keys())

    async def _find(self, kind: str):
        tools = await self._load()
        for name, tool in tools.items():
            if kind in name.lower():
                return tool
        raise RuntimeError(f"No Tavily MCP tool matching '{kind}'. Available: {sorted(tools)}")

    async def call(self, agent: str, kind: str, args: dict, tr=None) -> str:
        """kind is 'search' or 'extract'. agent is recorded for the trace."""
        if not self.enabled:
            raise RuntimeError("TAVILY_API_KEY is not set; live web retrieval is disabled.")
        tool = await self._find(kind)
        if kind == "extract" and "urls" in args:
            # Some servers want a list, some a single string: follow the tool's own schema.
            is_array = tool.args.get("urls", {}).get("type") == "array"
            urls = args["urls"] if isinstance(args["urls"], list) else [args["urls"]]
            args = {**args, "urls": urls if is_array else urls[0]}
        t = time.perf_counter()
        ok = False
        try:
            text = _flatten(await asyncio.wait_for(tool.ainvoke(args), CALL_TIMEOUT_S))
            # Tavily reports failures (e.g. a bad key) as ordinary tool text, not as an MCP error.
            try:
                body = json.loads(text)
            except ValueError:
                body = None
            if isinstance(body, dict) and body.get("error") and "results" not in body:
                detail = body.get("detail")
                detail = detail.get("error") if isinstance(detail, dict) else detail
                raise RuntimeError(f"Tavily {body.get('status', 'error')}: {detail or body['error']}")
            ok = True
            return text
        finally:
            ms = int((time.perf_counter() - t) * 1000)
            self.call_log.append({"agent": agent, "tool": tool.name, "ms": ms, "ok": ok})
            if tr is not None:
                await tr("mcp_call", f"agent={agent} tool={tool.name} ok={ok} {ms}ms", tool=tool.name)


hub = TavilyHub()


# ============================================================================
# 9. SUPERVISOR: makes the plan and decides which agent runs next
# ============================================================================

WORKERS = ("research", "summary", "data")
MAX_STEPS = 6

CLASSIFY_SYSTEM = """You are the planning supervisor of a multi-agent system. Decide which \
specialist agents should handle the user's query. Reply with ONLY a JSON object.

Agents:
- research: answers factual/conceptual questions about what the papers say, including comparisons \
across several papers and long multi-part questions, using the document corpus (RAG) and/or live web search.
- summary: ONLY when the user asks for a summary of a specific corpus document (named paper, file \
name or arXiv id), of a URL, or of the latest news.
- data: ONLY for statistics about the indexed collection itself, from a SQLite table with one row \
per document (title, source, chunk count): how many documents or chunks, which has the most, lists \
of titles. Never for what a paper says, even when the question mentions numbers, metrics or benchmarks.

Corpus description: {corpus}
Corpus documents: {sources}

JSON schema:
{{"agents": ["research"|"summary"|"data", ...],
  "needs_corpus": true|false,   // can the corpus plausibly help?
  "needs_live": true|false,     // does it need current / real-time web information?
  "complexity": "simple"|"complex",  // complex = multi-step reasoning, comparison, synthesis across sources
  "reason": "one short sentence"}}

Rules: use the fewest agents that cover the query (usually 1). Questions about current events,
news, prices or "latest" things need needs_live=true. General-knowledge questions the corpus does
not cover (geography, people, facts about the world) go to research with needs_live=true, never to
data. Output JSON only."""


def heuristic_plan(query: str) -> dict:
    """Fallback used if the classifier LLM call or its JSON fails. Also keeps the system
    usable if the cheap provider is down."""
    q = query.lower()
    agents: list[str] = []
    if re.search(r"https?://", q) or re.search(r"\b(summari[sz]e|summary|tl;?dr)\b", q):
        agents.append("summary")
    if re.search(r"\b(how many|count|average|avg|sum|total|per|rows?|table|sql|database|csv|highest|lowest|most|least)\b", q) and "summary" not in agents:
        agents.append("data")
    if not agents:
        agents.append("research")
    needs_live = bool(re.search(r"\b(latest|current|currently|today|news|recent|recently|now|this week|this month|2025|2026|price)\b", q))
    complex_ = len(q.split()) > 25 or bool(re.search(r"\b(compare|versus|vs\.?|trade-?offs?|analy[sz]e|evaluate|design|why)\b", q))
    return {
        "agents": agents,
        "needs_corpus": True,
        "needs_live": needs_live,
        "complexity": "complex" if complex_ else "simple",
        "reason": "heuristic fallback plan (classifier unavailable)",
    }


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# Citation-like labels ([C4], [S1], [U1], ...); earlier turns' labels mean nothing in this turn.
LABEL_RE = re.compile(r"\[([A-Z]{1,2}\d+)\]")


_FOLLOWUP_RE = re.compile(
    r"\b(it|its|they|them|their|those|these|he|she|again|previous|earlier|above|former|latter|same|"
    r"what about|how about|and the|elaborate|expand|continue)\b",
    re.I,
)


def is_followup(query: str) -> bool:
    """Only a follow-up question needs the earlier turns ("what datasets did it use?"). A
    self-contained one (a URL, a full standalone question) is answered without them, so
    the previous answer cannot leak into this one."""
    if re.search(r"https?://", query):
        return False
    return len(query.split()) <= 6 or bool(_FOLLOWUP_RE.search(query))


def _history_snippet(history: list[dict], n: int = 4, query: str = "") -> str:
    if not is_followup(query):
        return ""
    return "\n".join(f"{h['role']}: {LABEL_RE.sub('', h['content'])[:300]}" for h in (history or [])[-n:])


# NOTE: the parameter MUST be named `config` - LangGraph looks for that exact name.
# That is why the settings are reached through the alias `app_config` inside these functions.
async def supervisor_node(state: State, config: RunnableConfig) -> dict:
    run: Run = config["configurable"]["run"]
    tr = Tracer(run, "supervisor")

    # ---- first visit: classify + plan
    if not state.get("planned"):
        await tr("start", f"query: {state['query'][:120]}")
        decision = route_cheap("classify", "query classification is a short, structured task")
        await run.emit("route", **decision.to_dict(), agent="supervisor")
        await tr("route", f"classify -> {decision.provider} ({decision.dimension}): {decision.reason}")

        try:
            sources = rag.list_sources()
        except Exception:  # corpus not ingested yet; planning can still proceed
            sources = []

        plan = None
        try:
            system = CLASSIFY_SYSTEM.format(
                corpus=app_config.CORPUS_DESCRIPTION, sources=", ".join(sources[:40]) or "(none)"
            )
            user = state["query"]
            hist = _history_snippet(state.get("history", []), query=state["query"])
            if hist:
                user = f"Recent conversation:\n{hist}\n\nCurrent query: {state['query']}"
            raw = await call_llm_fo(run, tr, decision, [SystemMessage(dated(system)), HumanMessage(user)])
            plan = _parse_json(raw)
            if plan is None:
                await tr("warn", "classifier returned invalid JSON; using heuristic plan")
        except Exception as e:  # noqa: BLE001 - any provider failure must not kill the graph
            await tr("warn", f"classifier call failed ({type(e).__name__}: {e}); using heuristic plan")

        if plan is None:
            plan = heuristic_plan(state["query"])

        agents = [a for a in plan.get("agents", []) if a in WORKERS] or ["research"]
        agents = list(dict.fromkeys(agents))  # de-dupe, keep order
        if "summary" not in agents and re.search(r"\bsummar(y|ise|ize)\b|\btl;?dr\b", state["query"], re.I):
            agents.insert(0, "summary")  # an explicit summary request always gets the summary agent
            await tr("plan", "query asks for a summary -> summary agent added to the plan")
        if (
            "summary" not in agents
            and "research" in agents
            and plan.get("needs_live")
            and re.search(r"\b(compare|contrast|versus|vs\.?)\b", state["query"], re.I)
            and _NEWS_RE.search(_summary_target(state["query"]))
        ):
            agents.insert(0, "summary")  # news on one side of a comparison: the summary agent gathers it
            await tr("plan", "query compares live news with other material -> summary agent added to the plan")
        complexity = "complex" if plan.get("complexity") == "complex" or len(agents) > 1 else "simple"
        update = {
            "planned": True,
            "plan": agents,
            "needs_corpus": bool(plan.get("needs_corpus", True)),
            "needs_live": bool(plan.get("needs_live", False)),
            "complexity": complexity,
            "plan_reason": str(plan.get("reason", ""))[:300],
            "steps": 0,
            "routing": [{**decision.to_dict(), "agent": "supervisor"}],
        }
        await tr(
            "plan",
            f"agents={agents} corpus={update['needs_corpus']} live={update['needs_live']} "
            f"complexity={complexity} | {update['plan_reason']}",
        )
        state = {**state, **update}  # so the pick below sees the plan
    else:
        update = {}

    # ---- every visit: pick the next worker from state
    done = set(state.get("completed", []))
    steps = state.get("steps", 0) + 1
    remaining = [a for a in state["plan"] if a not in done]
    batch = remaining if remaining and steps <= MAX_STEPS else ["synthesize"]
    where = "in parallel" if len(batch) > 1 else "now"
    await tr("dispatch", f"completed={sorted(done)} -> next={batch} ({where})")
    update.update({"next": batch, "slots": {a: i for i, a in enumerate(batch)}, "steps": steps, "trace": tr.entries})
    return update


def next_node(state: State) -> list[str]:
    return state["next"]  # several names = LangGraph runs those workers in parallel


# ============================================================================
# 10. AGENTS: research, summary, data (different tools + different prompts)
# ============================================================================

EVIDENCE_CHARS = 1200
SQL_EVIDENCE_CHARS = 8000


# ---------------------------------------------------------------- shared helpers

def safe_worker(name: str):
    """A worker exception becomes an entry in state['errors']; the graph keeps going."""

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state: State, config: RunnableConfig) -> dict:  # must be named `config`
            run: Run = config["configurable"]["run"]
            tr = Tracer(run, name)
            try:
                update = await fn(state, config, run, tr)
                for c in update.get("citations", []):
                    c.setdefault("agent", name)
                return update
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                await tr("error", f"{name} agent failed: {msg}")
                return {
                    "completed": [name],
                    "errors": [{"agent": name, "error": msg}],
                    "trace": tr.entries,
                }

        # LangGraph inspects the signature to decide whether to pass `config`. wraps() sets
        # __wrapped__, which makes inspect.signature report the inner function instead.
        del wrapper.__wrapped__
        return wrapper

    return deco


# Workers of one parallel batch cannot see each other's citations, so each starts its labels in
# its own block; the synthesizer renumbers everything densely (C1, C2, ...) afterwards.
LABEL_BLOCK = {"C": 40, "W": 10, "S": 5}


def _next_label(state: State, prefix: str, agent: str = "") -> int:
    top = max((int(c["id"][1:]) for c in state.get("citations", []) if c["id"].startswith(prefix)), default=0)
    return top + 1 + state.get("slots", {}).get(agent, 0) * LABEL_BLOCK[prefix]


def _short(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _corpus_citation(label: str, ch: dict) -> dict:
    page = ch.get("page", -1)
    return {
        "id": label,
        "kind": "corpus",
        "source": ch.get("source"),
        "title": ch.get("title"),
        "chunk_id": ch["chunk_id"],
        "page": page if page and page > 0 else None,
        "snippet": _short(ch["text"], 240),
        "text": ch["text"],
    }


def _evidence_block(c: dict) -> str:
    where = c["source"] if c["kind"] != "web" else (c.get("url") or c["title"])
    if c["kind"] == "corpus" and c.get("page"):
        where += f", p.{c['page']}"
    # A SQL result is one table: cutting it to a chunk-sized snippet would hide most of its rows.
    limit = SQL_EVIDENCE_CHARS if c["kind"] == "sql" else EVIDENCE_CHARS
    return f"[{c['id']}] ({where})\n{_short(c['text'], limit)}"


# ---------------------------------------------------------------- research agent

RESEARCH_SYSTEM = """You are the RESEARCH agent. You receive evidence snippets, each labelled \
like [C1] (document corpus) or [W1] (live web). Write concise research notes that answer the \
question using ONLY that evidence.
- After every claim, cite the label(s) that directly state it, e.g. "... [C2][W1]".
- Only write a claim if a snippet says it, in or close to your wording. Do NOT add background \
knowledge, comparisons, evaluations or generalisations ("typically", "usually", "more efficient") \
that no snippet states.
- Prefer snippets from the paper the question is about; ignore snippets that merely mention it.
- Never invent a label and never cite a source that is not in the evidence.
- If the evidence does not answer the question (or part of it), say exactly what is missing instead \
of filling the gap."""


@safe_worker("research")
async def research_node(state: State, cfg: RunnableConfig, run: Run, tr: Tracer) -> dict:
    query = state["query"]
    await tr("start", "research agent: RAG retrieval + live web as needed")
    new_citations: list[dict] = []
    n_corpus = _next_label(state, "C", "research")
    n_web = _next_label(state, "W", "research")
    warnings: list[str] = []

    if state.get("needs_corpus", True):
        t = time.perf_counter()
        chunks = await retrieve_chunks(query)
        run.retrieval_ms += (time.perf_counter() - t) * 1000
        for i, ch in enumerate(chunks):
            new_citations.append(_corpus_citation(f"C{n_corpus + i}", ch))
        top = f"{chunks[0]['score']:.2f}" if chunks else "n/a"
        await tr("tool", f"rag.retrieve k={config.RAG_TOP_K} -> {len(chunks)} chunks (top score {top})", tool="rag.retrieve")

    if state.get("needs_live"):
        if not hub.enabled:
            warnings.append("live search requested but TAVILY_API_KEY is not set")
            await tr("warn", warnings[-1])
        else:
            try:
                raw = await web_search("research", query, tr)
                results = parse_search_results(raw)
                for i, r in enumerate(results):
                    new_citations.append(
                        {
                            "id": f"W{n_web + i}",
                            "kind": "web",
                            "source": r["url"] or "web",
                            "url": r["url"],
                            "title": r["title"],
                            "snippet": _short(r["content"], 240),
                            "text": r["content"],
                        }
                    )
            except Exception as e:  # noqa: BLE001 - degrade to corpus-only, don't fail the agent
                warnings.append(f"live search failed: {type(e).__name__}: {e}")
                await tr("warn", warnings[-1])

    if not new_citations:
        await tr("warn", "no evidence retrieved")
        return {
            "completed": ["research"],
            "agent_outputs": {"research": "No evidence was retrieved. " + " ".join(warnings)},
            "errors": [{"agent": "research", "error": w} for w in warnings],
            "trace": tr.entries,
        }

    decision = route_cheap("worker:research", "condensing retrieved snippets into cited notes")
    await run.emit("route", **decision.to_dict(), agent="research")
    await tr("route", f"{decision.provider} ({decision.dimension}): {decision.reason}")
    evidence = "\n\n".join(_evidence_block(c) for c in new_citations)
    notes = await call_llm_fo(
        run,
        tr,
        decision,
        [SystemMessage(dated(RESEARCH_SYSTEM)), HumanMessage(f"Question: {query}\n\nEvidence:\n{evidence}")],
        max_tokens=WORKER_NOTES_MAX_TOKENS,
    )
    await tr("done", f"{len(new_citations)} citations, notes {len(notes)} chars")
    return {
        "completed": ["research"],
        "agent_outputs": {"research": notes},
        "citations": new_citations,
        "routing": [{**decision.to_dict(), "agent": "research"}],
        "errors": [{"agent": "research", "error": w} for w in warnings],
        "trace": tr.entries,
    }


# ---------------------------------------------------------------- summary agent

SUMMARY_SYSTEM = """You are the SUMMARY agent. Summarize the provided material for the user's \
request. The material is labelled with citation labels like [C1], [W1].
- Preserve citations: attach the label(s) of the passage each point came from, e.g. "... [C3]".
- Cover the main points in the order they appear; do not add facts that are not in the material.
- Use a short paragraph or a tight bullet list, whichever the request implies."""

URL_RE = re.compile(r"https?://[^\s)>\]]+")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _match_source(query: str, sources: list[str], titles: dict[str, str] | None = None) -> str | None:
    """Pick the corpus document a query refers to: by file name, arXiv-style id, or paper name
    (the part of the title before ':' and any '(Acronym)')."""
    q = _norm(query)
    for s in sources:
        stem = Path(s).stem
        if s.lower() in query.lower() or _norm(stem) in q:
            return s
    for s, title in (titles or {}).items():
        for name in title_names(title):
            n = _norm(name)
            if len(n) >= 4 and re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", q):
                return s
    return None


_NEWS_RE = re.compile(r"\b(news|latest|recent|recently|today|this week|this month)\b", re.I)


def _summary_target(query: str) -> str:
    """The part of the request that asks for a summary, before any comparison clause.
    For "Compare X with Y" the first subject (X) comes after the keyword."""
    parts = re.split(r"\b(?:and\s+)?(?:compare|contrast|versus|vs\.?)\b", query, maxsplit=1, flags=re.I)
    if not parts[0].strip() and len(parts) > 1:
        return re.split(r"\b(?:with|to|against|and|versus|vs\.?)\b", parts[1], maxsplit=1, flags=re.I)[0]
    return parts[0]


@safe_worker("summary")
async def summary_node(state: State, cfg: RunnableConfig, run: Run, tr: Tracer) -> dict:
    query = state["query"]
    await tr("start", "summary agent: choose material, summarize with citations")
    n_corpus = _next_label(state, "C", "summary")
    n_web = _next_label(state, "W", "summary")
    new_citations: list[dict] = []

    urls = URL_RE.findall(query)
    sources = await asyncio.to_thread(rag.list_sources)
    titles = await asyncio.to_thread(rag.list_titles)
    # "Summarize the latest news ... and compare it with <paper>": the summary is of the news;
    # the paper named after "compare" is left to the research agent.
    news_first = bool(state.get("needs_live") and hub.enabled and _NEWS_RE.search(_summary_target(query)))
    src = None if news_first else _match_source(query, sources, titles)

    if urls:
        raw = await hub.call("summary", "extract", {"urls": urls[:1]}, tr)
        page_title, page_text = parse_extract_result(raw)
        new_citations.append(
            {
                "id": f"W{n_web}",
                "kind": "web",
                "source": urls[0],
                "url": urls[0],
                "title": page_title or urls[0],
                "snippet": _short(page_text, 240),
                "text": page_text[:12000],
            }
        )
        await tr("tool", f"tavily extract {urls[0]}", tool="extract")
    elif src:
        t = time.perf_counter()
        chunks = await asyncio.to_thread(rag.get_document_chunks, src, 24)
        run.retrieval_ms += (time.perf_counter() - t) * 1000
        for i, ch in enumerate(chunks):
            new_citations.append(_corpus_citation(f"C{n_corpus + i}", ch))
        await tr("tool", f"rag.get_document_chunks('{src}') -> {len(chunks)} chunks (evenly sampled)", tool="rag.get_document_chunks")
    elif state.get("needs_live") and hub.enabled:
        raw = await web_search("summary", query, tr)
        for i, r in enumerate(parse_search_results(raw)):
            new_citations.append(
                {
                    "id": f"W{n_web + i}",
                    "kind": "web",
                    "source": r["url"] or "web",
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": _short(r["content"], 240),
                    "text": r["content"],
                }
            )
    else:
        # Nothing specific to fetch: summarize what earlier agents already gathered.
        new_citations = []
        existing = state.get("citations", [])
        if not existing:
            chunks = await retrieve_chunks(query)
            for i, ch in enumerate(chunks):
                new_citations.append(_corpus_citation(f"C{n_corpus + i}", ch))
            await tr("tool", f"rag.retrieve -> {len(chunks)} chunks (no specific document named)", tool="rag.retrieve")
        else:
            await tr("tool", f"reusing {len(existing)} citations from earlier agents")

    material_cites = new_citations or state.get("citations", [])
    if not material_cites:
        raise RuntimeError("no material to summarize (no matching document, URL or retrieved chunks)")

    decision = route_cheap("worker:summary", "summarizing already-selected passages")
    await run.emit("route", **decision.to_dict(), agent="summary")
    await tr("route", f"{decision.provider} ({decision.dimension}): {decision.reason}")
    material = "\n\n".join(_evidence_block(c) for c in material_cites)
    summary = await call_llm_fo(
        run,
        tr,
        decision,
        [SystemMessage(dated(SUMMARY_SYSTEM)), HumanMessage(f"Request: {query}\n\nMaterial:\n{material}")],
        max_tokens=WORKER_NOTES_MAX_TOKENS,
    )
    await tr("done", f"summary {len(summary)} chars from {len(material_cites)} passages")
    return {
        "completed": ["summary"],
        "agent_outputs": {"summary": summary},
        "citations": new_citations,
        "routing": [{**decision.to_dict(), "agent": "summary"}],
        "trace": tr.entries,
    }


# ---------------------------------------------------------------- data agent

SQL_SYSTEM = """You are the DATA agent. Write ONE read-only SQLite query that answers the \
question using only the schema below. Rules: a single SELECT (or WITH ... SELECT) statement, \
no semicolons inside, no writes, add LIMIT 50 unless aggregating. If the question asks about \
each, every or all items, return every matching row (no smaller LIMIT). The query must read from \
the tables below; never SELECT a constant or answer from your own knowledge. The question may have \
several parts handled by other agents (paper content, web news, summaries): write SQL only for the \
part about the indexed collection's own statistics, and ignore the rest. If NO part of the question \
can be answered from the tables, reply with exactly NO_SQL. Otherwise reply with ONLY the SQL, no markdown \
fences, no explanation.

Schema:
{schema}"""

DATA_ANSWER_SYSTEM = """You are the DATA agent. Given a question, the SQL that was run, and its \
result rows, state the answer plainly using only those rows. Mention the key numbers. If the \
rows are empty, say so. If the rows are a list (one row per item), include every row, as a \
table; do not shorten it to a few examples. Cite the query as [S1] once (for example after the \
first sentence or under the table), not on every row."""

FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex|truncate)\b",
    re.I,
)


def validate_sql(sql: str) -> str:
    sql = sql.strip().strip("`").strip()
    sql = re.sub(r"^sql\s*\n", "", sql, flags=re.I).strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise ValueError("only SELECT queries are allowed")
    if FORBIDDEN_SQL.search(sql):
        raise ValueError("query contains a forbidden keyword")
    if not re.search(r"\bfrom\b", sql, re.I):
        # e.g. SELECT 'Canberra': the model's own knowledge dressed up as a database result
        raise ValueError("query does not read from any table")
    return sql


def _connect_ro() -> sqlite3.Connection:
    path = Path(config.SQLITE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {path}. Run `python app.py ingest` or `python app.py csv`."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def describe_schema() -> str:
    conn = _connect_ro()
    try:
        out = []
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
            out.append(f'TABLE {t} (' + ", ".join(f"{c[1]} {c[2]}" for c in cols) + ")")
            rows = conn.execute(f'SELECT * FROM "{t}" LIMIT 3').fetchall()
            for r in rows:
                out.append("  e.g. " + repr(r))
        return "\n".join(out)
    finally:
        conn.close()


def run_select(sql: str, limit: int = 50) -> tuple[list[str], list[tuple]]:
    conn = _connect_ro()
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        return cols, cur.fetchmany(limit)
    finally:
        conn.close()


def _clean_sql(raw: str) -> str:
    return re.sub(r"^```(?:sql)?|```$", "", raw.strip(), flags=re.M).strip()


async def _hand_to_research(state: State, tr: Tracer) -> dict:
    """The database cannot answer this question: add the research agent to the plan (with web
    search allowed) instead of inventing a query. The supervisor picks it up from state."""
    plan = list(state.get("plan", []))
    if "research" not in plan:
        plan.append("research")
    await tr("handoff", "not a question about the indexed collection -> handing over to research (corpus + web)")
    return {
        "completed": ["data"],
        "plan": plan,
        "needs_corpus": True,
        "needs_live": True,
        "trace": tr.entries,
    }


@safe_worker("data")
async def data_node(state: State, cfg: RunnableConfig, run: Run, tr: Tracer) -> dict:
    query = state["query"]
    await tr("start", "data agent: text-to-SQL over read-only SQLite")
    schema = await asyncio.to_thread(describe_schema)
    await tr("tool", f"sqlite.describe_schema ({schema.count('TABLE ')} tables)", tool="sqlite.schema")

    decision = route_strong("worker:data", "text-to-SQL must be exact (a wrong column or join gives a wrong number)")
    await run.emit("route", **decision.to_dict(), agent="data")
    await tr("route", f"{decision.provider} ({decision.dimension}): {decision.reason}")

    sql, cols, rows, last_err = "", [], [], ""
    for attempt in (1, 2):  # one retry, feeding the error back
        prompt = f"Question: {query}"
        if last_err:
            prompt += f"\n\nYour previous query was:\n{sql}\nIt failed with: {last_err}\nWrite a corrected query."
        raw = await call_llm_fo(run, tr, decision, [SystemMessage(SQL_SYSTEM.format(schema=schema)), HumanMessage(prompt)])
        if raw.strip().strip("`").upper() == "NO_SQL":
            return await _hand_to_research(state, tr)
        try:
            sql = validate_sql(_clean_sql(raw))
            cols, rows = await asyncio.to_thread(run_select, sql)
            await tr("tool", f"sqlite.run_select attempt {attempt}: {sql} -> {len(rows)} rows", tool="sqlite.select")
            last_err = ""
            break
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            sql = sql or _clean_sql(raw)
            await tr("warn", f"SQL attempt {attempt} failed: {last_err}")
    if last_err:
        if "does not read from any table" in last_err:
            return await _hand_to_research(state, tr)
        raise RuntimeError(f"could not produce a valid query: {last_err}")

    table =" | ".join(cols) + "\n" + "\n".join(" | ".join(str(v) for v in r) for r in rows)
    answer = await call_llm_fo(
        run,
        tr,
        decision,
        [
            SystemMessage(DATA_ANSWER_SYSTEM),
            HumanMessage(f"Question: {query}\n\nSQL:\n{sql}\n\nRows ({len(rows)}):\n{table[:SQL_EVIDENCE_CHARS]}"),
        ],
    )
    n_sql = _next_label(state, "S", "data")
    cite = {
        "id": f"S{n_sql}",
        "kind": "sql",
        "source": f"sqlite:{Path(config.SQLITE_PATH).name}",
        "title": "SQL query",
        "rows": len(rows),
        "snippet": _short(sql, 240),
        "text": f"{sql}\n\n{table[:SQL_EVIDENCE_CHARS]}",
    }
    await tr("done", f"{len(rows)} rows -> answer {len(answer)} chars")
    return {
        "completed": ["data"],
        "agent_outputs": {"data": answer},
        "citations": [cite],
        "routing": [{**decision.to_dict(), "agent": "data"}],
        "trace": tr.entries,
    }


# ============================================================================
# 11. FINAL ANSWER + GRAPH: merge the agents' work, stream the answer, wire the graph
# ============================================================================

SYNTH_SYSTEM = """You are the FINAL ANSWER writer in a multi-agent system. Specialist agents have \
gathered notes and evidence. Write the answer to the user's question.
- Ground every factual claim in the evidence. Cite with the bracket labels exactly as given, \
e.g. [C1], [W2], [S1]. Only use labels that appear in the evidence. A table or list taken from \
one SQL result needs its [S#] label once (e.g. after the table), not on every row. When the \
user asks about each or all items and the data agent returned a list, include every row.
- Cite even short or well-known facts when a snippet states them (e.g. "Canberra [W1]").
- For "this week", "this month" or "latest", use today's date below; do not guess the date, and \
say when an item's date is unclear.
- Use only what the notes and evidence state. Do not add background knowledge, qualifiers or \
evaluations ("typically", "more efficient", "widely used") that they do not contain, even if you \
believe them to be true.
- If an agent failed or the evidence does not cover something, say so plainly. Do not invent \
facts or citations. If there is no evidence at all, say you could not retrieve any, and keep \
any general-knowledge remark short and clearly marked as ungrounded.
- Answer only the CURRENT question. Earlier turns are context only, not evidence: never cite a \
label from them and never repeat or redo an earlier answer.
- Be direct and concise."""

MAX_EVIDENCE_BLOCKS = 12


def renumber_citations(citations: list[dict], outputs: dict, plan: list[str]) -> tuple[list[dict], dict]:
    """Workers that ran in parallel numbered their citations in separate blocks (C1, C41, W1, W11).
    Put the citations in plan order, number each kind 1..n, and rewrite the labels in the agents'
    notes to match, so the answer and the Citations tab show C1, C2, C3... as before."""
    rank = {a: i for i, a in enumerate(plan)}
    ordered = sorted(citations, key=lambda c: (rank.get(c.get("agent"), len(rank)), c["id"][0], int(c["id"][1:])))
    count: dict[str, int] = {}
    mapping: dict[str, str] = {}
    fixed = []
    for c in ordered:
        count[c["id"][0]] = count.get(c["id"][0], 0) + 1
        new = f"{c['id'][0]}{count[c['id'][0]]}"
        mapping[c["id"]] = new
        fixed.append({**c, "id": new})
    sub = lambda text: LABEL_RE.sub(lambda m: f"[{mapping.get(m.group(1), m.group(1))}]", text) if isinstance(text, str) else text
    return fixed, {name: sub(text) for name, text in outputs.items()}


def _history_text(history: list[dict], n: int = 4, query: str = "") -> str:
    if not is_followup(query):
        return ""
    return "\n".join(f"{h['role']}: {LABEL_RE.sub('', h['content'])[:400]}" for h in (history or [])[-n:])


async def synthesizer_node(state: State, config: RunnableConfig) -> dict:  # must be named `config`
    run: Run = config["configurable"]["run"]
    tr = Tracer(run, "synthesizer")
    citations = state.get("citations", [])
    outputs = state.get("agent_outputs", {})
    errors = state.get("errors", [])

    citations, outputs = renumber_citations(citations, outputs, state.get("plan", []))
    notes = "\n\n".join(f"--- notes from {name} agent ---\n{text}" for name, text in outputs.items())
    evidence = "\n\n".join(_evidence_block(c) for c in citations[:MAX_EVIDENCE_BLOCKS])
    err_text = "\n".join(f"- {e['agent']} agent: {e['error']}" for e in errors)
    context_chars = len(notes) + len(evidence)

    decision = route_synthesis(
        state.get("complexity", "simple"),
        len(outputs),
        context_chars,
        max((c.get("rows", 0) for c in citations if c["kind"] == "sql"), default=0),
    )
    await run.emit("route", **decision.to_dict(), agent="synthesizer")
    await tr("route", f"synthesize -> {decision.provider} ({decision.dimension}): {decision.reason}")

    user = f"Question: {state['query']}\n"
    hist = _history_text(state.get("history", []), query=state["query"])
    if hist:
        user = (
            "Earlier turns, only for resolving words like 'it' or 'that'. Do not repeat, redo or "
            f"continue them:\n{hist}\n\n" + user.replace("Question:", "CURRENT question (answer only this):", 1)
        )
    user += f"\nAgent notes:\n{notes or '(none)'}\n\nEvidence:\n{evidence or '(none)'}"
    if err_text:
        user += f"\n\nAgent problems:\n{err_text}"
    messages = [SystemMessage(dated(SYNTH_SYSTEM)), HumanMessage(user)]

    provider = decision.provider
    routing = [{**decision.to_dict(), "agent": "synthesizer"}]
    try:
        answer = await call_llm(run, provider, messages, stream_tokens=True)
    except Exception as e:  # noqa: BLE001 - provider outage: fail over, trying every other provider
        failed, err, answer = provider, e, None
        for alt in fallback_providers(provider):
            await tr("warn", f"{llm_error_text(failed, err)}; failing over to {alt}")
            await run.emit("reset")  # UI clears any partial tokens
            fo = {
                "stage": "synthesize",
                "provider": alt,
                "model": app_config.PROVIDERS[alt]["model"],
                "dimension": "fallback",
                "reason": f"{failed} errored mid-request; failover to {alt}",
                "agent": "synthesizer",
            }
            await run.emit("route", **fo)
            routing.append(fo)
            try:
                answer = await call_llm(run, alt, messages, stream_tokens=True)
                break
            except Exception as e2:  # noqa: BLE001
                failed, err = alt, e2
        if answer is None:
            raise err

    valid = {c["id"] for c in citations}
    bad = sorted(set(LABEL_RE.findall(answer)) - valid)
    if bad:  # a label with no evidence behind it: drop it and resend the cleaned answer
        answer = re.sub(r"[ \t]*\[(" + "|".join(map(re.escape, bad)) + r")\]", "", answer)
        await tr("warn", f"removed citation labels with no matching evidence: {bad}")
        await run.emit("reset")
        await run.emit("token", text=answer)
    used = set(re.findall(r"\[([CWS]\d+)\]", answer))
    await tr("done", f"answer {len(answer)} chars, cites {sorted(used) or 'none'}")
    await run.emit(
        "citations",
        items=[
            {k: v for k, v in c.items() if k != "text"} | {"used": c["id"] in used}
            for c in citations
        ],
    )
    return {"answer": answer, "routing": routing, "trace": tr.entries}


def build_graph():
    g = StateGraph(State)
    g.add_node("supervisor", supervisor_node)
    g.add_node("research", research_node)
    g.add_node("summary", summary_node)
    g.add_node("data", data_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        next_node,
        {"research": "research", "summary": "summary", "data": "data", "synthesize": "synthesizer"},  # a list fans out
    )
    for worker in ("research", "summary", "data"):
        g.add_edge(worker, "supervisor")  # workers always return to the supervisor
    g.add_edge("synthesizer", END)
    return g.compile()


# ============================================================================
# 12. WEB SERVER: /chat streams events to the browser
# ============================================================================
# SSE event types sent to the browser (JSON payloads):
#   status {message} | route {stage, agent, provider, model, dimension, reason}
#   trace {node, event, detail} | token {text} | reset {} (discard partial answer on failover)
#   citations {items} | error {message} | done {metrics, plan, completed, errors}

RATE_LIMIT_PER_MIN = int(config.env("RATE_LIMIT_PER_MIN", "12"))
MAX_CONCURRENT_RUNS = int(config.env("MAX_CONCURRENT_RUNS", "6"))
RUN_DEADLINE_S = float(config.env("RUN_DEADLINE_S", "240"))  # below Cloud Run's 300 s request timeout

_graph = None
_run_slots = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
_hits: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    ok, msg = provider_rule()
    if not ok and config.env("ALLOW_SINGLE_PROVIDER", "0") not in ("1", "true"):
        raise RuntimeError(f"Provider rule violated: {msg}. (ALLOW_SINGLE_PROVIDER=1 is for offline measurements only.)")
    print(f"[startup] provider rule {'OK' if ok else 'VIOLATED (ALLOW_SINGLE_PROVIDER)'}: {msg}")
    _graph = build_graph()
    try:  # open the vector store at startup so the first request doesn't pay for it
        await asyncio.to_thread(rag.get_collection)
    except Exception as e:  # noqa: BLE001
        print(f"[startup] vector store not available: {scrub(f'{type(e).__name__}: {e}')}")
    yield


app = FastAPI(title="Multi-Agent RAG", lifespan=lifespan)


HISTORY_TURN_CHARS = 4000


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    # A long earlier answer is shortened, not rejected: rejecting it broke every later request
    # in that browser session, because the page resends the last turns each time.
    content: str = Field(max_length=200_000)

    @field_validator("content")
    @classmethod
    def _shorten(cls, v: str) -> str:
        return v[:HISTORY_TURN_CHARS]


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=12)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(scrub_obj(data), default=str)}\n\n"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        return True
    q.append(now)
    return False


async def _run_graph(run: Run, query: str, history: list[dict]) -> None:
    final: dict = {}
    try:
        async with _run_slots:
            final = await asyncio.wait_for(
                _graph.ainvoke({"query": query, "history": history}, config={"configurable": {"run": run}}),
                RUN_DEADLINE_S,
            )
    except asyncio.TimeoutError:
        await run.emit(
            "error",
            message=f"This question took longer than {RUN_DEADLINE_S:.0f} s and was stopped. "
            "Try a shorter or more focused question.",
        )
    except Exception as e:  # noqa: BLE001
        await run.emit("error", message=f"{type(e).__name__}: {e}")
    finally:
        await run.emit(
            "done",
            metrics=run.metrics(),
            plan=final.get("plan"),
            completed=final.get("completed"),
            errors=final.get("errors"),
        )
        await run.queue.put(None)


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT_PER_MIN} requests/minute. Try again shortly.")

    run = Run()
    history = [t.model_dump() for t in req.history]
    task = asyncio.create_task(_run_graph(run, req.query, history))

    async def stream():
        try:
            yield _sse("status", {"message": "connected"})
            while True:
                try:
                    ev = await asyncio.wait_for(run.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if ev is None:
                    break
                yield _sse(ev.pop("type"), ev)
        finally:
            if not task.done():  # client went away: stop spending money
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def api_config():
    try:
        chunks = await asyncio.to_thread(lambda: rag.get_collection().count())
    except Exception:  # noqa: BLE001
        chunks = 0
    return {
        "providers": {p: config.PROVIDERS[p]["model"] for p in config.available_providers()},
        "cheap_provider": config.CHEAP_PROVIDER,
        "strong_provider": config.STRONG_PROVIDER,
        "two_provider_rule": provider_rule()[0],
        "tavily_enabled": hub.enabled,
        "corpus_chunks": chunks,
        "mcp_call_log": hub.call_log[-20:],
    }


@app.get("/api/chunk/{chunk_id:path}")
async def api_chunk(chunk_id: str):
    ch = await asyncio.to_thread(rag.get_chunk, chunk_id)
    if not ch:
        return JSONResponse({"error": "chunk not found"}, status_code=404)
    return ch


# ============================================================================
# 13. THE WEB PAGE (HTML + CSS + JavaScript, served at /)
# ============================================================================

INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Agent RAG</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #1c2330; --muted: #667085; --line: #e3e7ee;
    --accent: #3b5bdb; --accent-ink: #ffffff; --user: #eaf0ff; --code: #f1f3f7;
    --cost: #0b7a4b; --cap: #9a3412; --fb: #b42318;
    --c-supervisor: #3b5bdb; --c-research: #0b7a4b; --c-summary: #9a3412;
    --c-data: #7a3db8; --c-synthesizer: #b42318;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #11141a; --panel: #1a1f29; --ink: #e6e9ef; --muted: #98a2b3; --line: #2a3140;
      --accent: #7c93ff; --accent-ink: #11141a; --user: #232c45; --code: #232a37;
      --cost: #4ade80; --cap: #fb923c; --fb: #f87171;
      --c-supervisor: #7c93ff; --c-research: #4ade80; --c-summary: #fb923c;
      --c-data: #c084fc; --c-synthesizer: #f87171;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  header { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center;
           padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--line); }
  header h1 { font-size: 16px; margin: 0; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line);
          color: var(--muted); background: var(--bg); }
  #status { margin-left: auto; font-size: 12px; color: var(--muted); }
  main { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 12px;
         padding: 12px; height: calc(100vh - 53px); }
  @media (max-width: 900px) { main { grid-template-columns: minmax(0, 1fr); height: auto; } }
  .col { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
         display: flex; flex-direction: column; min-height: 0; min-width: 0; }
  #messages { flex: 1; overflow-y: auto; padding: 14px; min-height: 240px; }
  .msg { margin: 0 0 14px; max-width: 100%; }
  .msg .who { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  .bubble { white-space: pre-wrap; overflow-wrap: anywhere; padding: 8px 12px; border-radius: 10px;
            background: var(--bg); border: 1px solid var(--line); }
  .msg.user .bubble { background: var(--user); }
  .msg.error .bubble { border-color: var(--fb); color: var(--fb); }
  .cite { display: inline-block; font-size: 11px; font-weight: 600; padding: 0 5px; margin: 0 1px;
          border-radius: 4px; background: var(--accent); color: var(--accent-ink); cursor: pointer;
          text-decoration: none; vertical-align: baseline; }
  .routes { margin-top: 6px; display: grid; gap: 4px; }
  .route { overflow-wrap: anywhere; font-size: 12px; border: 1px solid var(--line); border-left-width: 4px; border-radius: 6px;
           padding: 4px 8px; background: var(--panel); }
  .route .agent { font-weight: 600; }
  .route .why { color: var(--muted); }
  .badge { font-size: 10px; text-transform: uppercase; padding: 0 6px; border-radius: 4px;
           border: 1px solid currentColor; margin-left: 4px; }
  .badge.cost { color: var(--cost); } .badge.capability { color: var(--cap); }
  .badge.fallback { color: var(--fb); }
  .metrics { margin-top: 4px; font-size: 11px; color: var(--muted); }
  form { display: flex; gap: 8px; padding: 10px; border-top: 1px solid var(--line); }
  textarea { flex: 1; resize: none; font: inherit; color: inherit; background: var(--bg);
             border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; height: 44px; }
  button { font: inherit; border: 0; border-radius: 8px; padding: 0 16px; background: var(--accent);
           color: var(--accent-ink); cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .examples { padding: 0 14px 10px; display: flex; flex-wrap: wrap; gap: 6px; }
  .examples button { background: var(--bg); color: var(--ink); border: 1px solid var(--line);
                     font-size: 12px; padding: 3px 10px; border-radius: 999px; }
  .tabs { display: flex; border-bottom: 1px solid var(--line); overflow-x: auto; }
  .tabs button { flex: none; }
  .tabs button { background: none; color: var(--muted); border-radius: 0; padding: 10px 14px;
                 border-bottom: 2px solid transparent; }
  .tabs button.on { color: var(--ink); border-bottom-color: var(--accent); }
  .pane { display: none; overflow-y: auto; padding: 10px 12px; flex: 1; min-height: 200px; }
  .pane.on { display: block; }
  .tline { font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; padding: 1px 0;
           border-bottom: 1px dashed var(--line); word-break: break-word; }
  .tline .t { color: var(--muted); } .tline .node { font-weight: 700; }
  .tline.warn { color: var(--cap); } .tline.error { color: var(--fb); }
  .turnhead { font-size: 12px; font-weight: 600; margin: 10px 0 4px; color: var(--muted); }
  .cit { border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; margin-bottom: 8px;
         font-size: 13px; overflow-wrap: anywhere; }
  .cit.flash { outline: 2px solid var(--accent); }
  .cit .lab { font-weight: 700; margin-right: 6px; }
  .cit .src { color: var(--muted); font-size: 12px; }
  .cit .snip { margin-top: 3px; color: var(--muted); font-size: 12px; }
  .cit .unused { opacity: .6; }
  a, .link { color: var(--accent); cursor: pointer; }
  #modal { position: fixed; inset: 0; background: rgba(0,0,0,.45); display: none;
           align-items: center; justify-content: center; padding: 16px; }
  #modal.on { display: flex; }
  #modal .box { background: var(--panel); border-radius: 10px; padding: 16px; max-width: 720px;
                max-height: 80vh; overflow: auto; width: 100%; }
  #modal pre { white-space: pre-wrap; background: var(--code); padding: 10px; border-radius: 6px;
               font-size: 13px; }
  .empty { color: var(--muted); font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>Multi-Agent RAG</h1>
  <div class="chips" id="chips"></div>
  <div id="status">connecting…</div>
</header>

<main>
  <section class="col">
    <div id="messages" aria-live="polite"></div>
    <div class="examples" id="examples"></div>
    <form id="form">
      <textarea id="q" placeholder="Ask a question about the corpus, the latest news, or the data…" required maxlength="2000"></textarea>
      <button id="send" type="submit">Send</button>
    </form>
  </section>

  <section class="col">
    <div class="tabs">
      <button class="on" data-tab="routing">Routing</button>
      <button data-tab="trace">Trace</button>
      <button data-tab="cites">Citations</button>
    </div>
    <div class="pane on" id="pane-routing"><div class="empty">Routing decisions for each turn appear here.</div></div>
    <div class="pane" id="pane-trace"><div class="empty">Orchestration trace appears here.</div></div>
    <div class="pane" id="pane-cites"><div class="empty">Citations appear here.</div></div>
  </section>
</main>

<div id="modal"><div class="box"><div id="modal-body"></div>
  <p><button id="modal-close" type="button">Close</button></p></div></div>

<script>
// ---------------------------------------------------------------- SSE parsing (unit-tested with node)
// SSE-PARSER-START
function parseSSE(buffer) {
  // Returns {events: [{event, data}], rest}. An event ends at a blank line; ": ..." lines are comments.
  const events = [];
  const parts = buffer.split("\n\n");
  const rest = parts.pop();
  for (const block of parts) {
    let event = "message", data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (data.length) {
      try { events.push({ event, data: JSON.parse(data.join("\n")) }); }
      catch (e) { /* ignore malformed event */ }
    }
  }
  return { events, rest };
}
// SSE-PARSER-END

if (typeof document !== "undefined") {
const $ = (id) => document.getElementById(id);
const history = [];            // [{role, content}] sent with each request (server is stateless)
let busy = false, turnNo = 0;
const AGENT_COLOR = (a) => `var(--c-${a}, var(--muted))`;

const EXAMPLES = [
  "What are the main ideas in the corpus?",
  "What is the latest news about AI agents?",
  "How many chunks does each document have?",
  "Summarize the first document in the corpus",
];

// ---- init
$("examples").innerHTML = "";
EXAMPLES.forEach((t) => {
  const b = document.createElement("button"); b.type = "button"; b.textContent = t;
  b.onclick = () => { $("q").value = t; $("form").requestSubmit(); };
  $("examples").appendChild(b);
});

document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x.dataset.tab === name));
  document.querySelectorAll(".pane").forEach((x) => x.classList.toggle("on", x.id === "pane-" + name));
}

async function loadConfig() {
  const t0 = performance.now();
  try {
    const r = await fetch("/api/config");
    const c = await r.json();
    const chips = $("chips"); chips.innerHTML = "";
    const add = (txt) => { const s = document.createElement("span"); s.className = "chip"; s.textContent = txt; chips.appendChild(s); };
    Object.entries(c.providers).forEach(([p, m]) => add(`${p}: ${m}${p === c.cheap_provider ? " (cheap)" : ""}${p === c.strong_provider ? " (strong)" : ""}`));
    if (c.two_provider_rule === false) add("WARNING: fewer than two providers");
    add(`corpus: ${c.corpus_chunks} chunks`);
    add(`live web: ${c.tavily_enabled ? "on" : "off"}`);
    $("status").textContent = `ready (${Math.round(performance.now() - t0)} ms)`;
  } catch (e) { $("status").textContent = "backend unreachable"; }
}
loadConfig();   // also warms a cold Cloud Run instance while the user is typing

// ---- rendering helpers
function addMsg(role, text) {
  const wrap = document.createElement("div"); wrap.className = "msg " + role;
  const who = document.createElement("div"); who.className = "who"; who.textContent = role === "user" ? "You" : role === "error" ? "Error" : "Assistant";
  const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = text;
  wrap.append(who, bubble); $("messages").appendChild(wrap); scrollDown();
  return { wrap, bubble };
}
function scrollDown() { const m = $("messages"); m.scrollTop = m.scrollHeight; }

function renderWithCites(el, text, turn) {
  // Turn [C1] / [W2] / [S1] into clickable chips; everything else stays plain text (no innerHTML).
  el.textContent = "";
  text.split(/(\[[CWS]\d+\])/g).forEach((part) => {
    const m = part.match(/^\[([CWS]\d+)\]$/);
    if (!m) { el.appendChild(document.createTextNode(part)); return; }
    const a = document.createElement("a"); a.className = "cite"; a.textContent = m[1];
    a.onclick = () => { showTab("cites"); const t = document.getElementById(`cit-${turn}-${m[1]}`);
      if (t) { t.scrollIntoView({ block: "center" }); t.classList.add("flash"); setTimeout(() => t.classList.remove("flash"), 1500); } };
    el.appendChild(a);
  });
}

function clearEmpty(pane) { const e = pane.querySelector(".empty"); if (e) e.remove(); }
function turnHeader(pane, turn, q) {
  clearEmpty(pane);
  const h = document.createElement("div"); h.className = "turnhead"; h.textContent = `Turn ${turn}: ${q.slice(0, 70)}`;
  pane.appendChild(h);
}

function routeEl(r) {
  const d = document.createElement("div"); d.className = "route"; d.style.borderLeftColor = AGENT_COLOR(r.agent);
  const head = document.createElement("div");
  const ag = document.createElement("span"); ag.className = "agent"; ag.textContent = r.agent;
  const arrow = document.createTextNode(` → ${r.provider} / ${r.model} `);
  const bd = document.createElement("span"); bd.className = "badge " + r.dimension; bd.textContent = r.dimension;
  head.append(ag, arrow, bd);
  const why = document.createElement("div"); why.className = "why"; why.textContent = r.reason;
  d.append(head, why); return d;
}

function traceEl(e) {
  const d = document.createElement("div"); d.className = "tline " + (e.event === "warn" ? "warn" : e.event === "error" ? "error" : "");
  const t = document.createElement("span"); t.className = "t"; t.textContent = `+${e.t_ms}ms `;
  const n = document.createElement("span"); n.className = "node"; n.style.color = AGENT_COLOR(e.node); n.textContent = e.node;
  d.append(t, n, document.createTextNode(` ${e.event}  ${e.detail || ""}`)); return d;
}

async function showChunk(id) {
  const body = $("modal-body"); body.textContent = "Loading…"; $("modal").classList.add("on");
  try {
    const r = await fetch("/api/chunk/" + encodeURI(id));
    const c = await r.json(); body.textContent = "";
    const h = document.createElement("h3"); h.textContent = `${c.source || id}${c.page > 0 ? ", p." + c.page : ""}`;
    const p = document.createElement("pre"); p.textContent = c.text || c.error;
    const i = document.createElement("div"); i.className = "src"; i.textContent = "chunk id: " + id;
    body.append(h, p, i);
  } catch (e) { body.textContent = "Could not load chunk."; }
}
$("modal-close").onclick = () => $("modal").classList.remove("on");
$("modal").onclick = (e) => { if (e.target.id === "modal") $("modal").classList.remove("on"); };

function renderCitations(items, turn) {
  const pane = $("pane-cites");
  if (!items.length) { const e = document.createElement("div"); e.className = "empty"; e.textContent = "No citations for this turn."; pane.appendChild(e); return; }
  [...items].sort((a, b) => Number(b.used) - Number(a.used)).forEach((c) => {
    const d = document.createElement("div"); d.className = "cit"; d.id = `cit-${turn}-${c.id}`;
    const top = document.createElement("div"); if (!c.used) top.className = "unused";
    const lab = document.createElement("span"); lab.className = "lab"; lab.textContent = c.id;
    top.appendChild(lab);
    if (c.kind === "corpus") {
      const a = document.createElement("a"); a.className = "link"; a.textContent = `${c.source}${c.page ? ", p." + c.page : ""}`;
      a.onclick = () => showChunk(c.chunk_id); top.appendChild(a);
    } else if (c.kind === "web") {
      const a = document.createElement("a"); a.textContent = c.title || c.url; a.href = c.url || "#"; a.target = "_blank"; a.rel = "noopener noreferrer"; top.appendChild(a);
    } else { top.appendChild(document.createTextNode(c.source)); }
    const tag = document.createElement("span"); tag.className = "src"; tag.textContent = `  ${c.kind}${c.used ? "" : " · retrieved, not cited"}`; top.appendChild(tag);
    const snip = document.createElement("div"); snip.className = "snip"; snip.textContent = c.snippet;
    d.append(top, snip); pane.appendChild(d);
  });
}

// ---- main chat flow
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); } });

$("form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const query = $("q").value.trim();
  if (!query || busy) return;
  busy = true; $("send").disabled = true; $("q").value = "";
  const turn = ++turnNo;
  addMsg("user", query);
  const out = addMsg("assistant", "");
  out.bubble.textContent = "…";
  const routesBox = document.createElement("div"); routesBox.className = "routes"; out.wrap.appendChild(routesBox);
  const metricsBox = document.createElement("div"); metricsBox.className = "metrics"; out.wrap.appendChild(metricsBox);
  ["routing", "trace", "cites"].forEach((p) => turnHeader($("pane-" + p), turn, query));

  let answer = "", gotEvent = false;
  const coldTimer = setTimeout(() => {
    if (!gotEvent) $("status").textContent = "waking the server (cold start can take 20–30 s)…";
  }, 3000);

  try {
    const resp = await fetch("/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history: history.slice(-6) }),
    });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try {
        const j = await resp.json();
        // FastAPI validation errors are a list of objects; show their messages, not "[object Object]".
        if (typeof j.detail === "string") msg = j.detail;
        else if (Array.isArray(j.detail)) msg += ": " + j.detail.map(d => d.msg || JSON.stringify(d)).join("; ");
      } catch (e) {}
      throw new Error(msg);
    }
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const { events, rest } = parseSSE(buf); buf = rest;
      for (const { event, data } of events) {
        gotEvent = true; clearTimeout(coldTimer); $("status").textContent = "streaming…";
        if (event === "token") { answer += data.text; renderWithCites(out.bubble, answer, turn); scrollDown(); }
        else if (event === "reset") { answer = ""; out.bubble.textContent = "…"; }
        else if (event === "route") { routesBox.appendChild(routeEl(data)); $("pane-routing").appendChild(routeEl(data)); scrollDown(); }
        else if (event === "trace") { $("pane-trace").appendChild(traceEl(data)); }
        else if (event === "citations") { renderCitations(data.items, turn); renderWithCites(out.bubble, answer, turn); }
        else if (event === "error") { addMsg("error", data.message); }
        else if (event === "done") {
          const m = data.metrics || {};
          const cost = m.cost_usd != null ? `$${m.cost_usd.toFixed(4)}` : "n/a";
          metricsBox.textContent = `TTFT ${m.ttft_ms ?? "n/a"} ms · total ${m.total_ms} ms · retrieval ${m.retrieval_ms} ms · llm ${m.llm_ms} ms · ${m.in_tokens}+${m.out_tokens} tokens · est. ${cost}`
            + (data.errors && data.errors.length ? ` · ${data.errors.length} agent warning(s)` : "");
        }
      }
    }
    if (!answer) out.bubble.textContent = "(no answer was produced — see the trace)";
    // Only the start of a long answer is needed to resolve follow-ups ("that", "it").
    history.push({ role: "user", content: query }, { role: "assistant", content: answer.slice(0, 3000) });
    $("status").textContent = "ready";
  } catch (e) {
    out.wrap.remove(); addMsg("error", "Request failed: " + e.message);
    $("status").textContent = "error";
  } finally {
    clearTimeout(coldTimer); busy = false; $("send").disabled = false; $("q").focus();
  }
});
}
</script>
</body>
</html>
'''


# ============================================================================
# 14. COMMAND LINE: ingest / csv / serve
# ============================================================================



def write_documents_table(stats: dict, db_path: str) -> None:
    """Small `documents` table (source, title, chunks, bytes) so the Data agent has data to query."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS documents")
        conn.execute(
            "CREATE TABLE documents (source TEXT PRIMARY KEY, title TEXT, chunks INTEGER, bytes INTEGER)"
        )
        conn.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?)",
            [(s, v["title"], v["chunks"], v["bytes"]) for s, v in sorted(stats.items())],
        )
        conn.commit()
    finally:
        conn.close()


def _ident(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s.strip()).strip("_").lower()
    return s if s and not s[0].isdigit() else f"c_{s}"


def _is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def load_csv_to_sqlite(csv_path: str, table: str, db_path: str) -> int:
    """Load a CSV as a table the Data agent can query. Numeric columns become numbers."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise SystemExit("CSV needs a header row and at least one data row")
    header = [_ident(h) for h in rows[0]]
    data = rows[1:]
    types = []
    for i in range(len(header)):
        vals = [r[i] for r in data if i < len(r) and r[i] != ""]
        if vals and all(_is_num(v) for v in vals):
            types.append("REAL" if any("." in v for v in vals) else "INTEGER")
        else:
            types.append("TEXT")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        t = _ident(table)
        conn.execute(f'DROP TABLE IF EXISTS "{t}"')
        conn.execute(f'CREATE TABLE "{t}" (' + ", ".join(f'"{h}" {ty}' for h, ty in zip(header, types)) + ")")
        conn.executemany(
            f'INSERT INTO "{t}" VALUES (' + ",".join("?" * len(header)) + ")",
            [[(r[i] if i < len(r) and r[i] != "" else None) for i in range(len(header))] for r in data],
        )
        conn.commit()
    finally:
        conn.close()
    return len(data)


def cli() -> None:
    ap = argparse.ArgumentParser(description="Multi-agent RAG: ingest documents or start the server.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="embed everything in ./corpus into Chroma + build the SQLite documents table")
    p_ing.add_argument("--corpus", default="./corpus")
    p_csv = sub.add_parser("csv", help="load a CSV as a SQLite table for the Data agent")
    p_csv.add_argument("file")
    p_csv.add_argument("table")
    p_srv = sub.add_parser("serve", help="start the web server")
    p_srv.add_argument("--port", type=int, default=int(env("PORT", "8080")))
    p_srv.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if args.cmd == "ingest":
        print(f"Embedding backend: {config.EMBEDDING_BACKEND} | index: {config.CHROMA_PERSIST_DIR}")
        stats = rag.ingest_directory(args.corpus)
        write_documents_table(stats, config.SQLITE_PATH)
        print(f"Wrote documents table to {config.SQLITE_PATH}")
    elif args.cmd == "csv":
        n = load_csv_to_sqlite(args.file, args.table, config.SQLITE_PATH)
        print(f"Loaded {n} rows into table '{args.table}' in {config.SQLITE_PATH}")
    else:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    cli()

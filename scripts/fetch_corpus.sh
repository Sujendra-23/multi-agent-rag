#!/usr/bin/env bash
# Re-download the 40 arXiv papers that make up the RAG corpus (see scripts/corpus_manifest.md).
set -euo pipefail
cd "$(dirname "$0")/../corpus"
for id in $(grep -oE "^\| [0-9]{4}\.[0-9]{5}" ../scripts/corpus_manifest.md | tr -d '| '); do
  [ -s "$id.pdf" ] || curl -sSL --max-time 90 -o "$id.pdf" "https://arxiv.org/pdf/$id"
  sleep 3
done

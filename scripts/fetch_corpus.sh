#!/usr/bin/env bash
# Re-download the 40 arXiv papers that make up the RAG corpus (see scripts/corpus_manifest.md).
# Each paper is fetched at the exact version listed in the manifest (e.g. 2210.03629v3), so the
# rebuilt index matches the deployed one even if a paper is revised later.
set -euo pipefail
cd "$(dirname "$0")/../corpus"
grep -oE "^\| [0-9]{4}\.[0-9]{5} \| v[0-9]+" ../scripts/corpus_manifest.md | tr -d '| ' | while read -r idv; do
  id=${idv%v*}
  [ -s "$id.pdf" ] || curl -sSL --max-time 90 -o "$id.pdf" "https://arxiv.org/pdf/$idv"
  sleep 3
done

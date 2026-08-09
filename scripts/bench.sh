#!/usr/bin/env bash
# Hi gate + essay tok/s regression bench.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec cargo run -p ksearch_cli --release -- bench "$@"

#!/usr/bin/env bash
# Incremental rebuild of libtriton (with the Furiosa plugin) in build/triton. Builds are serialized by a
# lock: several agents share the one build directory. Usage: compiler/build.sh [ninja targets...]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -eq 0 ] && set -- triton furiosa-mlir-opt  # libtriton and the opt tool the lit tests run
exec flock "$root/build/.build.lock" "$root/compiler/.venv/bin/ninja" -C "$root/build/triton" -j "${MAX_JOBS:-48}" "$@"

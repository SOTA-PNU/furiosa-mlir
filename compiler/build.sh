#!/usr/bin/env bash
# Incremental rebuild of libtriton (with the Furiosa plugin) in build/triton. Builds are serialized by a
# lock: several agents share the one build directory. Usage: compiler/build.sh [ninja targets...]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -eq 0 ] && set -- triton furiosa-mlir-opt  # libtriton and the opt tool the lit tests run
# Every core, as far as memory goes: an MLIR translation unit takes up to ~2 GB to compile.
mem_jobs=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 2097152 ))
jobs=$(( mem_jobs < $(nproc) ? mem_jobs : $(nproc) ))
exec flock "$root/build/.build.lock" "$root/compiler/.venv/bin/ninja" -C "$root/build/triton" -j "${MAX_JOBS:-$(( jobs > 0 ? jobs : 1 ))}" "$@"

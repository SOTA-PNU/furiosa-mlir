#!/usr/bin/env bash
# Runs a command that loads libtriton while holding compiler/build.sh's lock shared, so the library is
# never read while a build relinks it. A waiting build goes first: new readers queue at the pending gate
# behind it instead of holding the shared lock over it forever. Usage: compiler/with-build-lock.sh cmd...
set -euo pipefail
lock="$(cd "$(dirname "$0")/.." && pwd)/build"
mkdir -p "$lock"
flock -s "$lock/.build.pending" true
exec flock -s "$lock/.build.lock" "$@"

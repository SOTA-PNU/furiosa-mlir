#!/usr/bin/env bash
# Installs this binary package in place: a Python 3.12 venv (venv/) with torch (CPU), the package's Triton and
# furiosa_compiler on its path, and bin/ for the furiosa CLI. furiosa-opt builds need Rust: the crates pin their
# toolchain (rust-toolchain.toml), which rustup installs on the first build.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
"${PYTHON:-python3.12}" -m venv "$here/venv"
"$here/venv/bin/pip" install --quiet --extra-index-url https://download.pytorch.org/whl/cpu -r "$here/requirements.txt"
site="$("$here/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$here/python" > "$site/furiosa-mlir.pth"
# Triton finds its backends by entry point: register the furiosa one for the Triton in python/.
mkdir -p "$site/triton-3.9.0+furiosa.dist-info"
printf 'Metadata-Version: 2.1\nName: triton\nVersion: 3.9.0+furiosa\n' > "$site/triton-3.9.0+furiosa.dist-info/METADATA"
printf '[triton.backends]\nfuriosa = triton.backends.furiosa\n' > "$site/triton-3.9.0+furiosa.dist-info/entry_points.txt"
"$here/venv/bin/python" -c "
import furiosa_compiler.cli, triton
from triton.backends import backends
assert 'furiosa' in backends, backends
print('triton', triton.__version__, 'with the furiosa backend')"
command -v cargo >/dev/null || echo "note: no cargo on PATH; install rustup (https://rustup.rs) before running on the NPU"
echo "installed: add $here/bin to PATH, then: furiosa --help"

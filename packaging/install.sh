#!/usr/bin/env bash
# Installs this binary package in place: a Python 3.12 venv (venv/) with torch (CPU), the package's Triton and
# furiosa_compiler on its path, bin/ for the furiosa CLI, and the furiosa-opt tools in .tools/ (from crates.io).
# furiosa-opt builds need Rust: the crates pin their toolchain (rust-toolchain.toml), which rustup installs on the
# first build.
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
# The furiosa-opt tools the generated crates build with (crates.io; the version furiosa-opt-std is pinned to in
# furiosa_compiler/crate.py), where the compiler looks for them: .tools/bin.
if command -v cargo >/dev/null; then
    if command -v cargo-binstall >/dev/null; then
        cargo binstall -y --root "$here/.tools" cargo-furiosa-opt@0.10.0
    else
        cargo install --locked --root "$here/.tools" cargo-furiosa-opt@0.10.0
    fi
else
    echo "note: no cargo on PATH; install rustup (https://rustup.rs), then rerun install.sh to run on the NPU"
fi
echo "installed: add $here/bin to PATH, then: furiosa --help"

#!/usr/bin/env bash
# Binary package: dist/furiosa-mlir-<version>-linux-x86_64.tar.gz with the furiosa-enabled Triton (libtriton,
# stripped), furiosa_compiler, the furiosa CLI, furiosa-mlir-opt, the example models and the playground; and
# dist/furiosa_mlir-<version>-cp312-cp312-linux_x86_64.whl with the same Triton and furiosa_compiler. Neither
# carries the furiosa-opt tools: install.sh (or the user) installs them from crates.io. Build the compiler first
# (compiler/build.sh). DIST=<dir> puts them elsewhere. Install: docs/usage.md.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
name="furiosa-mlir-$version-linux-x86_64"
dist="${DIST:-$root/dist}"
stage="$dist/stage/$name"
rm -rf "$stage" && mkdir -p "$stage/python" "$stage/bin" "$stage/examples" "$stage/docs"

# Triton with the furiosa backend as files, and no other backend. The source tree links language/extra and
# tools/extra (cuda's libdevice, which the furiosa backend maps libdevice calls through) into third_party/:
# copy what the links point to, not the links.
rsync -a --copy-links --exclude=__pycache__ --exclude='/backends/furiosa' --exclude='/backends/nvidia' \
    --exclude='/backends/amd' third_party/triton/python/triton/ "$stage/python/triton/"
cp -aL compiler/backend "$stage/python/triton/backends/furiosa"
strip --strip-debug "$stage/python/triton/_C/libtriton.so"
# The JIT links the MLIR runner utilities; libtriton looks for them beside itself (compiler/lib/Runtime/Jit.cpp).
llvm_lib="$(sed -n 's|^LLVM_DIR:PATH=\(.*\)/cmake/llvm$|\1|p' build/triton/CMakeCache.txt)"
cp -L "$llvm_lib/libmlir_c_runner_utils.so" "$stage/python/triton/_C/"
mkdir -p "$stage/python/triton/lib"  # its dependencies, where its RUNPATH ($ORIGIN/../lib) looks
cp -L "$llvm_lib"/lib{mlir_apfloat_wrappers,mlir_float16_utils}.so.*git "$stage/python/triton/lib/"
cp -a compiler/furiosa_compiler "$stage/python/furiosa_compiler"
find "$stage/python" -name __pycache__ -prune -exec rm -rf {} +

install -m755 packaging/furiosa packaging/install.sh -t "$stage/bin"
mv "$stage/bin/install.sh" "$stage/install.sh"
install -m755 -s build/triton/third_party/furiosa/tools/furiosa-mlir-opt "$stage/bin/"
cp packaging/requirements.txt "$stage/"
# Triton (MIT) and the LLVM/MLIR libraries libtriton links (Apache-2.0 with LLVM exceptions) travel with it.
mkdir -p "$stage/licenses"
cp third_party/triton/LICENSE "$stage/licenses/LICENSE-triton.txt"
cp packaging/LICENSE-LLVM.txt "$stage/licenses/"

cp -a models playground "$stage/examples/"
find "$stage/examples" \( -name '*.mlir' -o -name __pycache__ \) -prune -exec rm -rf {} +
cp docs/usage.md README.md "$stage/docs/"
echo "$version $(git rev-parse --short HEAD)" > "$stage/VERSION"

tar czf "$dist/$name.tar.gz" -C "$dist/stage" "$name"

# The wheel: the same Triton (it provides the `triton` package, so install it into a fresh venv) and
# furiosa_compiler, the furiosa CLI as a console script, furiosa-mlir-opt, and the furiosa backend's entry point.
whl="$dist/stage/wheel"; info="$whl/furiosa_mlir-$version.dist-info"
mkdir -p "$info/licenses" "$whl/furiosa_mlir-$version.data/scripts"
cp -a "$stage/python/triton" "$stage/python/furiosa_compiler" "$whl/"
cp "$stage/bin/furiosa-mlir-opt" "$whl/furiosa_mlir-$version.data/scripts/"
cp "$stage/licenses/"* "$info/licenses/"
{ printf 'Metadata-Version: 2.1\nName: furiosa-mlir\nVersion: %s\n' "$version"
  printf 'Summary: Triton for the Furiosa RNGD NPU (Triton 3.9 with the furiosa backend)\nRequires-Python: >=3.12,<3.13\n'
  grep -v '^#' packaging/requirements.txt | sed 's/^/Requires-Dist: /'; } > "$info/METADATA"
printf 'Wheel-Version: 1.0\nGenerator: scripts/package.sh\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n' > "$info/WHEEL"
printf '[console_scripts]\nfuriosa = furiosa_compiler.cli:main\n\n[triton.backends]\nfuriosa = triton.backends.furiosa\n' \
  > "$info/entry_points.txt"
compiler/.venv/bin/python -m wheel pack "$whl" -d "$dist" >/dev/null
rm -rf "$dist/stage"
ls -la "$dist/$name.tar.gz" "$dist/furiosa_mlir-$version-cp312-cp312-linux_x86_64.whl"

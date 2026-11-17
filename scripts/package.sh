#!/usr/bin/env bash
# Binary package: dist/furiosa-mlir-<version>-linux-x86_64.tar.gz with the furiosa-enabled Triton (libtriton,
# stripped), furiosa_compiler, the furiosa CLI, furiosa-mlir-opt, the pinned furiosa-opt tools, the example
# models and the playground. Build the compiler first (compiler/build.sh). DIST=<dir> puts it elsewhere.
# Install: tar xzf ...; ./furiosa-mlir-<version>-linux-x86_64/install.sh (docs/usage.md).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
name="furiosa-mlir-$version-linux-x86_64"
dist="${DIST:-$root/dist}"
stage="$dist/stage/$name"
rm -rf "$stage" && mkdir -p "$stage/python" "$stage/bin" "$stage/examples" "$stage/docs"

# Triton with the furiosa backend as files (the source tree links it), and no other backend.
cp -a third_party/triton/python/triton "$stage/python/triton"
rm -rf "$stage/python/triton/backends/"{furiosa,nvidia,amd}
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
install -m755 .tools/bin/cargo-furiosa-opt .tools/bin/furiosa-opt-driver "$stage/bin/"
cp packaging/requirements.txt "$stage/"
cp -a models playground "$stage/examples/"
find "$stage/examples" \( -name '*.mlir' -o -name __pycache__ \) -prune -exec rm -rf {} +
cp docs/usage.md README.md "$stage/docs/"
echo "$version $(git rev-parse --short HEAD)" > "$stage/VERSION"

tar czf "$dist/$name.tar.gz" -C "$dist/stage" "$name"
rm -rf "$dist/stage"
ls -la "$dist/$name.tar.gz"

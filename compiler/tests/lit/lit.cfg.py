# lit configuration for the per-pass tests: each .mlir runs one or more Furiosa passes through
# furiosa-mlir-opt and checks the output with FileCheck (see compiler/README.md).

import glob
import os

import lit.formats

config.name = "furiosa"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

root = os.path.abspath(os.path.join(config.test_source_root, "..", "..", ".."))
tools = [os.path.join(root, "build", "triton", "third_party", "furiosa", "tools")]
# FileCheck and not come with the LLVM package Triton builds against.
tools += sorted(os.path.dirname(p) for p in glob.glob(os.path.expanduser("~/.triton/llvm/llvm-*/bin/FileCheck")))
config.environment["PATH"] = os.pathsep.join(tools + [os.environ.get("PATH", "")])

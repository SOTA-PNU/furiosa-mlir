"""furiosa compile and trace on a micro model, CPU reference only (no NPU): every kernel compiles, the crate is
written, and trace runs every launch and dumps its buffers.

    PYTHONPATH=compiler compiler/.venv/bin/python compiler/tests/test_cli.py
"""

import tempfile
from pathlib import Path

from furiosa_compiler import cli, tensors

MODEL = Path(__file__).resolve().parents[2] / "harness" / "cases" / "vector_add.py"


def main():
    with tempfile.TemporaryDirectory(dir="/mnt/nvme1n1p2/chacha" if Path("/mnt/nvme1n1p2/chacha").exists() else None) as tmp:
        out = Path(tmp)
        result = cli.compile_model(MODEL, out / "c")
        assert not any("error" in k for k in result["kernels"].values()), result["kernels"]
        assert (out / "c" / "crate" / "Cargo.toml").exists()
        assert cli.trace(MODEL, out / "t", 0, None, out / "dump", npu=False, rtol=2e-2, atol=1e-2)
        dumps = sorted(p.name for p in (out / "dump").glob("*.safetensors"))
        assert dumps[0] == "000_add_kernel_0.safetensors" and "inputs.safetensors" in dumps, dumps
        ins, outs = tensors.load(out / "dump" / "inputs.safetensors"), tensors.load(out / "dump" / dumps[0])
        assert all(abs(outs[n] - (ins["x"] + ins["y"])).max() == 0 for n in outs), "trace's reference is not x + y"
    print("ok")


if __name__ == "__main__":
    main()

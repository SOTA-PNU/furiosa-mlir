import subprocess
import sys

import torch

from harness import ROOT, load_model, run


def test_tiny_mlp_matches_torch():
    model = load_model("tests/fixtures/tiny_mlp.py")
    g = model.forward(batch=5)
    weights = {
        "fc1.weight": torch.randn(128, 64, device="cuda"),
        "fc1.bias": torch.randn(128, device="cuda"),
        "fc2.weight": torch.randn(10, 128, device="cuda"),
        "fc2.bias": torch.randn(10, device="cuda"),
    }
    x = torch.randn(5, 64, device="cuda")
    out = run(g, {"x": x, **weights})["logits"]
    h = torch.relu(x @ weights["fc1.weight"].T + weights["fc1.bias"])
    torch.testing.assert_close(out, h @ weights["fc2.weight"].T + weights["fc2.bias"], atol=1e-4, rtol=1e-4)


def test_dumped_ttir_is_what_the_jit_runs(tmp_path):
    """The dumper specializes each launch exactly as the JIT does when the harness launches it."""
    out = tmp_path / "tiny_mlp.triton.mlir"
    subprocess.run([sys.executable, "scripts/dump_mlir.py", "tests/fixtures/tiny_mlp.py", "-o", str(out)], cwd=ROOT, check=True)
    dumped = out.read_text()

    model = load_model("tests/fixtures/tiny_mlp.py")
    g = model.graphs()[0]
    tensors = {name: torch.randn(buf.shape, device="cuda") for name, buf in g.buffers.items() if buf.kind in ("input", "weight")}
    run(g, tensors)
    kernel_cache = next(iter(model.linear_kernel.device_caches.values()))[0]
    for compiled in kernel_cache.values():
        body = compiled.asm["ttir"].replace(f"{ROOT}/", "")
        ops = [line.split(" loc(")[0].strip() for line in body.splitlines() if line.startswith("    ") and "tt." in line]
        assert all(op in dumped for op in ops)

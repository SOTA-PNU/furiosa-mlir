"""The furiosa Triton driver: wherever this backend is installed, Triton launches run on the NPU.
``kernel[grid](...)`` compiles the kernel's TTIR through Triton, then the launcher specializes the
launch and runs it on the NPU through the generated crate (furiosa_compiler.npu)."""

import os
import sys
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

# Always the NPU: Triton picks the driver TRITON_DEFAULT_BACKEND names.
os.environ["TRITON_DEFAULT_BACKEND"] = "furiosa"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # compiler/, for furiosa_compiler


class FuriosaLauncher:
    """Called by Triton for every launch of a compiled kernel, with the grid and the bound arguments."""

    def __init__(self, src, metadata):
        self.fn = src.fn

    def __call__(self, grid_x, grid_y, grid_z, stream, function, packed_metadata, launch_metadata, enter_hook,
                 exit_hook, *args):
        from furiosa_compiler import npu

        npu.run(self.fn, (grid_x, grid_y, grid_z), dict(zip(self.fn.arg_names, args)))


class FuriosaUtils:
    """What Triton asks of a device before a launch; the NPU kernel is built per launch, so nothing loads."""

    @staticmethod
    def load_binary(name, kernel, shared, device):
        return None, None, 0, 0, 1 << 30

    @staticmethod
    def get_device_properties(device):
        return {"max_shared_mem": 1 << 30, "multiprocessor_count": 1, "max_num_regs": 1 << 30, "warpSize": 1}


class FuriosaDriver(DriverBase):

    def __init__(self):
        self.utils = FuriosaUtils()
        self.launcher_cls = FuriosaLauncher

    @staticmethod
    def is_active():
        return True

    def get_current_target(self):
        return GPUTarget("furiosa", "rngd", 1)

    def get_current_device(self):
        return 0

    def set_current_device(self, device):
        pass

    def get_current_stream(self, device=None):
        return 0

    def get_device_interface(self):
        import torch

        return torch.cpu

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def map_python_to_cpp_type(self, ty: str) -> str:
        raise NotImplementedError("furiosa launches go through the generated crate, not a C++ launcher")

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

"""Driver stub: Furiosa kernels run through the generated furiosa-opt crate, never through Triton's
launcher, so this driver is never the active one."""

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class FuriosaDriver(DriverBase):

    @staticmethod
    def is_active():
        return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        raise NotImplementedError("furiosa kernels are not launched from Python")

    def get_current_target(self):
        return GPUTarget("furiosa", "rngd", 1)

    def get_active_torch_device(self):
        raise NotImplementedError("furiosa kernels are not launched from Python")

    def get_benchmarker(self):
        raise NotImplementedError("furiosa kernels are not launched from Python")

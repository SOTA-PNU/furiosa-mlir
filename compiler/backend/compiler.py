"""Triton backend for the Furiosa RNGD NPU: TTIR -> whole-tensor linalg -> (later) fopt -> Rust.

Kernels are compiled per launch: the driver (compiler/furiosa_compile) passes every integer argument
as a constexpr, every float one as an fp32 value it records, and the launch grid, so the raising pass
sees a kernel with constant shapes.
"""

import functools
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from types import ModuleType
from typing import Dict, Tuple

from triton._C.libtriton import furiosa, ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language


# Starts the fopt / rust stage output of a kernel codegen cannot handle yet; the rest is the diagnostic.
CODEGEN_FAILED = "// furiosa codegen failed:"


def codegen_failure(error: Exception) -> str:
    return CODEGEN_FAILED + "\n" + "\n".join("// " + line for line in str(error).splitlines()) + "\n"


@dataclass(frozen=True)
class FuriosaOptions:
    # Read by Triton's frontend; the GPU-only knobs keep inert defaults.
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 1
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    supported_fp8_dtypes: Tuple[str, ...] = ("fp8e5", "fp8e4nv")
    deprecated_fp8_dot_operand_dtypes: Tuple[str, ...] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str, ...] = ("ieee", "tf32", "tf32x3")
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = False
    arch: str = "rngd"
    backend_name: str = "furiosa"
    instrumentation_mode: str = ""
    fpsan_homomorphic_casts: bool = False
    # The launch, recorded on the module for the raising pass: its grid, and the graph buffer (name,
    # kind, dtype, shape) behind each pointer argument, as JSON.
    grid: Tuple[int, ...] = (1, 1, 1)
    buffers: str = "[]"
    scalars: str = "[]"  # the values of the fp32 scalar arguments, in order, as JSON

    def hash(self):
        key = "_".join(f"{name}-{value}" for name, value in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class FuriosaBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "furiosa"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "ttir"  # the launch binary: the NPU launcher compiles each launch from its TTIR

    def hash(self) -> str:
        # triton_key() hashes libtriton but not this backend (its directory is no Python package), so the
        # stages defined here enter the cache key through their source.
        source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
        return f"furiosa-{self.target.arch}-{source}"

    def parse_options(self, opts) -> FuriosaOptions:
        known = FuriosaOptions.__dataclass_fields__
        args = {k: v for k, v in opts.items() if k in known and v is not None}
        if "grid" in args:
            args["grid"] = tuple(args["grid"]) + (1,) * (3 - len(args["grid"]))
        return FuriosaOptions(**args)

    def pack_metadata(self, metadata):
        return ()

    def get_codegen_implementation(self, options):
        # No minimum tl.dot shape: the NPU tiles contractions itself.
        return {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        # libdevice calls keep their declarations (extern_elementwise of __nv_* symbols, no library is
        # linked); the raising pass turns the ones with a math-dialect meaning into those ops.
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        furiosa.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        metadata["shared"], metadata["warp_size"] = 0, 1  # read by Triton's launch path
        metadata["name"] = mod.get_entry_func_name()
        return mod

    @staticmethod
    def make_linalg(mod, metadata, options):
        builder = ir.builder(mod.context)
        launch = json.dumps({"grid": list(options.grid), "buffers": json.loads(options.buffers),
                             "scalars": json.loads(options.scalars)})
        mod.set_attr("furiosa.launch", builder.get_string_attr(launch))
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        furiosa.passes.add_raise_to_linalg(pm)
        pm.run(mod, "make_linalg")
        metadata["name"] = mod.get_entry_func_name()
        return str(mod)

    @staticmethod
    def make_structured(src, metadata, options):
        """Promotes the raised kernel's affine reads to indexing-map inputs (furiosa-structure). A failure
        yields a comment carrying the diagnostic, so the raised stage stays usable."""
        try:
            return furiosa.passes.structure(src)
        except RuntimeError as e:
            return codegen_failure(e)

    @staticmethod
    def make_fopt(src, metadata, options):
        """Maps the raised kernel to furiosa-opt (fopt). A kernel no op class maps yet yields a comment carrying
        the diagnostic instead of failing compilation, so the earlier stages stay usable."""
        if src.startswith(CODEGEN_FAILED):
            return src
        try:
            return furiosa.codegen.linalg_to_fopt(src)
        except RuntimeError as e:
            return codegen_failure(e)

    @staticmethod
    def make_rust(src, metadata, options):
        """The fopt kernel as a Rust module body (src/kernel/<symbol>.rs of the crate)."""
        if src.startswith(CODEGEN_FAILED):
            return src
        try:
            return furiosa.codegen.fopt_to_rust(src)
        except RuntimeError as e:
            return codegen_failure(e)

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError(f"the furiosa backend compiles Triton kernels, not {language}")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        if options.buffers == "[]":
            # kernel[grid](...): Triton compiles before it knows the launch; the launcher compiles the
            # specialized launch through every stage (furiosa_compiler.npu).
            return
        stages["linalg"] = lambda src, metadata: self.make_linalg(src, metadata, options)
        stages["structured"] = lambda src, metadata: self.make_structured(src, metadata, options)
        stages["fopt"] = lambda src, metadata: self.make_fopt(src, metadata, options)
        stages["rust"] = lambda src, metadata: self.make_rust(src, metadata, options)

    @functools.lru_cache()
    def get_target_name(self, options) -> str:
        return "furiosa:rngd"

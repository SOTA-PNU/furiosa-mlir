// Bindings for the L1 JIT: triton._C.libtriton.furiosa.runtime.

#include "furiosa/Runtime/Jit.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstdint>
#include <stdexcept>

namespace nb = nanobind;
using mlir::furiosa::JitModule;

void init_furiosa_runtime(nb::module_ &m) {
  auto runtime = m.def_submodule("runtime");

  nb::class_<JitModule>(runtime, "Engine")
      .def(
          "lookup",
          [](const JitModule &self, const std::string &name) {
            std::string error;
            void *address = self.lookup(name, error);
            if (!address)
              throw std::runtime_error("no C interface for " + name + ": " + error);
            return reinterpret_cast<std::uintptr_t>(address);
          },
          "The address of _mlir_ciface_<name>.")
      .def_prop_ro("lowered", &JitModule::lowered, "The module lowered to the LLVM dialect.");

  runtime.def(
      "compile",
      [](const std::string &text) {
        std::string error;
        std::unique_ptr<JitModule> jit = JitModule::compile(text, error);
        if (!jit)
          throw std::runtime_error(error);
        return jit;
      },
      "JIT-compiles a raised kernel module for the host.");
}

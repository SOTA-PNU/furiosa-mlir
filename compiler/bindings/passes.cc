// Bindings for the raising passes.

#include "furiosa/Transforms/Passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_furiosa_passes(nb::module_ &m) {
  // The Furiosa passes declare the dialects they create, so the TTIR stages need none loaded.
  m.def("load_dialects", [](mlir::MLIRContext &context) {});

  auto passes = m.def_submodule("passes");
  passes.def("add_raise_to_linalg", [](mlir::PassManager &pm) { pm.addPass(mlir::furiosa::createRaiseToLinalgPass()); });
}

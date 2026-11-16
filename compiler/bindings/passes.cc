// Bindings for the raising passes.

#include "furiosa/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/raw_ostream.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>

namespace nb = nanobind;

namespace {

/// Parses raised linalg, runs furiosa-structure, and prints the result; diagnostics become the error.
std::string structure(const std::string &text) {
  mlir::DialectRegistry registry;
  mlir::furiosa::registerDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
    diag << d.getLocation() << ": " << d << "\n";
    return mlir::success();
  });
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    throw std::runtime_error("furiosa-structure: cannot parse the raised module:\n" + diagnostics);
  mlir::PassManager pm(&context);
  pm.addPass(mlir::furiosa::createStructurePass());
  if (mlir::failed(pm.run(*module)))
    throw std::runtime_error("furiosa-structure failed:\n" + diagnostics);
  std::string out;
  llvm::raw_string_ostream os(out);
  module->print(os, mlir::OpPrintingFlags().enableDebugInfo(true, false));
  return out;
}

} // namespace

void init_furiosa_passes(nb::module_ &m) {
  // The Furiosa passes declare the dialects they create, so the TTIR stages need none loaded.
  m.def("load_dialects", [](mlir::MLIRContext &context) {});

  auto passes = m.def_submodule("passes");
  passes.def("add_raise_to_linalg", [](mlir::PassManager &pm) { pm.addPass(mlir::furiosa::createRaiseToLinalgPass()); });
  passes.def("structure", &structure, "Gather-form raised linalg (text) -> structured linalg (text)");
}

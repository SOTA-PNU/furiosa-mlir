// Bindings for fopt mapping and Rust emission (triton._C.libtriton.furiosa.codegen).
//
// The backend stages pass text between them, so each function parses its input in a fresh context with
// the codegen dialects and raises Python's RuntimeError carrying the diagnostics on failure.

#include "furiosa/Codegen/Codegen.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>

namespace nb = nanobind;

namespace {

/// Parses `text`, runs `body` on the module, and turns diagnostics into an exception on failure.
template <typename F> std::string withModule(const std::string &text, F body) {
  mlir::DialectRegistry registry;
  mlir::furiosa::registerCodegenDialects(registry);
  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects(true); // e.g. tt.divisibility left on raised arguments
  context.printOpOnDiagnostic(false);       // the message names the problem; the IR is in the stage file
  std::string diagnostics;
  llvm::raw_string_ostream diag(diagnostics);
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
    diag << d.getLocation() << ": " << d << "\n";
    for (mlir::Diagnostic &note : d.getNotes())
      diag << "  note: " << note << "\n";
    return mlir::success();
  });
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    throw std::runtime_error("cannot parse the module:\n" + diagnostics);
  std::string out;
  llvm::raw_string_ostream os(out);
  if (mlir::failed(body(*module, os)))
    throw std::runtime_error(diagnostics.empty() ? "codegen failed" : diagnostics);
  return out;
}

} // namespace

void init_furiosa_codegen(nb::module_ &m) {
  auto codegen = m.def_submodule("codegen");

  codegen.def("linalg_to_fopt", [](const std::string &text) {
    return withModule(text, [](mlir::ModuleOp module, llvm::raw_ostream &os) {
      mlir::PassManager pm(module->getContext());
      pm.addPass(mlir::furiosa::createLinalgToFoptPass());
      if (mlir::failed(pm.run(module)))
        return mlir::failure();
      module.print(os);
      return mlir::success();
    });
  });

  codegen.def("split_steps", [](const std::string &text) {
    return withModule(text, [](mlir::ModuleOp module, llvm::raw_ostream &os) {
      if (mlir::failed(mlir::furiosa::splitSteps(module)))
        return mlir::failure();
      module.print(os, mlir::OpPrintingFlags().enableDebugInfo(true, false));
      return mlir::success();
    });
  });

  codegen.def("steps_plan", [](const std::string &text) {
    return withModule(text, [](mlir::ModuleOp module, llvm::raw_ostream &os) {
      mlir::FailureOr<std::string> plan = mlir::furiosa::stepsPlan(module);
      if (mlir::failed(plan))
        return mlir::failure();
      os << *plan;
      return mlir::success();
    });
  });

  codegen.def("fopt_signature", [](const std::string &text) {
    return withModule(text, [](mlir::ModuleOp module, llvm::raw_ostream &os) {
      mlir::FailureOr<std::string> sig = mlir::furiosa::foptSignature(module);
      if (mlir::failed(sig))
        return mlir::failure();
      os << *sig;
      return mlir::success();
    });
  });

  codegen.def("fopt_to_rust", [](const std::string &text) {
    return withModule(text, [](mlir::ModuleOp module, llvm::raw_ostream &os) {
      return mlir::furiosa::translateFoptToRust(module, os);
    });
  });
}

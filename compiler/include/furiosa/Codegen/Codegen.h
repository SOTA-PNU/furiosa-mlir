#ifndef FURIOSA_CODEGEN_CODEGEN_H
#define FURIOSA_CODEGEN_CODEGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir::furiosa {

/// Maps each raised kernel (compiler/README.md's linalg contract) to a furiosa-opt kernel in the fopt
/// dialect, op class by op class (lib/Codegen/Mapping/).
std::unique_ptr<Pass> createLinalgToFoptPass();

/// Writes every fopt kernel of `module` as one Rust module (the body of src/kernel/<symbol>.rs).
LogicalResult translateFoptToRust(ModuleOp module, llvm::raw_ostream &os);

/// The fopt kernel's interface as JSON, for the crate's host code: {"kernel", "chip", "pe", "inputs":
/// [pointer-arg index per parameter], "outputs": [pointer-arg index per result]}.
FailureOr<std::string> foptSignature(ModuleOp module);

/// The dialects the codegen passes read and create.
void registerCodegenDialects(DialectRegistry &registry);

} // namespace mlir::furiosa

#endif // FURIOSA_CODEGEN_CODEGEN_H

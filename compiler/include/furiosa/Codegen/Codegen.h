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
/// [pointer-arg index per parameter], "inouts": [parameters written in place, as &mut], "outputs":
/// [pointer-arg index per result]}.
FailureOr<std::string> foptSignature(ModuleOp module);

/// Evaluates every value of a linalg.generic body that depends only on the iteration indices at compile time,
/// over the dims it reads, and replaces it with a constant tensor input (lib/Codegen/Hoist.cpp).
void hoistIndexOnly(ModuleOp module);

/// Splits each kernel that no op class maps whole into steps (lib/Codegen/Steps.cpp): sub-kernel functions,
/// each one an op class maps, called in order from the kernel on its buffers and HBM temporaries.
LogicalResult splitSteps(ModuleOp module);

/// The step plan of a split module as JSON, or "null" when the kernel is not split: {"kernel", "temps":
/// [{"dtype", "shape"}], "consts": [{"dtype", "shape", "data": base64}], "steps": [{"func", "args":
/// [{"buffer": i} | {"temp": j} | {"const": k}], "module": text}]}.
FailureOr<std::string> stepsPlan(ModuleOp module);

/// The dialects the codegen passes read and create.
void registerCodegenDialects(DialectRegistry &registry);

} // namespace mlir::furiosa

#endif // FURIOSA_CODEGEN_CODEGEN_H

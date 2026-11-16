#ifndef FURIOSA_RUNTIME_JIT_H
#define FURIOSA_RUNTIME_JIT_H

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace mlir {
class ExecutionEngine;
class MLIRContext;
} // namespace mlir

namespace mlir::furiosa {

/// A raised kernel module JIT-compiled for the host (harness level L1).
///
/// Every public function gets a C interface `_mlir_ciface_<name>(desc *arg0, ..., desc *out0, ...)`:
/// one memref descriptor per tensor argument, then one caller-allocated descriptor per result.
/// Arguments are never written; results are fresh buffers the caller owns.
class JitModule {
public:
  /// Compiles `text`; on failure returns null and puts the diagnostics in `error`.
  static std::unique_ptr<JitModule> compile(llvm::StringRef text, std::string &error);
  ~JitModule();

  /// The address of `_mlir_ciface_<name>`, or null with `error` set.
  void *lookup(llvm::StringRef name, std::string &error) const;

  /// The module after lowering to the LLVM dialect, for debugging.
  const std::string &lowered() const { return loweredText; }

private:
  JitModule();
  std::unique_ptr<MLIRContext> context;
  std::unique_ptr<ExecutionEngine> engine;
  std::string loweredText;
};

} // namespace mlir::furiosa

#endif // FURIOSA_RUNTIME_JIT_H

#ifndef FURIOSA_TRANSFORMS_PASSES_H
#define FURIOSA_TRANSFORMS_PASSES_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::furiosa {

/// Raises a launch-specialized Triton kernel (every scalar argument a constant, the grid recorded in
/// the `furiosa.grid` attribute) to whole-tensor linalg on tensors.
std::unique_ptr<Pass> createRaiseToLinalgPass();

/// Promotes affine tensor.extract reads of gather-form raised linalg to indexing-map inputs.
std::unique_ptr<Pass> createStructurePass();

/// The dialects the Furiosa passes create.
void registerDialects(DialectRegistry &registry);

} // namespace mlir::furiosa

#endif // FURIOSA_TRANSFORMS_PASSES_H

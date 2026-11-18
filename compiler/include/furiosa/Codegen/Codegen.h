#ifndef FURIOSA_CODEGEN_CODEGEN_H
#define FURIOSA_CODEGEN_CODEGEN_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

namespace mlir::furiosa {

/// Maps each raised kernel (compiler/README.md's linalg contract) to a furiosa-opt kernel in the fopt
/// dialect, op class by op class (lib/Codegen/Mapping/).
std::unique_ptr<Pass> createLinalgToFoptPass();

/// furiosa-split-steps: splitSteps as a pass over a module of structured kernels (lib/Codegen/Steps.cpp).
std::unique_ptr<Pass> createSplitStepsPass();

/// splitSteps' stages one by one, for the per-pass tests (lib/Codegen/Steps.cpp): furiosa-normalize-kernels,
/// furiosa-normalize-contractions, furiosa-hoist-index-only, furiosa-kept-loops, furiosa-slice-kept-loops,
/// and furiosa-steps-plan (prints a split kernel's launch plan).
std::unique_ptr<Pass> createNormalizeKernelsPass();
std::unique_ptr<Pass> createNormalizeContractionsPass();
std::unique_ptr<Pass> createHoistIndexOnlyPass();
std::unique_ptr<Pass> createKeptLoopsPass();
std::unique_ptr<Pass> createSliceKeptLoopsPass();
std::unique_ptr<Pass> createStepsPlanPass();

/// furiosa-tile: tiles each kernel's output rows by the tiled lowering's plan into scf.for loops over its root
/// generic (lib/Codegen/Tile/TilePass.cpp).
std::unique_ptr<Pass> createTilePass();

/// furiosa-dma: lowers staged reads (fopt.stage, fopt.read) to HBM staging buffers and DMAs, choosing their layout
/// (lib/Codegen/Tile/Dma.cpp).
std::unique_ptr<Pass> createDmaPass();

/// furiosa-ve-fuse: groups the abstract Vector Engine ops (fopt.ve) into passes in the engine's stage order and
/// lowers them (lib/Codegen/Tile/VeFuse.cpp).
std::unique_ptr<Pass> createVeFusePass();

/// Writes every fopt kernel of `module` as one Rust module (the body of src/kernel/<symbol>.rs).
LogicalResult translateFoptToRust(ModuleOp module, llvm::raw_ostream &os);

/// The fopt kernel's interface as JSON, for the crate's host code: {"kernel", "chip", "pe", "inputs":
/// [pointer-arg index per parameter], "inouts": [parameters written in place, as &mut], "outputs":
/// [pointer-arg index per result]}.
FailureOr<std::string> foptSignature(ModuleOp module);

/// Rewrites reads through mixed-radix map results (d1 * 128 + d2) into projected reads of the buffer reshaped by
/// the radices and cut to a box, drops inputs that only carried loop extents, and splits a generic that only
/// chooses between two reads by an index into copies, unless `mapsAlone` says it maps as a step of its own
/// (lib/Codegen/Normalize.cpp).
void normalizeKernels(ModuleOp module, function_ref<bool(linalg::GenericOp)> mapsAlone = {});

/// maxnumf/minnumf (tl.maximum's default, relu, tl.max) take the other operand where one is NaN, as on the GPU,
/// while the Vector Engine's max and min propagate NaN: x > -inf (x < +inf for min) is false exactly where x is NaN
/// or the bound, where the bound loses to anything. So a value cleans to clean(x) = x > -inf ? x : -inf, and
/// maxnumf(x, c) = maximumf(clean(x), c) for a number c (relu), a reduction's combiner maximumf(acc, clean(v)) (a
/// row of NaN alone reduces to -inf). Runs on every kernel, before any is mapped whole (lib/Codegen/Normalize.cpp).
void guardNaNs(ModuleOp module);

/// Whether float `v` holds an integer by construction: rounded, converted from an integer, an integral constant,
/// or sums, products, negations, min/max and selects of such (lib/Codegen/Normalize.cpp).
bool isIntegral(Value v);

/// Takes contractions apart into the 2-D form op classes map: computed factors become producers, a non-zero
/// start is added after, batch dims split into slices, dims collapse (lib/Codegen/Contract.cpp).
/// `mapsAlone` says whether a batched contraction maps as a step of its own, so it need not be unbatched.
void normalizeContractions(ModuleOp module, function_ref<bool(linalg::GenericOp)> mapsAlone = {});

/// A Rust f32 literal of `value`, exact (lib/Codegen/EmitRust.cpp).
std::string rustF32(float value);

/// `v` as type `t`: the same elements in row-major order, through collapse/expand_shape (moves no data).
Value reshapeTo(OpBuilder &b, Location loc, Value v, RankedTensorType t);

/// Unrolls the kernels' top-level scf.for loops into host-launched iterations, then structures their reads
/// (lib/Codegen/Loops.cpp).
LogicalResult unrollKeptLoops(ModuleOp module);

/// A padded tensor only boxes are cut from becomes the source inserted into a constant (lib/Codegen/Normalize.cpp).
void padsToInserts(ModuleOp module);

/// Tiles the generics unrollKeptLoops lifted back to their iteration's slice (lib/Codegen/Loops.cpp).
void sliceKeptLoops(ModuleOp module);

/// Evaluates every value of a linalg.generic body that depends only on the iteration indices at compile time,
/// over the dims it reads, and replaces it with a constant tensor input (lib/Codegen/Hoist.cpp).
void hoistIndexOnly(ModuleOp module);

/// For an index-only i1 value of a generic body that depends on one loop dim: that dim, and the value at each of
/// its iterations (`ranges` are the loop extents).
std::optional<std::pair<unsigned, SmallVector<bool>>> truthOverOneDim(Value cond, Block &body,
                                                                    ArrayRef<int64_t> ranges);

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

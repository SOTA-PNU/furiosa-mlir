// Shared by the op-class mappings (one file per class): what a mapping receives and the helpers to build a
// furiosa-opt kernel in the fopt dialect.

#ifndef FURIOSA_CODEGEN_MAPPING_MAPPING_H
#define FURIOSA_CODEGEN_MAPPING_MAPPING_H

#include "furiosa/Codegen/Fopt.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"

#include <optional>
#include <string>

namespace mlir::furiosa::mapping {

/// A raised kernel (compiler/README.md's linalg contract) as a mapping sees it.
struct RaisedKernel {
  func::FuncOp func;
  /// The pointer-argument names (from the Triton signature when the raised function kept them).
  SmallVector<std::string> argNames;
  /// The pointer-argument indices the kernel writes, in result order (`furiosa.written`).
  SmallVector<int64_t> written;
};

/// SPMD over chips: the kernels of a module with `furiosa.chips = N` run on N chips, each on its own copy of the
/// data (every type's chip mapping is the CHIP axis), and furiosa-shard may split one parallel dim of a kernel over
/// them, marking the pointer arguments it splits with `furiosa.shard` (the dim) (LinalgToFopt.cpp).
inline constexpr llvm::StringLiteral kChipAxis = "CHIP";
inline constexpr llvm::StringLiteral kChipsAttr = "furiosa.chips";
inline constexpr llvm::StringLiteral kShardAttr = "furiosa.shard";
/// The chips the module of `op` runs its kernels on (1 without `furiosa.chips`).
int64_t chipCount(Operation *op);
/// The chip mapping of the kernels in the module of `op`: m![1], or the CHIP axis.
fopt::MapAttr chipMapping(Operation *op);

/// Builds one fopt kernel: a func.func with the #[device] and axes! attributes.
class KernelBuilder {
public:
  KernelBuilder(RaisedKernel &raised);

  MLIRContext *getContext() { return ctx; }
  OpBuilder &builder() { return b; }
  Location loc() { return raised.func.getLoc(); }

  /// Declares an axis of the kernel's axes! and returns it as a mapping.
  fopt::MapAttr axis(StringRef name, int64_t size);
  /// How many axes are declared so far.
  size_t numAxes() const { return axes.size(); }
  fopt::MapAttr one() { return fopt::MapAttr::broadcast(ctx, 1); }
  /// The chip mapping (one chip, or the chip axis of an SPMD kernel: chipMapping) and the cluster mapping (cluster 0
  /// of 2) every kernel uses.
  fopt::MapAttr chip() { return chips > 1 ? axis(kChipAxis, chips) : one(); }
  fopt::MapAttr cluster() { return one().padTo(2); }

  /// A pointer argument the kernel reads, as `&HbmTensor<D, Chip, element>`; returns its parameter index.
  unsigned input(int64_t ptrArg, Type elementType, fopt::MapAttr element);
  /// The kernel parameter `i` (after beginBody): the argument, or this chip's shard of it when furiosa-shard split
  /// the argument over the chips.
  Value param(unsigned i) { return params[i]; }
  /// A pointer argument the kernel writes, as a returned `HbmTensor<D, Chip, element>`.
  void output(int64_t ptrArg, Type elementType, fopt::MapAttr element);
  /// A pointer argument the kernel updates in place, as `&mut HbmTensor<D, Chip, element>` (the kernel
  /// writes part of it, or reads it too); returns its parameter index.
  unsigned inout(int64_t ptrArg, Type elementType, fopt::MapAttr element);

  /// Starts the body once every input and output is declared.
  void beginBody();
  /// Ends the body returning `results` (one per output, in order) and replaces the raised function.
  void finish(ArrayRef<Value> results);

  fopt::HbmTensorType hbmType(Type elementType, fopt::MapAttr element) {
    return fopt::HbmTensorType::get(ctx, elementType, chip(), element);
  }
  fopt::DmTensorType dmType(Type elementType, fopt::MapAttr slice, fopt::MapAttr element) {
    return fopt::DmTensorType::get(ctx, elementType, chip(), cluster(), slice, element);
  }
  fopt::StreamType streamType(Type elementType, fopt::MapAttr slice, fopt::MapAttr time, fopt::MapAttr packet) {
    return fopt::StreamType::get(ctx, elementType, chip(), cluster(), slice, time, packet);
  }

private:
  RaisedKernel &raised;
  MLIRContext *ctx;
  OpBuilder b;
  int64_t chips;
  SmallVector<std::pair<std::string, int64_t>> axes;
  SmallVector<Type> inputTypes, outputTypes;
  SmallVector<int64_t> inputArgs, outputArgs;
  SmallVector<bool> inoutParams;
  func::FuncOp kernel;
  SmallVector<Value> params;

  /// The whole buffer's HBM type behind a parameter or result of `shard` type, for a pointer argument whose dim
  /// furiosa-shard split over the chips (null when it did not).
  fopt::HbmTensorType wholeType(fopt::HbmTensorType shard, RankedTensorType tensor, Attribute dim);
  /// One HBM-to-HBM DMA of tensor `from` into tensor `to` (a chip's block of a whole buffer, or back).
  void copy(Value from, Value to);
};

/// The loops furiosa-tile makes before a class maps a kernel (lib/Codegen/Tile/TilePass.cpp): tile sizes of the
/// root generic's loops (0 for a loop not tiled) and of one generic's reduction loops (K or chunk tiles), each a
/// whole range of its loop. A class that returns a tiling maps the tiled kernel (Tile/TileLoops.h reads the loops).
struct Tiling {
  linalg::GenericOp root;
  SmallVector<int64_t> rowSizes;
  linalg::GenericOp reduce;
  SmallVector<int64_t> reduceSizes;
};

/// An op class: `match` says whether the raised kernel is one; `map` builds its furiosa-opt kernel or
/// explains why it cannot; `tiling`, when set, the loops furiosa-tile makes for a kernel the class maps.
struct OpClass {
  const char *name;
  bool (*match)(RaisedKernel &);
  LogicalResult (*map)(RaisedKernel &);
  std::optional<Tiling> (*tiling)(RaisedKernel &) = nullptr;
};

/// Elementwise linalg.generic over f32 (Mapping/Elementwise.cpp).
OpClass elementwiseClass();
/// x(m, k) . w(n, k) over bf16 with fp32 accumulation, then an optional cast (Mapping/Contraction.cpp).
OpClass contractionClass();
/// Row-wise elementwise generics and row reductions over [rows, columns] (Mapping/Rowwise.cpp).
OpClass rowwiseClass();
/// Any kernel the tiled lowering folds: contractions and elementwise tails over row tiles (Tile/Lower.cpp).
OpClass tiledClass();
/// Rows of a table read at data-dependent indices, as an indirect DMA (Mapping/Gather.cpp).
OpClass gatherClass();
/// A buffer copied into a box of another, in place, as an HBM-to-HBM DMA (Mapping/Copy.cpp).
OpClass copyClass();

/// Every op class, in the order a kernel tries them (LinalgToFopt.cpp).
ArrayRef<OpClass> opClasses();
/// The first op class that matches the kernel, or null.
const OpClass *matchClass(RaisedKernel &kernel);
/// The pointer-argument names (from NameLocs on the arguments) and `furiosa.written` of a raised function.
RaisedKernel raisedKernel(func::FuncOp func);
/// Whether op class `c` maps `func` (tried on a copy, its diagnostics dropped), with at most `maxDmas` DMA
/// statements (a device loop's body counted once).
bool mapsOnCopy(func::FuncOp func, const OpClass &c, unsigned maxDmas = ~0u);
/// Maps a raised kernel with the first matching op class that maps it.
LogicalResult mapKernel(func::FuncOp func);

} // namespace mlir::furiosa::mapping

#endif // FURIOSA_CODEGEN_MAPPING_MAPPING_H

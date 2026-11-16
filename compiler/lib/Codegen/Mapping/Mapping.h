// Shared by the op-class mappings (one file per class): what a mapping receives and the helpers to build a
// furiosa-opt kernel in the fopt dialect.

#ifndef FURIOSA_CODEGEN_MAPPING_MAPPING_H
#define FURIOSA_CODEGEN_MAPPING_MAPPING_H

#include "furiosa/Codegen/Fopt.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"

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

/// Builds one fopt kernel: a func.func with the #[device] and axes! attributes.
class KernelBuilder {
public:
  KernelBuilder(RaisedKernel &raised);

  MLIRContext *getContext() { return ctx; }
  OpBuilder &builder() { return b; }
  Location loc() { return raised.func.getLoc(); }

  /// Declares an axis of the kernel's axes! and returns it as a mapping.
  fopt::MapAttr axis(StringRef name, int64_t size);
  fopt::MapAttr one() { return fopt::MapAttr::broadcast(ctx, 1); }
  /// The chip mapping (one chip) and cluster mapping (cluster 0 of 2) every kernel uses for now.
  fopt::MapAttr chip() { return one(); }
  fopt::MapAttr cluster() { return one().padTo(2); }

  /// A pointer argument the kernel reads, as `&HbmTensor<D, Chip, element>`; returns its parameter index.
  unsigned input(int64_t ptrArg, Type elementType, fopt::MapAttr element);
  /// The kernel parameter `i` (after beginBody).
  Value param(unsigned i) { return kernel.getArgument(i); }
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
  SmallVector<std::pair<std::string, int64_t>> axes;
  SmallVector<Type> inputTypes, outputTypes;
  SmallVector<int64_t> inputArgs, outputArgs;
  SmallVector<bool> inoutParams;
  func::FuncOp kernel;
};

/// An op class: `match` says whether the raised kernel is one; `map` builds its furiosa-opt kernel or
/// explains why it cannot.
struct OpClass {
  const char *name;
  bool (*match)(RaisedKernel &);
  LogicalResult (*map)(RaisedKernel &);
};

/// Elementwise linalg.generic over f32 (Mapping/Elementwise.cpp).
OpClass elementwiseClass();
/// x(m, k) . w(n, k) over bf16 with fp32 accumulation, then an optional cast (Mapping/Contraction.cpp).
OpClass contractionClass();
/// Row-wise elementwise generics and row reductions over [rows, columns] (Mapping/Rowwise.cpp).
OpClass rowwiseClass();
/// Rows of a table read at data-dependent indices, as an indirect DMA (Mapping/Gather.cpp).
OpClass gatherClass();

} // namespace mlir::furiosa::mapping

#endif // FURIOSA_CODEGEN_MAPPING_MAPPING_H

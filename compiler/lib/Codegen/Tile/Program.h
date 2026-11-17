// A raised kernel as one scalar expression over an iteration space of axes (lib/Codegen/Tile/).
//
// The structured linalg of a kernel (its generics, fills, pads, slices and reshapes) is folded into a single
// expression DAG evaluated at every point of the kernel's output space. Leaves read buffers at affine indices;
// reductions introduce their own axes. Every index is linear in the axes: dims that the maps divide or reduce by
// constants are split into mixed-radix digits first, so `(d floordiv 5) * 2` becomes `2 * d_hi` over the digit
// `d_hi = d floordiv 5`. Unit-extent dims disappear.

#ifndef FURIOSA_CODEGEN_TILE_PROGRAM_H
#define FURIOSA_CODEGEN_TILE_PROGRAM_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::furiosa::tile {

/// An axis of the iteration space: an output digit or a reduction digit.
struct Axis {
  int64_t size;
  bool reduction;
};

/// sum(coefficient * axis) + offset, with no zero coefficients.
struct Lin {
  SmallVector<std::pair<unsigned, int64_t>> terms;
  int64_t offset = 0;

  int64_t coeff(unsigned axis) const;
  bool uses(unsigned axis) const { return coeff(axis) != 0; }
  void add(unsigned axis, int64_t c);
  bool operator==(const Lin &o) const;
};

/// A read of a kernel argument. Indices are in the buffer's own coordinates; a read outside [0, shape) returns
/// `fill` (the value of the tensor.pad it came through). `fill` is unset when no read can leave the buffer.
struct Read {
  unsigned arg;
  Type type;
  SmallVector<int64_t> shape;
  SmallVector<Lin> index;
  std::optional<double> fill;

  /// The axes the read depends on.
  SmallVector<unsigned> axes() const;
};

struct Node {
  enum Kind { Leaf, Const, Op, Reduce } kind;
  unsigned read = 0;          // Leaf: index into Program::reads
  double value = 0;           // Const
  std::string op;             // Op: add sub mul div max min neg abs exp sqrt rsqrt tanh log erf ..., select, cmp.<pred>;
                              // Reduce: add | max
  SmallVector<unsigned> args; // Op operands; Reduce: {body}
  SmallVector<unsigned> axes; // Reduce: the axes it reduces
};

struct Program {
  SmallVector<Axis> axes;
  /// The output axes, outermost first.
  SmallVector<unsigned> out;
  SmallVector<Read> reads;
  std::vector<Node> nodes;
  unsigned root = 0;
  /// The store: the written argument and its index over the output axes.
  unsigned outArg = 0;
  Type outType;
  SmallVector<int64_t> outShape;
  SmallVector<Lin> outIndex;

  /// Where each axis comes from: (dim, place), the axis being (dim / place) % size; the dims' extents, the
  /// first of them the output dims, which are the root generic's loops `outLoop` (for the tiling pass).
  SmallVector<std::pair<unsigned, int64_t>> axisDim;
  SmallVector<int64_t> dimExtent;
  SmallVector<unsigned> outLoop;
  Operation *rootOp = nullptr;
  /// Tile loops the kernel already has (furiosa-tile): axis `first` tiled by `second`, its loop's step.
  SmallVector<std::pair<unsigned, int64_t>> loopTiles;

  /// The axes a node depends on (reduction axes it reduces excluded).
  SmallVector<unsigned> axesOf(unsigned node) const;
  void print(llvm::raw_ostream &os) const;
};

/// Folds a raised kernel into a Program, or explains why it cannot (`why`).
FailureOr<Program> analyze(func::FuncOp func, ArrayRef<int64_t> written, std::string &why);

} // namespace mlir::furiosa::tile

#endif // FURIOSA_CODEGEN_TILE_PROGRAM_H

// The loops furiosa-tile makes (TilePass.cpp), as the op classes that map tiled kernels read them back.
//
// Row tile loops: a nest of static scf.for (0 to trips * step by step), each carrying the stored tensor as its one
// iteration argument, whose innermost body inserts one tile of it at the loops' offsets. K (or chunk) loops: an
// scf.for marked `furiosa.tile_dim = <the generic loop it tiles>`, accumulating one tile of a generic's reduction
// per iteration in its iteration argument.

#ifndef FURIOSA_CODEGEN_TILE_TILELOOPS_H
#define FURIOSA_CODEGEN_TILE_TILELOOPS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"

#include <optional>
#include <string>

namespace mlir::furiosa::tile {

/// One tile loop: 0 to trips * step by step.
struct TileLoop {
  scf::ForOp op;
  Value iv;
  int64_t step = 1, trips = 1;
  /// K loops: the loop of the generic it tiles (`furiosa.tile_dim`).
  std::optional<unsigned> dim;
  /// The accumulator: the iteration argument, its start and what the body yields.
  Value iterArg, init, yield;
};

/// A nest of row tile loops around the stored value.
struct TileNest {
  SmallVector<TileLoop> loops; // outermost first
  tensor::InsertSliceOp insert;
  /// The inserted tile, and per dim of the stored tensor its offset as an expression of the loops' induction
  /// variables (dim i = loops[i].iv).
  Value tile;
  SmallVector<AffineExpr> offsets;
  /// The loop tiling each stored dim (nullopt: the dim is whole in every tile).
  SmallVector<std::optional<unsigned>> loopOf;
};

/// The row tile-loop nest whose result is `stored`, or nothing when `stored` is not a tile loop's result; failure
/// (with `why`) when it is a loop but not a nest of whole tiles of the stored tensor.
FailureOr<std::optional<TileNest>> tileNest(Value stored, std::string &why);

/// A K loop's description, or nothing when `v` is not a K loop's result.
std::optional<TileLoop> kLoop(Value v);

/// An index (a constant, one of `ivs`, or an affine.apply of those) as an expression of `ivs` (dim i = ivs[i]).
FailureOr<AffineExpr> offsetExpr(OpFoldResult index, ArrayRef<Value> ivs, MLIRContext *ctx);

} // namespace mlir::furiosa::tile

#endif // FURIOSA_CODEGEN_TILE_TILELOOPS_H

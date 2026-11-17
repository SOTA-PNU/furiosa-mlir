// Hardware-abstract data movement between the tiled lowering (Lower.cpp) and furiosa-dma (Dma.cpp): what a
// staged read holds (fopt.stage's spec), encoded as attributes.

#ifndef FURIOSA_CODEGEN_TILE_DMA_H
#define FURIOSA_CODEGEN_TILE_DMA_H

#include "Plan.h"

#include "furiosa/Codegen/Fopt.h"

namespace mlir::furiosa::tile {

/// A kernel buffer read at a linear index over the program's axes, and the layout its tiles are cut from.
struct StageSpec {
  Read read;                         // arg is unused: the stage's operand is the buffer
  SmallVector<int64_t> sizes;        // per program axis
  SmallVector<int64_t> extents;      // per program axis, padded by the layout
  SmallVector<unsigned> rows;        // the layout's row axes, in order
  SmallVector<fopt::MapAttr> canonical; // their canonical digits (rows, a reduction's rc/rf, the column)
  std::optional<KLayout> k;          // a contraction operand's K split
  int64_t kTile = 0;                 // the K extent of one pass (0: K is not tiled)
  fopt::MapAttr kf;                  // the flat K axis
  SmallVector<unsigned> ext;         // inside a Vector Engine reduction: its axes
  fopt::MapAttr rf, rc;              // its flat axis and chunk axis
  std::string tag;                   // prefix of the staging axes
  int64_t axesAt = 0;                // where the staging axes go in the kernel's axes

  int64_t kTiles() const { return kTile ? k->size() / kTile : 1; }
  DictionaryAttr encode(MLIRContext *ctx) const;
  static FailureOr<StageSpec> decode(DictionaryAttr spec);
};

/// Lowers every fopt.stage and fopt.read of `func` to fopt DMAs (furiosa-dma).
LogicalResult lowerDataMovement(func::FuncOp func);

} // namespace mlir::furiosa::tile

#endif // FURIOSA_CODEGEN_TILE_DMA_H

// The tiling plan of a Program (Program.h), the cost model of the tiled lowering: which reduction is a
// contraction and how its K splits, the column and its width per slice, and how the rows split into tile, slice
// and local digits (padding rows and columns that split badly). It holds every capacity the lowering relies on:
// TRF lanes, the DM budget, intra-slice accumulators, slices and lanes.

#ifndef FURIOSA_CODEGEN_TILE_PLAN_H
#define FURIOSA_CODEGEN_TILE_PLAN_H

#include "Program.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <map>
#include <optional>
#include <string>
#include <tuple>

namespace mlir::furiosa::tile {

constexpr int64_t kSlices = 256, kLanes = 8, kTrfLaneBytes = 8 * 1024, kDmBudget = 384 * 1024;

inline int64_t roundUp(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

/// A digit of an axis: (axis / place) % size.
struct Digit {
  unsigned axis;
  int64_t place, size;
  int64_t extent = 0; // the axis padded to this extent (0: unpadded)
};

/// A contraction: the sum `node` over `k` of the product of two reads. `stream` reads the rows `m`, `trf` the
/// column axis `n` (or none).
struct Gemm {
  unsigned node;
  unsigned stream, trf;
  SmallVector<unsigned> m, n, k;
  SmallVector<unsigned> streamExtras, trfExtras; // further factors of each operand (reads)
  SmallVector<unsigned> b;                       // batch axes: output axes both operands index
  /// The TRF operand's axes: the batch axes, then the column.
  SmallVector<unsigned> trfAxes() const {
    SmallVector<unsigned> a(b);
    llvm::append_range(a, n);
    return a;
  }
};

/// The K layout of a contraction's operands: taps (axes peeled into one DMA each) times the inner axis.
struct KLayout {
  SmallVector<unsigned> taps;
  std::optional<unsigned> inner;
  int64_t tapPad = 1, innerPad = 1;
  int64_t size() const { return tapPad * innerPad; }
};

/// A reduction with the reductions nested directly in it merged (same combiner): (op, axes, body).
std::tuple<std::string, SmallVector<unsigned>, unsigned> flatReduce(const Program &p, unsigned node);

class Plan {
public:
  Plan(func::FuncOp func, Program &p, bool quiet) : func(func), p(p), quiet(quiet) {}

  /// Plans the layout; failure (with a diagnostic unless quiet) when the program cannot be lowered.
  LogicalResult plan();
  /// The planned tile digits (rows, and the column's when its tiles pad), most significant first.
  ArrayRef<Digit> tiles() const { return tileD; }
  /// The planned K tiles as tile sizes of the contraction's reduction axes, {axis, size} per tiled axis: whole
  /// taps (the outer taps one at a time) or a part of the inner axis (every tap one at a time). Nothing when K is
  /// not tiled, or not by whole axes (padded taps or inner).
  std::optional<SmallVector<std::pair<unsigned, int64_t>>> kAxisTiles();

protected:
  func::FuncOp func;
  Program &p;
  bool quiet;

  std::optional<Gemm> gemm;
  KLayout kl;
  int64_t kTile = 0; // the K extent of one contraction pass (KF when K is not tiled)
  SmallVector<unsigned> rows;
  std::optional<unsigned> col;               // the column axis
  int64_t colLocalSize = 1, colPad = kLanes; // local columns, and padded to whole packets
  std::optional<Digit> colBlock, colLocalD;
  int64_t colExtent = 0;                     // the column axis padded (whole column tiles or packets; 0: none)
  std::map<unsigned, int64_t> rowExtent;     // row axes padded to these extents (rows that split badly)
  SmallVector<Digit> tileD, sliceD, localD;  // row digits by role, most significant first
  int64_t rfMax = 0;                         // the longest Vector Engine reduction chunk

  LogicalResult fail(const Twine &msg);
  LogicalResult findGemm();
  LogicalResult tryGemm(unsigned node);
  LogicalResult planLayout();
  int64_t assignRows(int64_t rowSlices, int64_t perRow, SmallVector<Digit> &local, SmallVector<Digit> &slice,
                     SmallVector<Digit> &tile);

  int64_t kTiles() { return kTile ? kl.size() / kTile : 1; }
  /// The extent the layout pads axis `a` to (its size unless padded).
  int64_t extentOf(unsigned a) {
    if (col && a == *col)
      return colExtent ? colExtent : p.axes[a].size;
    auto it = rowExtent.find(a);
    return it != rowExtent.end() ? it->second : p.axes[a].size;
  }
  /// Whether a contraction operand's taps each sit alone in a buffer dim (a weight's kernel position, no window).
  bool windowFree(const Read &r) {
    return !kl.taps.empty() && llvm::all_of(kl.taps, [&](unsigned a) {
      int dims = 0;
      for (const Lin &l : r.index)
        if (l.uses(a)) {
          if (l.terms.size() != 1)
            return false;
          ++dims;
        }
      return dims == 1;
    });
  }
  /// A windowed operand is tap-major in HBM, [taps, rows, inner], so each tap's DMA writes one contiguous block
  /// (the NPU compiler's cost grows superlinearly in the contiguous runs an HBM destination has), when its inner
  /// run is a whole 8-byte DMA unit (each tile's move to DM puts K innermost).
  bool tapMajorRead(const Read &r) {
    return !kl.taps.empty() && !windowFree(r) && kl.innerPad * r.type.getIntOrFloatBitWidth() >= 64;
  }
};

} // namespace mlir::furiosa::tile

#endif // FURIOSA_CODEGEN_TILE_PLAN_H

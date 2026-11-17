// The tiling plan (Plan.h).

#include "Plan.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <functional>
#include <set>

using namespace mlir;
using namespace mlir::furiosa::tile;

LogicalResult Plan::fail(const Twine &msg) {
  if (std::getenv("FURIOSA_TILED_DEBUG")) // diagnostics of trial mappings are swallowed
    llvm::errs() << "tiled: " << func.getName() << ": " << msg << "\n";
  if (!quiet)
    func.emitError("tiled: ") << msg;
  return failure();
}

/// A reduction with the reductions nested directly in it merged (same combiner): (op, axes, body).
std::tuple<std::string, SmallVector<unsigned>, unsigned> mlir::furiosa::tile::flatReduce(const Program &p,
                                                                                         unsigned node) {
  const Node &n = p.nodes[node];
  SmallVector<unsigned> axes(n.axes);
  unsigned body = n.args[0];
  while (p.nodes[body].kind == Node::Reduce && p.nodes[body].op == n.op) {
    llvm::append_range(axes, p.nodes[body].axes);
    body = p.nodes[body].args[0];
  }
  return {n.op, axes, body};
}

LogicalResult Plan::findGemm() {
  // Reductions that head a nest (not the body of a reduction with the same combiner).
  std::set<unsigned> nested;
  for (const Node &n : p.nodes)
    if (n.kind == Node::Reduce && p.nodes[n.args[0]].kind == Node::Reduce && p.nodes[n.args[0]].op == n.op)
      nested.insert(n.args[0]);
  SmallVector<unsigned> reduces;
  for (auto [i, n] : llvm::enumerate(p.nodes))
    if (n.kind == Node::Reduce && !nested.count(i) && !std::get<1>(flatReduce(p, i)).empty())
      reduces.push_back(i);
  // A sum of the product of two reads is a contraction; every other reduction runs on the Vector Engine.
  for (unsigned r : reduces) {
    if (failed(tryGemm(r)))
      return failure();
  }
  return success();
}

LogicalResult Plan::tryGemm(unsigned node) {
  auto [op, axes, body] = flatReduce(p, node);
  if (op != "add")
    return success();
  Node red = p.nodes[node];
  red.axes = axes;
  red.args = {body};
  SmallVector<unsigned> factors;
  std::function<bool(unsigned)> collect = [&](unsigned n) {
    const Node &x = p.nodes[n];
    if (x.kind == Node::Op && x.op == "mul")
      return collect(x.args[0]) && collect(x.args[1]);
    if (x.kind == Node::Leaf) {
      factors.push_back(x.read);
      return true;
    }
    return false;
  };
  if (!collect(red.args[0]) || factors.size() < 2)
    return success(); // not a contraction: the Vector Engine reduces it
  std::set<unsigned> k(red.axes.begin(), red.axes.end());
  auto outAxes = [&](unsigned read) {
    SmallVector<unsigned> o;
    for (unsigned a : p.reads[read].axes())
      if (!k.count(a))
        o.push_back(a);
    return o;
  };
  auto rowsOf = [&](ArrayRef<unsigned> axes) {
    int64_t n = 1;
    for (unsigned a : axes)
      n *= p.axes[a].size;
    return n;
  };
  // Factors fall into two operands by the output axes they share; a factor with only contraction axes (a
  // per-K scale) joins the smaller operand. Each operand is its largest factor times the others, multiplied in
  // on the Vector Engine.
  // An output axis two factors share is a batch axis: one contraction per batch index, laid out over the
  // slices (both operands distributed by it) and tiles, never in time.
  std::map<unsigned, int> uses;
  for (unsigned f : factors)
    for (unsigned a : outAxes(f))
      ++uses[a];
  SmallVector<unsigned> batch;
  for (auto [a, n] : uses)
    if (n > 1)
      batch.push_back(a);
  auto nonBatch = [&](unsigned read) {
    SmallVector<unsigned> o;
    for (unsigned a : outAxes(read))
      if (!llvm::is_contained(batch, a))
        o.push_back(a);
    return o;
  };
  SmallVector<SmallVector<unsigned>> groups; // factor indices
  SmallVector<std::set<unsigned>> groupAxes;
  SmallVector<unsigned> free;
  for (unsigned f = 0; f < factors.size(); ++f) {
    SmallVector<unsigned> o = nonBatch(factors[f]);
    if (o.empty()) {
      free.push_back(f);
      continue;
    }
    SmallVector<unsigned> joined;
    for (unsigned gi = 0; gi < groups.size(); ++gi)
      if (llvm::any_of(o, [&](unsigned a) { return groupAxes[gi].count(a); }))
        joined.push_back(gi);
    if (joined.empty()) {
      groups.push_back({f});
      groupAxes.push_back(std::set<unsigned>(o.begin(), o.end()));
      continue;
    }
    // Merge every group this factor touches into the first.
    unsigned first = joined.front();
    groups[first].push_back(f);
    groupAxes[first].insert(o.begin(), o.end());
    for (int gi = joined.size() - 1; gi >= 1; --gi) {
      llvm::append_range(groups[first], groups[joined[gi]]);
      groupAxes[first].insert(groupAxes[joined[gi]].begin(), groupAxes[joined[gi]].end());
      groups.erase(groups.begin() + joined[gi]);
      groupAxes.erase(groupAxes.begin() + joined[gi]);
    }
  }
  if (groups.size() == 1 && groups[0].size() >= 1 && !free.empty()) {
    // One operand with output axes and per-K factors: the largest per-K factor is the other operand.
    groups.push_back({free.pop_back_val()});
    groupAxes.push_back({});
  }
  if (groups.size() != 2)
    return groups.size() == 1 ? fail("a contraction with a batch axis (both operands index it)") : success();
  SmallVector<unsigned> o[2] = {SmallVector<unsigned>(groupAxes[0].begin(), groupAxes[0].end()),
                                SmallVector<unsigned>(groupAxes[1].begin(), groupAxes[1].end())};
  for (unsigned f : free)
    groups[rowsOf(o[0]) <= rowsOf(o[1]) ? 0 : 1].push_back(f);
  // The main factor of each operand reads every contraction axis.
  unsigned mainOf[2];
  for (int gi = 0; gi < 2; ++gi) {
    auto best = llvm::max_element(groups[gi], [&](unsigned x, unsigned y) {
      return p.reads[factors[x]].axes().size() < p.reads[factors[y]].axes().size();
    });
    mainOf[gi] = *best;
    for (unsigned a : red.axes)
      if (!llvm::is_contained(p.reads[factors[*best]].axes(), a))
        return fail("a contraction axis one operand does not read");
  }
  // The TRF operand holds one column axis; the stream operand has the rest (possibly none).
  int s;
  if (o[0].size() > 1 && o[1].size() > 1)
    return fail("a contraction whose operands both have several output axes");
  if (o[0].empty() || o[0].size() > 1)
    s = 0;
  else if (o[1].empty() || o[1].size() > 1)
    s = 1;
  else
    s = rowsOf(o[0]) >= rowsOf(o[1]) ? 0 : 1;
  if (o[1 - s].empty())
    return success(); // no column (one per batch index, as in a depthwise convolution): the Vector Engine reduces it
  if (o[1 - s].size() != 1)
    return fail("a contraction without a column axis");
  if (gemm)
    return fail("more than one contraction");
  // Its rows and columns are the tile's: an output axis that an enclosing reduction reduces (a softmax max over
  // attention scores in the same kernel) would need the contraction's rows in that reduction's time.
  for (unsigned a : llvm::concat<unsigned>(o[0], o[1], batch))
    if (!llvm::is_contained(p.out, a))
      return fail("a contraction whose output axis an enclosing reduction reduces");
  Gemm g;
  g.node = node;
  g.stream = factors[mainOf[s]];
  g.trf = factors[mainOf[1 - s]];
  for (unsigned f : groups[s])
    if (f != mainOf[s])
      g.streamExtras.push_back(factors[f]);
  for (unsigned f : groups[1 - s])
    if (f != mainOf[1 - s])
      g.trfExtras.push_back(factors[f]);
  g.m = o[s];
  g.n = o[1 - s];
  g.b = batch;
  llvm::sort(g.b, [&](unsigned x, unsigned y) { return llvm::find(p.out, x) < llvm::find(p.out, y); });
  g.k = SmallVector<unsigned>(red.axes);
  llvm::sort(g.m, [&](unsigned x, unsigned y) { return llvm::find(p.out, x) < llvm::find(p.out, y); });
  gemm = g;
  return success();
}

LogicalResult Plan::planLayout() {
  if (gemm) {
    rows = gemm->b;
    llvm::append_range(rows, gemm->m);
    col = gemm->n[0];
  } else {
    // The column: the output axis contiguous (stride 1) in the most reads, else the innermost output axis; an
    // even one, since DMA rows are 8-byte units.
    std::map<unsigned, int> votes;
    for (const Read &r : p.reads)
      if (!r.index.empty())
        for (auto [a, c] : r.index.back().terms)
          if (c == 1)
            ++votes[a];
    unsigned best = p.out.back();
    for (unsigned a : p.out)
      if (p.axes[best].size % 2 && !(p.axes[a].size % 2))
        best = a;
    for (auto [a, v] : votes)
      if (llvm::is_contained(p.out, a) && !(p.axes[a].size % 2) && (v > votes[best] || p.axes[best].size % 2))
        best = a;
    col = best;
    for (unsigned a : p.out)
      if (a != best)
        rows.push_back(a);
  }
  int64_t c = p.axes[*col].size;
  colLocalSize = 0;
  // An odd column pads to whole packets (DMA rows are 8-byte units): cells past it are computed and never stored.
  if (c % 2)
    c = colExtent = roundUp(c, kLanes);
  if (gemm) {
    // The inner K axis: one the stream operand reads alone in a dim (preferably its contiguous last dim, else the
    // largest), so it moves as a whole in each tap's DMA; every other K axis is a tap.
    const Read &s = p.reads[gemm->stream];
    auto alone = [&](unsigned a, const Lin &l) { return l.terms.size() == 1 && l.terms[0].first == a; };
    for (unsigned a : gemm->k)
      if (alone(a, s.index.back()) && s.index.back().terms[0].second == 1)
        kl.inner = a;
    if (!kl.inner)
      for (unsigned a : gemm->k)
        if (llvm::any_of(s.index, [&](const Lin &l) { return alone(a, l); }) &&
            (!kl.inner || p.axes[a].size > p.axes[*kl.inner].size))
          kl.inner = a;
    for (unsigned a : gemm->k)
      if (!kl.inner || a != *kl.inner)
        kl.taps.push_back(a);
    int64_t t = 1;
    for (unsigned a : kl.taps)
      t *= p.axes[a].size;
    int64_t in = kl.inner ? p.axes[*kl.inner].size : 1;
    while (kl.innerPad < in && kl.innerPad < 32)
      kl.innerPad *= 2;
    if (in > 32)
      kl.innerPad = roundUp(in, 32);
    kl.tapPad = roundUp(t, std::max<int64_t>(1, 32 / kl.innerPad));
    // K tiles and columns per slice: the largest K tile (a multiple of 32 dividing K) for which some whole number
    // of lane groups dividing the columns fits a TRF lane (groups x tile bf16) on at most 256 column blocks.
    // K tiles beyond the first accumulate through a Vector Engine add.
    // Among the K tiles (multiples of 32 dividing K) and lane-group widths that fit a TRF lane on at most 256
    // column blocks, the one with the fewest passes: row tiles x (K tiles + their adds + the epilogue).
    int64_t kf = kl.size();
    // Tap-major operands (windowed reads) take K tiles of whole taps or of part of the inner axis.
    SmallVector<unsigned> operands = {gemm->stream, gemm->trf};
    llvm::append_range(operands, gemm->streamExtras);
    llvm::append_range(operands, gemm->trfExtras);
    bool tapMajorK = llvm::any_of(operands, [&](unsigned r) { return tapMajorRead(p.reads[r]); });
    auto kTileFits = [&](int64_t t) {
      return kf % t == 0 && (!tapMajorK || t % kl.innerPad == 0 || kl.innerPad % t == 0);
    };
    colLocalSize = 0;
    int64_t bestCost = INT64_MAX;
    for (int64_t t = kf; t >= 32; t -= 32) {
      if (!kTileFits(t))
        continue;
      int64_t maxGroups = kTrfLaneBytes / (2 * t);
      if (maxGroups < 1)
        continue;
      SmallVector<int64_t> widths;
      if (c % kLanes == 0) {
        for (int64_t g = std::min(maxGroups, c / kLanes); g >= 1; --g)
          if ((c / kLanes) % g == 0 && c / (g * kLanes) <= kSlices)
            widths.push_back(g * kLanes);
      } else {
        widths.push_back(0); // decided below: at most one lane group per slice
      }
      for (int64_t w : widths) {
        int64_t blocks = w ? c / w : 1;
        SmallVector<Digit> l, sl, tl;
        int64_t tiles = assignRows(kSlices / int64_t(llvm::PowerOf2Ceil(blocks)),
                                   4 * roundUp(std::max<int64_t>(w, 8), kLanes) * 4 + 4 * t, l, sl, tl);
        int64_t cost = tiles * (2 * (kf / t) + 2);
        if (cost < bestCost) {
          bestCost = cost;
          kTile = t;
          colLocalSize = w;
        }
      }
    }
    if (!colLocalSize && c % kLanes == 0) {
      // More columns than 256 blocks of whole lane groups: tiles of 256 blocks, the last one padded. The K tile
      // and lane groups fill a TRF lane.
      for (int64_t t = kf; t >= 32; t -= 32)
        if (kTileFits(t) && kTrfLaneBytes / (2 * t) >= 1) {
          kTile = t;
          colLocalSize = std::min<int64_t>(kTrfLaneBytes / (2 * t), 128 / kLanes) * kLanes;
          break;
        }
    }
    if (!kTile)
      return fail("no K tile fits the TRF for " + std::to_string(c) + " columns");
  } else if (c % kLanes == 0) {
    // Vector Engine work: whole columns when few, else blocks over the slices (all of them when there are no rows).
    colLocalSize = c;
    if (c > 64)
      for (int64_t d = kLanes; d < c; d += kLanes)
        if (c % d == 0 && c / d <= (rows.empty() ? kSlices : 16)) {
          colLocalSize = d;
          break;
        }
    // The intra-slice reduce holds 8 accumulators per slice: a reduced tile's column packets, each split in two
    // for the float region, must fit them.
    if (llvm::any_of(p.nodes, [](const Node &n) { return n.kind == Node::Reduce; }))
      for (int64_t d = std::min<int64_t>(colLocalSize, 4 * kLanes); colLocalSize > 4 * kLanes; d -= kLanes)
        if (c % d == 0)
          colLocalSize = d;
  }
  // A column count that is not a multiple of the packet: up to 8 columns is one padded packet; beyond, one packet
  // per slice block of the largest power of two dividing it (a padded packet digit cannot be split for the float
  // region and joined back, and DMA layouts do not factor padded non-power-of-two digits).
  if (!colLocalSize && c <= kLanes)
    colLocalSize = c;
  if (!colLocalSize)
    for (int64_t d = kLanes; d >= 2; d /= 2)
      if (c % d == 0) {
        colLocalSize = d;
        break;
      }
  colPad = roundUp(colLocalSize, kLanes);
  // DMA rows to DM are whole 8-byte units: bf16 needs four live columns per unit, in the output and in the reads
  // tiles hold column-innermost (contraction operands keep K innermost).
  auto kInner = [&](unsigned r) {
    return gemm && (r == gemm->stream || r == gemm->trf || llvm::is_contained(gemm->streamExtras, r) ||
                    llvm::is_contained(gemm->trfExtras, r));
  };
  bool bf16 = p.outType.isBF16();
  for (auto [i, r] : llvm::enumerate(p.reads))
    bf16 |= r.type.isBF16() && !kInner(i);
  if (bf16 && colLocalSize % 4)
    return fail(std::to_string(colLocalSize) + " bf16 columns per slice (DMA rows are 8-byte units)");
  int64_t blocks = (c + colLocalSize - 1) / colLocalSize;
  if (blocks > kSlices) {
    // Column tiles of 256 blocks; the column axis pads to whole tiles.
    int64_t tileCols = colLocalSize * kSlices;
    int64_t tiles = (c + tileCols - 1) / tileCols;
    colExtent = tiles * tileCols == p.axes[*col].size ? 0 : tiles * tileCols;
    blocks = kSlices;
    tileD.push_back({*col, tileCols, tiles, colExtent});
  } else if (c % colLocalSize) {
    return fail(std::to_string(c) + " columns do not split into blocks of " + std::to_string(colLocalSize));
  }
  if (blocks > 1)
    colBlock = Digit{*col, colLocalSize, blocks, colExtent};
  colLocalD = Digit{*col, 1, colLocalSize, colExtent};
  int64_t rowSlices = kSlices / int64_t(llvm::PowerOf2Ceil(blocks));

  // Rows: slices first (as many as the rows allow), local digits up to the DM budget, then tiles. A tile the
  // Vector Engine reduces holds the reduction's extent per row.
  int64_t reduced = 1;
  for (auto [i, n] : llvm::enumerate(p.nodes))
    if (n.kind == Node::Reduce && !(gemm && i == gemm->node)) {
      int64_t size = 1;
      for (unsigned a : std::get<1>(flatReduce(p, i)))
        size *= p.axes[a].size;
      reduced = std::max(reduced, size);
    }
  // A reduction whose tile row would not fit DM reduces in chunks (at most rfMax long).
  rfMax = std::max<int64_t>(1, std::min<int64_t>(reduced, kDmBudget / (4 * colPad * 4)));
  SmallVector<Digit> rowTiles;
  int64_t perRow = 4 * colPad * 4 * rfMax + (gemm ? 4 * kTile : 0);
  int64_t tiles = assignRows(rowSlices, perRow, localD, sliceD, rowTiles);
  // Each tile is a pass sequence of its own in the kernel. Rows that split into no slices or local digits that
  // fit DM (a prime count) pad, the outermost such axis first, to the smallest multiple of a power of two that
  // splits; the padded rows are computed and never stored.
  for (int64_t m = 8; tiles > kSlices && m <= kSlices; m *= 2)
    for (unsigned a : rows) {
      if (gemm && llvm::is_contained(gemm->b, a))
        continue;
      if (roundUp(p.axes[a].size, m) == p.axes[a].size)
        continue;
      rowExtent = {{a, roundUp(p.axes[a].size, m)}};
      tiles = assignRows(rowSlices, perRow, localD, sliceD, rowTiles);
      if (tiles <= kSlices)
        break;
      rowExtent.clear();
      tiles = assignRows(rowSlices, perRow, localD, sliceD, rowTiles);
    }
  if (tiles > kSlices)
    return fail(std::to_string(tiles) + " row tiles (rows that split into no slices or local digits that fit DM)");
  llvm::append_range(tileD, rowTiles);
  return success();
}

/// Splits the rows into local digits (up to the DM budget), slice digits (up to `rowSlices`, inner ones counted as
/// powers of two) and tile digits; returns the number of row tiles.
int64_t Plan::assignRows(int64_t rowSlices, int64_t perRow, SmallVector<Digit> &local,
                            SmallVector<Digit> &slice, SmallVector<Digit> &tile) {
  int64_t maxLocal = std::max<int64_t>(1, kDmBudget / perRow);
  int64_t total = 1;
  for (unsigned a : rows)
    total *= extentOf(a);
  struct Cursor {
    unsigned axis;
    int64_t place, left;
  };
  SmallVector<Cursor> cur;
  for (unsigned a : rows)
    cur.push_back({a, 1, extentOf(a)});
  int64_t have = 1;
  local.clear();
  for (int i = cur.size() - 1; i >= 0; --i) {
    Cursor &cu = cur[i];
    if ((total + have - 1) / have <= rowSlices || have >= maxLocal)
      break;
    // The TRF holds one batch index per slice: batch axes stay out of time.
    if (gemm && llvm::is_contained(gemm->b, cu.axis))
      break;
    int64_t need = (total / have + rowSlices - 1) / rowSlices;
    int64_t d = 0;
    if (cu.left <= std::min(need, maxLocal / have))
      d = cu.left;
    else {
      for (int64_t x = need; x <= cu.left && have * x <= maxLocal; ++x)
        if (cu.left % x == 0) {
          d = x;
          break;
        }
      if (!d)
        for (int64_t x = std::min(cu.left, maxLocal / have); x > 1; --x)
          if (cu.left % x == 0) {
            d = x;
            break;
          }
    }
    if (d <= 1)
      break;
    local.push_back({cu.axis, cu.place, d});
    have *= d;
    cu.place *= d;
    cu.left /= d;
    if (cu.left > 1)
      break;
  }
  std::reverse(local.begin(), local.end());
  slice.clear();
  int64_t slices = 1;
  for (int i = cur.size() - 1; i >= 0; --i) {
    Cursor &cu = cur[i];
    if (cu.left == 1)
      continue;
    int64_t d = 0;
    for (int64_t x = std::min(cu.left, rowSlices / slices); x > 1; --x)
      if (cu.left % x == 0) {
        d = x;
        break;
      }
    if (d <= 1)
      break;
    slice.push_back({cu.axis, cu.place, d});
    slices *= int64_t(llvm::PowerOf2Ceil(d));
    cu.place *= d;
    cu.left /= d;
    if (cu.left > 1)
      break;
  }
  std::reverse(slice.begin(), slice.end());
  tile.clear();
  int64_t tiles = 1;
  for (const Cursor &cu : cur)
    if (cu.left > 1) {
      tile.push_back({cu.axis, cu.place, cu.left});
      tiles *= cu.left;
    }
  return tiles;
}

LogicalResult Plan::plan() {
  if (p.out.empty())
    return fail("a scalar output (no output axis to lay out)");
  if (failed(findGemm()) || failed(planLayout()))
    return failure();
  auto supported = [](Type t) { return t.isF32() || t.isBF16(); };
  for (const Read &r : p.reads)
    if (!supported(r.type))
      return fail("a buffer that is neither f32 nor bf16");
  if (!supported(p.outType))
    return fail("the written buffer is neither f32 nor bf16");
  for (auto [j, l] : llvm::enumerate(p.outIndex)) {
    if (p.outShape[j] == 1)
      continue;
    if (l.offset != 0)
      return fail("a store at an offset");
    int64_t expect = p.outShape[j];
    SmallVector<std::pair<unsigned, int64_t>> terms(l.terms);
    llvm::sort(terms, [](auto x, auto y) { return x.second > y.second; });
    for (auto [a, c] : terms) {
      if (expect % p.axes[a].size || expect / p.axes[a].size != c)
        return fail("a store dim that is not a row-major group of output axes");
      expect /= p.axes[a].size;
    }
    if (expect != 1)
      return fail("a store that does not cover its buffer");
  }
  // Tile loops the kernel already has (furiosa-tile tiled it by this plan) must be the plan's tiles.
  for (auto [a, t] : p.loopTiles)
    if (!llvm::any_of(tileD, [&, a = a, t = t](const Digit &d) {
          return d.axis == a && d.place == t && d.place * d.size == extentOf(a);
        }))
      return fail("a tile loop that is not one of the plan's tiles");
  return success();
}


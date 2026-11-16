// Op class "tiled": a kernel analyze() folds into a Program (Program.h) whose reductions are contractions, or
// that has none, lowered over tiles of a 2-D [rows, columns] layout.
//
//  - Layout: one output axis is the column (a contraction's TRF-side axis, else the axis most reads keep
//    contiguous); the others are rows. Columns split into blocks over slices and a local part (eight to a
//    packet); rows split into slice digits, local digits (time) and tile digits (one pass sequence each).
//  - Reads are made dense in HBM first, in that canonical digit layout. HBM-to-HBM DMAs are byte-granular, so
//    any stride, window, padding, permutation or broadcast costs one DMA per tap: a windowed read (a contraction
//    axis added to an output axis, as in convolutions) copies one tap per DMA out of a filled, padded copy of
//    its buffer. Each tile then moves to DM with one aligned DMA.
//  - A contraction runs on the Contraction Engine: the stream operand's rows over slices and time, the TRF
//    operand's columns over lanes (and slices, by block), K in the packet and time. f32 operands are cast to
//    bf16 as they are fetched and accumulate in f32 (the dtype policy).
//  - Everything else is elementwise and runs on the Vector Engine, one op per pass: two tiles pair through an
//    interleaved fetch, column vectors are VRF operands, constants are immediates, and selects become
//    tag-branched immediates.
//  - The result returns through a canonical HBM temporary and one relayout DMA into the written buffer.
//
// NPU rules this follows (verified on npu3): only whole tensors (or whole read views) are relabeled with
// reshape; DM tensors are never tiled (each tile gets fresh ones); padding cells are never read as data, so K
// padding is live cells filled from a memset tensor.

#include "../Mapping/Mapping.h"
#include "Program.h"

#include "furiosa/Codegen/Fopt.h"

#include <cmath>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;
using namespace mlir::furiosa::mapping;
using namespace mlir::furiosa::tile;

namespace {

constexpr int64_t kSlices = 256, kLanes = 8, kTrfLaneBytes = 8 * 1024, kDmBudget = 384 * 1024;

int64_t roundUp(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

/// A digit of an axis: (axis / place) % size.
struct Digit {
  unsigned axis;
  int64_t place, size;
};

/// A contraction: the sum `node` over `k` of the product of two reads. `stream` reads the rows `m`, `trf` the
/// column axis `n` (or none).
struct Gemm {
  unsigned node;
  unsigned stream, trf;
  SmallVector<unsigned> m, n, k;
};

/// The K layout of a contraction's operands: taps (axes peeled into one DMA each) times the inner axis.
struct KLayout {
  SmallVector<unsigned> taps;
  std::optional<unsigned> inner;
  int64_t tapPad = 1, innerPad = 1;
  int64_t size() const { return tapPad * innerPad; }
};

/// A value of the Vector Engine program: a DM tile over [local rows..., local columns], a DM column vector over
/// [local columns], or a constant.
struct Val {
  enum Kind { Tile, Vec, Scalar } kind;
  Value dm;
  double value = 0;
};

class Lowering {
public:
  Lowering(RaisedKernel &kernel, Program &p, bool quiet)
      : kernel(kernel), p(p), quiet(quiet), kb(kernel), ctx(kb.getContext()), b(kb.builder()),
        f32(Float32Type::get(ctx)) {}

  /// Plans the layout; false (with a diagnostic unless quiet) when the class cannot map the kernel.
  LogicalResult plan();
  /// Builds the fopt kernel.
  LogicalResult emit();

private:
  RaisedKernel &kernel;
  Program &p;
  bool quiet;
  KernelBuilder kb;
  MLIRContext *ctx;
  OpBuilder &b;
  Type f32;

  std::optional<Gemm> gemm;
  KLayout kl;
  SmallVector<unsigned> rows;
  std::optional<unsigned> col;              // the column axis
  int64_t colLocalSize = 1, colPad = kLanes; // local columns, and padded to whole packets
  std::optional<Digit> colBlock, colLocalD;
  SmallVector<Digit> tileD, sliceD, localD; // row digits by role, most significant first
  SmallVector<MapAttr> outDims;

  SmallVector<MapAttr> axisMap;
  std::map<unsigned, unsigned> paramOf;
  std::map<double, Value> fillDm;
  MapAttr kf, interleave;
  unsigned fresh = 0;

  LogicalResult fail(const Twine &msg) {
    if (!quiet)
      kernel.func.emitError("tiled: ") << msg;
    return failure();
  }

  LogicalResult findGemm();
  LogicalResult planLayout();
  LogicalResult declare();

  // Mappings.
  MapAttr ax(unsigned a) { return axisMap[a]; }
  MapAttr pairOf(ArrayRef<MapAttr> parts) { return MapAttr::pair(ctx, parts); }
  MapAttr digit(const Digit &d) {
    MapAttr m = d.place > 1 ? ax(d.axis).stride(d.place) : ax(d.axis);
    return d.place * d.size < p.axes[d.axis].size ? m.modulo(d.size) : m;
  }
  /// The slice mapping: row slice digits, then the column block. Every digit inside the outermost is padded to a
  /// power of two (the NPU addresses slices by bit fields; a padded pair with a non-power-of-two inner digit
  /// places data on the wrong slices), the whole padded to the 256 slices.
  MapAttr sliceMap() {
    SmallVector<MapAttr> s;
    for (const Digit &d : sliceD)
      s.push_back(digit(d));
    if (colBlock)
      s.push_back(digit(*colBlock));
    for (unsigned i = 1; i < s.size(); ++i)
      if (int64_t p2 = llvm::PowerOf2Ceil(s[i].size()); p2 != s[i].size())
        s[i] = s[i].padTo(p2);
    MapAttr m = pairOf(s);
    return m.size() < kSlices ? m.padTo(kSlices) : m;
  }
  /// The canonical layout of a read over `axes`: row digits (tile, slice, local), then the column's block and
  /// local digits (local padded to the packet when `padCol`).
  SmallVector<MapAttr> canonical(ArrayRef<unsigned> axes, bool padCol) {
    SmallVector<MapAttr> r;
    auto in = [&](unsigned a) { return llvm::is_contained(axes, a); };
    for (const Digit &d : tileD)
      if (in(d.axis))
        r.push_back(digit(d));
    for (const Digit &d : sliceD)
      if (in(d.axis))
        r.push_back(digit(d));
    for (const Digit &d : localD)
      if (in(d.axis))
        r.push_back(digit(d));
    if (col && in(*col)) {
      if (colBlock)
        r.push_back(digit(*colBlock));
      r.push_back(padCol ? colLocal() : digit(*colLocalD));
    }
    return r;
  }
  MapAttr colLocal() {
    if (!col)
      return kb.one().padTo(colPad);
    MapAttr c = digit(*colLocalD);
    return colLocalSize == colPad ? c : c.padTo(colPad);
  }
  /// The DM element of a tile: local rows, then local columns.
  MapAttr tileElement() {
    SmallVector<MapAttr> e;
    for (const Digit &d : localD)
      e.push_back(digit(d));
    e.push_back(colLocal());
    return pairOf(e);
  }

  // Emission helpers.
  Location loc() { return kb.loc(); }
  Value zeros(double value);
  Value view(Value hbm, bool mut) {
    auto t = cast<HbmTensorType>(hbm.getType());
    return ViewOp::create(b, loc(), HbmViewType::get(ctx, t.getElementType(), t.getChip(), t.getElement(), mut), hbm);
  }
  Value tileView(Value v, MapAttr index, int64_t len, int64_t start, MapAttr element) {
    auto t = cast<HbmViewType>(v.getType());
    return TileOp::create(b, loc(), HbmViewType::get(ctx, t.getElementType(), t.getChip(), element, t.getIsMut()), v,
                          index, b.getI64IntegerAttr(len), b.getI64IntegerAttr(start));
  }
  Value reshapeView(Value v, MapAttr element) {
    auto t = cast<HbmViewType>(v.getType());
    if (t.getElement() == element)
      return v;
    return ViewReshapeOp::create(b, loc(), HbmViewType::get(ctx, t.getElementType(), t.getChip(), element, false), v);
  }
  Value allocHbm(MapAttr element) { return AllocHbmOp::create(b, loc(), kb.hbmType(f32, element)); }
  Value reshapeHbm(Value t, MapAttr element) {
    if (cast<HbmTensorType>(t.getType()).getElement() == element)
      return t;
    return ReshapeHbmOp::create(b, loc(), kb.hbmType(f32, element), t);
  }
  void copy(Value src, Value dst) { HbmToHbmViewOp::create(b, loc(), src, dst); }
  void fill(Value hbm, double value) { DmToHbmViewOp::create(b, loc(), zeros(value), view(hbm, true)); }
  /// Tiles view `v` (of mapping `parts`) to one position of each digit in `at`.
  Value tileDigits(Value v, SmallVector<MapAttr> parts, ArrayRef<std::pair<MapAttr, int64_t>> at, bool mut) {
    for (auto [d, index] : at) {
      auto it = llvm::find(parts, d);
      assert(it != parts.end() && "a tiled digit of the view");
      *it = kb.one().padTo(d.size(), mut ? PadKind::Bottom : PadKind::Top);
      v = tileView(v, d, 1, index, pairOf(parts));
    }
    return v;
  }
  SmallVector<std::pair<MapAttr, int64_t>> tileAt(ArrayRef<unsigned> axes, ArrayRef<int64_t> tileIndex) {
    SmallVector<std::pair<MapAttr, int64_t>> at;
    for (auto [i, d] : llvm::enumerate(tileD))
      if (llvm::is_contained(axes, d.axis))
        at.push_back({digit(d), tileIndex[i]});
    return at;
  }

  // Operands.
  FailureOr<Value> materialize(const Read &r, ArrayRef<unsigned> rowAxes, const KLayout *k);
  Value toDm(Value hbm, ArrayRef<unsigned> axes, ArrayRef<int64_t> tileIndex, MapAttr element, bool withK = false) {
    auto parts = canonical(axes, false);
    if (withK)
      parts.push_back(kf);
    Value v = tileDigits(view(hbm, false), parts, tileAt(axes, tileIndex), false);
    return ViewToDmOp::create(b, loc(), kb.dmType(f32, sliceMap(), element), v);
  }

  // Vector Engine.
  std::map<unsigned, Val> vals;
  std::map<unsigned, Value> hbmOfRead;
  std::pair<SmallVector<MapAttr>, MapAttr> timePacket(MapAttr element);
  Value vrfOf(Value vecDm);
  struct AluSpec {
    std::string method, op, mode;
    std::optional<double> imm;
    Value vrf;
    ArrayAttr branches;
    bool fp = false; // runs in the 4-way float region
  };
  Value pass(Value in, const AluSpec &spec, ArrayAttr tagCmps = {});
  Value zip(Value lhs, Value rhs, const AluSpec &spec);
  FailureOr<Val> evaluate(unsigned node, ArrayRef<int64_t> tileIndex);
  FailureOr<Val> binary(const std::string &op, Val l, Val r);
  FailureOr<Val> select(const Node &n, ArrayRef<int64_t> tileIndex);
  FailureOr<Val> leaf(unsigned read, ArrayRef<int64_t> tileIndex);

  // Contraction.
  Value trf;
  FailureOr<Value> contract(ArrayRef<int64_t> tileIndex);
};

//===----------------------------------------------------------------------===//
// Planning
//===----------------------------------------------------------------------===//

LogicalResult Lowering::findGemm() {
  SmallVector<unsigned> reduces;
  for (auto [i, n] : llvm::enumerate(p.nodes))
    if (n.kind == Node::Reduce)
      reduces.push_back(i);
  if (reduces.empty())
    return success();
  if (reduces.size() > 1)
    return fail("more than one reduction");
  const Node &red = p.nodes[reduces[0]];
  if (red.op != "add")
    return fail("a " + red.op + " reduction (only sums of products are mapped yet)");
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
  if (!collect(red.args[0]) || factors.size() != 2)
    return fail("a reduction that is not a product of two reads");
  std::set<unsigned> k(red.axes.begin(), red.axes.end());
  auto outAxes = [&](unsigned read) {
    SmallVector<unsigned> o;
    for (unsigned a : p.reads[read].axes())
      if (!k.count(a))
        o.push_back(a);
    return o;
  };
  SmallVector<unsigned> o[2] = {outAxes(factors[0]), outAxes(factors[1])};
  for (unsigned a : o[0])
    if (llvm::is_contained(o[1], a))
      return fail("a contraction with a batch axis (both operands index it)");
  for (unsigned f : factors)
    for (unsigned a : red.axes)
      if (!llvm::is_contained(p.reads[f].axes(), a))
        return fail("a contraction axis one operand does not read");
  auto rowsOf = [&](ArrayRef<unsigned> axes) {
    int64_t n = 1;
    for (unsigned a : axes)
      n *= p.axes[a].size;
    return n;
  };
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
  if (o[1 - s].size() != 1)
    return fail("a contraction without a column axis");
  Gemm g;
  g.node = reduces[0];
  g.stream = factors[s];
  g.trf = factors[1 - s];
  g.m = o[s];
  g.n = o[1 - s];
  g.k = SmallVector<unsigned>(red.axes);
  llvm::sort(g.m, [&](unsigned x, unsigned y) { return llvm::find(p.out, x) < llvm::find(p.out, y); });
  gemm = g;
  return success();
}

LogicalResult Lowering::planLayout() {
  if (gemm) {
    rows = gemm->m;
    col = gemm->n[0];
  } else {
    // The column: the output axis contiguous in the most reads, else the innermost output axis.
    std::map<unsigned, int> votes;
    for (const Read &r : p.reads)
      if (!r.index.empty() && r.index.back().terms.size() == 1 && r.index.back().terms[0].second == 1)
        ++votes[r.index.back().terms[0].first];
    unsigned best = p.out.back();
    for (auto [a, v] : votes)
      if (llvm::is_contained(p.out, a) && v > votes[best])
        best = a;
    col = best;
    for (unsigned a : p.out)
      if (a != best)
        rows.push_back(a);
  }
  int64_t c = p.axes[*col].size;
  colLocalSize = 0;
  if (c % 2)
    return fail("an odd column count " + std::to_string(c) + " (DMA rows are 8-byte units)");
  if (gemm) {
    const Read &s = p.reads[gemm->stream];
    for (unsigned a : gemm->k) {
      const Lin &last = s.index.back();
      if (!kl.inner && last.terms.size() == 1 && last.terms[0].first == a && last.terms[0].second == 1)
        kl.inner = a;
      else
        kl.taps.push_back(a);
    }
    int64_t t = 1;
    for (unsigned a : kl.taps)
      t *= p.axes[a].size;
    int64_t in = kl.inner ? p.axes[*kl.inner].size : 1;
    while (kl.innerPad < in && kl.innerPad < 32)
      kl.innerPad *= 2;
    if (in > 32)
      kl.innerPad = roundUp(in, 32);
    kl.tapPad = roundUp(t, std::max<int64_t>(1, 32 / kl.innerPad));
    // Columns per slice: the most whole lane groups the TRF holds that divide the columns.
    int64_t maxGroups = kTrfLaneBytes / (2 * kl.size());
    if (maxGroups < 1)
      return fail("K = " + std::to_string(kl.size()) + " exceeds a TRF lane (K tiling is not implemented yet)");
    colLocalSize = 0;
    if (c % kLanes == 0)
      for (int64_t g = std::min(maxGroups, c / kLanes); g >= 1; --g)
        if ((c / kLanes) % g == 0 && c / (g * kLanes) <= kSlices) {
          colLocalSize = g * kLanes;
          break;
        }
  } else if (c % kLanes == 0) {
    colLocalSize = c;
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
  if (c / colLocalSize > kSlices)
    return fail(std::to_string(c) + " columns need more than 256 slices (column tiling is not implemented yet)");
  colPad = roundUp(colLocalSize, kLanes);
  int64_t blocks = c / colLocalSize;
  if (blocks > 1)
    colBlock = Digit{*col, colLocalSize, blocks};
  colLocalD = Digit{*col, 1, colLocalSize};
  int64_t rowSlices = kSlices / int64_t(llvm::PowerOf2Ceil(blocks));

  // Rows: slices first (as many as the rows allow), local digits up to the DM budget, then tiles.
  int64_t perRow = 4 * colPad * 4 + (gemm ? 4 * kl.size() : 0);
  int64_t maxLocal = std::max<int64_t>(1, kDmBudget / perRow);
  int64_t total = 1;
  for (unsigned a : rows)
    total *= p.axes[a].size;
  struct Cursor {
    unsigned axis;
    int64_t place, left;
  };
  SmallVector<Cursor> cur;
  for (unsigned a : rows)
    cur.push_back({a, 1, p.axes[a].size});
  int64_t local = 1;
  SmallVector<Digit> got;
  for (int i = cur.size() - 1; i >= 0; --i) {
    Cursor &cu = cur[i];
    if ((total + local - 1) / local <= rowSlices || local >= maxLocal)
      break;
    int64_t need = (total / local + rowSlices - 1) / rowSlices;
    int64_t d = 0;
    if (cu.left <= std::min(need, maxLocal / local))
      d = cu.left;
    else {
      for (int64_t x = need; x <= cu.left && local * x <= maxLocal; ++x)
        if (cu.left % x == 0) {
          d = x;
          break;
        }
      if (!d)
        for (int64_t x = std::min(cu.left, maxLocal / local); x > 1; --x)
          if (cu.left % x == 0) {
            d = x;
            break;
          }
    }
    if (d <= 1)
      break;
    got.push_back({cu.axis, cu.place, d});
    local *= d;
    cu.place *= d;
    cu.left /= d;
    if (cu.left > 1)
      break;
  }
  std::reverse(got.begin(), got.end());
  localD = got;
  got.clear();
  // Slice digits, innermost first; each one taken counts as a power of two (it may end up inside another).
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
    got.push_back({cu.axis, cu.place, d});
    slices *= int64_t(llvm::PowerOf2Ceil(d));
    cu.place *= d;
    cu.left /= d;
    if (cu.left > 1)
      break;
  }
  std::reverse(got.begin(), got.end());
  sliceD = got;
  for (const Cursor &cu : cur)
    if (cu.left > 1)
      tileD.push_back({cu.axis, cu.place, cu.left});
  return success();
}

LogicalResult Lowering::plan() {
  if (failed(findGemm()) || failed(planLayout()))
    return failure();
  for (const Read &r : p.reads)
    if (!r.type.isF32())
      return fail("a buffer that is not f32 (only f32 graphs are mapped yet)");
  if (!p.outType.isF32())
    return fail("the written buffer is not f32");
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
  return success();
}

LogicalResult Lowering::declare() {
  for (auto [i, a] : llvm::enumerate(p.axes))
    axisMap.push_back(kb.axis("A" + std::to_string(i), a.size));
  if (gemm)
    kf = kb.axis("KF", kl.size());
  interleave = kb.axis("I", 2);
  std::set<unsigned> readArgs;
  for (const Read &r : p.reads)
    readArgs.insert(r.arg);
  for (unsigned arg : readArgs) {
    const Read *r = nullptr;
    for (const Read &x : p.reads)
      if (x.arg == arg)
        r = &x;
    SmallVector<MapAttr> dims;
    for (auto [j, s] : llvm::enumerate(r->shape))
      if (s > 1)
        dims.push_back(kb.axis("B" + std::to_string(arg) + "_" + std::to_string(j), s));
    paramOf[arg] = kb.input(arg, r->type, pairOf(dims));
  }
  for (auto [j, l] : llvm::enumerate(p.outIndex)) {
    if (p.outShape[j] == 1)
      continue;
    SmallVector<std::pair<unsigned, int64_t>> terms(l.terms);
    llvm::sort(terms, [](auto x, auto y) { return x.second > y.second; });
    for (auto [a, c] : terms)
      outDims.push_back(ax(a));
  }
  kb.output(p.outArg, p.outType, pairOf(outDims));
  return success();
}

//===----------------------------------------------------------------------===//
// Operands
//===----------------------------------------------------------------------===//

Value Lowering::zeros(double value) {
  auto it = fillDm.find(value);
  if (it != fillDm.end())
    return it->second;
  Value dm = AllocDmOp::create(b, loc(), kb.dmType(f32, kb.one().padTo(kSlices), kb.one().padTo(2)));
  MemsetOp::create(b, loc(), dm, b.getF32FloatAttr(value));
  return fillDm[value] = dm;
}

/// Read `r` as a dense HBM tensor in the canonical layout over `rowAxes`, then KF (a contraction operand, `k`)
/// or nothing more (the column, when the read has it, is one of `rowAxes`). Each tap of a windowed read is one
/// DMA from a source view that names the buffer's digits by fresh axes (a reshape of the whole read view); the
/// destination names its rows by the same expressions, and a whole-tensor reshape renames it canonically.
FailureOr<Value> Lowering::materialize(const Read &r, ArrayRef<unsigned> rowAxes, const KLayout *k) {
  unsigned nd = r.shape.size();
  SmallVector<unsigned> taps;
  if (k)
    taps = k->taps;
  auto isTap = [&](unsigned a) { return llvm::is_contained(taps, a); };
  std::optional<unsigned> innerAxis = k ? k->inner : std::nullopt;
  int64_t innerSize = innerAxis ? p.axes[*innerAxis].size : 1;

  // Per dim: the low padding, the extent (every tap's window inside, divisible by its coefficients) and the
  // terms that are not taps, by decreasing coefficient.
  SmallVector<int64_t> low(nd, 0), extent(nd);
  SmallVector<SmallVector<std::pair<unsigned, int64_t>>> rem(nd);
  for (unsigned j = 0; j < nd; ++j) {
    const Lin &l = r.index[j];
    int64_t lo = l.offset, hi = l.offset;
    for (auto [a, c] : l.terms) {
      if (c < 0)
        return fail("a read with a negative stride");
      hi += c * (p.axes[a].size - 1);
      if (!isTap(a))
        rem[j].push_back({a, c});
    }
    llvm::sort(rem[j], [](auto x, auto y) { return x.second > y.second; });
    low[j] = lo < 0 ? -lo : 0;
    extent[j] = std::max(hi + 1, r.shape[j]) + low[j];
    if (!rem[j].empty())
      extent[j] = roundUp(extent[j], rem[j].front().second);
  }
  bool copyNeeded = r.fill.has_value();
  for (unsigned j = 0; j < nd; ++j)
    copyNeeded |= low[j] || extent[j] != r.shape[j];

  Value param = kb.param(paramOf[r.arg]);
  Value source = param;
  std::string tag = "S" + std::to_string(fresh++) + "_";
  if (copyNeeded) {
    // A copy with the padded extents, filled with the padding value, the buffer in its interior.
    SmallVector<MapAttr> padded, interior;
    for (unsigned j = 0; j < nd; ++j) {
      if (extent[j] == 1)
        continue;
      MapAttr axis = kb.axis(tag + "P" + std::to_string(j), extent[j]);
      padded.push_back(axis);
      interior.push_back(r.shape[j] == extent[j] ? axis : axis.resize(r.shape[j]));
    }
    Value xp = allocHbm(pairOf(padded));
    fill(xp, r.fill.value_or(0.0));
    Value d = view(xp, true);
    SmallVector<MapAttr> cur(padded);
    SmallVector<MapAttr> srcDims;
    for (unsigned j = 0, i = 0; j < nd; ++j) {
      if (extent[j] == 1)
        continue;
      if (r.shape[j] != extent[j]) {
        cur[i] = interior[i].padTo(extent[j], PadKind::Bottom);
        d = tileView(d, padded[i], r.shape[j], low[j], pairOf(cur));
      }
      if (r.shape[j] > 1)
        srcDims.push_back(interior[i]);
      ++i;
    }
    copy(reshapeView(view(param, false), pairOf(srcDims)), d);
    source = xp;
  }

  // The digits of each dim, most significant first: one per remaining term, then the residual below the
  // smallest coefficient.
  struct DigitInfo {
    MapAttr expr;
    int64_t radix, coeff;
    std::optional<unsigned> axis;
  };
  SmallVector<SmallVector<DigitInfo>> dig(nd);
  MapAttr cq = (k && k->innerPad > 1) ? kb.axis(tag + "CQ", k->innerPad) : MapAttr();
  for (unsigned j = 0; j < nd; ++j) {
    if (extent[j] == 1)
      continue;
    int64_t above = extent[j];
    for (auto [a, c] : rem[j]) {
      if (above % c)
        return fail("a read whose strides do not nest in its buffer dim");
      int64_t radix = above / c;
      MapAttr e;
      if (innerAxis && a == *innerAxis && cq) {
        if (radix > k->innerPad)
          return fail("an inner contraction digit wider than its padded extent");
        e = radix == k->innerPad ? cq : cq.resize(radix);
      } else {
        e = kb.axis(tag + "D" + std::to_string(j) + "_" + std::to_string(dig[j].size()), radix);
      }
      dig[j].push_back({e, radix, c, a});
      above = c;
    }
    if (above > 1)
      dig[j].push_back({kb.axis(tag + "D" + std::to_string(j) + "_" + std::to_string(dig[j].size()), above), above, 1,
                        std::nullopt});
  }
  SmallVector<MapAttr> srcParts;
  for (auto &ds : dig)
    for (auto &g : ds)
      srcParts.push_back(g.expr);

  // The destination: rows named as the source digits they come from (resized to their window); an axis the
  // read does not use is a fresh one (the DMA broadcasts it). Then the taps and the inner axis.
  std::map<unsigned, MapAttr> named;
  for (auto &ds : dig)
    for (auto &g : ds)
      if (g.axis && !isTap(*g.axis) && !(innerAxis && *g.axis == *innerAxis)) {
        int64_t size = p.axes[*g.axis].size;
        named[*g.axis] = g.radix == size ? g.expr : g.expr.resize(size);
      }
  SmallVector<MapAttr> dstParts;
  for (unsigned a : rowAxes) {
    auto it = named.find(a);
    dstParts.push_back(it != named.end() ? it->second : kb.axis(tag + "R" + std::to_string(a), p.axes[a].size));
  }
  MapAttr tq;
  if (k) {
    if (k->tapPad > 1) {
      tq = kb.axis(tag + "TQ", k->tapPad);
      dstParts.push_back(tq);
    }
    if (cq)
      dstParts.push_back(cq);
  }
  Value dst = allocHbm(pairOf(dstParts));
  int64_t numTaps = 1;
  for (unsigned a : taps)
    numTaps *= p.axes[a].size;
  if (k && (k->tapPad > numTaps || k->innerPad > innerSize))
    fill(dst, 0.0);

  for (int64_t t = 0; t < numTaps; ++t) {
    std::map<unsigned, int64_t> at;
    int64_t rest = t;
    for (int i = taps.size() - 1; i >= 0; --i) {
      at[taps[i]] = rest % p.axes[taps[i]].size;
      rest /= p.axes[taps[i]].size;
    }
    Value s = reshapeView(view(source, false), pairOf(srcParts));
    SmallVector<MapAttr> cur(srcParts);
    unsigned pos = 0;
    for (unsigned j = 0; j < nd; ++j) {
      if (extent[j] == 1)
        continue;
      int64_t off = r.index[j].offset + low[j];
      for (auto [a, c] : r.index[j].terms)
        if (isTap(a))
          off += c * at[a];
      for (auto [i, g] : llvm::enumerate(dig[j])) {
        int64_t start = off / g.coeff;
        if (i > 0)
          start %= g.radix;
        int64_t len = !g.axis ? 1 : (innerAxis && *g.axis == *innerAxis) ? innerSize : p.axes[*g.axis].size;
        if (start + len > g.radix)
          return fail("a window that carries across digits");
        if (!(start == 0 && len == g.radix)) {
          cur[pos] = len == 1 ? kb.one().padTo(g.radix) : g.expr.resize(len).padTo(g.radix);
          s = tileView(s, g.expr, len, start, pairOf(cur));
        }
        ++pos;
      }
    }
    Value d = view(dst, true);
    if (k) {
      SmallVector<MapAttr> dc(dstParts);
      if (tq) {
        int64_t index = 0;
        for (unsigned a : taps)
          index = index * p.axes[a].size + at[a];
        *llvm::find(dc, tq) = kb.one().padTo(k->tapPad, PadKind::Bottom);
        d = tileView(d, tq, 1, index, pairOf(dc));
      }
      if (cq && innerSize < k->innerPad) {
        *llvm::find(dc, cq) = cq.resize(innerSize).padTo(k->innerPad, PadKind::Bottom);
        d = tileView(d, cq, innerSize, 0, pairOf(dc));
      }
    }
    copy(s, d);
  }

  SmallVector<MapAttr> canon = canonical(rowAxes, false);
  if (k)
    canon.push_back(kf);
  return reshapeHbm(dst, pairOf(canon));
}

//===----------------------------------------------------------------------===//
// Vector Engine
//===----------------------------------------------------------------------===//

/// The stream time and packet of a DM tile or vector: its last part is the local column, eight to a packet.
std::pair<SmallVector<MapAttr>, MapAttr> Lowering::timePacket(MapAttr element) {
  SmallVector<MapAttr> parts;
  if (element.getKind() == MapKind::Pair)
    for (Attribute a : element.getOperands())
      parts.push_back(cast<MapAttr>(a));
  else
    parts.push_back(element);
  MapAttr c = parts.pop_back_val();
  if (c.size() > kLanes) {
    parts.push_back(c.stride(kLanes));
    c = c.modulo(kLanes);
  }
  return {parts, c};
}

Value Lowering::vrfOf(Value vecDm) {
  auto t = cast<DmTensorType>(vecDm.getType());
  auto [time, packet] = timePacket(t.getElement());
  auto st = [&](MapAttr tm, MapAttr pk) { return kb.streamType(f32, t.getSlice(), tm, pk); };
  Value s = BeginOp::create(b, loc(), st(kb.one(), t.getElement()), vecDm, ContextKind::Sub);
  s = FetchOp::create(b, loc(), st(pairOf(time), packet), s);
  s = CollectOp::create(b, loc(), st(pairOf(time), packet), s);
  return ToVrfOp::create(b, loc(), VrfTensorType::get(ctx, f32, kb.chip(), kb.cluster(), t.getSlice(), t.getElement()),
                         s);
}

/// One Vector Engine pass: fetch a DM tile or vector, one ALU op, commit.
Value Lowering::pass(Value in, const AluSpec &spec, ArrayAttr tagCmps) {
  auto t = cast<DmTensorType>(in.getType());
  auto [timeParts, packet] = timePacket(t.getElement());
  MapAttr time = pairOf(timeParts);
  auto st = [&](MapAttr tm, MapAttr pk) { return kb.streamType(f32, t.getSlice(), tm, pk); };
  Value s = BeginOp::create(b, loc(), st(kb.one(), t.getElement()), in, ContextKind::Main);
  s = FetchOp::create(b, loc(), st(time, packet), s);
  s = CollectOp::create(b, loc(), st(time, packet), s);
  s = VectorInitOp::create(b, loc(), st(time, packet), s);
  if (tagCmps)
    s = VectorTagCmpOp::create(b, loc(), st(time, packet), s, tagCmps);
  else
    s = VectorIntraSliceTagOp::create(b, loc(), st(time, packet), s);
  // The float region is four lanes wide: split the packet's halves into time.
  SmallVector<MapAttr> t4(timeParts);
  t4.push_back(packet.stride(4));
  MapAttr p4 = packet.modulo(4);
  if (spec.fp)
    s = VectorNarrowSplitOp::create(b, loc(), st(pairOf(t4), p4), s);
  StreamType opType = spec.fp ? st(pairOf(t4), p4) : st(time, packet);
  s = VectorAluOp::create(b, loc(), opType, s, b.getStringAttr(spec.method), b.getStringAttr(spec.op),
                          spec.mode.empty() ? StringAttr() : b.getStringAttr(spec.mode),
                          spec.imm ? b.getF32FloatAttr(*spec.imm) : FloatAttr(), spec.vrf, UnitAttr(), spec.branches);
  // Widening joins the split halves back into the packet (whose digits the column layout keeps unpadded, so
  // both the CPU check and the NPU lowering see the same packet).
  MapAttr wide = packet;
  if (spec.fp)
    s = VectorWidenConcatOp::create(b, loc(), st(time, wide), s);
  s = VectorFinalOp::create(b, loc(), st(time, wide), s);
  s = CommitTrimOp::create(b, loc(), st(time, wide), s);
  return CommitOp::create(b, loc(), t, s);
}

/// One pass pairing two DM tensors of one type through an interleaved fetch and a zip op.
Value Lowering::zip(Value lhs, Value rhs, const AluSpec &spec) {
  auto t = cast<DmTensorType>(lhs.getType());
  auto [timeParts, packet] = timePacket(t.getElement());
  MapAttr time = pairOf(timeParts);
  auto st = [&](MapAttr tm, MapAttr pk) { return kb.streamType(f32, t.getSlice(), tm, pk); };
  MapAttr ti = pairOf({time, interleave});
  Value s = BeginInterleavedOp::create(b, loc(), st(interleave, t.getElement()), lhs, rhs, ContextKind::Main,
                                       interleave);
  s = FetchOp::create(b, loc(), st(ti, packet), s);
  s = CollectOp::create(b, loc(), st(ti, packet), s);
  s = VectorInitOp::create(b, loc(), st(ti, packet), s);
  s = VectorIntraSliceUnzipOp::create(b, loc(), st(time, packet), s, interleave);
  SmallVector<MapAttr> t4(timeParts);
  t4.push_back(packet.stride(4));
  MapAttr p4 = packet.modulo(4);
  if (spec.fp)
    s = VectorNarrowSplitOp::create(b, loc(), st(pairOf(t4), p4), s);
  s = VectorAluOp::create(b, loc(), spec.fp ? st(pairOf(t4), p4) : st(time, packet), s, b.getStringAttr(spec.method),
                          b.getStringAttr(spec.op), spec.mode.empty() ? StringAttr() : b.getStringAttr(spec.mode),
                          FloatAttr(), Value(), UnitAttr(), ArrayAttr());
  // Widening joins the split halves back into the packet (whose digits the column layout keeps unpadded, so
  // both the CPU check and the NPU lowering see the same packet).
  MapAttr wide = packet;
  if (spec.fp)
    s = VectorWidenConcatOp::create(b, loc(), st(time, wide), s);
  s = VectorFinalOp::create(b, loc(), st(time, wide), s);
  s = CommitTrimOp::create(b, loc(), st(time, wide), s);
  return CommitOp::create(b, loc(), t, s);
}

FailureOr<Val> Lowering::leaf(unsigned read, ArrayRef<int64_t> tileIndex) {
  const Read &r = p.reads[read];
  SmallVector<unsigned> axes = r.axes();
  bool colOnly = llvm::all_of(axes, [&](unsigned a) { return a == *col; });
  SmallVector<unsigned> layout = colOnly ? SmallVector<unsigned>{*col} : rows;
  if (!colOnly)
    layout.push_back(*col);
  auto it = hbmOfRead.find(read);
  if (it == hbmOfRead.end()) {
    auto h = materialize(r, layout, nullptr);
    if (failed(h))
      return failure();
    it = hbmOfRead.insert({read, *h}).first;
  }
  if (colOnly) // with no rows a column vector is the whole tile
    return Val{rows.empty() ? Val::Tile : Val::Vec, toDm(it->second, layout, tileIndex, colLocal())};
  return Val{Val::Tile, toDm(it->second, layout, tileIndex, tileElement())};
}

FailureOr<Val> Lowering::binary(const std::string &op, Val l, Val r) {
  if (l.kind == Val::Scalar && r.kind == Val::Scalar) {
    double a = l.value, c = r.value;
    double v = op == "add" ? a + c : op == "sub" ? a - c : op == "mul" ? a * c : op == "div" ? a / c
               : op == "max" ? std::max(a, c) : std::min(a, c);
    return Val{Val::Scalar, Value(), v};
  }
  // x + 0 and x * 1 are x (up to the sign of a zero sum).
  if (op == "add" && r.kind == Val::Scalar && r.value == 0)
    return l;
  if (op == "add" && l.kind == Val::Scalar && l.value == 0)
    return r;
  if (op == "mul" && r.kind == Val::Scalar && r.value == 1)
    return l;
  if (op == "mul" && l.kind == Val::Scalar && l.value == 1)
    return r;
  bool commutative = op == "add" || op == "mul" || op == "max" || op == "min";
  // The stream is a tile over a vector or a constant, a vector over a constant.
  bool swapped = false;
  if ((r.kind == Val::Tile && l.kind != Val::Tile) || (r.kind == Val::Vec && l.kind == Val::Scalar)) {
    std::swap(l, r);
    swapped = true;
  }
  bool reversed = swapped && !commutative;
  AluSpec spec;
  if (op == "add" || op == "sub" || op == "mul" || op == "div") {
    std::string fpOp = op == "add" ? "FpBinaryOp::AddF" : op == "sub" ? "FpBinaryOp::SubF"
                       : op == "mul" ? "FpBinaryOp::MulF(FpMulAlu::Mul0)" : "FpBinaryOp::DivF";
    spec.fp = true;
    if (r.kind == l.kind) {
      spec.method = "vector_fp_zip";
      spec.op = fpOp;
      if (reversed)
        spec.mode = "Mode10";
      return Val{l.kind, zip(l.dm, r.dm, spec)};
    }
    if (op == "div") {
      spec.method = "vector_fp_div";
    } else {
      spec.method = "vector_fp_binary";
      spec.op = fpOp;
    }
    if (reversed)
      spec.mode = "Mode10";
  } else if (op == "max" || op == "min") {
    spec.op = op == "max" ? "ClipBinaryOpF32::Max" : "ClipBinaryOpF32::Min";
    if (r.kind == l.kind) {
      spec.method = "vector_clip_zip";
      return Val{l.kind, zip(l.dm, r.dm, spec)};
    }
    spec.method = "vector_clip";
  } else {
    return fail("no Vector Engine op for '" + op + "'");
  }
  if (r.kind == Val::Scalar)
    spec.imm = r.value;
  else
    spec.vrf = vrfOf(r.dm);
  return Val{l.kind, pass(l.dm, spec)};
}

/// The tag comparison of `x pred k` (or `k pred x`): the Cmp case and the guard bits under which it holds.
static std::optional<std::pair<std::string, std::string>> tagOf(StringRef pred, double k, bool constFirst) {
  StringRef q = pred.size() == 3 ? pred.drop_front() : pred; // olt -> lt
  if (constFirst)
    q = q == "lt" ? "gt" : q == "gt" ? "lt" : q == "le" ? "ge" : q == "ge" ? "le" : q;
  std::string base;
  if (q == "lt" || q == "ge")
    base = "Less";
  else if (q == "gt" || q == "le")
    base = "Greater";
  else if (q == "eq" || q == "ne")
    base = "Equal";
  else
    return std::nullopt;
  std::string bits = (q == "lt" || q == "gt" || q == "eq") ? "1---" : "0---";
  llvm::SmallString<16> lit;
  llvm::APFloat(float(k)).toString(lit, 0, 0);
  std::string s(lit);
  if (s.find_first_of(".eE") == std::string::npos)
    s += ".0";
  return std::make_pair(base + "(" + s + "f32)", bits);
}

FailureOr<Val> Lowering::select(const Node &n, ArrayRef<int64_t> tileIndex) {
  const Node &c = p.nodes[n.args[0]];
  if (c.kind != Node::Op || !StringRef(c.op).starts_with("cmp."))
    return fail("a select whose condition is not a comparison");
  bool constFirst = p.nodes[c.args[0]].kind == Node::Const;
  unsigned xs = c.args[constFirst ? 1 : 0];
  const Node &kn = p.nodes[c.args[constFirst ? 0 : 1]];
  if (kn.kind != Node::Const)
    return fail("a comparison with a non-constant bound");
  auto tag = tagOf(StringRef(c.op).drop_front(4), kn.value, constFirst);
  if (!tag)
    return fail("an unsupported comparison '" + c.op + "'");
  auto x = evaluate(xs, tileIndex);
  if (failed(x))
    return failure();
  if (x->kind == Val::Scalar)
    return fail("a select over a constant condition");
  ArrayAttr cmps = b.getStrArrayAttr({tag->first, "False", "False", "False"});
  // The chosen constant per element: max(k, k) with tag-branched immediates.
  auto constTile = [&](double whenTrue, double otherwise) {
    AluSpec spec;
    spec.method = "vector_clip";
    spec.op = "ClipBinaryOpF32::Max";
    spec.mode = "Mode11";
    spec.branches = b.getArrayAttr({b.getArrayAttr({b.getStringAttr(tag->second), b.getF32FloatAttr(whenTrue)}),
                                    b.getArrayAttr({b.getStringAttr("all"), b.getF32FloatAttr(otherwise)})});
    return Val{x->kind, pass(x->dm, spec, cmps)};
  };
  const Node &ta = p.nodes[n.args[1]], &fb = p.nodes[n.args[2]];
  if (ta.kind == Node::Const && fb.kind == Node::Const)
    return constTile(ta.value, fb.value);
  // select(c, -y, y) = y * select(c, -1, 1); select(c, y, -y) likewise.
  if (ta.kind == Node::Op && ta.op == "neg" && ta.args[0] == n.args[2]) {
    auto y = evaluate(n.args[2], tileIndex);
    if (failed(y))
      return failure();
    return binary("mul", *y, constTile(-1, 1));
  }
  if (fb.kind == Node::Op && fb.op == "neg" && fb.args[0] == n.args[1]) {
    auto y = evaluate(n.args[1], tileIndex);
    if (failed(y))
      return failure();
    return binary("mul", *y, constTile(1, -1));
  }
  // Otherwise a * s + b * (1 - s) with s = select(c, 1, 0): exact for finite branches.
  auto a = evaluate(n.args[1], tileIndex);
  auto bb = evaluate(n.args[2], tileIndex);
  if (failed(a) || failed(bb))
    return failure();
  auto as = binary("mul", *a, constTile(1, 0));
  if (failed(as))
    return failure();
  auto bs = binary("mul", *bb, constTile(0, 1));
  if (failed(bs))
    return failure();
  return binary("add", *as, *bs);
}

FailureOr<Val> Lowering::evaluate(unsigned node, ArrayRef<int64_t> tileIndex) {
  if (auto it = vals.find(node); it != vals.end())
    return it->second;
  const Node &n = p.nodes[node];
  FailureOr<Val> v = failure();
  switch (n.kind) {
  case Node::Const:
    v = Val{Val::Scalar, Value(), n.value};
    break;
  case Node::Leaf:
    v = leaf(n.read, tileIndex);
    break;
  case Node::Reduce: {
    if (!gemm || node != gemm->node)
      return fail("an unmapped reduction");
    auto acc = contract(tileIndex);
    if (failed(acc))
      return failure();
    v = Val{Val::Tile, *acc};
    break;
  }
  case Node::Op: {
    if (n.op == "select") {
      v = select(n, tileIndex);
      break;
    }
    if (StringRef(n.op).starts_with("cmp."))
      return fail("a comparison outside a select");
    SmallVector<Val> args;
    for (unsigned a : n.args) {
      auto x = evaluate(a, tileIndex);
      if (failed(x))
        return failure();
      args.push_back(*x);
    }
    if (n.args.size() == 2) {
      v = binary(n.op, args[0], args[1]);
      break;
    }
    Val x = args[0];
    if (x.kind == Val::Scalar) {
      double y = x.value;
      double r = n.op == "neg" ? -y : n.op == "abs" ? std::fabs(y) : n.op == "exp" ? std::exp(y)
                 : n.op == "sqrt" ? std::sqrt(y) : n.op == "tanh" ? std::tanh(y) : n.op == "log" ? std::log(y)
                 : n.op == "rsqrt" ? 1 / std::sqrt(y) : NAN;
      if (std::isnan(r) && !std::isnan(y))
        return fail("no constant folding for '" + n.op + "'");
      v = Val{Val::Scalar, Value(), r};
      break;
    }
    if (n.op == "neg") {
      v = binary("mul", x, Val{Val::Scalar, Value(), -1.0});
      break;
    }
    if (n.op == "abs") {
      // |x| = x * (x < 0 ? -1 : 1), the sign from a tag comparison.
      AluSpec spec;
      spec.method = "vector_fp_binary";
      spec.op = "FpBinaryOp::MulF(FpMulAlu::Mul0)";
      spec.fp = true;
      spec.branches = b.getArrayAttr({b.getArrayAttr({b.getStringAttr("1---"), b.getF32FloatAttr(-1)}),
                                      b.getArrayAttr({b.getStringAttr("all"), b.getF32FloatAttr(1)})});
      v = Val{x.kind, pass(x.dm, spec, b.getStrArrayAttr({"Less(0.0f32)", "False", "False", "False"}))};
      break;
    }
    if (n.op == "rsqrt") {
      AluSpec sq;
      sq.method = "vector_fp_unary";
      sq.op = "FpUnaryOp::Sqrt";
      sq.fp = true;
      v = binary("div", Val{Val::Scalar, Value(), 1.0}, Val{x.kind, pass(x.dm, sq)});
      break;
    }
    static const std::map<std::string, std::string> unary = {
        {"exp", "Exp"}, {"sqrt", "Sqrt"}, {"tanh", "Tanh"}, {"log", "Log"},
        {"erf", "Erf"}, {"sin", "Sin"},   {"cos", "Cos"}};
    auto it = unary.find(n.op);
    if (it == unary.end())
      return fail("no Vector Engine op for '" + n.op + "'");
    AluSpec spec;
    spec.method = "vector_fp_unary";
    spec.op = "FpUnaryOp::" + it->second;
    spec.fp = true;
    v = Val{x.kind, pass(x.dm, spec)};
    break;
  }
  }
  if (failed(v))
    return failure();
  return vals[node] = *v;
}

//===----------------------------------------------------------------------===//
// Contraction
//===----------------------------------------------------------------------===//

FailureOr<Value> Lowering::contract(ArrayRef<int64_t> tileIndex) {
  const Gemm &g = *gemm;
  MapAttr slice = sliceMap();
  auto st = [&](Type et, MapAttr tm, MapAttr pk) { return kb.streamType(et, slice, tm, pk); };
  Type bf16 = BFloat16Type::get(ctx);
  MapAttr cl = colLocal();
  MapAttr lane = colPad > kLanes ? cl.modulo(kLanes) : cl;
  MapAttr group = colPad > kLanes ? cl.stride(kLanes) : kb.one();
  if (!trf) {
    // The TRF operand, loaded once: columns over lanes (and lane groups), K in the element.
    auto h = materialize(p.reads[g.trf], g.n, &kl);
    if (failed(h))
      return failure();
    Value bDm = toDm(*h, g.n, tileIndex, pairOf({cl, kf}), true);
    MapAttr tt = pairOf({lane, group, kf.stride(16)});
    Value s = BeginOp::create(b, loc(), st(f32, kb.one(), pairOf({cl, kf})), bDm, ContextKind::Sub);
    s = FetchOp::create(b, loc(), st(f32, tt, kf.modulo(16)), s);
    s = FetchCastOp::create(b, loc(), st(bf16, tt, kf.modulo(16)), s);
    s = CollectOp::create(b, loc(), st(bf16, tt, kf.modulo(16)), s);
    trf = ToTrfOp::create(b, loc(), TrfTensorType::get(ctx, bf16, kb.chip(), kb.cluster(), slice, lane,
                                                       pairOf({group, kf})),
                          s);
  }
  auto it = hbmOfRead.find(g.stream);
  if (it == hbmOfRead.end()) {
    auto h = materialize(p.reads[g.stream], rows, &kl);
    if (failed(h))
      return failure();
    it = hbmOfRead.insert({g.stream, *h}).first;
  }
  SmallVector<MapAttr> localE;
  for (const Digit &d : localD)
    localE.push_back(digit(d));
  SmallVector<MapAttr> ae(localE);
  ae.push_back(kf);
  Value aDm = toDm(it->second, rows, tileIndex, pairOf(ae), true);
  SmallVector<MapAttr> t16(localE), t32(localE), tg(localE);
  t16.push_back(kf.stride(16));
  t32.push_back(kf.stride(32));
  t32.push_back(group);
  tg.push_back(group);
  Value s = BeginOp::create(b, loc(), st(f32, kb.one(), pairOf(ae)), aDm, ContextKind::Main);
  s = FetchOp::create(b, loc(), st(f32, pairOf(t16), kf.modulo(16)), s);
  s = FetchCastOp::create(b, loc(), st(bf16, pairOf(t16), kf.modulo(16)), s);
  s = CollectOp::create(b, loc(), st(bf16, pairOf(t16), kf.modulo(16)), s);
  s = ContractOuterOp::create(b, loc(), st(bf16, pairOf(t32), kf.modulo(32)), s, trf);
  s = ContractPacketOp::create(b, loc(), st(f32, pairOf(t32), kb.one()), s);
  s = ContractTimeOp::create(b, loc(), st(f32, pairOf(tg), kb.one()), s);
  s = ContractLaneOp::create(b, loc(), st(f32, pairOf(tg), lane), s, LaneMode::Interleaved);
  s = CommitTrimOp::create(b, loc(), st(f32, pairOf(tg), lane), s);
  return CommitOp::create(b, loc(), kb.dmType(f32, slice, tileElement()), s).getResult();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

LogicalResult Lowering::emit() {
  if (failed(declare()))
    return failure();
  kb.beginBody();
  SmallVector<unsigned> layout(rows);
  layout.push_back(*col);
  SmallVector<MapAttr> outParts = canonical(layout, true);
  Value outCanonical = allocHbm(pairOf(outParts));
  int64_t numTiles = 1;
  for (const Digit &d : tileD)
    numTiles *= d.size;
  for (int64_t t = 0; t < numTiles; ++t) {
    SmallVector<int64_t> index(tileD.size());
    int64_t rest = t;
    for (int i = tileD.size() - 1; i >= 0; --i) {
      index[i] = rest % tileD[i].size;
      rest /= tileD[i].size;
    }
    vals.clear();
    auto v = evaluate(p.root, index);
    if (failed(v))
      return failure();
    if (v->kind != Val::Tile)
      return fail("the stored value does not vary over the rows");
    DmToHbmViewOp::create(b, loc(), v->dm, tileDigits(view(outCanonical, true), outParts, tileAt(layout, index), true));
  }
  // The relayout into the written buffer's own layout.
  Value out = allocHbm(pairOf(outDims));
  copy(view(outCanonical, false), view(out, true));
  kb.finish({out});
  return success();
}

bool match(RaisedKernel &kernel) {
  std::string why;
  FailureOr<Program> p = analyze(kernel.func, kernel.written, why);
  if (failed(p))
    return false;
  Lowering l(kernel, *p, /*quiet=*/true);
  return succeeded(l.plan());
}

LogicalResult map(RaisedKernel &kernel) {
  std::string why;
  FailureOr<Program> p = analyze(kernel.func, kernel.written, why);
  if (failed(p))
    return kernel.func.emitError("tiled: ") << why;
  Lowering l(kernel, *p, /*quiet=*/false);
  if (failed(l.plan()))
    return failure();
  return l.emit();
}

} // namespace

OpClass mlir::furiosa::mapping::tiledClass() { return {"tiled", match, map}; }

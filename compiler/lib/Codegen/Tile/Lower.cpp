// Op class "tiled": a kernel analyze() folds into a Program (Program.h) whose reductions are contractions, or
// that has none, lowered over tiles of a 2-D [rows, columns] layout.
//
//  - Layout: one output axis is the column (a contraction's TRF-side axis, else the axis most reads keep
//    contiguous); the others are rows. Columns split into blocks over slices and a local part (eight to a
//    packet); rows split into slice digits, local digits (time) and tile digits (one pass sequence each).
//  - Reads are made dense in HBM first, in that canonical digit layout. HBM-to-HBM DMAs are byte-granular, so
//    any stride, window, padding, permutation or broadcast costs one DMA per tap: a windowed read (a contraction
//    axis added to an output axis, as in convolutions) copies one tap per DMA out of a filled, padded copy of
//    its buffer, into a tap-major [taps, rows, inner] buffer (one contiguous block per tap). Each tile then
//    moves to DM with one aligned DMA.
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
#include <cstdlib>
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
  int64_t kTile = 0; // the K extent of one contraction pass (KF when K is not tiled)
  SmallVector<unsigned> rows;
  std::optional<unsigned> col;              // the column axis
  int64_t colLocalSize = 1, colPad = kLanes; // local columns, and padded to whole packets
  std::optional<Digit> colBlock, colLocalD;
  int64_t colExtent = 0; // the column axis padded to whole column tiles (0: no padding)
  MapAttr cpAxis;         // the padded column as a live axis
  std::map<unsigned, int64_t> rowExtent; // row axes padded to these extents (rows that split badly)
  std::map<unsigned, MapAttr> rpAxis;    // the padded rows as live axes
  SmallVector<Digit> tileD, sliceD, localD; // row digits by role, most significant first
  SmallVector<MapAttr> outDims;

  SmallVector<MapAttr> axisMap;
  std::map<unsigned, unsigned> paramOf;
  std::map<std::pair<double, bool>, Value> fillDm;
  MapAttr kf, interleave;
  unsigned fresh = 0;

  LogicalResult fail(const Twine &msg) {
    if (std::getenv("FURIOSA_TILED_DEBUG")) // diagnostics of trial mappings are swallowed
      llvm::errs() << "tiled: " << kernel.func.getName() << ": " << msg << "\n";
    if (!quiet)
      kernel.func.emitError("tiled: ") << msg;
    return failure();
  }

  LogicalResult findGemm();
  LogicalResult tryGemm(unsigned node);
  LogicalResult planLayout();
  int64_t assignRows(int64_t rowSlices, int64_t perRow, SmallVector<Digit> &local, SmallVector<Digit> &slice,
                     SmallVector<Digit> &tile);
  LogicalResult declare();

  // Mappings.
  MapAttr ax(unsigned a) { return axisMap[a]; }
  MapAttr pairOf(ArrayRef<MapAttr> parts) { return MapAttr::pair(ctx, parts); }
  MapAttr digit(const Digit &d) {
    int64_t extent = d.extent ? d.extent : extentOf(d.axis);
    // A column padded to whole tiles is a live axis of the padded extent inside the kernel (digits of a padded
    // axis do not tile); only the final relayout reads it as the column.
    MapAttr base = extent <= p.axes[d.axis].size ? ax(d.axis) : col && d.axis == *col ? cpAxis : rpAxis[d.axis];
    MapAttr m = d.place > 1 ? base.stride(d.place) : base;
    return d.place * d.size < extent ? m.modulo(d.size) : m;
  }
  /// The extent the layout pads axis `a` to (its size unless it is the tiled column).
  int64_t extentOf(unsigned a) {
    if (col && a == *col)
      return colExtent ? colExtent : p.axes[a].size;
    auto it = rowExtent.find(a);
    return it != rowExtent.end() ? it->second : p.axes[a].size;
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
      if (in(d.axis) && !(col && d.axis == *col))
        r.push_back(digit(d));
    for (const Digit &d : sliceD)
      if (in(d.axis))
        r.push_back(digit(d));
    for (const Digit &d : localD)
      if (in(d.axis))
        r.push_back(digit(d));
    if (rf && llvm::any_of(ext, in)) {
      if (rc)
        r.push_back(rc);
      r.push_back(rf);
    }
    if (col && in(*col)) {
      for (const Digit &d : tileD)
        if (d.axis == *col)
          r.push_back(digit(d));
      if (colBlock)
        r.push_back(digit(*colBlock));
      r.push_back(padCol ? colLocal() : digit(*colLocalD));
    }
    return r;
  }
  // The reduction being evaluated on the Vector Engine: its axes, flattened into one axis in time.
  SmallVector<unsigned> ext;
  MapAttr rf;
  // A reduction longer than one DM tile reduces in chunks: the chunk axis (outside rf) and the chunk at hand.
  MapAttr rc;
  int64_t rcIndex = 0;
  int64_t rfMax = 0; // the longest reduction chunk the plan allows
  int64_t kTiles() { return kTile ? kl.size() / kTile : 1; }
  /// KF in canonical HBM operands: one axis, or tile and local digits when K is tiled.
  SmallVector<MapAttr> kParts() {
    if (kTiles() == 1)
      return {kf};
    return {kf.stride(kTile), kf.modulo(kTile)};
  }
  /// The K of one contraction pass.
  MapAttr kLocal() { return kTiles() == 1 ? kf : kf.modulo(kTile); }
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
  /// A tap-major operand's K digits before and after the rows, and the digits K tile `kIndex` selects; a K tile
  /// is whole taps or part of the inner axis.
  struct KSplit {
    SmallVector<MapAttr> hi, lo;
    SmallVector<std::pair<MapAttr, int64_t>> at;
  };
  KSplit kSplit(int64_t kIndex) {
    int64_t ip = kl.innerPad, kt = kTiles() > 1 ? kTile : kl.size();
    KSplit s;
    if (kt % ip == 0) {
      if (kTiles() > 1) {
        s.hi.push_back(kf.stride(kTile));
        s.at.push_back({kf.stride(kTile), kIndex});
      }
      MapAttr inTile = kTiles() > 1 ? kf.modulo(kTile) : kf;
      if (kt > ip)
        s.hi.push_back(ip == 1 ? inTile : inTile.stride(ip));
      if (ip > 1)
        s.lo.push_back(kf.modulo(ip));
    } else {
      int64_t per = ip / kTile;
      s.hi = {kf.stride(ip)};
      s.lo = {kf.modulo(ip).stride(kTile), kf.modulo(kTile)};
      s.at = {{s.hi[0], kIndex / per}, {s.lo[0], kIndex % per}};
    }
    return s;
  }
  std::set<void *> tapMajor; // tap-major operands (HBM tensors, by value)
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
    if (rf)
      e.push_back(rf);
    e.push_back(colLocal());
    return pairOf(e);
  }

  // Emission helpers.
  Location loc() { return kb.loc(); }
  Value zeros(double value, Type et);
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
  Value allocHbm(MapAttr element, Type et) { return AllocHbmOp::create(b, loc(), kb.hbmType(et, element)); }
  Value reshapeHbm(Value t, MapAttr element) {
    if (cast<HbmTensorType>(t.getType()).getElement() == element)
      return t;
    return ReshapeHbmOp::create(b, loc(), kb.hbmType(cast<HbmTensorType>(t.getType()).getElementType(), element), t);
  }
  void copy(Value src, Value dst) { HbmToHbmViewOp::create(b, loc(), src, dst); }
  void fill(Value hbm, double value) {
    DmToHbmViewOp::create(b, loc(), zeros(value, cast<HbmTensorType>(hbm.getType()).getElementType()), view(hbm, true));
  }
  /// A DM tile or vector converted between f32 and bf16 by one pass (fetch_cast up, the Cast Engine down).
  Value convert(Value dm, Type to);
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
  Value toDm(Value hbm, ArrayRef<unsigned> axes, ArrayRef<int64_t> tileIndex, MapAttr element, bool withK = false,
             int64_t kIndex = 0) {
    auto parts = canonical(axes, false);
    auto at = tileAt(axes, tileIndex);
    bool major = tapMajor.count(hbm.getAsOpaquePointer());
    if (major && !withK) { // the Vector Engine reduction's flat axis (chunk axis first) leads
      parts.erase(llvm::find(parts, rf));
      parts.insert(parts.begin(), rf);
      if (rc) {
        parts.erase(llvm::find(parts, rc));
        parts.insert(parts.begin(), rc);
      }
    }
    if (rc && llvm::is_contained(parts, rc))
      at.push_back({rc, rcIndex});
    if (withK && major) {
      KSplit s = kSplit(kIndex);
      parts.insert(parts.begin(), s.hi.begin(), s.hi.end());
      llvm::append_range(parts, s.lo);
      llvm::append_range(at, s.at);
    } else if (withK) {
      llvm::append_range(parts, kParts());
      if (kTiles() > 1)
        at.push_back({kf.stride(kTile), kIndex});
    }
    Value v = tileDigits(view(hbm, false), parts, at, false);
    return ViewToDmOp::create(
        b, loc(), kb.dmType(cast<HbmTensorType>(hbm.getType()).getElementType(), sliceMap(), element), v);
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
  FailureOr<Val> reduce(unsigned node, ArrayRef<int64_t> tileIndex);

  // Contraction.
  FailureOr<Value> contract(ArrayRef<int64_t> tileIndex);
  /// A further factor of a contraction operand, in that operand's K layout (cached).
  FailureOr<Value> extraHbm(unsigned read, ArrayRef<unsigned> rowAxes) {
    auto it = hbmOfRead.find(~read - (1u << 20));
    if (it != hbmOfRead.end())
      return it->second;
    auto h = materialize(p.reads[read], rowAxes, &kl);
    if (failed(h))
      return failure();
    return hbmOfRead[~read - (1u << 20)] = *h;
  }
};

//===----------------------------------------------------------------------===//
// Planning
//===----------------------------------------------------------------------===//

/// A reduction with the reductions nested directly in it merged (same combiner): (op, axes, body).
std::tuple<std::string, SmallVector<unsigned>, unsigned> flatReduce(const Program &p, unsigned node) {
  const Node &n = p.nodes[node];
  SmallVector<unsigned> axes(n.axes);
  unsigned body = n.args[0];
  while (p.nodes[body].kind == Node::Reduce && p.nodes[body].op == n.op) {
    llvm::append_range(axes, p.nodes[body].axes);
    body = p.nodes[body].args[0];
  }
  return {n.op, axes, body};
}

LogicalResult Lowering::findGemm() {
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

LogicalResult Lowering::tryGemm(unsigned node) {
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

LogicalResult Lowering::planLayout() {
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
int64_t Lowering::assignRows(int64_t rowSlices, int64_t perRow, SmallVector<Digit> &local,
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

LogicalResult Lowering::plan() {
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
  return success();
}

LogicalResult Lowering::declare() {
  for (auto [i, a] : llvm::enumerate(p.axes))
    axisMap.push_back(kb.axis("A" + std::to_string(i), a.size));
  if (gemm)
    kf = kb.axis("KF", kl.size());
  if (colExtent)
    cpAxis = kb.axis("CP", colExtent);
  for (auto [a, e] : rowExtent)
    rpAxis[a] = kb.axis("RP" + std::to_string(a), e);
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

Value Lowering::zeros(double value, Type et) {
  auto key = std::make_pair(value, et.isBF16());
  auto it = fillDm.find(key);
  if (it != fillDm.end())
    return it->second;
  // Eight bytes of the value, broadcast by the DMA over the destination.
  Value dm = AllocDmOp::create(b, loc(), kb.dmType(et, kb.one().padTo(kSlices), kb.one().padTo(et.isBF16() ? 4 : 2)));
  MemsetOp::create(b, loc(), dm, b.getF32FloatAttr(value));
  return fillDm[key] = dm;
}

Value Lowering::convert(Value dm, Type to) {
  auto t = cast<DmTensorType>(dm.getType());
  if (t.getElementType() == to)
    return dm;
  auto [timeParts, packet] = timePacket(t.getElement());
  MapAttr time = pairOf(timeParts);
  auto st = [&](Type et, MapAttr tm, MapAttr pk) { return kb.streamType(et, t.getSlice(), tm, pk); };
  Type from = t.getElementType();
  Value s = BeginOp::create(b, loc(), st(from, kb.one(), t.getElement()), dm, ContextKind::Main);
  s = FetchOp::create(b, loc(), st(from, time, packet), s);
  if (from.isBF16()) {
    s = FetchCastOp::create(b, loc(), st(to, time, packet), s);
    s = CollectOp::create(b, loc(), st(to, time, packet), s);
  } else {
    s = CollectOp::create(b, loc(), st(from, time, packet), s);
    s = CastOp::create(b, loc(), st(to, time, packet.size() < 16 ? packet.padTo(16) : packet), s);
  }
  s = CommitTrimOp::create(b, loc(), st(to, time, packet), s);
  return CommitOp::create(b, loc(), kb.dmType(to, t.getSlice(), t.getElement()), s);
}

/// Read `r` as a dense HBM tensor in the canonical layout over `rowAxes`, then KF (a contraction operand, `k`)
/// or nothing more (the column, when the read has it, is one of `rowAxes`). Each tap of a windowed read is one
/// DMA from a source view that names the buffer's digits by fresh axes (a reshape of the whole read view); the
/// destination names its rows by the same expressions, and a whole-tensor reshape renames it canonically.
FailureOr<Value> Lowering::materialize(const Read &r, ArrayRef<unsigned> rowAxes, const KLayout *k) {
  unsigned nd = r.shape.size();
  // A read windowed by the reduction the Vector Engine evaluates (one of its axes spans past the step of a
  // coarser term of the same dim, so the two are no digits of the dim) copies one tap of that reduction per DMA,
  // like a contraction operand, into a buffer whose flat reduction axis leads.
  KLayout reduction;
  auto overlaps = [&](const Lin &l) {
    for (auto [a, c] : l.terms)
      if (llvm::is_contained(ext, a))
        for (auto [b2, c2] : l.terms)
          if (b2 != a && c2 >= c && c * p.axes[a].size > c2)
            return true;
    return false;
  };
  if (!k && llvm::any_of(r.index, overlaps)) {
    reduction.taps = ext;
    reduction.tapPad = rc ? rc.size() * rf.size() : rf.size();
    k = &reduction;
  }
  bool veTaps = k == &reduction;
  SmallVector<unsigned> taps;
  if (k)
    taps = k->taps;
  auto isTap = [&](unsigned a) { return llvm::is_contained(taps, a); };
  std::optional<unsigned> innerAxis = k ? k->inner : std::nullopt;
  int64_t innerSize = innerAxis ? p.axes[*innerAxis].size : 1;
  // Taps that are each alone in a buffer dim (no window: a weight's kernel position) move in one DMA as axes of
  // their own, then into the padded K layout in a second one, instead of one DMA per tap.
  SmallVector<unsigned> peeled;
  bool windowFree = k && !veTaps && this->windowFree(r);
  if (windowFree)
    std::swap(peeled, taps);
  bool major = veTaps || (k && tapMajorRead(r));
  // A tap-major operand's padded inner digit is read from a copy of the buffer padded with zeros in that dim
  // (one more DMA, at most as many runs), so each tap's destination stays one block with no holes.
  std::optional<unsigned> innerDim;
  if (major && innerAxis && innerSize < k->innerPad && r.fill.value_or(0.0) == 0.0)
    for (unsigned j = 0; j < nd; ++j)
      if (r.index[j].offset == 0 && r.index[j].terms.size() == 1 && r.index[j].terms[0].first == *innerAxis &&
          r.index[j].terms[0].second == 1)
        innerDim = j;
  int64_t innerLen = innerDim ? k->innerPad : innerSize;

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
    if (innerDim == j)
      extent[j] = k->innerPad;
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
    Value xp = allocHbm(pairOf(padded), r.type);
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
    if (isTap(a))
      continue;
    auto it = named.find(a);
    MapAttr m = it != named.end() ? it->second : kb.axis(tag + "R" + std::to_string(a), p.axes[a].size);
    // The tiled column pads to whole column tiles (cells the DMAs never write).
    dstParts.push_back(extentOf(a) > p.axes[a].size ? m.padTo(extentOf(a)) : m);
  }
  MapAttr tq;
  MapAttr innerExpr; // the inner axis as the source names it
  for (auto &ds : dig)
    for (auto &g : ds)
      if (g.axis && innerAxis && *g.axis == *innerAxis)
        innerExpr = g.expr;
  SmallVector<MapAttr> rowParts(dstParts);
  if (k && windowFree) {
    for (unsigned a : peeled)
      dstParts.push_back(named.at(a));
    if (innerExpr)
      dstParts.push_back(innerExpr);
  } else if (k) {
    if (k->tapPad > 1) {
      tq = kb.axis(tag + "TQ", k->tapPad);
      dstParts.insert(major ? dstParts.begin() : dstParts.end(), tq);
    }
    if (cq)
      dstParts.push_back(cq);
  }
  Value dst = allocHbm(pairOf(dstParts), r.type);
  int64_t numTaps = 1;
  for (unsigned a : taps)
    numTaps *= p.axes[a].size;
  if (k && !windowFree && (k->tapPad > numTaps || k->innerPad > innerLen))
    fill(dst, 0.0);

  // Every tap's view starts (one per source digit) and its index into the taps axis.
  auto startsOf = [&](std::map<unsigned, int64_t> &at, SmallVector<int64_t> &st, int64_t &index) -> LogicalResult {
    st.clear();
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
        int64_t len = !g.axis ? 1 : (innerAxis && *g.axis == *innerAxis) ? innerLen : p.axes[*g.axis].size;
        if (start + len > g.radix)
          return fail("a window that carries across digits");
        st.push_back(start);
      }
    }
    index = 0;
    for (unsigned a : taps)
      index = index * p.axes[a].size + at[a];
    return success();
  };
  SmallVector<int64_t> lens, radix;
  for (auto &ds : dig)
    for (auto &g : ds) {
      lens.push_back(!g.axis ? 1 : (innerAxis && *g.axis == *innerAxis) ? innerLen : p.axes[*g.axis].size);
      radix.push_back(g.radix);
    }
  // The taps split into classes (a residue of each tap axis modulo its period) in which every start is affine in
  // the quotients; each class is one nest of kept loops around a single DMA (unrolled DMAs cost the NPU compiler
  // superlinearly). A stride-s window has period s on its tap axis; a stride-1 window period 1.
  auto affineClass = [&](unsigned axis, int64_t m) -> bool {
    int64_t n = p.axes[axis].size;
    for (int64_t rr = 0; rr < m; ++rr) {
      std::map<unsigned, int64_t> at;
      SmallVector<int64_t> s0, s1, sq;
      int64_t i0, i1, iq;
      at[axis] = rr;
      if (failed(startsOf(at, s0, i0)))
        return false;
      if (rr + m >= n)
        continue;
      at[axis] = rr + m;
      if (failed(startsOf(at, s1, i1)))
        return false;
      for (int64_t v = rr + 2 * m; v < n; v += m) {
        at[axis] = v;
        if (failed(startsOf(at, sq, iq)))
          return false;
        int64_t q = (v - rr) / m;
        for (unsigned x = 0; x < s0.size(); ++x)
          if (sq[x] != s0[x] + q * (s1[x] - s0[x]))
            return false;
        if (iq != i0 + q * (i1 - i0))
          return false;
      }
    }
    return true;
  };
  SmallVector<int64_t> period;
  for (unsigned a : taps) {
    int64_t m = 1;
    while (m < p.axes[a].size && !affineClass(a, m))
      ++m;
    period.push_back(m);
  }
  // Class combinations: residues per tap axis, mixed radix.
  int64_t numClasses = 1;
  for (int64_t m : period)
    numClasses *= m;
  for (int64_t cls = 0; cls < numClasses; ++cls) {
    SmallVector<int64_t> residue(taps.size()), count(taps.size());
    int64_t rest = cls;
    for (int i = taps.size() - 1; i >= 0; --i) {
      residue[i] = rest % period[i];
      rest /= period[i];
      count[i] = (p.axes[taps[i]].size - residue[i] + period[i] - 1) / period[i];
    }
    std::map<unsigned, int64_t> at;
    for (auto [i, a] : llvm::enumerate(taps))
      at[a] = residue[i];
    SmallVector<int64_t> base, step;
    int64_t baseIndex;
    if (failed(startsOf(at, base, baseIndex)))
      return failure();
    // Per looped tap axis: its slopes and loop.
    SmallVector<SmallVector<int64_t>> slopes;
    SmallVector<int64_t> indexSlopes;
    SmallVector<Value> ivs;
    OpBuilder::InsertionGuard guard(b);
    for (auto [i, a] : llvm::enumerate(taps)) {
      if (count[i] == 1)
        continue;
      std::map<unsigned, int64_t> next(at);
      next[a] += period[i];
      int64_t nextIndex;
      if (failed(startsOf(next, step, nextIndex)))
        return failure();
      SmallVector<int64_t> sl;
      for (unsigned x = 0; x < base.size(); ++x)
        sl.push_back(step[x] - base[x]);
      slopes.push_back(sl);
      indexSlopes.push_back(nextIndex - baseIndex);
      auto loop = ForOp::create(b, loc(), b.getI64IntegerAttr(count[i]));
      Block *body = new Block();
      loop.getBody().push_back(body);
      ivs.push_back(body->addArgument(b.getIndexType(), loc()));
      b.setInsertionPointToStart(body);
    }
    auto tileAt = [&](Value v, MapAttr index, int64_t len, int64_t start, ArrayRef<int64_t> coeffs,
                      MapAttr element) -> Value {
      SmallVector<Value> used;
      SmallVector<int64_t> cs;
      for (auto [iv, c] : llvm::zip(ivs, coeffs))
        if (c) {
          used.push_back(iv);
          cs.push_back(c);
        }
      if (used.empty())
        return tileView(v, index, len, start, element);
      auto t = cast<HbmViewType>(v.getType());
      return TileAtOp::create(b, loc(), HbmViewType::get(ctx, t.getElementType(), t.getChip(), element, t.getIsMut()),
                              v, index, b.getI64IntegerAttr(len), b.getI64IntegerAttr(start), used,
                              b.getDenseI64ArrayAttr(cs))
          .getResult();
    };
    Value sv = reshapeView(view(source, false), pairOf(srcParts));
    SmallVector<MapAttr> cur(srcParts);
    for (unsigned x = 0; x < srcParts.size(); ++x) {
      if (lens[x] == radix[x])
        continue;
      SmallVector<int64_t> coeffs;
      for (auto &sl : slopes)
        coeffs.push_back(sl[x]);
      cur[x] = lens[x] == 1 ? kb.one().padTo(radix[x]) : srcParts[x].resize(lens[x]).padTo(radix[x]);
      sv = tileAt(sv, srcParts[x], lens[x], base[x], coeffs, pairOf(cur));
    }
    Value d = view(dst, true);
    if (k && !windowFree) {
      SmallVector<MapAttr> dc(dstParts);
      if (tq) {
        *llvm::find(dc, tq) = kb.one().padTo(k->tapPad, PadKind::Bottom);
        d = tileAt(d, tq, 1, baseIndex, indexSlopes, pairOf(dc));
      }
      if (cq && innerLen < k->innerPad) {
        *llvm::find(dc, cq) = cq.resize(innerSize).padTo(k->innerPad, PadKind::Bottom);
        d = tileView(d, cq, innerSize, 0, pairOf(dc));
      }
    }
    copy(sv, d);
  }
  if (k && windowFree) {
    // Into the padded K layout: the tap axes read as one (a reshape of the whole read view).
    int64_t numPeeled = 1;
    for (unsigned a : peeled)
      numPeeled *= p.axes[a].size;
    SmallVector<MapAttr> padded(rowParts), flat(rowParts);
    tq = kb.axis(tag + "TQ", k->tapPad);
    padded.push_back(tq);
    flat.push_back(numPeeled == k->tapPad ? tq : tq.resize(numPeeled));
    if (cq) {
      padded.push_back(cq);
      flat.push_back(innerExpr ? innerExpr : cq.resize(1));
    }
    Value full = allocHbm(pairOf(padded), r.type);
    if (k->tapPad > numPeeled || k->innerPad > innerSize)
      fill(full, 0.0);
    Value d = view(full, true);
    SmallVector<MapAttr> dc(padded);
    if (numPeeled < k->tapPad) {
      *llvm::find(dc, tq) = tq.resize(numPeeled).padTo(k->tapPad, PadKind::Bottom);
      d = tileView(d, tq, numPeeled, 0, pairOf(dc));
    }
    if (cq && innerSize < k->innerPad) {
      *llvm::find(dc, cq) = cq.resize(innerSize).padTo(k->innerPad, PadKind::Bottom);
      d = tileView(d, cq, innerSize, 0, pairOf(dc));
    }
    copy(reshapeView(view(dst, false), pairOf(flat)), d);
    dst = full;
  }

  SmallVector<MapAttr> canon = canonical(rowAxes, false);
  if (veTaps) {
    canon.erase(llvm::find(canon, rf));
    canon.insert(canon.begin(), rf);
    if (rc) {
      canon.erase(llvm::find(canon, rc));
      canon.insert(canon.begin(), rc);
    }
  } else if (major) {
    KSplit s = kSplit(0);
    canon.insert(canon.begin(), s.hi.begin(), s.hi.end());
    llvm::append_range(canon, s.lo);
  } else if (k) {
    llvm::append_range(canon, kParts());
  }
  Value out = reshapeHbm(dst, pairOf(canon));
  if (major)
    tapMajor.insert(out.getAsOpaquePointer());
  return out;
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
  // A column-only read is a VRF vector over the rows, or with no rows the whole tile (but not inside a
  // reduction, whose tiles carry its axis too).
  bool colOnly = llvm::all_of(axes, [&](unsigned a) { return a == *col; }) && !(rows.empty() && !ext.empty());
  SmallVector<unsigned> layout = colOnly ? SmallVector<unsigned>{*col} : rows;
  if (!colOnly) {
    llvm::append_range(layout, ext);
    layout.push_back(*col);
  }
  // Reads keyed by the reduction they sit in (their layout includes its axis).
  unsigned key = read * 1024 + (colOnly ? 0 : unsigned(ext.size() ? ext.front() + 1 : 0));
  auto it = hbmOfRead.find(key);
  if (it == hbmOfRead.end()) {
    auto h = materialize(r, layout, nullptr);
    if (failed(h))
      return failure();
    it = hbmOfRead.insert({key, *h}).first;
  }
  if (colOnly) // with no rows a column vector is the whole tile
    return Val{rows.empty() ? Val::Tile : Val::Vec, convert(toDm(it->second, layout, tileIndex, colLocal()), f32)};
  return Val{Val::Tile, convert(toDm(it->second, layout, tileIndex, tileElement()), f32)};
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
  // Otherwise a * s + b * (1 - s) with s = select(c, 1, 0): exact for finite branches (an infinite one would
  // give inf * 0 = NaN on the other side).
  if ((ta.kind == Node::Const && !std::isfinite(ta.value)) || (fb.kind == Node::Const && !std::isfinite(fb.value)))
    return fail("a select between a tile and an infinite constant");
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
    if (n.axes.empty()) {
      v = evaluate(n.args[0], tileIndex);
      break;
    }
    if (!gemm || node != gemm->node) {
      v = reduce(node, tileIndex);
      break;
    }
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

/// A reduction on the Vector Engine: its body evaluated over tiles extended by the reduction axes (flattened
/// into one axis in time), then one intra-slice reduce pass over that axis.
FailureOr<Val> Lowering::reduce(unsigned node, ArrayRef<int64_t> tileIndex) {
  auto [op, axes, body] = flatReduce(p, node);
  if (!ext.empty())
    return fail("a reduction inside a reduction");
  int64_t size = 1;
  for (unsigned a : axes)
    size *= p.axes[a].size;
  // Chunks: the largest divisor of the reduction that the plan allows per tile.
  int64_t chunk = std::min(size, rfMax);
  while (size % chunk)
    --chunk;
  std::string n = std::to_string(fresh++);
  MapAttr flat = kb.axis("RF" + n, chunk);
  MapAttr chunks = size > chunk ? kb.axis("RC" + n, size / chunk) : MapAttr();
  std::string rop = op == "add" ? "Add" : op == "max" ? "Max" : "Min";
  std::optional<Val> acc;
  for (int64_t c = 0; c < size / chunk; ++c) {
    ext = axes;
    rf = flat;
    rc = chunks;
    rcIndex = c;
    auto saved = std::move(vals);
    vals.clear();
    FailureOr<Val> x = evaluate(body, tileIndex);
    vals = std::move(saved);
    MapAttr element = tileElement();
    ext.clear();
    rf = rc = MapAttr();
    if (failed(x))
      return failure();
    if (x->kind != Val::Tile)
      return fail("a reduction whose body does not vary over its axes");
    // Fetch [local..., RF, (col / 8)] x packet, narrow to four lanes, reduce RF away, widen back.
    auto t = cast<DmTensorType>(x->dm.getType());
    auto [timeParts, packet] = timePacket(element);
    SmallVector<MapAttr> outParts;
    for (MapAttr m : timeParts)
      if (m != flat)
        outParts.push_back(m);
    MapAttr time = pairOf(timeParts), outTime = pairOf(outParts);
    auto st = [&](MapAttr tm, MapAttr pk) { return kb.streamType(f32, t.getSlice(), tm, pk); };
    Value s = BeginOp::create(b, loc(), st(kb.one(), element), x->dm, ContextKind::Main);
    s = FetchOp::create(b, loc(), st(time, packet), s);
    s = CollectOp::create(b, loc(), st(time, packet), s);
    s = VectorInitOp::create(b, loc(), st(time, packet), s);
    s = VectorIntraSliceTagOp::create(b, loc(), st(time, packet), s);
    SmallVector<MapAttr> t4(timeParts), o4(outParts);
    t4.push_back(packet.stride(4));
    o4.push_back(packet.stride(4));
    MapAttr p4 = packet.modulo(4);
    s = VectorNarrowSplitOp::create(b, loc(), st(pairOf(t4), p4), s);
    s = VectorIntraSliceReduceOp::create(b, loc(), st(pairOf(o4), p4), s, b.getStringAttr(rop),
                                         b.getStringAttr(flat.getName()));
    s = VectorWidenConcatOp::create(b, loc(), st(outTime, packet), s);
    s = VectorFinalOp::create(b, loc(), st(outTime, packet), s);
    s = CommitTrimOp::create(b, loc(), st(outTime, packet), s);
    Val part{Val::Tile, CommitOp::create(b, loc(), kb.dmType(f32, t.getSlice(), tileElement()), s).getResult()};
    if (!acc) {
      acc = part;
      continue;
    }
    // The chunks' partial results combine with the reduction's own op.
    AluSpec combine;
    combine.method = "vector_clip_zip";
    combine.op = "ClipBinaryOpF32::" + rop;
    acc = Val{Val::Tile, zip(acc->dm, part.dm, combine)};
  }
  return *acc;
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
  MapAttr k = kLocal();
  auto trfHbm = hbmOfRead.find(~g.trf);
  if (trfHbm == hbmOfRead.end()) {
    auto h = materialize(p.reads[g.trf], g.trfAxes(), &kl);
    if (failed(h))
      return failure();
    trfHbm = hbmOfRead.insert({~g.trf, *h}).first;
  }
  auto aHbm = hbmOfRead.find(~g.stream);
  if (aHbm == hbmOfRead.end()) {
    auto h = materialize(p.reads[g.stream], rows, &kl);
    if (failed(h))
      return failure();
    aHbm = hbmOfRead.insert({~g.stream, *h}).first;
  }
  SmallVector<MapAttr> localE;
  for (const Digit &d : localD)
    localE.push_back(digit(d));
  SmallVector<MapAttr> ae(localE);
  ae.push_back(k);
  SmallVector<MapAttr> t16(localE), t32(localE), tg(localE);
  t16.push_back(k.stride(16));
  t32.push_back(k.stride(32));
  t32.push_back(group);
  tg.push_back(group);
  Value acc;
  for (int64_t kt = 0; kt < kTiles(); ++kt) {
    // The TRF operand: columns over lanes (and lane groups), this K tile in the element.
    Value bDm = toDm(trfHbm->second, g.trfAxes(), tileIndex, pairOf({cl, k}), true, kt);
    for (unsigned e : g.trfExtras) {
      auto h = extraHbm(e, g.trfAxes());
      if (failed(h))
        return failure();
      AluSpec mul{"vector_fp_zip", "FpBinaryOp::MulF(FpMulAlu::Mul0)"};
      mul.fp = true;
      bDm = zip(convert(bDm, f32), convert(toDm(*h, g.trfAxes(), tileIndex, pairOf({cl, k}), true, kt), f32), mul);
    }
    MapAttr tt = pairOf({lane, group, k.stride(16)});
    Type bt = cast<DmTensorType>(bDm.getType()).getElementType();
    Value s = BeginOp::create(b, loc(), st(bt, kb.one(), pairOf({cl, k})), bDm, ContextKind::Sub);
    s = FetchOp::create(b, loc(), st(bt, tt, k.modulo(16)), s);
    if (!bt.isBF16())
      s = FetchCastOp::create(b, loc(), st(bf16, tt, k.modulo(16)), s);
    s = CollectOp::create(b, loc(), st(bf16, tt, k.modulo(16)), s);
    Value trf = ToTrfOp::create(
        b, loc(), TrfTensorType::get(ctx, bf16, kb.chip(), kb.cluster(), slice, lane, pairOf({group, k})), s);
    // The stream operand's tile.
    Value aDm = toDm(aHbm->second, rows, tileIndex, pairOf(ae), true, kt);
    for (unsigned e : g.streamExtras) {
      auto h = extraHbm(e, rows);
      if (failed(h))
        return failure();
      AluSpec mul{"vector_fp_zip", "FpBinaryOp::MulF(FpMulAlu::Mul0)"};
      mul.fp = true;
      aDm = zip(convert(aDm, f32), convert(toDm(*h, rows, tileIndex, pairOf(ae), true, kt), f32), mul);
    }
    Type at = cast<DmTensorType>(aDm.getType()).getElementType();
    s = BeginOp::create(b, loc(), st(at, kb.one(), pairOf(ae)), aDm, ContextKind::Main);
    s = FetchOp::create(b, loc(), st(at, pairOf(t16), k.modulo(16)), s);
    if (!at.isBF16())
      s = FetchCastOp::create(b, loc(), st(bf16, pairOf(t16), k.modulo(16)), s);
    s = CollectOp::create(b, loc(), st(bf16, pairOf(t16), k.modulo(16)), s);
    s = ContractOuterOp::create(b, loc(), st(bf16, pairOf(t32), k.modulo(32)), s, trf);
    s = ContractPacketOp::create(b, loc(), st(f32, pairOf(t32), kb.one()), s);
    s = ContractTimeOp::create(b, loc(), st(f32, pairOf(tg), kb.one()), s);
    s = ContractLaneOp::create(b, loc(), st(f32, pairOf(tg), lane), s, LaneMode::Interleaved);
    s = CommitTrimOp::create(b, loc(), st(f32, pairOf(tg), lane), s);
    Value partial = CommitOp::create(b, loc(), kb.dmType(f32, slice, tileElement()), s);
    if (!acc) {
      acc = partial;
      continue;
    }
    AluSpec add;
    add.method = "vector_clip_zip";
    add.op = "ClipBinaryOpF32::Add";
    acc = zip(acc, partial, add);
  }
  return acc;
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
  Value outCanonical = allocHbm(pairOf(outParts), p.outType);
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
    DmToHbmViewOp::create(b, loc(), convert(v->dm, p.outType),
                          tileDigits(view(outCanonical, true), outParts, tileAt(layout, index), true));
  }
  // The relayout into the written buffer's own layout: a padded row or column is read back as its axis (the
  // canonical digits are row-major per axis: tile, slice, local).
  Value out = allocHbm(pairOf(outDims), p.outType);
  if (colExtent || !rowExtent.empty()) {
    SmallVector<MapAttr> asAxes;
    for (unsigned a : rows) {
      if (extentOf(a) > p.axes[a].size) {
        asAxes.push_back(ax(a).padTo(extentOf(a)));
        continue;
      }
      for (const SmallVector<Digit> *role : {&tileD, &sliceD, &localD})
        for (const Digit &d : *role)
          if (d.axis == a)
            asAxes.push_back(digit(d));
    }
    if (colExtent) {
      asAxes.push_back(ax(*col).padTo(colExtent));
    } else {
      if (colBlock)
        asAxes.push_back(digit(*colBlock));
      asAxes.push_back(colLocal());
    }
    outCanonical = reshapeHbm(outCanonical, pairOf(asAxes));
  }
  copy(view(outCanonical, false), view(out, true));
  kb.finish({out});
  return success();
}

bool match(RaisedKernel &kernel) {
  // FURIOSA_TILED_DEBUG=1 reports why the class passes on a kernel (to stderr).
  bool debug = std::getenv("FURIOSA_TILED_DEBUG");
  std::string why;
  FailureOr<Program> p = analyze(kernel.func, kernel.written, why);
  if (failed(p)) {
    if (debug)
      llvm::errs() << "tiled: " << kernel.func.getName() << ": " << why << "\n";
    return false;
  }
  if (debug) {
    llvm::errs() << "tiled: " << kernel.func.getName() << ":\n";
    p->print(llvm::errs());
  }
  Lowering l(kernel, *p, /*quiet=*/!debug);
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

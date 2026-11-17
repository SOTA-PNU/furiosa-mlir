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
#include "Dma.h"
#include "Plan.h"

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


/// A value of the Vector Engine program: a DM tile over [local rows..., local columns], a DM column vector over
/// [local columns], or a constant.
struct Val {
  enum Kind { Tile, Vec, Scalar } kind;
  Value dm;
  double value = 0;
};

class Lowering : public Plan {
public:
  Lowering(RaisedKernel &kernel, Program &p, bool quiet)
      : Plan(kernel.func, p, quiet), kernel(kernel), kb(kernel), ctx(kb.getContext()), b(kb.builder()),
        f32(Float32Type::get(ctx)) {}

  /// Builds the fopt kernel.
  LogicalResult emit();

private:
  RaisedKernel &kernel;
  KernelBuilder kb;
  MLIRContext *ctx;
  OpBuilder &b;
  Type f32;

  MapAttr cpAxis;                     // the padded column as a live axis
  std::map<unsigned, MapAttr> rpAxis; // the padded rows as live axes
  SmallVector<MapAttr> outDims;

  SmallVector<MapAttr> axisMap;
  std::map<unsigned, unsigned> paramOf;
  MapAttr kf, interleave;
  unsigned fresh = 0;

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
  /// KF in canonical HBM operands: one axis, or tile and local digits when K is tiled.
  SmallVector<MapAttr> kParts() {
    if (kTiles() == 1)
      return {kf};
    return {kf.stride(kTile), kf.modulo(kTile)};
  }
  /// The K of one contraction pass.
  MapAttr kLocal() { return kTiles() == 1 ? kf : kf.modulo(kTile); }
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

  // Operands: staged reads (fopt.stage) and their tiles (fopt.read), lowered by furiosa-dma (Dma.cpp), which
  // decides the staging layout.
  /// Read `r` staged in HBM for tiles of the layout over `rowAxes`, then KF (a contraction operand, `k`) or
  /// nothing more (the column, when the read has it, is one of `rowAxes`).
  FailureOr<Value> materialize(const Read &r, ArrayRef<unsigned> rowAxes, const KLayout *k) {
    StageSpec spec;
    spec.read = r;
    for (unsigned a = 0; a < p.axes.size(); ++a) {
      spec.sizes.push_back(p.axes[a].size);
      spec.extents.push_back(extentOf(a));
    }
    spec.rows = SmallVector<unsigned>(rowAxes);
    spec.canonical = canonical(rowAxes, false);
    SmallVector<MapAttr> logical(spec.canonical);
    if (k) {
      spec.k = *k;
      spec.kTile = kTiles() > 1 ? kTile : 0;
      spec.kf = kf;
      llvm::append_range(logical, kParts());
    }
    if (rf) {
      spec.ext = ext;
      spec.rf = rf;
      spec.rc = rc;
    }
    spec.tag = "S" + std::to_string(fresh++) + "_";
    spec.axesAt = kb.numAxes();
    return StageOp::create(b, loc(), kb.hbmType(r.type, pairOf(logical)), kb.param(paramOf[r.arg]), spec.encode(ctx))
        .getResult();
  }
  /// One tile of a staged read in DM: the tile `tileIndex` of the layout over `axes`, K tile `kIndex` of a
  /// contraction operand.
  Value toDm(Value staged, ArrayRef<unsigned> axes, ArrayRef<int64_t> tileIndex, MapAttr element, bool withK = false,
             int64_t kIndex = 0) {
    SmallVector<Attribute> at;
    for (auto [d, index] : tileAt(axes, tileIndex))
      at.push_back(b.getArrayAttr({d, b.getI64IntegerAttr(index)}));
    Type et = cast<HbmTensorType>(staged.getType()).getElementType();
    return ReadOp::create(b, loc(), kb.dmType(et, sliceMap(), element), staged, b.getArrayAttr(at),
                          b.getBoolAttr(withK), b.getI64IntegerAttr(kIndex), b.getI64IntegerAttr(rcIndex));
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

/// The loops the plan tiles a kernel into: its tile digits as tile sizes of the root generic's loops, its K tiles
/// as tile sizes of the contraction's reduction loops. Nothing when there are none, or one is no whole range of a
/// loop (a padded axis, an axis that is an inner digit of its dim).
std::optional<Tiling> tiling(RaisedKernel &kernel) {
  func::FuncOp func = kernel.func;
  std::string why;
  FailureOr<Program> p = analyze(func, kernel.written, why);
  if (failed(p) || !p->rootOp || p->outLoop.empty())
    return std::nullopt;
  Plan plan(func, *p, /*quiet=*/true);
  if (failed(plan.plan()))
    return std::nullopt;
  auto debug = [&](const Twine &msg) {
    if (std::getenv("FURIOSA_TILED_DEBUG"))
      llvm::errs() << "tile: " << func.getName() << ": " << msg << "\n";
    return std::nullopt;
  };
  Tiling t;
  t.root = cast<linalg::GenericOp>(p->rootOp);
  t.rowSizes.assign(t.root.getNumLoops(), 0);
  bool rows = false;
  for (const Digit &d : plan.tiles()) {
    auto [dim, place] = p->axisDim[d.axis];
    int64_t size = place * d.place;
    if (d.extent || dim >= p->outLoop.size() || size * d.size != p->dimExtent[dim] || t.rowSizes[p->outLoop[dim]])
      return debug("tile digit of axis " + Twine(d.axis) + " is no whole range of a loop");
    t.rowSizes[p->outLoop[dim]] = size;
    rows = true;
  }
  if (auto k = plan.kAxisTiles()) {
    for (auto [a, size] : *k) {
      auto [dim, place] = p->axisDim[a];
      auto it = p->reductionLoop.find(dim);
      if (place != 1 || p->axes[a].size != p->dimExtent[dim] || it == p->reductionLoop.end() ||
          (t.reduce && t.reduce != it->second.first))
        return debug("K tile of axis " + Twine(a) + " is no whole reduction loop of the contraction");
      t.reduce = cast<linalg::GenericOp>(it->second.first);
      t.reduceSizes.resize(t.reduce.getNumLoops(), 0);
      t.reduceSizes[it->second.second] = size;
    }
  }
  if (!rows && !t.reduce)
    return std::nullopt;
  return t;
}

} // namespace

OpClass mlir::furiosa::mapping::tiledClass() { return {"tiled", match, map, tiling}; }

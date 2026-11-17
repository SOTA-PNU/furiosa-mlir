// Pass furiosa-dma: lowers the tiled lowering's hardware-abstract data movement (fopt.stage, fopt.read) to fopt
// DMAs. A stage is a kernel buffer read at a linear index over the program's axes; the pass decides how it is
// staged in HBM: a padded copy filled with the read's padding value, its digits, the K layout of a contraction
// operand (tap-major [taps, rows, inner] when the inner run is a whole DMA unit, so each tap's DMA writes one
// contiguous block; the NPU compiler's cost grows superlinearly in an HBM destination's runs), the inner digit
// zero-padded in the source copy, the taps' DMAs as kept loops over affine tap classes. Each read is a tile view
// of the staged tensor and one HBM-to-DM DMA.

#include "Dma.h"

#include "furiosa/Codegen/Codegen.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <map>

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;
using namespace mlir::furiosa::tile;

//===----------------------------------------------------------------------===//
// The stage spec as attributes
//===----------------------------------------------------------------------===//

namespace {

ArrayAttr linsAttr(Builder &b, ArrayRef<Lin> lins) {
  SmallVector<Attribute> out;
  for (const Lin &l : lins) {
    SmallVector<int64_t> v{l.offset};
    for (auto [a, c] : l.terms) {
      v.push_back(a);
      v.push_back(c);
    }
    out.push_back(b.getDenseI64ArrayAttr(v));
  }
  return b.getArrayAttr(out);
}

SmallVector<unsigned> unsignedOf(DenseI64ArrayAttr a) {
  SmallVector<unsigned> out;
  for (int64_t x : a.asArrayRef())
    out.push_back(x);
  return out;
}

DenseI64ArrayAttr i64s(Builder &b, ArrayRef<unsigned> v) {
  return b.getDenseI64ArrayAttr(SmallVector<int64_t>(v.begin(), v.end()));
}

} // namespace

DictionaryAttr StageSpec::encode(MLIRContext *ctx) const {
  Builder b(ctx);
  NamedAttrList a;
  a.append("index", linsAttr(b, read.index));
  a.append("shape", b.getDenseI64ArrayAttr(read.shape));
  if (read.fill)
    a.append("fill", b.getF64FloatAttr(*read.fill));
  a.append("sizes", b.getDenseI64ArrayAttr(sizes));
  a.append("extents", b.getDenseI64ArrayAttr(extents));
  a.append("rows", i64s(b, rows));
  a.append("canonical", b.getArrayAttr(SmallVector<Attribute>(canonical.begin(), canonical.end())));
  if (k) {
    a.append("taps", i64s(b, k->taps));
    if (k->inner)
      a.append("inner", b.getI64IntegerAttr(*k->inner));
    a.append("tap_pad", b.getI64IntegerAttr(k->tapPad));
    a.append("inner_pad", b.getI64IntegerAttr(k->innerPad));
    a.append("k_tile", b.getI64IntegerAttr(kTile));
    a.append("kf", kf);
  }
  if (rf) {
    a.append("ext", i64s(b, ext));
    a.append("rf", rf);
    if (rc)
      a.append("rc", rc);
  }
  a.append("tag", b.getStringAttr(tag));
  a.append("axes_at", b.getI64IntegerAttr(axesAt));
  return a.getDictionary(ctx);
}

FailureOr<StageSpec> StageSpec::decode(DictionaryAttr d) {
  StageSpec s;
  auto index = d.getAs<ArrayAttr>("index");
  auto shape = d.getAs<DenseI64ArrayAttr>("shape");
  auto sizes = d.getAs<DenseI64ArrayAttr>("sizes");
  auto extents = d.getAs<DenseI64ArrayAttr>("extents");
  auto rows = d.getAs<DenseI64ArrayAttr>("rows");
  auto canonical = d.getAs<ArrayAttr>("canonical");
  auto tag = d.getAs<StringAttr>("tag");
  auto axesAt = d.getAs<IntegerAttr>("axes_at");
  if (!index || !shape || !sizes || !extents || !rows || !canonical || !tag || !axesAt)
    return failure();
  for (Attribute x : index) {
    ArrayRef<int64_t> v = cast<DenseI64ArrayAttr>(x).asArrayRef();
    Lin l;
    l.offset = v[0];
    for (size_t i = 1; i + 1 < v.size(); i += 2)
      l.terms.push_back({unsigned(v[i]), v[i + 1]});
    s.read.index.push_back(l);
  }
  s.read.shape = SmallVector<int64_t>(shape.asArrayRef());
  if (auto fill = d.getAs<FloatAttr>("fill"))
    s.read.fill = fill.getValueAsDouble();
  s.sizes = SmallVector<int64_t>(sizes.asArrayRef());
  s.extents = SmallVector<int64_t>(extents.asArrayRef());
  s.rows = unsignedOf(rows);
  for (Attribute x : canonical)
    s.canonical.push_back(cast<MapAttr>(x));
  if (auto taps = d.getAs<DenseI64ArrayAttr>("taps")) {
    KLayout k;
    k.taps = unsignedOf(taps);
    if (auto inner = d.getAs<IntegerAttr>("inner"))
      k.inner = inner.getInt();
    k.tapPad = d.getAs<IntegerAttr>("tap_pad").getInt();
    k.innerPad = d.getAs<IntegerAttr>("inner_pad").getInt();
    s.k = k;
    s.kTile = d.getAs<IntegerAttr>("k_tile").getInt();
    s.kf = d.getAs<MapAttr>("kf");
  }
  if (auto rf = d.getAs<MapAttr>("rf")) {
    s.rf = rf;
    s.rc = d.getAs<MapAttr>("rc");
    s.ext = unsignedOf(d.getAs<DenseI64ArrayAttr>("ext"));
  }
  s.tag = tag.str();
  s.axesAt = axesAt.getInt();
  return s;
}

//===----------------------------------------------------------------------===//
// Staging
//===----------------------------------------------------------------------===//

namespace {

/// A padded copy several stages of one buffer share: the union of their windows, made by the first of them.
struct SharedCopy {
  SmallVector<int64_t> low, extent;
  Value copy;
};

/// Lowers one stage and its reads (the layout decisions of the data movement).
class Stager {
public:
  Stager(StageOp stage, StageSpec &spec, std::map<std::pair<double, bool>, Value> &fillDm, SharedCopy *shared)
      : stage(stage), spec(spec), fillDm(fillDm), shared(shared), ctx(stage.getContext()), b(stage),
        source(stage.getSource()) {}

  /// The staged tensor (inserted before the stage), or failure.
  FailureOr<Value> run() {
    spec.read.type = cast<HbmTensorType>(source.getType()).getElementType();
    return materialize(spec.read, spec.rows, spec.k ? &*spec.k : nullptr);
  }
  /// Lowers `read` of the staged tensor `hbm` (a tile view and one DMA to DM).
  Value lower(ReadOp read, Value hbm);

  bool tapMajor = false;
  SmallVector<std::pair<std::string, int64_t>> newAxes; // in declaration order

private:
  StageOp stage;
  StageSpec &spec;
  std::map<std::pair<double, bool>, Value> &fillDm;
  SharedCopy *shared;
  MLIRContext *ctx;
  OpBuilder b;
  Value source;

  LogicalResult fail(const Twine &msg) { return stage.emitError("tiled: ") << msg; }
  Location loc() { return stage.getLoc(); }
  MapAttr one() { return MapAttr::broadcast(ctx, 1); }
  MapAttr declare(StringRef name, int64_t size) {
    if (!llvm::any_of(newAxes, [&](auto &x) { return x.first == name; }))
      newAxes.emplace_back(name.str(), size);
    return MapAttr::axis(ctx, name, size);
  }
  MapAttr pairOf(ArrayRef<MapAttr> parts) { return MapAttr::pair(ctx, parts); }
  HbmTensorType hbmType(Type et, MapAttr element) { return HbmTensorType::get(ctx, et, one(), element); }
  DmTensorType dmType(Type et, MapAttr slice, MapAttr element) {
    return DmTensorType::get(ctx, et, one(), one().padTo(2), slice, element);
  }

  int64_t kTiles() { return spec.kTiles(); }
  /// KF in canonical HBM operands: one axis, or tile and local digits when K is tiled.
  SmallVector<MapAttr> kParts() {
    if (kTiles() == 1)
      return {spec.kf};
    return {spec.kf.stride(spec.kTile), spec.kf.modulo(spec.kTile)};
  }
  /// A tap-major operand's K digits before and after the rows, and the digits K tile `kIndex` selects; a K tile
  /// is whole taps or part of the inner axis.
  struct KSplit {
    SmallVector<MapAttr> hi, lo;
    SmallVector<std::pair<MapAttr, int64_t>> at;
  };
  KSplit kSplit(int64_t kIndex) {
    const KLayout &kl = *spec.k;
    MapAttr kf = spec.kf;
    int64_t kTile = spec.kTile;
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
  /// Whether a contraction operand's taps each sit alone in a buffer dim (a weight's kernel position, no window).
  bool windowFree(const Read &r) {
    const KLayout &kl = *spec.k;
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
  /// A windowed operand is tap-major in HBM, [taps, rows, inner], so each tap's DMA writes one contiguous block,
  /// when its inner run is a whole 8-byte DMA unit (each tile's move to DM puts K innermost).
  bool tapMajorRead(const Read &r) {
    const KLayout &kl = *spec.k;
    return !kl.taps.empty() && !windowFree(r) && kl.innerPad * r.type.getIntOrFloatBitWidth() >= 64;
  }

  Value zeros(double value, Type et) {
    auto key = std::make_pair(value, et.isBF16());
    auto it = fillDm.find(key);
    if (it != fillDm.end())
      return it->second;
    // Eight bytes of the value, broadcast by the DMA over the destination.
    Value dm = AllocDmOp::create(b, loc(), dmType(et, one().padTo(kSlices), one().padTo(et.isBF16() ? 4 : 2)));
    MemsetOp::create(b, loc(), dm, b.getF32FloatAttr(value));
    return fillDm[key] = dm;
  }
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
  Value allocHbm(MapAttr element, Type et) { return AllocHbmOp::create(b, loc(), hbmType(et, element)); }
  Value reshapeHbm(Value t, MapAttr element) {
    if (cast<HbmTensorType>(t.getType()).getElement() == element)
      return t;
    return ReshapeHbmOp::create(b, loc(), hbmType(cast<HbmTensorType>(t.getType()).getElementType(), element), t);
  }
  void copy(Value src, Value dst) { HbmToHbmViewOp::create(b, loc(), src, dst); }
  void fill(Value hbm, double value) {
    DmToHbmViewOp::create(b, loc(), zeros(value, cast<HbmTensorType>(hbm.getType()).getElementType()), view(hbm, true));
  }
  /// Tiles view `v` (of mapping `parts`) to one position of each digit in `at`.
  Value tileDigits(Value v, SmallVector<MapAttr> parts, ArrayRef<std::pair<MapAttr, int64_t>> at, bool mut) {
    for (auto [d, index] : at) {
      auto it = llvm::find(parts, d);
      assert(it != parts.end() && "a tiled digit of the view");
      *it = one().padTo(d.size(), mut ? PadKind::Bottom : PadKind::Top);
      v = tileView(v, d, 1, index, pairOf(parts));
    }
    return v;
  }

  FailureOr<Value> materialize(const Read &r, ArrayRef<unsigned> rowAxes, const KLayout *k);
};

FailureOr<Value> Stager::materialize(const Read &r, ArrayRef<unsigned> rowAxes, const KLayout *k) {
  unsigned nd = r.shape.size();
  // A read windowed by the reduction the Vector Engine evaluates (one of its axes spans past the step of a
  // coarser term of the same dim, so the two are no digits of the dim) copies one tap of that reduction per DMA,
  // like a contraction operand, into a buffer whose flat reduction axis leads.
  KLayout reduction;
  auto overlaps = [&](const Lin &l) {
    for (auto [a, c] : l.terms)
      if (llvm::is_contained(spec.ext, a))
        for (auto [b2, c2] : l.terms)
          if (b2 != a && c2 >= c && c * spec.sizes[a] > c2)
            return true;
    return false;
  };
  if (!k && llvm::any_of(r.index, overlaps)) {
    reduction.taps = spec.ext;
    reduction.tapPad = spec.rc ? spec.rc.size() * spec.rf.size() : spec.rf.size();
    k = &reduction;
  }
  bool veTaps = k == &reduction;
  SmallVector<unsigned> taps;
  if (k)
    taps = k->taps;
  auto isTap = [&](unsigned a) { return llvm::is_contained(taps, a); };
  std::optional<unsigned> innerAxis = k ? k->inner : std::nullopt;
  int64_t innerSize = innerAxis ? spec.sizes[*innerAxis] : 1;
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
      hi += c * (spec.sizes[a] - 1);
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
  // A copy shared with other stages of the buffer: the union of their windows.
  if (shared) {
    low = shared->low;
    extent = shared->extent;
  }
  bool copyNeeded = r.fill.has_value();
  for (unsigned j = 0; j < nd; ++j)
    copyNeeded |= low[j] || extent[j] != r.shape[j];

  Value param = source;
  Value source = param;
  std::string tag = spec.tag;
  if (copyNeeded && shared && shared->copy) {
    source = shared->copy;
  } else if (copyNeeded) {
    // A copy with the padded extents, filled with the padding value, the buffer in its interior.
    SmallVector<MapAttr> padded, interior;
    for (unsigned j = 0; j < nd; ++j) {
      if (extent[j] == 1)
        continue;
      MapAttr axis = declare(tag + "P" + std::to_string(j), extent[j]);
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
    if (shared)
      shared->copy = xp;
  }

  // The digits of each dim, most significant first: one per remaining term, then the residual below the
  // smallest coefficient.
  struct DigitInfo {
    MapAttr expr;
    int64_t radix, coeff;
    std::optional<unsigned> axis;
  };
  SmallVector<SmallVector<DigitInfo>> dig(nd);
  MapAttr cq = (k && k->innerPad > 1) ? declare(tag + "CQ", k->innerPad) : MapAttr();
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
        e = declare(tag + "D" + std::to_string(j) + "_" + std::to_string(dig[j].size()), radix);
      }
      dig[j].push_back({e, radix, c, a});
      above = c;
    }
    if (above > 1)
      dig[j].push_back({declare(tag + "D" + std::to_string(j) + "_" + std::to_string(dig[j].size()), above), above, 1,
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
        int64_t size = spec.sizes[*g.axis];
        named[*g.axis] = g.radix == size ? g.expr : g.expr.resize(size);
      }
  SmallVector<MapAttr> dstParts;
  for (unsigned a : rowAxes) {
    if (isTap(a))
      continue;
    auto it = named.find(a);
    MapAttr m = it != named.end() ? it->second : declare(tag + "R" + std::to_string(a), spec.sizes[a]);
    // The tiled column pads to whole column tiles (cells the DMAs never write).
    dstParts.push_back(spec.extents[a] > spec.sizes[a] ? m.padTo(spec.extents[a]) : m);
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
      tq = declare(tag + "TQ", k->tapPad);
      dstParts.insert(major ? dstParts.begin() : dstParts.end(), tq);
    }
    if (cq)
      dstParts.push_back(cq);
  }
  Value dst = allocHbm(pairOf(dstParts), r.type);
  int64_t numTaps = 1;
  for (unsigned a : taps)
    numTaps *= spec.sizes[a];
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
        int64_t len = !g.axis ? 1 : (innerAxis && *g.axis == *innerAxis) ? innerLen : spec.sizes[*g.axis];
        if (start + len > g.radix)
          return fail("a window that carries across digits");
        st.push_back(start);
      }
    }
    index = 0;
    for (unsigned a : taps)
      index = index * spec.sizes[a] + at[a];
    return success();
  };
  SmallVector<int64_t> lens, radix;
  for (auto &ds : dig)
    for (auto &g : ds) {
      lens.push_back(!g.axis ? 1 : (innerAxis && *g.axis == *innerAxis) ? innerLen : spec.sizes[*g.axis]);
      radix.push_back(g.radix);
    }
  // The taps split into classes (a residue of each tap axis modulo its period) in which every start is affine in
  // the quotients; each class is one nest of kept loops around a single DMA (unrolled DMAs cost the NPU compiler
  // superlinearly). A stride-s window has period s on its tap axis; a stride-1 window period 1.
  auto affineClass = [&](unsigned axis, int64_t m) -> bool {
    int64_t n = spec.sizes[axis];
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
    while (m < spec.sizes[a] && !affineClass(a, m))
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
      count[i] = (spec.sizes[taps[i]] - residue[i] + period[i] - 1) / period[i];
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
      cur[x] = lens[x] == 1 ? one().padTo(radix[x]) : srcParts[x].resize(lens[x]).padTo(radix[x]);
      sv = tileAt(sv, srcParts[x], lens[x], base[x], coeffs, pairOf(cur));
    }
    Value d = view(dst, true);
    if (k && !windowFree) {
      SmallVector<MapAttr> dc(dstParts);
      if (tq) {
        *llvm::find(dc, tq) = one().padTo(k->tapPad, PadKind::Bottom);
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
      numPeeled *= spec.sizes[a];
    SmallVector<MapAttr> padded(rowParts), flat(rowParts);
    tq = declare(tag + "TQ", k->tapPad);
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

  SmallVector<MapAttr> canon(spec.canonical);
  if (veTaps) {
    canon.erase(llvm::find(canon, spec.rf));
    canon.insert(canon.begin(), spec.rf);
    if (spec.rc) {
      canon.erase(llvm::find(canon, spec.rc));
      canon.insert(canon.begin(), spec.rc);
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
    tapMajor = true;
  return out;
}

Value Stager::lower(ReadOp read, Value hbm) {
  b.setInsertionPoint(read);
  SmallVector<MapAttr> parts(spec.canonical);
  SmallVector<std::pair<MapAttr, int64_t>> at;
  llvm::DenseMap<Attribute, Value> ivOf; // tile digits a device loop indexes
  for (Attribute x : read.getAt()) {
    auto entry = cast<ArrayAttr>(x);
    auto digit = cast<MapAttr>(entry[0]);
    at.push_back({digit, cast<IntegerAttr>(entry[1]).getInt()});
    if (int64_t iv = cast<IntegerAttr>(entry[2]).getInt(); iv >= 0)
      ivOf[digit] = read.getIvs()[iv];
  }
  bool withK = read.getWithK();
  if (tapMajor && !withK) { // the Vector Engine reduction's flat axis (chunk axis first) leads
    parts.erase(llvm::find(parts, spec.rf));
    parts.insert(parts.begin(), spec.rf);
    if (spec.rc) {
      parts.erase(llvm::find(parts, spec.rc));
      parts.insert(parts.begin(), spec.rc);
    }
  }
  if (spec.rc && llvm::is_contained(parts, spec.rc))
    at.push_back({spec.rc, read.getRcIndex()});
  if (withK && tapMajor) {
    KSplit s = kSplit(read.getKIndex());
    parts.insert(parts.begin(), s.hi.begin(), s.hi.end());
    llvm::append_range(parts, s.lo);
    llvm::append_range(at, s.at);
  } else if (withK) {
    llvm::append_range(parts, kParts());
    if (kTiles() > 1)
      at.push_back({spec.kf.stride(spec.kTile), read.getKIndex()});
  }
  Value v = view(hbm, false);
  for (auto [d, index] : at) {
    auto it = llvm::find(parts, d);
    assert(it != parts.end() && "a tiled digit of the view");
    *it = one().padTo(d.size(), PadKind::Top);
    if (Value iv = ivOf.lookup(d)) {
      auto vt = cast<HbmViewType>(v.getType());
      v = TileAtOp::create(b, loc(), HbmViewType::get(ctx, vt.getElementType(), vt.getChip(), pairOf(parts), false),
                           v, d, b.getI64IntegerAttr(1), b.getI64IntegerAttr(index), ValueRange{iv},
                           b.getDenseI64ArrayAttr({1}))
              .getResult();
      continue;
    }
    v = tileView(v, d, 1, index, pairOf(parts));
  }
  return ViewToDmOp::create(b, loc(), read.getType(), v);
}

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

LogicalResult mlir::furiosa::tile::lowerDataMovement(func::FuncOp func) {
  SmallVector<StageOp> stages(func.getOps<StageOp>());
  if (stages.empty())
    return success();
  std::map<std::pair<double, bool>, Value> fillDm;
  // Stages of one buffer that read it alike but for their offsets (unrolled taps of a window, each its own read)
  // share one padded copy holding all of their windows.
  std::vector<StageSpec> specs;
  for (StageOp stage : stages) {
    FailureOr<StageSpec> spec = StageSpec::decode(stage.getSpec());
    if (failed(spec))
      return stage.emitError("fopt.stage: a malformed spec");
    specs.push_back(*spec);
  }
  std::map<std::string, SharedCopy> shares;
  SmallVector<std::string> keyOf(stages.size());
  std::map<std::string, int> members;
  for (auto [i, spec] : llvm::enumerate(specs)) {
    if (spec.k || spec.rf)
      continue;
    std::string key;
    llvm::raw_string_ostream os(key);
    os << stages[i].getSource().getAsOpaquePointer() << "|" << (spec.read.fill ? *spec.read.fill : 0.0)
       << (spec.read.fill ? "f" : "-") << "|";
    for (int64_t x : spec.read.shape)
      os << x << ",";
    for (const Lin &l : spec.read.index) {
      os << "|";
      for (auto [a, c] : l.terms)
        os << a << "*" << c << "+";
    }
    keyOf[i] = key;
    ++members[key];
  }
  for (auto [i, spec] : llvm::enumerate(specs)) {
    if (keyOf[i].empty() || members[keyOf[i]] < 2)
      continue;
    // The union of the windows: the most low padding any needs, the farthest index any reads.
    unsigned nd = spec.read.shape.size();
    SharedCopy &share = shares[keyOf[i]];
    if (share.low.empty()) {
      share.low.assign(nd, 0);
      share.extent.assign(nd, 0);
    }
    for (unsigned j = 0; j < nd; ++j) {
      const Lin &l = spec.read.index[j];
      int64_t lo = l.offset, hi = l.offset;
      for (auto [a, c] : l.terms)
        hi += c * (spec.sizes[a] - 1);
      share.low[j] = std::max(share.low[j], lo < 0 ? -lo : 0);
      share.extent[j] = std::max(share.extent[j], std::max(hi + 1, spec.read.shape[j]));
    }
  }
  for (auto &[key, share] : shares)
    for (auto [i, spec] : llvm::enumerate(specs))
      if (keyOf[i] == key) {
        // Extents past the low padding, rounded to the coarsest stride as each read's own would be.
        for (unsigned j = 0; j < share.extent.size(); ++j) {
          SmallVector<int64_t> cs;
          for (auto [a, c] : spec.read.index[j].terms)
            cs.push_back(c);
          int64_t e = share.extent[j] + share.low[j];
          share.extent[j] = cs.empty() ? e : roundUp(e, *llvm::max_element(cs));
        }
        break;
      }
  // The staging axes go where the lowering would have declared them: after the kernel axes declared before
  // the stage, which later stages' axes push further.
  SmallVector<std::pair<int64_t, SmallVector<std::pair<std::string, int64_t>>>> inserts;
  for (auto [i, stage] : llvm::enumerate(stages)) {
    StageSpec *spec = &specs[i];
    auto share = keyOf[i].empty() ? shares.end() : shares.find(keyOf[i]);
    Stager stager(stage, *spec, fillDm, share == shares.end() ? nullptr : &share->second);
    FailureOr<Value> hbm = stager.run();
    if (failed(hbm))
      return failure();
    for (Operation *user : llvm::make_early_inc_range(stage->getUsers())) {
      auto read = dyn_cast<ReadOp>(user);
      if (!read)
        return stage.emitError("fopt.stage: used by an op other than fopt.read");
      read.replaceAllUsesWith(stager.lower(read, *hbm));
      read.erase();
    }
    stage.erase();
    inserts.push_back({spec->axesAt, stager.newAxes});
  }
  SmallVector<Attribute> axes(func->getAttrOfType<ArrayAttr>(kAxesAttr).getValue());
  Builder b(func.getContext());
  int64_t shift = 0;
  for (auto &[at, decls] : inserts) {
    SmallVector<Attribute> attrs;
    for (auto &[name, size] : decls)
      attrs.push_back(b.getArrayAttr({b.getStringAttr(name), b.getI64IntegerAttr(size)}));
    axes.insert(axes.begin() + at + shift, attrs.begin(), attrs.end());
    shift += attrs.size();
  }
  func->setAttr(kAxesAttr, b.getArrayAttr(axes));
  return success();
}

namespace {

struct DmaPass : PassWrapper<DmaPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DmaPass)

  StringRef getArgument() const override { return "furiosa-dma"; }
  StringRef getDescription() const override {
    return "Lower staged reads (fopt.stage, fopt.read) to HBM staging buffers and DMAs, choosing their layout";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registerCodegenDialects(registry); }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>())
      if (failed(lowerDataMovement(func)))
        return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createDmaPass() { return std::make_unique<DmaPass>(); }

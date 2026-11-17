// Normalizing structured kernels for the op classes. A read whose indexing-map result is a mixed-radix linear
// combination of loop dims (d1 * 128 + d2: a buffer row read as heads x head dim) becomes a projected read of
// the buffer reshaped by those radices (tensor.expand_shape), cut to the box the dims reach
// (tensor.extract_slice). An input that only carried a loop's extent is dropped once another operand spans
// that loop.

#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

#include <map>

using namespace mlir;

namespace {

/// constant + sum(coefficient * dim) of an affine expression, or failure.
FailureOr<std::pair<int64_t, std::map<unsigned, int64_t>>> linearForm(AffineExpr e) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return std::make_pair(c.getValue(), std::map<unsigned, int64_t>{});
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return std::make_pair(int64_t(0), std::map<unsigned, int64_t>{{d.getPosition(), 1}});
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return failure();
  if (bin.getKind() == AffineExprKind::Add) {
    auto l = linearForm(bin.getLHS()), r = linearForm(bin.getRHS());
    if (failed(l) || failed(r))
      return failure();
    for (auto [d, c] : r->second)
      l->second[d] += c;
    l->first += r->first;
    return l;
  }
  if (bin.getKind() == AffineExprKind::Mul) {
    auto k = dyn_cast<AffineConstantExpr>(bin.getRHS());
    auto x = linearForm(bin.getLHS());
    if (!k || failed(x))
      return failure();
    for (auto &[d, c] : x->second)
      c *= k.getValue();
    x->first *= k.getValue();
    return x;
  }
  return failure();
}

/// An index computed in a generic body as an affine function of the loop indices, or null.
AffineExpr affineIndex(Value v, Block &body, MLIRContext *ctx) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
      return getAffineConstantExpr(i.getInt(), ctx);
  Operation *op = v.getDefiningOp();
  if (!op || op->getBlock() != &body)
    return {};
  if (auto index = dyn_cast<linalg::IndexOp>(op))
    return getAffineDimExpr(index.getDim(), ctx);
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
    return affineIndex(op->getOperand(0), body, ctx);
  AffineExpr a = op->getNumOperands() == 2 ? affineIndex(op->getOperand(0), body, ctx) : AffineExpr();
  AffineExpr b = op->getNumOperands() == 2 ? affineIndex(op->getOperand(1), body, ctx) : AffineExpr();
  if (!a || !b)
    return {};
  if (isa<arith::AddIOp>(op))
    return a + b;
  if (isa<arith::SubIOp>(op))
    return a - b;
  if (isa<arith::MulIOp>(op) && (isa<AffineConstantExpr>(a) || isa<AffineConstantExpr>(b)))
    return a * b;
  return {};
}

/// Reads a generic body makes with tensor.extract at affine addresses of a buffer defined outside it become
/// inputs read through that affine map (delinearize then makes them projections). Structure promotes such
/// reads itself unless they are masked by a bound it cannot pad; the mask is maskToPad's.
linalg::GenericOp promoteExtracts(linalg::GenericOp g) {
  Block &body = g.getRegion().front();
  MLIRContext *ctx = g.getContext();
  SmallVector<tensor::ExtractOp> extracts;
  SmallVector<AffineMap> maps;
  body.walk([&](tensor::ExtractOp e) {
    Value t = e.getTensor();
    if (t.getParentBlock() == &body || e->getParentOp() != g.getOperation())
      return;
    SmallVector<AffineExpr> exprs;
    for (Value index : e.getIndices()) {
      AffineExpr x = affineIndex(index, body, ctx);
      if (!x)
        return;
      exprs.push_back(x);
    }
    extracts.push_back(e);
    maps.push_back(AffineMap::get(g.getNumLoops(), 0, exprs, ctx));
  });
  if (extracts.empty())
    return g;
  OpBuilder b(g);
  SmallVector<Value> inputs(g.getDpsInputs());
  SmallVector<AffineMap> allMaps;
  for (OpOperand *in : g.getDpsInputOperands())
    allMaps.push_back(g.getMatchingIndexingMap(in));
  for (auto [e, m] : llvm::zip(extracts, maps)) {
    inputs.push_back(e.getTensor());
    allMaps.push_back(m);
  }
  for (OpOperand &out : g.getDpsInitsMutable())
    allMaps.push_back(g.getMatchingIndexingMap(&out));
  auto replacement = linalg::GenericOp::create(b, g.getLoc(), g.getResultTypes(), inputs, g.getDpsInits(), allMaps,
                                               g.getIteratorTypesArray());
  replacement->setDiscardableAttrs(g->getDiscardableAttrDictionary());
  replacement.getRegion().takeBody(g.getRegion());
  Block &moved = replacement.getRegion().front();
  unsigned at = g.getNumDpsInputs();
  for (auto [k, e] : llvm::enumerate(extracts)) {
    BlockArgument arg = moved.insertArgument(at + k, e.getType(), e.getLoc());
    e.replaceAllUsesWith(Value(arg));
    e.erase();
  }
  g.replaceAllUsesWith(replacement.getResults());
  g.erase();
  return replacement;
}

/// Rewrites input `i` of `g` to a projected read when its map's results are mixed-radix combinations.
void delinearizeRead(linalg::GenericOp g, unsigned i) {
  OpOperand *operand = g.getDpsInputOperand(i);
  AffineMap map = g.getMatchingIndexingMap(operand);
  if (map.isProjectedPermutation())
    return;
  SmallVector<int64_t> ranges = g.getStaticLoopRanges();
  // The buffer the operand reads, and the box offsets of an extract_slice in between.
  Value base = operand->get();
  auto type = cast<RankedTensorType>(base.getType());
  SmallVector<int64_t> offsets(type.getRank(), 0);
  if (auto slice = base.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (slice.getSourceType().getRank() == type.getRank() && slice.hasUnitStride() &&
        !ShapedType::isDynamicShape(slice.getStaticOffsets())) {
      base = slice.getSource();
      offsets.assign(slice.getStaticOffsets().begin(), slice.getStaticOffsets().end());
    }
  }
  auto baseType = cast<RankedTensorType>(base.getType());
  if (!baseType.hasStaticShape() || map.getNumResults() != unsigned(baseType.getRank()))
    return;
  // Per result: its digits (radix, box offset, box size, the dim or -1 for a fixed index).
  struct Digit {
    int64_t radix, offset, size;
    int dim;
  };
  SmallVector<SmallVector<Digit>> digits;
  for (auto [k, e] : llvm::enumerate(map.getResults())) {
    auto form = linearForm(e);
    if (failed(form))
      return;
    int64_t n = baseType.getDimSize(k), at = offsets[k] + form->first;
    SmallVector<std::pair<int64_t, unsigned>> terms; // (coefficient, dim), largest first
    for (auto [d, c] : form->second)
      if (c) {
        if (c < 0)
          return;
        terms.push_back({c, d});
      }
    llvm::sort(terms, [](auto x, auto y) { return x.first > y.first; });
    SmallVector<Digit> ds;
    if (terms.empty()) {
      ds.push_back({n, at, 1, -1});
    } else {
      if (terms.back().first != 1 || n % terms.front().first)
        return;
      int64_t radix = n / terms.front().first, rest = at;
      for (size_t t = 0; t < terms.size(); ++t) {
        int64_t c = terms[t].first;
        if (t + 1 < terms.size() && c % terms[t + 1].first)
          return;
        int64_t off = rest / c;
        rest %= c;
        int64_t extent = ranges[terms[t].second];
        if (off + extent > radix)
          return; // the box would carry into the next digit
        ds.push_back({radix, off, extent, int(terms[t].second)});
        radix = t + 1 < terms.size() ? c / terms[t + 1].first : 1;
      }
    }
    digits.push_back(ds);
  }
  // expand_shape by the digits, the box, and the projected map.
  OpBuilder b(g);
  Location loc = g.getLoc();
  SmallVector<int64_t> shape, boxOffsets, boxSizes;
  SmallVector<ReassociationIndices> groups;
  SmallVector<AffineExpr> results;
  for (auto &ds : digits) {
    ReassociationIndices group;
    for (Digit d : ds) {
      group.push_back(shape.size());
      shape.push_back(d.radix);
      boxOffsets.push_back(d.offset);
      boxSizes.push_back(d.size);
      results.push_back(d.dim < 0 ? b.getAffineConstantExpr(0) : b.getAffineDimExpr(d.dim));
    }
    groups.push_back(group);
  }
  Value expanded = base;
  if (shape.size() != size_t(baseType.getRank()))
    expanded = tensor::ExpandShapeOp::create(b, loc, RankedTensorType::get(shape, baseType.getElementType()), base,
                                             groups);
  Value box = expanded;
  if (boxSizes != shape || llvm::any_of(boxOffsets, [](int64_t o) { return o != 0; })) {
    auto folds = [&](ArrayRef<int64_t> xs) {
      SmallVector<OpFoldResult> r;
      for (int64_t x : xs)
        r.push_back(b.getIndexAttr(x));
      return r;
    };
    SmallVector<int64_t> strides(shape.size(), 1);
    box = tensor::ExtractSliceOp::create(b, loc, expanded, folds(boxOffsets), folds(boxSizes), folds(strides));
  }
  operand->set(box);
  SmallVector<AffineMap> maps = g.getIndexingMapsArray();
  maps[i] = AffineMap::get(map.getNumDims(), 0, results, b.getContext());
  g.setIndexingMapsAttr(b.getAffineMapArrayAttr(maps));
}

/// A masked read: an input read only through `select(p, in, c)`, with p an index-only bound on one loop dim the
/// input spans, true on an interval of it, becomes a read of the input's box on that interval padded with c,
/// written as the box inserted into a filled tensor (tensor.pad's meaning, in ops the copy steps map).
void maskToPad(linalg::GenericOp g, unsigned i) {
  Block &body = g.getRegion().front();
  BlockArgument in = body.getArgument(i);
  SmallVector<arith::SelectOp> selects;
  for (Operation *user : in.getUsers()) {
    auto sel = dyn_cast<arith::SelectOp>(user);
    if (!sel || (sel.getTrueValue() == in) == (sel.getFalseValue() == in))
      return;
    selects.push_back(sel);
  }
  if (selects.empty())
    return;
  arith::SelectOp first = selects.front();
  bool keepTrue = first.getTrueValue() == in;
  Value other = keepTrue ? first.getFalseValue() : first.getTrueValue();
  auto c = other.getDefiningOp<arith::ConstantOp>();
  for (arith::SelectOp sel : selects)
    if (sel.getCondition() != first.getCondition() || (sel.getTrueValue() == in) != keepTrue ||
        (keepTrue ? sel.getFalseValue() : sel.getTrueValue()) != other)
      return;
  if (!c)
    return;
  SmallVector<int64_t> ranges = g.getStaticLoopRanges();
  auto truth = furiosa::truthOverOneDim(first.getCondition(), body, ranges);
  if (!truth)
    return;
  auto [dim, holds] = *truth;
  int64_t lo = -1, hi = -1;
  for (auto [x, h] : llvm::enumerate(holds)) {
    if (h != keepTrue)
      continue;
    if (lo < 0)
      lo = x;
    else if (hi >= 0)
      return; // not one interval
    if (x + 1 == int64_t(holds.size()) || holds[x + 1] != keepTrue)
      hi = x + 1;
  }
  if (lo < 0)
    return;
  OpOperand *operand = g.getDpsInputOperand(i);
  AffineMap map = g.getMatchingIndexingMap(operand);
  auto type = cast<RankedTensorType>(operand->get().getType());
  int t = -1;
  for (auto [k, e] : llvm::enumerate(map.getResults()))
    if (auto d = dyn_cast<AffineDimExpr>(e); d && d.getPosition() == dim)
      t = k;
  if (t < 0 || type.getDimSize(t) != ranges[dim])
    return;
  if (lo > 0 || hi < ranges[dim]) {
    OpBuilder b(g);
    Location loc = g.getLoc();
    auto folds = [&](ArrayRef<int64_t> xs) {
      SmallVector<OpFoldResult> r;
      for (int64_t x : xs)
        r.push_back(b.getIndexAttr(x));
      return r;
    };
    SmallVector<int64_t> offsets(type.getRank(), 0), sizes(type.getShape()), strides(type.getRank(), 1);
    offsets[t] = lo;
    sizes[t] = hi - lo;
    // A box of a box is one box of the outer source.
    Value source = operand->get();
    SmallVector<int64_t> at(offsets);
    if (auto inner = source.getDefiningOp<tensor::ExtractSliceOp>();
        inner && inner.hasUnitStride() && inner.getSourceType().getRank() == type.getRank() &&
        !ShapedType::isDynamicShape(inner.getStaticOffsets())) {
      source = inner.getSource();
      for (int64_t k = 0; k < type.getRank(); ++k)
        at[k] += inner.getStaticOffsets()[k];
    }
    Value box = tensor::ExtractSliceOp::create(b, loc, source, folds(at), folds(sizes), folds(strides));
    Value fillValue = arith::ConstantOp::create(b, loc, c.getValue());
    Value empty = tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
    Value filled = linalg::FillOp::create(b, loc, fillValue, empty).getResult(0);
    operand->set(tensor::InsertSliceOp::create(b, loc, box, filled, folds(offsets), folds(sizes), folds(strides)));
  }
  for (arith::SelectOp sel : selects) {
    sel.replaceAllUsesWith(Value(in));
    sel.erase();
  }
}

/// Drops inputs that are never read and only carried loop extents another operand now spans.
void dropCarriers(linalg::GenericOp g) {
  Block &body = g.getRegion().front();
  for (int i = g.getNumDpsInputs() - 1; i >= 0; --i) {
    if (!body.getArgument(i).use_empty() || !g.getDpsInputs()[i].getDefiningOp<tensor::EmptyOp>())
      continue;
    SmallVector<AffineMap> maps = g.getIndexingMapsArray();
    llvm::SmallBitVector spanned(g.getNumLoops());
    for (auto [k, m] : llvm::enumerate(maps))
      if (int(k) != i)
        for (AffineExpr e : m.getResults())
          if (auto d = dyn_cast<AffineDimExpr>(e))
            spanned.set(d.getPosition());
    if (!spanned.all())
      continue;
    g.getInputsMutable().erase(i);
    body.eraseArgument(i);
    maps.erase(maps.begin() + i);
    g.setIndexingMapsAttr(OpBuilder(g).getAffineMapArrayAttr(maps));
  }
}

/// A generic that only copies a value another linalg op computed into an output of as many elements, in the
/// same order, is a reshape: it moves no data (the producer writes where the copy would have).
void copiesToReshapes(ModuleOp module) {
  SmallVector<linalg::GenericOp> copies;
  module.walk([&](linalg::GenericOp g) {
    if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 || g.getNumReductionLoops() ||
        !g.getDpsInputs()[0].getDefiningOp<linalg::LinalgOp>())
      return;
    Block &body = g.getRegion().front();
    if (&body.front() != body.getTerminator() || body.getTerminator()->getOperand(0) != body.getArgument(0))
      return;
    auto in = cast<RankedTensorType>(g.getDpsInputs()[0].getType());
    auto out = cast<RankedTensorType>(g.getResultTypes()[0]);
    if (in.getNumElements() != out.getNumElements() || !g.getIndexingMapsArray()[1].isIdentity())
      return;
    int last = -1;
    for (AffineExpr e : g.getIndexingMapsArray()[0].getResults()) {
      auto d = dyn_cast<AffineDimExpr>(e);
      if (!d || int(d.getPosition()) <= last)
        return; // as many elements, read in order: the loops the input skips are unit ones
      last = d.getPosition();
    }
    copies.push_back(g);
  });
  for (linalg::GenericOp g : copies) {
    OpBuilder b(g);
    g.getResult(0).replaceAllUsesWith(
        furiosa::reshapeTo(b, g.getLoc(), g.getDpsInputs()[0], cast<RankedTensorType>(g.getResultTypes()[0])));
    g.erase();
  }
}

/// A full reduction over more than kRow elements reduces in two levels (upstream splitReduction): rows of at
/// most kRow first, then their partials, so no op class holds one long row. Rows and partials both come in
/// multiples of 16 (the row-wise class's row granule).
void splitLongReductions(ModuleOp module) {
  constexpr int64_t kRow = 1024;
  SmallVector<linalg::GenericOp> todo;
  module.walk([&](linalg::GenericOp g) {
    if (g.getNumLoops() == 1 && g.getNumReductionLoops() == 1 && g.getNumDpsInits() == 1 &&
        cast<RankedTensorType>(g.getResultTypes()[0]).getRank() == 0 && g.getStaticLoopRanges()[0] > kRow)
      todo.push_back(g);
  });
  IRRewriter rewriter(module.getContext());
  for (linalg::GenericOp g : todo) {
    int64_t n = g.getStaticLoopRanges()[0], row = 0;
    for (int64_t parts = 16; parts * 16 <= n && !row; parts += 16)
      if (n % parts == 0 && (n / parts) % 16 == 0 && n / parts <= kRow)
        row = n / parts;
    if (!row)
      continue;
    rewriter.setInsertionPoint(g);
    (void)linalg::splitReduction(rewriter, cast<linalg::LinalgOp>(g.getOperation()), [&](linalg::LinalgOp) {
      return linalg::SplitReductionOptions{n / row, 0, false};
    });
  }
}

/// A padded tensor that only boxes are cut from (structure's masked tiles) is the source inserted into a constant
/// of the padding value, which the copy steps map: one copy into the padded tensor, whose extent the boxes
/// then divide (the NPU tiles an axis only in windows that grow into a divisor of it).
void padsToInserts(ModuleOp module) {
  SmallVector<tensor::PadOp> pads;
  module.walk([&](tensor::PadOp pad) {
    Attribute c;
    Value padding = pad.getConstantPaddingValue();
    if (padding && matchPattern(padding, m_Constant(&c)) && !ShapedType::isDynamicShape(pad.getStaticLow()) &&
        pad.getResultType().hasStaticShape() && !pad.getResult().use_empty() &&
        llvm::all_of(pad->getUsers(), [](Operation *u) { return isa<tensor::ExtractSliceOp>(u); }))
      pads.push_back(pad);
  });
  for (tensor::PadOp pad : pads) {
    OpBuilder b(pad);
    Location loc = pad.getLoc();
    Attribute c;
    matchPattern(pad.getConstantPaddingValue(), m_Constant(&c));
    SmallVector<OpFoldResult> low, sizes, ones;
    for (auto [d, l] : llvm::enumerate(pad.getStaticLow())) {
      low.push_back(b.getIndexAttr(l));
      sizes.push_back(b.getIndexAttr(pad.getSourceType().getDimSize(d)));
      ones.push_back(b.getIndexAttr(1));
    }
    Value filled = arith::ConstantOp::create(b, loc, DenseElementsAttr::get(pad.getResultType(), c));
    pad.replaceAllUsesWith(tensor::InsertSliceOp::create(b, loc, pad.getSource(), filled, low, sizes, ones).getResult());
    pad.erase();
  }
}

} // namespace

void mlir::furiosa::normalizeKernels(ModuleOp module) {
  copiesToReshapes(module);
  splitLongReductions(module);
  padsToInserts(module);
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp g) { generics.push_back(g); });
  for (linalg::GenericOp g : generics) {
    g = promoteExtracts(g);
    for (unsigned i = 0; i < g.getNumDpsInputs(); ++i) {
      delinearizeRead(g, i);
      maskToPad(g, i);
    }
    dropCarriers(g);
  }
  // The slices and carriers the reads no longer use.
  for (auto func : module.getOps<func::FuncOp>())
    for (bool erased = true; erased;) {
      erased = false;
      SmallVector<Operation *> ops;
      for (Operation &op : func.getBody().front().without_terminator())
        ops.push_back(&op);
      for (Operation *op : llvm::reverse(ops))
        if (isOpTriviallyDead(op)) {
          op->erase();
          erased = true;
        }
    }
}

Value mlir::furiosa::reshapeTo(OpBuilder &b, Location loc, Value v, RankedTensorType t) {
  auto from = cast<RankedTensorType>(v.getType());
  if (from == t)
    return v;
  auto all = [](int64_t rank) {
    ReassociationIndices r;
    for (int64_t d = 0; d < rank; ++d)
      r.push_back(d);
    return rank ? SmallVector<ReassociationIndices>{r} : SmallVector<ReassociationIndices>{};
  };
  if (from.getRank() == 0)
    return tensor::ExpandShapeOp::create(b, loc, t, v, all(0));
  if (t.getRank() == 0)
    return tensor::CollapseShapeOp::create(b, loc, t, v, all(0));
  if (from.getRank() != 1)
    v = tensor::CollapseShapeOp::create(b, loc, RankedTensorType::get({from.getNumElements()}, from.getElementType()), v,
                                        all(from.getRank()));
  if (t.getRank() != 1)
    v = tensor::ExpandShapeOp::create(b, loc, t, v, all(t.getRank()));
  return v;
}

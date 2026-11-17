// Normalizing structured kernels for the op classes. A read whose indexing-map result is a mixed-radix linear
// combination of loop dims (d1 * 128 + d2: a buffer row read as heads x head dim) becomes a projected read of
// the buffer reshaped by those radices (tensor.expand_shape), cut to the box the dims reach
// (tensor.extract_slice). An input that only carried a loop's extent is dropped once another operand spans
// that loop.

#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
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
void delinearize(linalg::GenericOp g, unsigned i) {
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

} // namespace

void mlir::furiosa::normalizeKernels(ModuleOp module) {
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp g) { generics.push_back(g); });
  for (linalg::GenericOp g : generics) {
    g = promoteExtracts(g);
    for (unsigned i = 0; i < g.getNumDpsInputs(); ++i) {
      delinearize(g, i);
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

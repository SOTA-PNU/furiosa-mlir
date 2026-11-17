// Normalizing structured kernels for the op classes. A read whose indexing-map result is a mixed-radix linear
// combination of loop dims (d1 * 128 + d2: a buffer row read as heads x head dim) becomes a projected read of
// the buffer reshaped by those radices (tensor.expand_shape), cut to the box the dims reach
// (tensor.extract_slice). An input that only carried a loop's extent is dropped once another operand spans
// that loop.

#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include <cmath>
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

/// The value an index-only body value takes where loop index `dim` is `at`: linalg.index of that dim, integer
/// constants and integer arithmetic; nothing for anything else (another index, a block argument).
std::optional<int64_t> evalAt(Value v, unsigned dim, int64_t at) {
  if (APInt c; matchPattern(v, m_ConstantInt(&c)))
    return c.getSExtValue();
  Operation *op = v.getDefiningOp();
  if (auto index = dyn_cast_or_null<linalg::IndexOp>(op))
    return index.getDim() == dim ? std::optional<int64_t>(at) : std::nullopt;
  if (isa_and_nonnull<arith::IndexCastOp>(op))
    return evalAt(op->getOperand(0), dim, at);
  if (!isa_and_nonnull<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
    return std::nullopt;
  std::optional<int64_t> a = evalAt(op->getOperand(0), dim, at), b = evalAt(op->getOperand(1), dim, at);
  if (!a || !b)
    return std::nullopt;
  return isa<arith::AddIOp>(op) ? *a + *b : isa<arith::SubIOp>(op) ? *a - *b : *a * *b;
}

/// A selection reduction: one accumulator takes the greatest (least) value, the other the payload at the same
/// element, ties going to the least (greatest) payload. tl.argmax/argmin's combiner, read by what it does:
/// select both accumulators by order(acc, x) | (acc == x & tie(acc_payload, x_payload)).
struct Selection {
  unsigned value, payload; // result indices
  BlockArgument in;        // the compared input's block argument
  Value term;              // the payload's term (the element's)
  bool max, leastPayload;
};

std::optional<Selection> matchSelection(linalg::GenericOp g) {
  if (g.getNumDpsInits() != 2 || g.getNumReductionLoops() != 1)
    return std::nullopt;
  Block &body = *g.getBody();
  unsigned nIn = g.getNumDpsInputs();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  auto s0 = yield.getOperand(0).getDefiningOp<arith::SelectOp>();
  auto s1 = yield.getOperand(1).getDefiningOp<arith::SelectOp>();
  if (!s0 || !s1 || s0.getCondition() != s1.getCondition() || s0.getTrueValue() != body.getArgument(nIn) ||
      s1.getTrueValue() != body.getArgument(nIn + 1))
    return std::nullopt;
  // cmp(out, in) as +1 (greater), -1 (less), 0 (equal); nothing for another predicate or operands.
  auto rel = [](Value cond, Value out, Value in) -> std::optional<int> {
    Value l, r;
    int sign;
    if (auto f = cond.getDefiningOp<arith::CmpFOp>()) {
      using P = arith::CmpFPredicate;
      l = f.getLhs(), r = f.getRhs();
      P p = f.getPredicate();
      if (p != P::OGT && p != P::OLT && p != P::OEQ)
        return std::nullopt;
      sign = p == P::OGT ? 1 : p == P::OLT ? -1 : 0;
    } else if (auto i = cond.getDefiningOp<arith::CmpIOp>()) {
      using P = arith::CmpIPredicate;
      l = i.getLhs(), r = i.getRhs();
      P p = i.getPredicate();
      if (p != P::sgt && p != P::slt && p != P::eq)
        return std::nullopt;
      sign = p == P::sgt ? 1 : p == P::slt ? -1 : 0;
    } else {
      return std::nullopt;
    }
    if (l == out && r == in)
      return sign;
    if (l == in && r == out)
      return -sign;
    return std::nullopt;
  };
  auto readsOuts = [&](Value v) {
    SmallVector<Value> work{v};
    llvm::DenseSet<Value> seen;
    while (!work.empty()) {
      Value x = work.pop_back_val();
      if (!seen.insert(x).second)
        continue;
      if (auto arg = dyn_cast<BlockArgument>(x); arg && arg.getOwner() == &body && arg.getArgNumber() >= nIn)
        return true;
      if (Operation *op = x.getDefiningOp(); op && op->getBlock() == &body)
        llvm::append_range(work, op->getOperands());
    }
    return false;
  };
  Value c = s0.getCondition();
  for (unsigned v : {0u, 1u}) {
    Value outV = body.getArgument(nIn + v), outP = body.getArgument(nIn + 1 - v);
    Value inV = (v ? s1 : s0).getFalseValue(), inP = (v ? s0 : s1).getFalseValue();
    auto inArg = dyn_cast<BlockArgument>(inV);
    if (!inArg || inArg.getOwner() != &body || inArg.getArgNumber() >= nIn || readsOuts(inP))
      continue;
    std::optional<int> order = rel(c, outV, inV), tie;
    if (order == 0)
      order.reset();
    if (auto either = c.getDefiningOp<arith::OrIOp>(); !order && either)
      for (int side : {0, 1}) {
        std::optional<int> o = rel(either->getOperand(side), outV, inV);
        auto both = either->getOperand(1 - side).getDefiningOp<arith::AndIOp>();
        if (!o || *o == 0 || !both)
          continue;
        for (int k : {0, 1}) {
          std::optional<int> eq = rel(both->getOperand(k), outV, inV), t = rel(both->getOperand(1 - k), outP, inP);
          if (eq == 0 && t && *t != 0)
            order = o, tie = t;
        }
      }
    if (order)
      return Selection{v, 1 - v, inArg, inP, *order > 0, !tie || *tie < 0};
  }
  return std::nullopt;
}

/// A selection reduction seeded by its first element (how the raise starts a reduction with no identity: the
/// accumulators from element 0, the reduction over elements 1..) becomes two reductions with identities over all
/// the elements: the greatest (least) value m, then the least (greatest) payload of the elements equal to m.
/// Seeded means: the compared input is a box of some tensor T from element 1 along the reduced dim; the
/// accumulators come from one parallel generic that reads T's element 0 there and yields it and, for the payload,
/// a constant the payload's term (an index-only value of the reduced dim) takes at element 0.
void selectionReductions(ModuleOp module) {
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp g) { generics.push_back(g); });
  for (linalg::GenericOp g : generics) {
    std::optional<Selection> sel = matchSelection(g);
    if (!sel)
      continue;
    unsigned r = 0;
    for (auto [d, it] : llvm::enumerate(g.getIteratorTypesArray()))
      if (it == utils::IteratorType::reduction)
        r = d;
    OpOperand *x = g.getDpsInputOperand(sel->in.getArgNumber());
    AffineMap xMap = g.getMatchingIndexingMap(x);
    std::optional<unsigned> k = xMap.getResultPosition(getAffineDimExpr(r, g.getContext()));
    auto box = x->get().getDefiningOp<tensor::ExtractSliceOp>();
    if (!k || !box || !box.hasUnitStride() || box.getSourceType().getRank() != box.getType().getRank() ||
        ShapedType::isDynamicShape(box.getStaticOffsets()) || ShapedType::isDynamicShape(box.getStaticSizes()))
      continue;
    Value t = box.getSource();
    auto tType = cast<RankedTensorType>(t.getType());
    // A box of `t` from element `from` along dim k, `len` long, whole along the others.
    // A box of `t` (or of what `t` pads at its high ends, which holds the same leading elements) from element
    // `from` along dim k, `len` long, whole along the others.
    auto pad = t.getDefiningOp<tensor::PadOp>();
    Value unpadded = pad && llvm::all_of(pad.getStaticLow(), [](int64_t l) { return l == 0; }) ? pad.getSource() : t;
    auto isBox = [&](tensor::ExtractSliceOp s, int64_t from, int64_t len) {
      if (!s || (s.getSource() != t && s.getSource() != unpadded) || !s.hasUnitStride() ||
          s.getType().getRank() != tType.getRank())
        return false;
      for (int64_t d = 0; d < tType.getRank(); ++d)
        if (s.getStaticOffsets()[d] != (d == *k ? from : 0) || s.getStaticSizes()[d] != (d == *k ? len : tType.getDimSize(d)))
          return false;
      return true;
    };
    int64_t n = tType.getDimSize(*k);
    Value initV = g.getDpsInits()[sel->value], initP = g.getDpsInits()[sel->payload];
    auto seed = initV.getDefiningOp<linalg::GenericOp>();
    if (!isBox(box, 1, n - 1) || !seed || initP.getDefiningOp() != seed || seed.getNumDpsInputs() != 1 ||
        seed.getNumReductionLoops() != 0 || !isBox(seed.getDpsInputs()[0].getDefiningOp<tensor::ExtractSliceOp>(), 0, 1))
      continue;
    // The seed reads element 0 where the reduction reads its element: the reduced dim at 0, each other dim the
    // seed's loop the generic's output places it at.
    AffineMap outMap = g.getIndexingMapsArray()[g.getNumDpsInputs() + sel->value];
    AffineMap seedMap = seed.getIndexingMapsArray()[0];
    bool aligned = seedMap.getNumResults() == xMap.getNumResults() &&
                   llvm::all_of(ArrayRef(seed.getIndexingMapsArray()).drop_front(), [](AffineMap m) { return m.isIdentity(); });
    for (unsigned i = 0; aligned && i < xMap.getNumResults(); ++i) {
      AffineExpr e = xMap.getResult(i), s = seedMap.getResult(i);
      if (i == *k || isa<AffineConstantExpr>(e)) {
        aligned = s == (i == *k ? getAffineConstantExpr(0, g.getContext()) : e);
        continue;
      }
      auto d = dyn_cast<AffineDimExpr>(e);
      std::optional<unsigned> at = d ? outMap.getResultPosition(d) : std::nullopt;
      aligned = at && s == getAffineDimExpr(*at, g.getContext());
    }
    auto seedYield = cast<linalg::YieldOp>(seed.getBody()->getTerminator());
    unsigned vAt = cast<OpResult>(initV).getResultNumber(), pAt = cast<OpResult>(initP).getResultNumber();
    APInt seedPayload;
    if (!aligned || seedYield.getOperand(vAt) != seed.getBody()->getArgument(0) ||
        !matchPattern(seedYield.getOperand(pAt), m_ConstantInt(&seedPayload)) ||
        evalAt(sel->term, r, -1) != seedPayload.getSExtValue())
      continue;

    Type vt = getElementTypeOrSelf(initV.getType()), pt = getElementTypeOrSelf(initP.getType());
    bool fv = isa<FloatType>(vt);
    // Narrow integer values (int8) compare as i32: m is computed wide, from the i32 identity.
    Type wt = !fv && vt.getIntOrFloatBitWidth() < 32 ? Type(IntegerType::get(g.getContext(), 32)) : vt;
    bool narrow = wt != vt;
    if ((!fv && !wt.isInteger(32)) || !pt.isInteger(32) || (narrow && !g.getResult(sel->value).use_empty()))
      continue;
    OpBuilder b(g);
    Location loc = g.getLoc();
    auto fill = [&](Value like, TypedAttr value) {
      auto type = cast<RankedTensorType>(like.getType());
      Value empty = tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
      return linalg::FillOp::create(b, loc, arith::ConstantOp::create(b, loc, value).getResult(), empty).getResult(0);
    };
    auto extreme = [&](Type et, bool high) -> TypedAttr {
      if (auto ft = dyn_cast<FloatType>(et))
        return b.getFloatAttr(ft, APFloat::getInf(ft.getFloatSemantics(), !high));
      return b.getIntegerAttr(et, high ? APInt::getSignedMaxValue(32) : APInt::getSignedMinValue(32));
    };
    SmallVector<utils::IteratorType> iters = g.getIteratorTypesArray();
    // m: the greatest (least) element, from the least (greatest) value.
    auto wideType = cast<RankedTensorType>(initV.getType()).clone(wt);
    Value wideInit = fill(tensor::EmptyOp::create(b, loc, wideType.getShape(), wt), extreme(wt, !sel->max));
    auto widened = [&](OpBuilder &nb, Location l, Value x) {
      return narrow ? arith::ExtSIOp::create(nb, l, wt, x).getResult() : x;
    };
    auto value = linalg::GenericOp::create(
        b, loc, TypeRange{wideType}, ValueRange{t}, ValueRange{wideInit},
        ArrayRef<AffineMap>{xMap, outMap}, iters, [&](OpBuilder &nb, Location l, ValueRange a) {
          SmallVector<Value> args{widened(nb, l, a[0]), a[1]};
          Value m = fv ? (sel->max ? arith::MaximumFOp::create(nb, l, args[1], args[0]).getResult()
                                   : arith::MinimumFOp::create(nb, l, args[1], args[0]).getResult())
                       : (sel->max ? arith::MaxSIOp::create(nb, l, args[1], args[0]).getResult()
                                   : arith::MinSIOp::create(nb, l, args[1], args[0]).getResult());
          linalg::YieldOp::create(nb, l, m);
        });
    // The payload: the least (greatest) term of the elements equal to m. The term moves with the elements: its
    // index is one less over t than over the box.
    TypedAttr none = extreme(pt, sel->leastPayload);
    auto payload = linalg::GenericOp::create(
        b, loc, TypeRange{initP.getType()}, ValueRange{t, value.getResult(0)}, ValueRange{fill(initP, none)},
        ArrayRef<AffineMap>{xMap, outMap, g.getIndexingMapsArray()[g.getNumDpsInputs() + sel->payload]}, iters,
        [&](OpBuilder &nb, Location l, ValueRange args) {
          IRMapping map;
          Value index = linalg::IndexOp::create(nb, l, r);
          Value before = arith::SubIOp::create(nb, l, index, arith::ConstantIndexOp::create(nb, l, 1));
          std::function<Value(Value)> clone = [&](Value v) -> Value {
            if (Value m = map.lookupOrNull(v))
              return m;
            Operation *op = v.getDefiningOp();
            if (auto i = dyn_cast<linalg::IndexOp>(op); i && i.getDim() == r) {
              map.map(v, before);
              return before;
            }
            for (Value o : op->getOperands())
              clone(o);
            return nb.clone(*op, map)->getResult(cast<OpResult>(v).getResultNumber());
          };
          Value term = clone(sel->term);
          Value x = widened(nb, l, args[0]);
          Value eq = fv ? arith::CmpFOp::create(nb, l, arith::CmpFPredicate::OEQ, x, args[1]).getResult()
                        : arith::CmpIOp::create(nb, l, arith::CmpIPredicate::eq, x, args[1]).getResult();
          Value candidate = arith::SelectOp::create(nb, l, eq, term, arith::ConstantOp::create(nb, l, none));
          Value p = sel->leastPayload ? arith::MinSIOp::create(nb, l, args[2], candidate).getResult()
                                      : arith::MaxSIOp::create(nb, l, args[2], candidate).getResult();
          linalg::YieldOp::create(nb, l, p);
        });
    if (!narrow)
      g.getResult(sel->value).replaceAllUsesWith(value.getResult(0));
    g.getResult(sel->payload).replaceAllUsesWith(payload.getResult(0));
    g.erase();
  }
}

/// A reduction whose combiner the Vector Engine cannot reduce (products, bitwise xor/and/or) becomes a tree of
/// elementwise ops: its terms over the whole iteration space, then the two halves along the reduced dim combined
/// until one element is left, combined with the init last. The combiner is associative and commutative, so the
/// order does not matter (beyond rounding, for products).
void treeReductions(ModuleOp module) {
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp g) { generics.push_back(g); });
  for (linalg::GenericOp g : generics) {
    if (g.getNumDpsInits() != 1 || g.getNumReductionLoops() != 1)
      continue;
    Block &body = *g.getBody();
    Value out = body.getArgument(g.getNumDpsInputs());
    Operation *comb = cast<linalg::YieldOp>(body.getTerminator()).getOperand(0).getDefiningOp();
    if (!comb || !isa<arith::MulFOp, arith::MulIOp, arith::XOrIOp, arith::AndIOp, arith::OrIOp>(comb) ||
        !out.hasOneUse() || (comb->getOperand(0) != out && comb->getOperand(1) != out))
      continue;
    Value term = comb->getOperand(0) == out ? comb->getOperand(1) : comb->getOperand(0);
    unsigned r = 0;
    for (auto [d, it] : llvm::enumerate(g.getIteratorTypesArray()))
      if (it == utils::IteratorType::reduction)
        r = d;
    SmallVector<int64_t> extents = g.getStaticLoopRanges();
    // shortcut: halves need a power-of-two extent; pad with the combiner's identity to take other extents.
    if (llvm::any_of(extents, ShapedType::isDynamic) || !llvm::isPowerOf2_64(extents[r]))
      continue;
    AffineMap outMap = g.getIndexingMapsArray().back();
    OpBuilder b(g);
    Location loc = g.getLoc();
    MLIRContext *ctx = g.getContext();
    Type et = getElementTypeOrSelf(g.getResult(0).getType());
    unsigned rank = extents.size();
    SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
    auto combined = [&](OpBuilder &nb, Location l, Value a, Value c) -> Value {
      IRMapping m;
      m.map(comb->getOperand(0), a);
      m.map(comb->getOperand(1), c);
      return nb.clone(*comb, m)->getResult(0);
    };
    // The terms, one per point of the iteration space (laid out in loop order).
    SmallVector<AffineMap> maps(g.getIndexingMapsArray());
    maps.back() = AffineMap::getMultiDimIdentityMap(rank, ctx);
    Value cur = tensor::EmptyOp::create(b, loc, extents, et);
    auto terms = linalg::GenericOp::create(
        b, loc, TypeRange{cur.getType()}, g.getDpsInputs(), ValueRange{cur}, maps, parallel,
        [&](OpBuilder &nb, Location l, ValueRange args) {
          IRMapping m;
          for (unsigned i = 0; i < g.getNumDpsInputs(); ++i)
            m.map(body.getArgument(i), args[i]);
          for (Operation &op : body.without_terminator())
            if (&op != comb)
              nb.clone(op, m);
          linalg::YieldOp::create(nb, l, m.lookupOrDefault(term));
        });
    cur = terms.getResult(0);
    // Halves combined until one element is left along the reduced dim.
    AffineMap id = AffineMap::getMultiDimIdentityMap(rank, ctx);
    for (int64_t len = extents[r]; len > 1; len /= 2) {
      SmallVector<int64_t> half(extents), at(rank, 0), ones(rank, 1);
      half[r] = len / 2;
      auto box = [&]() {
        return tensor::ExtractSliceOp::create(b, loc, cur, getAsIndexOpFoldResult(ctx, at),
                                              getAsIndexOpFoldResult(ctx, half), getAsIndexOpFoldResult(ctx, ones))
            .getResult();
      };
      Value lo = box();
      at[r] = len / 2;
      Value hi = box();
      Value into = tensor::EmptyOp::create(b, loc, half, et);
      cur = linalg::GenericOp::create(b, loc, TypeRange{into.getType()}, ValueRange{lo, hi}, ValueRange{into},
                                      ArrayRef<AffineMap>{id, id, id}, parallel,
                                      [&](OpBuilder &nb, Location l, ValueRange args) {
                                        linalg::YieldOp::create(nb, l, combined(nb, l, args[0], args[1]));
                                      })
                .getResult(0);
    }
    // The total, combined with the init, over the output's dims.
    unsigned outRank = outMap.getNumResults();
    SmallVector<AffineExpr> read;
    for (unsigned d = 0; d < rank; ++d) {
      std::optional<unsigned> at = outMap.getResultPosition(getAffineDimExpr(d, ctx));
      read.push_back(d == r ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(at.value_or(0), ctx));
    }
    Value init = g.getDpsInits()[0];
    Value fresh = tensor::EmptyOp::create(b, loc, cast<RankedTensorType>(init.getType()).getShape(), et);
    AffineMap outId = AffineMap::getMultiDimIdentityMap(outRank, ctx);
    auto total = linalg::GenericOp::create(
        b, loc, TypeRange{init.getType()}, ValueRange{cur, init}, ValueRange{fresh},
        ArrayRef<AffineMap>{AffineMap::get(outRank, 0, read, ctx), outId, outId},
        SmallVector<utils::IteratorType>(outRank, utils::IteratorType::parallel),
        [&](OpBuilder &nb, Location l, ValueRange args) {
          linalg::YieldOp::create(nb, l, combined(nb, l, args[0], args[1]));
        });
    g.getResult(0).replaceAllUsesWith(total.getResult(0));
    g.erase();
  }
}

/// Math ops the Vector Engine lacks, in ones it has: asin, acos, atan and atan2 by MLIR's polynomial
/// approximations (whose fma and copysign expand below), copysign(a, b) = b < 0 ? -|a| : |a|, fma(a, b, c) =
/// a*b + c (rounded twice), log2(x) = ln(x) * log2(e) (within the engine's ln ulp budget, not bit-exact), and
/// floor/ceil from the nearest integer t (the engine converts to integers rounding to nearest): floor =
/// t - (t > x), ceil = t + (t < x), exact for any integer t within 1 of x; values beyond 2^(mantissa bits) are
/// integers already and pass through (the conversion would saturate).
void expandMath(ModuleOp module) {
  RewritePatternSet approximations(module.getContext());
  populateMathPolynomialApproximationPatterns(approximations, [](StringRef name) {
    return llvm::is_contained({math::AsinOp::getOperationName(), math::AcosOp::getOperationName(),
                               math::AtanOp::getOperationName(), math::Atan2Op::getOperationName()},
                              name);
  });
  (void)applyPatternsGreedily(module, std::move(approximations));
  SmallVector<Operation *> ops;
  module.walk([&](Operation *op) {
    if (isa<math::FmaOp, math::Log2Op, math::FloorOp, math::CeilOp, math::CopySignOp>(op) &&
        isa<FloatType>(op->getResult(0).getType()))
      ops.push_back(op);
  });
  for (Operation *op : ops) {
    OpBuilder b(op);
    Location loc = op->getLoc();
    auto ft = cast<FloatType>(op->getResult(0).getType());
    auto constant = [&](double v) { return arith::ConstantOp::create(b, loc, b.getFloatAttr(ft, v)).getResult(); };
    Value v, x = op->getOperand(0);
    if (auto fma = dyn_cast<math::FmaOp>(op)) {
      v = arith::AddFOp::create(b, loc, arith::MulFOp::create(b, loc, fma.getA(), fma.getB()), fma.getC());
    } else if (auto sign = dyn_cast<math::CopySignOp>(op)) {
      // shortcut: the sign is b < 0, so b = -0 (or a NaN with its sign bit set) gives +|a|; compare the sign bit
      // if such b matter.
      Value mag = math::AbsFOp::create(b, loc, x);
      Value neg = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, sign.getRhs(), constant(0.0));
      v = arith::SelectOp::create(b, loc, neg, arith::NegFOp::create(b, loc, mag), mag);
    } else if (isa<math::Log2Op>(op)) {
      v = arith::MulFOp::create(b, loc, math::LogOp::create(b, loc, x), constant(1.4426950408889634));
    } else {
      bool floor = isa<math::FloorOp>(op);
      using P = arith::CmpFPredicate;
      Value t = math::RoundOp::create(b, loc, x);
      // shortcut: the engine compares t with x through t - x, which flushes subnormals, so floor of a negative
      // subnormal gives -0 (not -1); compare the operands themselves if such inputs matter.
      Value past = arith::CmpFOp::create(b, loc, floor ? P::OGT : P::OLT, t, x);
      Value step = arith::SelectOp::create(b, loc, past, constant(1.0), constant(0.0));
      Value near = floor ? arith::SubFOp::create(b, loc, t, step).getResult()
                         : arith::AddFOp::create(b, loc, t, step).getResult();
      double integral = std::ldexp(1.0, int(APFloat::semanticsPrecision(ft.getFloatSemantics())) - 1);
      Value low = arith::CmpFOp::create(b, loc, P::OLT, x, constant(-integral));
      Value high = arith::CmpFOp::create(b, loc, P::OGT, x, constant(integral));
      v = arith::SelectOp::create(b, loc, high, x, arith::SelectOp::create(b, loc, low, x, near));
    }
    op->getResult(0).replaceAllUsesWith(v);
    op->erase();
  }
}

} // namespace

void mlir::furiosa::normalizeKernels(ModuleOp module) {
  expandMath(module);
  selectionReductions(module);
  treeReductions(module);
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

/// A padded tensor that only boxes are cut from (structure's masked tiles) is the source inserted into a constant
/// of the padding value, which the copy steps map: one copy into the padded tensor, whose extent the boxes
/// then divide (the NPU tiles an axis only in windows that grow into a divisor of it).
/// Whether `v` is only cut into boxes (extract_slices, possibly of reshapes of it).
static bool onlyBoxed(Value v) {
  return llvm::all_of(v.getUsers(), [](Operation *u) {
    return isa<tensor::ExtractSliceOp>(u) ||
           (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(u) && onlyBoxed(u->getResult(0)));
  });
}

void mlir::furiosa::padsToInserts(ModuleOp module) {
  SmallVector<tensor::PadOp> pads;
  module.walk([&](tensor::PadOp pad) {
    Attribute c;
    Value padding = pad.getConstantPaddingValue();
    if (padding && matchPattern(padding, m_Constant(&c)) && !ShapedType::isDynamicShape(pad.getStaticLow()) &&
        pad.getResultType().hasStaticShape() && !pad.getResult().use_empty() &&
        onlyBoxed(pad.getResult()))
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

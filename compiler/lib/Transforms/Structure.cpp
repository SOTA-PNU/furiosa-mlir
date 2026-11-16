// furiosa-structure: gather-form raised linalg -> structured linalg (compiler/README.md).
//
// Raising reads buffers with tensor.extract at addresses computed from linalg.index. This pass turns every
// such read whose address is affine in the loop indices into an ordinary input with an indexing map:
//
//  1. The index arithmetic (constants, linalg.index, add/sub/mul by a constant, floor/ceil div and the
//     mod idiom by a constant) becomes an AffineExpr over the generic's loop dims.
//  2. A flat address into a collapsed buffer is split back over the buffer's declared shape: each term goes
//     to the dim whose stride is the largest one dividing its coefficient. The split is kept only if every
//     index stays inside its dim over the loop ranges; otherwise the read stays flat.
//  3. A masked read (select(mask, extract(clamp(addr)), other)) becomes a read of the buffer padded with
//     `other`, when the mask is exactly "every index inside its dim"; any other mask keeps the select.
//  4. Upstream patterns then drop operands no longer read (the reduction-extent carrier), a generic that
//     only yields a constant becomes linalg.fill, two reduction dims always addressed together as a*B+b
//     merge into one, and CSE/canonicalization tidy the rest.
//
// Reads that are not affine (data-dependent gathers) stay as tensor.extract.

#include "furiosa/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <functional>
#include <optional>

namespace mlir::furiosa {
namespace {

struct Range {
  int64_t lo, hi; // inclusive
};

int64_t floorDiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

Range bounds(AffineExpr e, ArrayRef<Range> dims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return {c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return dims[d.getPosition()];
  auto bin = cast<AffineBinaryOpExpr>(e);
  Range l = bounds(bin.getLHS(), dims), r = bounds(bin.getRHS(), dims);
  switch (e.getKind()) {
  case AffineExprKind::Add:
    return {l.lo + r.lo, l.hi + r.hi};
  case AffineExprKind::Mul: {
    int64_t c[4] = {l.lo * r.lo, l.lo * r.hi, l.hi * r.lo, l.hi * r.hi};
    return {*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
  }
  case AffineExprKind::FloorDiv:
    return {floorDiv(l.lo, r.lo), floorDiv(l.hi, r.lo)};
  case AffineExprKind::CeilDiv:
    return {-floorDiv(-l.lo, r.lo), -floorDiv(-l.hi, r.lo)};
  case AffineExprKind::Mod:
    return (l.lo >= 0 && l.hi < r.lo) ? l : Range{0, r.lo - 1};
  default:
    llvm_unreachable("affine kind");
  }
}

/// The index arithmetic of a generic body as an AffineExpr over its loop dims, or nullopt.
std::optional<AffineExpr> toAffine(Value v, MLIRContext *ctx) {
  if (auto c = getConstantIntValue(v))
    return getAffineConstantExpr(*c, ctx);
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto ix = dyn_cast<linalg::IndexOp>(op))
    return getAffineDimExpr(ix.getDim(), ctx);
  auto binary = [&](auto fn) -> std::optional<AffineExpr> {
    auto l = toAffine(op->getOperand(0), ctx), r = toAffine(op->getOperand(1), ctx);
    if (!l || !r)
      return std::nullopt;
    return fn(*l, *r);
  };
  auto byConstant = [](AffineExpr r) -> std::optional<int64_t> {
    if (auto c = dyn_cast<AffineConstantExpr>(r); c && c.getValue() > 0)
      return c.getValue();
    return std::nullopt;
  };
  if (isa<arith::AddIOp>(op))
    return binary([](AffineExpr l, AffineExpr r) -> std::optional<AffineExpr> { return l + r; });
  if (isa<arith::SubIOp>(op))
    return binary([](AffineExpr l, AffineExpr r) -> std::optional<AffineExpr> { return l - r; });
  if (isa<arith::MulIOp>(op))
    return binary([](AffineExpr l, AffineExpr r) -> std::optional<AffineExpr> {
      if (isa<AffineConstantExpr>(l) || isa<AffineConstantExpr>(r))
        return l * r;
      return std::nullopt;
    });
  if (isa<arith::FloorDivSIOp>(op))
    return binary([&](AffineExpr l, AffineExpr r) -> std::optional<AffineExpr> {
      if (auto c = byConstant(r))
        return l.floorDiv(*c);
      return std::nullopt;
    });
  if (isa<arith::CeilDivSIOp>(op))
    return binary([&](AffineExpr l, AffineExpr r) -> std::optional<AffineExpr> {
      if (auto c = byConstant(r))
        return l.ceilDiv(*c);
      return std::nullopt;
    });
  // The mod idiom: select(remsi(a, r) < 0, remsi(a, r) + r, remsi(a, r)).
  if (auto sel = dyn_cast<arith::SelectOp>(op)) {
    auto rem = sel.getFalseValue().getDefiningOp<arith::RemSIOp>();
    auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
    auto add = sel.getTrueValue().getDefiningOp<arith::AddIOp>();
    if (rem && cmp && add && cmp.getPredicate() == arith::CmpIPredicate::slt && cmp.getLhs() == rem.getResult() &&
        matchPattern(cmp.getRhs(), m_Zero()) && add.getLhs() == rem.getResult() && add.getRhs() == rem.getRhs()) {
      auto a = toAffine(rem.getLhs(), ctx), r = toAffine(rem.getRhs(), ctx);
      if (a && r)
        if (auto c = byConstant(*r))
          return *a % *c;
    }
  }
  return std::nullopt;
}

/// Coefficients of a linear expr's terms, with the non-linear sub-exprs as atoms.
void terms(AffineExpr e, int64_t scale, SmallVectorImpl<std::pair<AffineExpr, int64_t>> &out, int64_t &constant) {
  if (auto c = dyn_cast<AffineConstantExpr>(e)) {
    constant += scale * c.getValue();
    return;
  }
  if (e.getKind() == AffineExprKind::Add) {
    auto bin = cast<AffineBinaryOpExpr>(e);
    terms(bin.getLHS(), scale, out, constant);
    terms(bin.getRHS(), scale, out, constant);
    return;
  }
  if (e.getKind() == AffineExprKind::Mul) {
    auto bin = cast<AffineBinaryOpExpr>(e);
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
      return terms(bin.getLHS(), scale * c.getValue(), out, constant);
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS()))
      return terms(bin.getRHS(), scale * c.getValue(), out, constant);
  }
  out.push_back({e, scale});
}

/// Splits a flat address over a row-major shape: each term goes to the dim with the largest stride dividing
/// its coefficient; the constant is split greedily from the outermost dim (truncating toward zero, so small
/// negative offsets such as padding stay in their own dim).
SmallVector<AffineExpr> split(AffineExpr flat, ArrayRef<int64_t> shape, unsigned numDims) {
  MLIRContext *ctx = flat.getContext();
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  SmallVector<std::pair<AffineExpr, int64_t>> ts;
  int64_t constant = 0;
  terms(flat, 1, ts, constant);
  SmallVector<AffineExpr> idx(shape.size(), getAffineConstantExpr(0, ctx));
  for (auto [atom, coeff] : ts) {
    unsigned j = 0;
    while (j + 1 < shape.size() && coeff % strides[j] != 0)
      ++j;
    idx[j] = idx[j] + atom * (coeff / strides[j]);
  }
  for (unsigned j = 0; j < shape.size(); ++j) {
    int64_t q = constant / strides[j]; // toward zero
    idx[j] = idx[j] + q;
    constant -= q * strides[j];
  }
  for (AffineExpr &e : idx)
    e = simplifyAffineExpr(e, numDims, 0);
  return idx;
}

/// The conditions of a mask built from `cmpi sge, x, 0` and `andi`, or nullopt.
std::optional<SmallVector<AffineExpr>> maskConditions(Value v, MLIRContext *ctx) {
  if (auto c = getConstantIntValue(v)) {
    if (*c)
      return SmallVector<AffineExpr>{};
    return std::nullopt;
  }
  if (auto andi = v.getDefiningOp<arith::AndIOp>()) {
    auto l = maskConditions(andi.getLhs(), ctx), r = maskConditions(andi.getRhs(), ctx);
    if (!l || !r)
      return std::nullopt;
    l->append(r->begin(), r->end());
    return l;
  }
  if (auto cmp = v.getDefiningOp<arith::CmpIOp>()) {
    if (cmp.getPredicate() != arith::CmpIPredicate::sge || !matchPattern(cmp.getRhs(), m_Zero()))
      return std::nullopt;
    if (auto e = toAffine(cmp.getLhs(), ctx))
      return SmallVector<AffineExpr>{*e};
  }
  return std::nullopt;
}

/// One read to promote: its replacement value in the body is a new input of `source` through `map`.
struct Promotion {
  Operation *replaced; // the extract, or the select of a masked read
  Value source;
  AffineMap map;
};

/// Plans the promotion of `extract`, or returns nullopt to leave it.
std::optional<Promotion> plan(tensor::ExtractOp extract, linalg::GenericOp generic, ArrayRef<Range> ranges,
                              OpBuilder &b) {
  MLIRContext *ctx = extract.getContext();
  Value source = extract.getTensor();
  if (generic->isAncestor(source.getParentBlock()->getParentOp()) && source.getParentBlock() != generic->getBlock())
    return std::nullopt; // defined inside the body
  unsigned nd = ranges.size();

  // A masked read: select(mask, extract(min(max(addr, 0), n - 1)), other).
  Value address = extract.getIndices().size() == 1 ? extract.getIndices()[0] : Value();
  arith::SelectOp select;
  std::optional<SmallVector<AffineExpr>> conds;
  Attribute other;
  if (address && extract->hasOneUse())
    if (auto sel = dyn_cast<arith::SelectOp>(*extract->getUsers().begin());
        sel && sel.getTrueValue() == extract.getResult()) {
      auto mn = address.getDefiningOp<arith::MinSIOp>();
      auto mx = mn ? mn.getLhs().getDefiningOp<arith::MaxSIOp>() : arith::MaxSIOp();
      if (mx && matchPattern(mx.getRhs(), m_Zero()) && matchPattern(sel.getFalseValue(), m_Constant(&other)) &&
          (conds = maskConditions(sel.getCondition(), ctx))) {
        select = sel;
        address = mx.getLhs();
      }
    }

  // The indices over the source's own shape, unfolding a full collapse to 1-D.
  SmallVector<AffineExpr> idx;
  Value base = source;
  if (address) {
    auto flat = toAffine(address, ctx);
    if (!flat)
      return std::nullopt;
    auto collapse = source.getDefiningOp<tensor::CollapseShapeOp>();
    if (collapse && cast<RankedTensorType>(collapse.getSrc().getType()).getRank() > 1) {
      auto shape = cast<RankedTensorType>(collapse.getSrc().getType()).getShape();
      SmallVector<AffineExpr> multi = split(*flat, shape, nd);
      bool fits = true;
      for (auto [e, s] : llvm::zip(multi, shape)) {
        Range r = bounds(e, ranges);
        fits &= select ? true : (r.lo >= 0 && r.hi < s);
      }
      if (fits) {
        idx = multi;
        base = collapse.getSrc();
      }
    }
    if (idx.empty())
      idx.push_back(*flat);
  } else {
    for (Value i : extract.getIndices()) {
      auto e = toAffine(i, ctx);
      if (!e)
        return std::nullopt;
      idx.push_back(*e);
    }
  }
  auto baseType = cast<RankedTensorType>(base.getType());

  if (!select) {
    for (auto [e, s] : llvm::zip(idx, baseType.getShape())) {
      Range r = bounds(e, ranges);
      if (r.lo < 0 || r.hi >= s)
        return std::nullopt; // a clamp or wrap we did not see; leave the read
    }
    return Promotion{extract, base, AffineMap::get(nd, 0, idx, ctx)};
  }

  // The mask must say exactly "every index inside its dim": each live condition is one of the bounds the
  // indices can break, and each such bound is in the mask.
  SmallVector<AffineExpr> needed;
  SmallVector<int64_t> low(idx.size(), 0), high(idx.size(), 0);
  for (auto [j, e] : llvm::enumerate(idx)) {
    Range r = bounds(e, ranges);
    int64_t s = baseType.getDimSize(j);
    if (r.lo < 0) {
      needed.push_back(simplifyAffineExpr(e, nd, 0));
      low[j] = -r.lo;
    }
    if (r.hi >= s) {
      needed.push_back(simplifyAffineExpr(getAffineConstantExpr(s - 1, ctx) - e, nd, 0));
      high[j] = r.hi - (s - 1);
    }
  }
  SmallVector<AffineExpr> live;
  for (AffineExpr c : *conds)
    if (bounds(c, ranges).lo < 0)
      live.push_back(simplifyAffineExpr(c, nd, 0));
  for (AffineExpr c : live)
    if (!llvm::is_contained(needed, c))
      return std::nullopt;
  for (AffineExpr n : needed)
    if (!llvm::is_contained(live, n))
      return std::nullopt;

  Value padded = base;
  if (!needed.empty()) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(generic);
    SmallVector<int64_t> shape(baseType.getShape());
    for (unsigned j = 0; j < shape.size(); ++j)
      shape[j] += low[j] + high[j];
    Value pad = arith::ConstantOp::create(b, generic.getLoc(), cast<TypedAttr>(other));
    padded = tensor::PadOp::create(b, generic.getLoc(), RankedTensorType::get(shape, baseType.getElementType()), base,
                                   getAsIndexOpFoldResult(ctx, low), getAsIndexOpFoldResult(ctx, high), pad);
  }
  SmallVector<AffineExpr> shifted;
  for (auto [j, e] : llvm::enumerate(idx))
    shifted.push_back(simplifyAffineExpr(e + low[j], nd, 0));
  return Promotion{select, padded, AffineMap::get(nd, 0, shifted, ctx)};
}

/// Rebuilds `generic` with one more input per promotion.
linalg::GenericOp promote(linalg::GenericOp generic, ArrayRef<Promotion> promotions, OpBuilder &b) {
  SmallVector<Value> inputs(generic.getDpsInputs());
  SmallVector<AffineMap> maps;
  for (OpOperand *in : generic.getDpsInputOperands())
    maps.push_back(generic.getMatchingIndexingMap(in));
  for (const Promotion &p : promotions) {
    inputs.push_back(p.source);
    maps.push_back(p.map);
  }
  for (OpOperand &out : generic.getDpsInitsMutable())
    maps.push_back(generic.getMatchingIndexingMap(&out));
  b.setInsertionPoint(generic);
  auto updated = linalg::GenericOp::create(b, generic.getLoc(), generic.getResultTypes(), inputs,
                                           generic.getDpsInits(), maps, generic.getIteratorTypesArray());
  updated.getRegion().takeBody(generic.getRegion());
  Block &body = updated.getRegion().front();
  unsigned at = generic.getNumDpsInputs();
  for (auto [k, p] : llvm::enumerate(promotions)) {
    Type et = cast<RankedTensorType>(p.source.getType()).getElementType();
    BlockArgument arg = body.insertArgument(at + k, et, generic.getLoc());
    p.replaced->getResult(0).replaceAllUsesWith(arg);
  }
  for (const Promotion &p : promotions) {
    Operation *op = p.replaced;
    SmallVector<Operation *> dead{op};
    if (auto sel = dyn_cast<arith::SelectOp>(op))
      dead.push_back(sel.getTrueValue().getDefiningOp());
    for (Operation *d : dead)
      if (d && d->use_empty())
        d->erase();
  }
  generic.replaceAllUsesWith(updated.getResults());
  generic.erase();
  return updated;
}

/// Merges reduction dims a and a+1 (extents A, B) into one of extent A*B when every map reads them only as
/// a*B + b and the body does not read them.
bool mergeReductionDims(linalg::GenericOp &generic, OpBuilder &b) {
  MLIRContext *ctx = generic.getContext();
  auto iters = generic.getIteratorTypesArray();
  SmallVector<int64_t> extents = generic.getStaticLoopRanges();
  unsigned nd = iters.size();
  for (unsigned a = 0; a + 1 < nd; ++a) {
    if (iters[a] != utils::IteratorType::reduction || iters[a + 1] != utils::IteratorType::reduction)
      continue;
    bool readsIndex = false;
    generic.getRegion().walk([&](linalg::IndexOp ix) { readsIndex |= ix.getDim() == a || ix.getDim() == a + 1; });
    if (readsIndex)
      continue;
    int64_t bExt = extents[a + 1];
    // New dims: 0..a-1, k = a, then the old a+2.. shifted down by one.
    SmallVector<AffineExpr> sub;
    for (unsigned d = 0; d < nd; ++d) {
      if (d < a)
        sub.push_back(getAffineDimExpr(d, ctx));
      else if (d == a)
        sub.push_back(getAffineDimExpr(a, ctx).floorDiv(bExt));
      else if (d == a + 1)
        sub.push_back(getAffineDimExpr(a, ctx) % bExt);
      else
        sub.push_back(getAffineDimExpr(d - 1, ctx));
    }
    // Inputs the body never reads (the extent carrier) are dropped with the merge.
    Block &body = generic.getRegion().front();
    SmallVector<Value> inputs;
    SmallVector<unsigned> keptArgs;
    SmallVector<AffineMap> oldMaps;
    for (OpOperand *in : generic.getDpsInputOperands())
      if (!body.getArgument(in->getOperandNumber()).use_empty()) {
        inputs.push_back(in->get());
        keptArgs.push_back(in->getOperandNumber());
        oldMaps.push_back(generic.getMatchingIndexingMap(in));
      }
    for (OpOperand &out : generic.getDpsInitsMutable())
      oldMaps.push_back(generic.getMatchingIndexingMap(&out));
    SmallVector<AffineMap> maps;
    bool clean = true;
    for (AffineMap m : oldMaps) {
      SmallVector<AffineExpr> results;
      for (AffineExpr e : m.getResults()) {
        AffineExpr r = simplifyAffineExpr(e.replaceDims(sub), nd - 1, 0);
        r.walk([&](AffineExpr x) {
          if (x.getKind() == AffineExprKind::FloorDiv || x.getKind() == AffineExprKind::Mod)
            if (cast<AffineBinaryOpExpr>(x).getLHS().isFunctionOfDim(a))
              clean = false;
        });
        results.push_back(r);
      }
      maps.push_back(AffineMap::get(nd - 1, 0, results, ctx));
    }
    if (!clean)
      continue;
    SmallVector<utils::IteratorType> newIters(iters);
    newIters.erase(newIters.begin() + a + 1);
    // linalg infers loop ranges by inverting the concatenated maps; the merge must keep that possible.
    if (!inversePermutation(concatAffineMaps(maps, ctx)))
      continue;
    b.setInsertionPoint(generic);
    auto merged = linalg::GenericOp::create(b, generic.getLoc(), generic.getResultTypes(), inputs,
                                            generic.getDpsInits(), maps, newIters);
    merged.getRegion().takeBody(generic.getRegion());
    Block &mergedBody = merged.getRegion().front();
    for (int i = generic.getNumDpsInputs() - 1; i >= 0; --i)
      if (!llvm::is_contained(keptArgs, unsigned(i)))
        mergedBody.eraseArgument(i);
    // Body indices above the merged pair move down by one.
    merged.getRegion().walk([&](linalg::IndexOp ix) {
      if (ix.getDim() > a + 1)
        ix.setDim(ix.getDim() - 1);
    });
    generic.replaceAllUsesWith(merged.getResults());
    generic.erase();
    generic = merged;
    return true;
  }
  return false;
}

struct StructurePass : PassWrapper<StructurePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StructurePass)

  StringRef getArgument() const override { return "furiosa-structure"; }
  StringRef getDescription() const override {
    return "Promote affine tensor.extract reads of gather-form linalg to indexing-map inputs";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registerDialects(registry); }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder b(ctx);
    SmallVector<linalg::GenericOp> generics;
    module.walk([&](linalg::GenericOp g) { generics.push_back(g); });
    for (linalg::GenericOp generic : generics) {
      SmallVector<Range> ranges;
      for (int64_t e : generic.getStaticLoopRanges())
        ranges.push_back({0, e - 1});
      SmallVector<Promotion> promotions;
      generic.getRegion().walk([&](tensor::ExtractOp extract) {
        if (auto p = plan(extract, generic, ranges, b))
          promotions.push_back(*p);
      });
      if (!promotions.empty())
        generic = promote(generic, promotions, b);
    }

    // Upstream cleanups: unread operands (the extent carrier), dead index arithmetic, duplicates.
    RewritePatternSet patterns(ctx);
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    linalg::GenericOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();

    // Constant-yielding generics become linalg.fill; reduction dims addressed together merge.
    generics.clear();
    module.walk([&](linalg::GenericOp g) { generics.push_back(g); });
    for (linalg::GenericOp generic : generics) {
      if (auto value = linalg::isaFillOpInterface(generic)) {
        b.setInsertionPoint(generic);
        Value v = *value;
        if (v.getParentBlock() == &generic.getRegion().front()) {
          if (auto c = v.getDefiningOp<arith::ConstantOp>())
            v = arith::ConstantOp::create(b, generic.getLoc(), c.getValue());
          else
            continue;
        }
        auto fill = linalg::FillOp::create(b, generic.getLoc(), ValueRange{v}, generic.getDpsInits());
        generic.replaceAllUsesWith(fill.getResults());
        generic.erase();
        continue;
      }
      while (mergeReductionDims(generic, b)) {
      }
    }

    OpPassManager pm(ModuleOp::getOperationName());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createStructurePass() { return std::make_unique<StructurePass>(); }

} // namespace mlir::furiosa

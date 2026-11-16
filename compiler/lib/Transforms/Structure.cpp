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

#include "llvm/ADT/MapVector.h"

#include <functional>
#include <map>
#include <set>
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

/// An affine expression as coefficients of atoms (dims, digits, other non-linear sub-expressions) plus a constant.
struct Form {
  llvm::MapVector<AffineExpr, int64_t> terms;
  int64_t constant = 0;

  void add(const Form &o, int64_t scale) {
    for (auto [atom, c] : o.terms)
      terms[atom] += c * scale;
    constant += o.constant * scale;
  }
  bool operator==(const Form &o) const {
    auto live = [](const Form &f) {
      llvm::DenseMap<AffineExpr, int64_t> m;
      for (auto [a, c] : f.terms)
        if (c)
          m[a] = c;
      return m;
    };
    return constant == o.constant && live(*this) == live(o);
  }
  AffineExpr expr(MLIRContext *ctx) const {
    AffineExpr e = getAffineConstantExpr(constant, ctx);
    for (auto [atom, c] : terms)
      if (c)
        e = e + atom * c;
    return e;
  }
};

/// The dims an expression mentions.
std::set<unsigned> dimsOf(AffineExpr e) {
  std::set<unsigned> out;
  e.walk([&](AffineExpr x) {
    if (auto d = dyn_cast<AffineDimExpr>(x))
      out.insert(d.getPosition());
  });
  return out;
}

/// The divisors each dim is divided or reduced by in `exprs` (x floordiv c, x mod c with x mentioning that dim
/// alone), when they nest.
std::map<unsigned, SmallVector<int64_t>> divisors(ArrayRef<AffineExpr> exprs) {
  std::map<unsigned, std::set<int64_t>> found;
  for (AffineExpr e : exprs)
    e.walk([&](AffineExpr x) {
      if (x.getKind() != AffineExprKind::FloorDiv && x.getKind() != AffineExprKind::Mod)
        return;
      auto bin = cast<AffineBinaryOpExpr>(x);
      auto c = dyn_cast<AffineConstantExpr>(bin.getRHS());
      std::set<unsigned> dims = dimsOf(bin.getLHS());
      if (c && c.getValue() > 1 && dims.size() == 1)
        found[*dims.begin()].insert(c.getValue());
    });
  std::map<unsigned, SmallVector<int64_t>> out;
  for (auto &[d, cs] : found) {
    SmallVector<int64_t> chain(cs.begin(), cs.end());
    bool nests = true;
    for (unsigned i = 1; i < chain.size(); ++i)
      nests &= chain[i] % chain[i - 1] == 0;
    if (nests)
      out[d] = chain;
  }
  return out;
}

/// `e` as a Form whose atoms are mixed-radix digits for every dim in `digits`: with divisors c1 < ... < cK,
/// digit 0 = d mod c1, digit k = (d floordiv ck) mod (ck+1 / ck), digit K = d floordiv cK, so d is the sum of its
/// digits times their place values. A floordiv or mod by a divisor of a sum of one dim's digits at their place
/// values (d itself, d mod c, d floordiv c, ...) keeps the digits above or below that place.
Form linearForm(AffineExpr e, const std::map<unsigned, SmallVector<int64_t>> &digits) {
  MLIRContext *ctx = e.getContext();
  llvm::DenseMap<AffineExpr, std::pair<unsigned, int64_t>> place; // digit -> (dim, place value)
  std::map<unsigned, SmallVector<std::pair<AffineExpr, int64_t>>> table;
  for (auto &[dim, cs] : digits) {
    AffineExpr d = getAffineDimExpr(dim, ctx);
    for (unsigned k = 0; k <= cs.size(); ++k) {
      int64_t value = k == 0 ? 1 : cs[k - 1];
      AffineExpr digit = k == 0 ? d % cs[0]
                         : k == cs.size() ? d.floorDiv(cs[k - 1])
                                          : d.floorDiv(cs[k - 1]) % (cs[k] / cs[k - 1]);
      table[dim].push_back({digit, value});
      place[digit] = {dim, value};
    }
  }
  std::function<Form(AffineExpr)> walk = [&](AffineExpr x) -> Form {
    Form f;
    if (auto c = dyn_cast<AffineConstantExpr>(x)) {
      f.constant = c.getValue();
      return f;
    }
    if (auto d = dyn_cast<AffineDimExpr>(x)) {
      auto it = table.find(d.getPosition());
      if (it == table.end()) {
        f.terms[x] = 1;
        return f;
      }
      for (auto [digit, value] : it->second)
        f.terms[digit] += value;
      return f;
    }
    auto bin = cast<AffineBinaryOpExpr>(x);
    if (x.getKind() == AffineExprKind::Add) {
      f = walk(bin.getLHS());
      f.add(walk(bin.getRHS()), 1);
      return f;
    }
    if (x.getKind() == AffineExprKind::Mul) {
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
        f.add(walk(bin.getLHS()), c.getValue());
        return f;
      }
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
        f.add(walk(bin.getRHS()), c.getValue());
        return f;
      }
    }
    if (x.getKind() == AffineExprKind::FloorDiv || x.getKind() == AffineExprKind::Mod) {
      auto c = dyn_cast<AffineConstantExpr>(bin.getRHS());
      Form inner = walk(bin.getLHS());
      // The inner form must be digits of one dim at exactly their place values, and c one of its divisors.
      std::optional<unsigned> dim;
      bool digitsOnly = c && inner.constant == 0;
      for (auto [atom, coeff] : inner.terms) {
        if (!coeff)
          continue;
        auto it = place.find(atom);
        if (it == place.end() || it->second.second != coeff || (dim && *dim != it->second.first)) {
          digitsOnly = false;
          break;
        }
        dim = it->second.first;
      }
      if (digitsOnly && dim && llvm::is_contained(digits.at(*dim), c.getValue())) {
        for (auto [atom, coeff] : inner.terms) {
          if (!coeff)
            continue;
          int64_t value = place[atom].second;
          bool above = value >= c.getValue();
          if (x.getKind() == AffineExprKind::FloorDiv && above)
            f.terms[atom] += value / c.getValue();
          if (x.getKind() == AffineExprKind::Mod && !above)
            f.terms[atom] += value;
        }
        return f;
      }
    }
    f.terms[x] = 1;
    return f;
  };
  return walk(e);
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

/// Splits a flat address over a row-major shape: each term goes to a dim whose stride is the largest one
/// dividing its coefficient (several dims share that stride around size-1 dims: every choice is returned); the
/// constant is split greedily from the outermost dim (truncating toward zero, so small negative offsets such as
/// padding stay in their own dim).
SmallVector<SmallVector<AffineExpr>> splits(const Form &flat, ArrayRef<int64_t> shape, MLIRContext *ctx) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  SmallVector<std::pair<AffineExpr, int64_t>> ts;
  for (auto [atom, c] : flat.terms)
    if (c)
      ts.push_back({atom, c});
  int64_t constant = flat.constant;
  // The candidate dims of each term.
  SmallVector<SmallVector<unsigned>> choices;
  for (auto [atom, coeff] : ts) {
    unsigned j = 0;
    while (j + 1 < shape.size() && coeff % strides[j] != 0)
      ++j;
    SmallVector<unsigned> c{j};
    for (unsigned k = j + 1; k < shape.size() && strides[k] == strides[j]; ++k)
      c.push_back(k);
    choices.push_back(c);
  }
  SmallVector<SmallVector<AffineExpr>> out;
  SmallVector<unsigned> pick(ts.size(), 0);
  while (out.size() < 16) {
    SmallVector<AffineExpr> idx(shape.size(), getAffineConstantExpr(0, ctx));
    for (auto [t, choice] : llvm::enumerate(pick)) {
      unsigned j = choices[t][choice];
      idx[j] = idx[j] + ts[t].first * (ts[t].second / strides[j]);
    }
    int64_t rest = constant;
    for (unsigned j = 0; j < shape.size(); ++j) {
      int64_t q = rest / strides[j]; // toward zero
      idx[j] = idx[j] + q;
      rest -= q * strides[j];
    }
    out.push_back(idx);
    // Next combination.
    unsigned t = 0;
    while (t < pick.size() && ++pick[t] == choices[t].size())
      pick[t++] = 0;
    if (t == pick.size())
      break;
  }
  return out;
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

  // Candidate index lists over the source's own shape (unfolding a full collapse to 1-D) or the flat source.
  SmallVector<std::pair<Value, SmallVector<AffineExpr>>> candidates;
  std::map<unsigned, SmallVector<int64_t>> digits;
  if (address) {
    auto flat = toAffine(address, ctx);
    if (!flat)
      return std::nullopt;
    SmallVector<AffineExpr> all{*flat};
    if (conds)
      all.append(conds->begin(), conds->end());
    digits = divisors(all);
    Form form = linearForm(*flat, digits);
    auto collapse = source.getDefiningOp<tensor::CollapseShapeOp>();
    if (collapse && cast<RankedTensorType>(collapse.getSrc().getType()).getRank() > 1) {
      auto shape = cast<RankedTensorType>(collapse.getSrc().getType()).getShape();
      for (SmallVector<AffineExpr> &multi : splits(form, shape, ctx))
        candidates.push_back({collapse.getSrc(), multi});
    }
    candidates.push_back({source, {*flat}});
  } else {
    SmallVector<AffineExpr> idx;
    for (Value i : extract.getIndices()) {
      auto e = toAffine(i, ctx);
      if (!e)
        return std::nullopt;
      idx.push_back(*e);
    }
    candidates.push_back({source, idx});
  }

  for (auto &[base, idx] : candidates) {
    auto baseType = cast<RankedTensorType>(base.getType());
    if (!select) {
      bool fits = true;
      for (auto [e, s] : llvm::zip(idx, baseType.getShape())) {
        Range r = bounds(e, ranges);
        fits &= r.lo >= 0 && r.hi < s;
      }
      if (fits)
        return Promotion{extract, base, AffineMap::get(nd, 0, idx, ctx)};
      continue;
    }

    // The mask must say exactly "every index inside its dim": each live condition is one of the bounds the
    // indices can break, and each such bound is in the mask.
    SmallVector<Form> needed;
    SmallVector<int64_t> low(idx.size(), 0), high(idx.size(), 0);
    for (auto [j, e] : llvm::enumerate(idx)) {
      Range r = bounds(e, ranges);
      int64_t s = baseType.getDimSize(j);
      if (r.lo < 0) {
        needed.push_back(linearForm(e, digits));
        low[j] = -r.lo;
      }
      if (r.hi >= s) {
        needed.push_back(linearForm(getAffineConstantExpr(s - 1, ctx) - e, digits));
        high[j] = r.hi - (s - 1);
      }
    }
    SmallVector<Form> live;
    for (AffineExpr c : *conds) {
      Form f = linearForm(c, digits);
      if (bounds(f.expr(ctx), ranges).lo < 0) // judged on digits: interval bounds of d - c*(d floordiv c) are loose
        live.push_back(f);
    }
    if (!llvm::all_of(live, [&](const Form &c) { return llvm::is_contained(needed, c); }) ||
        !llvm::all_of(needed, [&](const Form &n) { return llvm::is_contained(live, n); }))
      continue;

    Value padded = base;
    if (!needed.empty()) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(generic);
      SmallVector<int64_t> shape(baseType.getShape());
      for (unsigned j = 0; j < shape.size(); ++j)
        shape[j] += low[j] + high[j];
      Value pad = arith::ConstantOp::create(b, generic.getLoc(), cast<TypedAttr>(other));
      padded = tensor::PadOp::create(b, generic.getLoc(), RankedTensorType::get(shape, baseType.getElementType()),
                                     base, getAsIndexOpFoldResult(ctx, low), getAsIndexOpFoldResult(ctx, high), pad);
    }
    SmallVector<AffineExpr> shifted;
    for (auto [j, e] : llvm::enumerate(idx))
      shifted.push_back(e + low[j]);
    return Promotion{select, padded, AffineMap::get(nd, 0, shifted, ctx)};
  }
  return std::nullopt;
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

// Raises a launch-specialized Triton kernel to whole-tensor linalg on tensors (compiler/README.md).
//
// The driver folds every scalar argument to a constant and records the launch (grid, and the buffer
// behind each pointer argument) in the module's `furiosa.launch` attribute. Then:
//
//  1. Every integer, pointer and mask value is evaluated symbolically once per program instance: an
//     AffineExpr whose dims are the value's tensor axes and whose symbols are the iteration counts of
//     scf.for loops. Program ids are concrete per instance, so even non-affine program-id arithmetic
//     (min, select) folds. Integer data that reaches an address (a loaded index) is kept as a term.
//  2. Each store defines the coordinate space: its address, linear in the stored axes, picks a view of
//     the buffer (one view dim per axis stride) and a tile origin per instance. Enumerating every
//     instance and point proves which box the grid writes.
//  3. The stored value becomes a linalg.generic over that box in gather form: loads are tensor.extract
//     at addresses rewritten into box coordinates. Substituting axis = coordinate - origin must give one
//     expression for all instances; a leftover per-instance constant must be an affine function of the
//     origins, which coordinates determine. Masks become selects over clamped addresses.
//  4. tt.reduce, tt.dot and loop-carried accumulations become their own generics with reduction dims,
//     built before their consumer and pruned to the dims they use; consumers extract from them.
//
// Anything else fails with a diagnostic naming the op.

#include "furiosa/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/JSON.h"

#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>

namespace mlir::furiosa {
namespace {

namespace tt = mlir::triton;

// ---------------------------------------------------------------------------------------------------
// Affine helpers
// ---------------------------------------------------------------------------------------------------

struct Interval {
  int64_t lo, hi; // inclusive
};

int64_t floorDiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

Interval bounds(AffineExpr e, ArrayRef<Interval> dims, ArrayRef<Interval> syms = {}) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return {c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return dims[d.getPosition()];
  if (auto sy = dyn_cast<AffineSymbolExpr>(e))
    return syms[sy.getPosition()];
  auto bin = cast<AffineBinaryOpExpr>(e);
  Interval l = bounds(bin.getLHS(), dims, syms), r = bounds(bin.getRHS(), dims, syms);
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
    return (l.lo >= 0 && l.hi < r.lo) ? l : Interval{0, r.lo - 1};
  default:
    llvm_unreachable("affine kind");
  }
}

int64_t evalAt(AffineExpr e, ArrayRef<int64_t> dims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return c.getValue();
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return dims[d.getPosition()];
  auto bin = cast<AffineBinaryOpExpr>(e);
  int64_t l = evalAt(bin.getLHS(), dims), r = evalAt(bin.getRHS(), dims);
  switch (e.getKind()) {
  case AffineExprKind::Add:
    return l + r;
  case AffineExprKind::Mul:
    return l * r;
  case AffineExprKind::FloorDiv:
    return floorDiv(l, r);
  case AffineExprKind::CeilDiv:
    return -floorDiv(-l, r);
  case AffineExprKind::Mod:
    return ((l % r) + r) % r;
  default:
    llvm_unreachable("affine kind");
  }
}

/// The exact range of `e` over the box `dims` x `syms`, enumerating the dims and symbols it reads when
/// they span at most 2^20 points (intervals lose correlations: c - c mod 2 >= 0); else bounds().
Interval exactBounds(AffineExpr e, ArrayRef<Interval> dims, ArrayRef<Interval> syms) {
  SmallVector<Interval> box(dims.begin(), dims.end());
  box.append(syms.begin(), syms.end());
  SmallVector<AffineExpr> symAsDim;
  for (unsigned k = 0; k < syms.size(); ++k)
    symAsDim.push_back(getAffineDimExpr(dims.size() + k, e.getContext()));
  AffineExpr f = e.replaceSymbols(symAsDim);
  SmallVector<unsigned> used;
  int64_t points = 1;
  for (unsigned i = 0; i < box.size(); ++i)
    if (f.isFunctionOfDim(i)) {
      used.push_back(i);
      points *= box[i].hi - box[i].lo + 1;
      if (points > (1 << 20))
        return bounds(e, dims, syms);
    }
  SmallVector<int64_t> at;
  for (const Interval &b : box)
    at.push_back(b.lo);
  Interval r{std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()};
  while (true) {
    int64_t v = evalAt(f, at);
    r = {std::min(r.lo, v), std::max(r.hi, v)};
    unsigned k = 0;
    for (; k < used.size(); ++k) { // next point, odometer order
      if (++at[used[k]] <= box[used[k]].hi)
        break;
      at[used[k]] = box[used[k]].lo;
    }
    if (k == used.size())
      return r;
  }
}

/// Constant coefficients of the dims of a purely linear `e`, plus its constant.
std::optional<std::pair<SmallVector<int64_t>, int64_t>> linearize(AffineExpr e, unsigned numDims) {
  SmallVector<int64_t> coeffs(numDims, 0);
  int64_t constant = 0;
  std::function<bool(AffineExpr, int64_t)> walk = [&](AffineExpr x, int64_t scale) -> bool {
    if (auto c = dyn_cast<AffineConstantExpr>(x)) {
      constant += scale * c.getValue();
      return true;
    }
    if (auto d = dyn_cast<AffineDimExpr>(x)) {
      coeffs[d.getPosition()] += scale;
      return true;
    }
    auto bin = dyn_cast<AffineBinaryOpExpr>(x);
    if (!bin)
      return false;
    if (x.getKind() == AffineExprKind::Add)
      return walk(bin.getLHS(), scale) && walk(bin.getRHS(), scale);
    if (x.getKind() == AffineExprKind::Mul) {
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        return walk(bin.getLHS(), scale * c.getValue());
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS()))
        return walk(bin.getRHS(), scale * c.getValue());
    }
    return false;
  };
  if (!walk(e, 1))
    return std::nullopt;
  return std::make_pair(coeffs, constant);
}

/// `e` split into its constant term and the rest.
std::pair<AffineExpr, int64_t> splitConstant(AffineExpr e) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return {getAffineConstantExpr(0, e.getContext()), c.getValue()};
  if (e.getKind() == AffineExprKind::Add) {
    auto bin = cast<AffineBinaryOpExpr>(e);
    auto [l, lc] = splitConstant(bin.getLHS());
    auto [r, rc] = splitConstant(bin.getRHS());
    return {l + r, lc + rc};
  }
  return {e, 0};
}

/// Rewrites nested floordiv/mod into single ones ((x floordiv a) floordiv b = x floordiv ab, and
/// (x mod a) mod b = x mod b when b divides a), which MLIR's simplifier does not, then simplifies, so
/// equal digits of one index cancel.
AffineExpr normalize(AffineExpr e) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return e;
  AffineExpr l = normalize(bin.getLHS()), r = normalize(bin.getRHS());
  auto c = dyn_cast<AffineConstantExpr>(r);
  switch (e.getKind()) {
  case AffineExprKind::Add:
    return l + r;
  case AffineExprKind::Mul:
    return l * r;
  case AffineExprKind::FloorDiv:
    if (c && l.getKind() == AffineExprKind::FloorDiv) {
      auto inner = cast<AffineBinaryOpExpr>(l);
      if (auto ic = dyn_cast<AffineConstantExpr>(inner.getRHS()))
        return inner.getLHS().floorDiv(ic.getValue() * c.getValue());
    }
    return l.floorDiv(r);
  case AffineExprKind::CeilDiv:
    return l.ceilDiv(r);
  case AffineExprKind::Mod:
    if (c && l.getKind() == AffineExprKind::Mod) {
      auto inner = cast<AffineBinaryOpExpr>(l);
      if (auto ic = dyn_cast<AffineConstantExpr>(inner.getRHS()); ic && ic.getValue() % c.getValue() == 0)
        return inner.getLHS() % c.getValue();
    }
    return l % r;
  default:
    return e;
  }
}

AffineExpr simplify(AffineExpr e, unsigned numDims, unsigned numSyms = 0) {
  return simplifyAffineExpr(normalize(simplifyAffineExpr(e, numDims, numSyms)), numDims, numSyms);
}

/// `e` with every floordiv/mod by a constant k whose quotient the ranges of the dims and symbols fix
/// resolved: x = k*q + r with r in [m*k, (m+1)*k) gives x floordiv k = q + m and x mod k = r - m*k
/// (so (14*d0 + d1) floordiv 14 = d0 when d1 < 14), which MLIR's range-free simplifier cannot do.
AffineExpr resolveDigits(AffineExpr e, ArrayRef<Interval> dims, ArrayRef<Interval> syms) {
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return e;
  AffineExpr l = resolveDigits(bin.getLHS(), dims, syms), r = resolveDigits(bin.getRHS(), dims, syms);
  auto k = dyn_cast<AffineConstantExpr>(r);
  bool digit = e.getKind() == AffineExprKind::FloorDiv || e.getKind() == AffineExprKind::Mod;
  if (!digit || !k || k.getValue() <= 0) {
    switch (e.getKind()) {
    case AffineExprKind::Add:
      return l + r;
    case AffineExprKind::Mul:
      return l * r;
    case AffineExprKind::FloorDiv:
      return l.floorDiv(r);
    case AffineExprKind::CeilDiv:
      return l.ceilDiv(r);
    default:
      return l % r;
    }
  }
  // Split l into the terms whose coefficients k divides (q, scaled down) and the rest (r).
  AffineExpr zero = getAffineConstantExpr(0, e.getContext()), q = zero, rest = zero;
  std::function<void(AffineExpr, int64_t)> split = [&](AffineExpr x, int64_t c) {
    if (x.getKind() == AffineExprKind::Add) {
      auto add = cast<AffineBinaryOpExpr>(x);
      split(add.getLHS(), c);
      split(add.getRHS(), c);
      return;
    }
    if (x.getKind() == AffineExprKind::Mul)
      if (auto m = dyn_cast<AffineConstantExpr>(cast<AffineBinaryOpExpr>(x).getRHS())) {
        split(cast<AffineBinaryOpExpr>(x).getLHS(), c * m.getValue());
        return;
      }
    if (auto cst = dyn_cast<AffineConstantExpr>(x)) {
      int64_t v = c * cst.getValue(), m = floorDiv(v, k.getValue());
      q = q + m;
      rest = rest + (v - m * k.getValue());
      return;
    }
    if (c % k.getValue() == 0)
      q = q + x * (c / k.getValue());
    else
      rest = rest + x * c;
  };
  split(l, 1);
  Interval in = bounds(rest, dims, syms);
  int64_t m = floorDiv(in.lo, k.getValue());
  if (m != floorDiv(in.hi, k.getValue()))
    return e.getKind() == AffineExprKind::FloorDiv ? l.floorDiv(k) : l % k;
  return e.getKind() == AffineExprKind::FloorDiv ? q + m : rest - m * k.getValue();
}

/// Coefficients c (one per feature, then a constant) with sum_f c_f * x[n][f] + c_last == y[n] exactly
/// for every row n, if they exist (least squares, rounded, then checked).
std::optional<SmallVector<int64_t>> fitAffine(ArrayRef<SmallVector<int64_t>> x, ArrayRef<int64_t> y) {
  unsigned nf = x.empty() ? 0 : x[0].size(), nv = nf + 1;
  std::vector<std::vector<double>> a(nv, std::vector<double>(nv, 0.0));
  std::vector<double> rhs(nv, 0.0);
  // Features that are the same in every row stay out of the fit (the constant absorbs them).
  SmallVector<bool> varies(nf, false);
  for (unsigned f = 0; f < nf; ++f)
    for (auto &r : x)
      varies[f] = varies[f] || r[f] != x[0][f];
  for (unsigned n = 0; n < x.size(); ++n) {
    SmallVector<double> row;
    for (unsigned f = 0; f < nf; ++f)
      row.push_back(varies[f] ? (double)x[n][f] : 0.0);
    row.push_back(1.0);
    for (unsigned i = 0; i < nv; ++i) {
      rhs[i] += row[i] * y[n];
      for (unsigned j = 0; j < nv; ++j)
        a[i][j] += row[i] * row[j];
    }
  }
  std::vector<double> sol(nv, 0.0);
  std::vector<int> pivot(nv, -1);
  for (unsigned col = 0, row = 0; col < nv && row < nv; ++col) {
    unsigned best = row;
    for (unsigned r = row; r < nv; ++r)
      if (std::abs(a[r][col]) > std::abs(a[best][col]))
        best = r;
    if (std::abs(a[best][col]) < 1e-9)
      continue;
    std::swap(a[best], a[row]);
    std::swap(rhs[best], rhs[row]);
    for (unsigned r = 0; r < nv; ++r)
      if (r != row) {
        double f = a[r][col] / a[row][col];
        for (unsigned c = 0; c < nv; ++c)
          a[r][c] -= f * a[row][c];
        rhs[r] -= f * rhs[row];
      }
    pivot[col] = row++;
  }
  for (unsigned col = 0; col < nv; ++col)
    if (pivot[col] >= 0)
      sol[col] = rhs[pivot[col]] / a[pivot[col]][col];
  SmallVector<int64_t> c;
  for (double v : sol)
    c.push_back(std::llround(v));
  for (unsigned n = 0; n < x.size(); ++n) {
    int64_t v = c[nf];
    for (unsigned f = 0; f < nf; ++f)
      v += c[f] * x[n][f];
    if (v != y[n])
      return std::nullopt;
  }
  return c;
}

/// For a reshape that only inserts or drops unit dims, the source axis of each result axis (-1: new).
std::optional<SmallVector<int>> unitReshape(ArrayRef<int64_t> src, ArrayRef<int64_t> dst) {
  SmallVector<int> oldOf;
  unsigned k = 0;
  for (int64_t d : dst) {
    if (d == 1) {
      oldOf.push_back(-1);
      continue;
    }
    while (k < src.size() && src[k] == 1)
      ++k;
    if (k == src.size() || src[k] != d)
      return std::nullopt;
    oldOf.push_back(k++);
  }
  for (; k < src.size(); ++k)
    if (src[k] != 1)
      return std::nullopt;
  return oldOf;
}

std::string show(AffineExpr e) {
  std::string out;
  llvm::raw_string_ostream os(out);
  e.print(os);
  return out;
}

SmallVector<int64_t> shapeOf(Type t) {
  if (auto r = dyn_cast<RankedTensorType>(t))
    return SmallVector<int64_t>(r.getShape());
  return {};
}

// ---------------------------------------------------------------------------------------------------
// Per-instance symbolic evaluation
// ---------------------------------------------------------------------------------------------------

/// An integer data value added into an index or address. axisOf maps each axis of the index to the
/// axis of `value` it reads (-1: the value does not vary along it).
struct DataTerm {
  Value value;
  SmallVector<int> axisOf;
  int sign = 1;
  bool operator==(const DataTerm &o) const { return value == o.value && axisOf == o.axisOf && sign == o.sign; }
};

/// The symbolic value of an integer, pointer or i1 SSA value in one program instance. Dims are the
/// value's tensor axes; symbols are the iteration counts of scf.for loops (numbered per kernel).
struct Sym {
  enum Kind { Int, Ptr, Mask } kind = Int;
  SmallVector<int64_t> shape;
  AffineExpr expr;               // Int: the value; Ptr: element offset from the base argument
  int base = -1;                 // Ptr: pointer-argument index
  SmallVector<DataTerm> data;    // Int/Ptr: data terms added to expr
  SmallVector<AffineExpr> conds; // Mask: every expr >= 0
  bool never = false;            // Mask: statically false
};

class Evaluator {
public:
  Evaluator(MLIRContext *ctx, ArrayRef<int64_t> pid, ArrayRef<int64_t> grid, const DenseMap<Value, int> &argIndex,
            const DenseMap<Operation *, unsigned> &loopIds)
      : ctx(ctx), pid(pid.begin(), pid.end()), grid(grid.begin(), grid.end()), argIndex(argIndex), loopIds(loopIds) {}

  /// The symbolic mode: program id d is the symbol `pidSymbol + d` instead of a constant; whatever is
  /// not affine in it fails quietly (the mode is only a fallback).
  Evaluator(MLIRContext *ctx, unsigned pidSymbol, ArrayRef<int64_t> grid, const DenseMap<Value, int> &argIndex,
            const DenseMap<Operation *, unsigned> &loopIds)
      : ctx(ctx), pid{0, 0, 0}, grid(grid.begin(), grid.end()), argIndex(argIndex), loopIds(loopIds),
        symbolic(true), pidSymbol(pidSymbol) {}

  ArrayRef<int64_t> programId() const { return pid; }

  LogicalResult run(tt::FuncOp fn) {
    for (auto [arg, index] : argIndex) {
      Sym s;
      s.kind = Sym::Ptr;
      s.base = index;
      s.expr = cst(0);
      syms[arg] = s;
    }
    return evalBlock(fn.getBody().front());
  }

  const Sym *get(Value v) const {
    auto it = syms.find(v);
    return it == syms.end() ? nullptr : &it->second;
  }

  DenseMap<unsigned, int64_t> trips;    // per loop id: its trip count in this instance
  DenseMap<Operation *, bool> taken;     // per scf.if: whether this instance took its then-branch
  DenseSet<Operation *> executed;        // the stores (and atomics) this instance runs

private:
  MLIRContext *ctx;
  SmallVector<int64_t, 3> pid, grid;
  const DenseMap<Value, int> &argIndex;
  const DenseMap<Operation *, unsigned> &loopIds;
  bool symbolic = false;
  unsigned pidSymbol = 0;
  DenseMap<Value, Sym> syms;

  /// A value this mode cannot express: an error with constant program ids; in the symbolic mode the
  /// value just stays unknown (rewriteSymbolic then declines anything that needs it).
  LogicalResult fail(Operation *op, const Twine &msg) {
    if (symbolic)
      return success();
    return op->emitError(msg);
  }

  AffineExpr cst(int64_t c) { return getAffineConstantExpr(c, ctx); }

  void setInt(Value v, AffineExpr e) {
    Sym s;
    s.shape = shapeOf(v.getType());
    s.expr = e;
    syms[v] = s;
  }

  LogicalResult evalBlock(Block &block) {
    for (Operation &op : block)
      if (failed(evalOp(&op)))
        return failure();
    return success();
  }

  static std::optional<int64_t> constantInt(Attribute a) {
    if (auto i = dyn_cast<IntegerAttr>(a))
      return i.getValue().getSExtValue();
    if (auto d = dyn_cast<DenseIntElementsAttr>(a))
      if (d.isSplat())
        return d.getSplatValue<APInt>().getSExtValue();
    return std::nullopt;
  }

  /// Applies an axis remap to every expression of `s`: new axis i reads old axis oldOf[i] (-1: none).
  Sym remapAxes(const Sym &s, ArrayRef<int> oldOf, ArrayRef<int64_t> newShape) {
    SmallVector<AffineExpr> repl(s.shape.size(), cst(0));
    for (auto [i, old] : llvm::enumerate(oldOf))
      if (old >= 0)
        repl[old] = getAffineDimExpr(i, ctx);
    Sym out = s;
    out.shape.assign(newShape.begin(), newShape.end());
    if (out.expr)
      out.expr = out.expr.replaceDims(repl);
    for (AffineExpr &c : out.conds)
      c = c.replaceDims(repl);
    for (DataTerm &t : out.data) {
      SmallVector<int> axisOf;
      for (int old : oldOf)
        axisOf.push_back(old >= 0 ? t.axisOf[old] : -1);
      t.axisOf = axisOf;
    }
    return out;
  }

  static SmallVector<int> identityAxes(unsigned rank) {
    SmallVector<int> v;
    for (unsigned i = 0; i < rank; ++i)
      v.push_back(i);
    return v;
  }

  LogicalResult evalOp(Operation *op) {
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      Type et = getElementTypeOrSelf(c.getType());
      auto v = constantInt(c.getValue());
      if (!et.isInteger() || !v)
        return success();
      if (et.isInteger(1)) {
        Sym s;
        s.kind = Sym::Mask;
        s.shape = shapeOf(c.getType());
        s.never = (*v == 0);
        syms[c] = s;
      } else {
        setInt(c, cst(*v));
      }
      return success();
    }
    if (auto p = dyn_cast<tt::GetProgramIdOp>(op))
      return setInt(p, symbolic ? getAffineSymbolExpr(pidSymbol + p.getAxisAsInt(), ctx) : cst(pid[p.getAxisAsInt()])),
             success();
    if (auto p = dyn_cast<tt::GetNumProgramsOp>(op))
      return setInt(p, cst(grid[p.getAxisAsInt()])), success();
    if (auto r = dyn_cast<tt::MakeRangeOp>(op))
      return setInt(r, getAffineDimExpr(0, ctx) + cst(r.getStart())), success();
    if (auto bc = dyn_cast<tt::BitcastOp>(op)) { // a pointer to an element type of the same width
      const Sym *src = get(bc.getSrc());
      auto from = dyn_cast<tt::PointerType>(getElementTypeOrSelf(bc.getSrc().getType()));
      auto to = dyn_cast<tt::PointerType>(getElementTypeOrSelf(bc.getType()));
      if (src && src->kind == Sym::Ptr && from && to &&
          from.getPointeeType().getIntOrFloatBitWidth() == to.getPointeeType().getIntOrFloatBitWidth())
        syms[bc.getResult()] = *src;
      return success();
    }
    if (isa<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp, tt::TransOp>(op)) {
      const Sym *src = get(op->getOperand(0));
      if (!src)
        return success(); // data
      SmallVector<int64_t> dst = shapeOf(op->getResult(0).getType());
      SmallVector<int> oldOf;
      if (isa<tt::SplatOp>(op)) {
        oldOf.assign(dst.size(), -1);
      } else if (isa<tt::BroadcastOp>(op)) {
        for (unsigned i = 0; i < dst.size(); ++i)
          oldOf.push_back(src->shape[i] == dst[i] ? (int)i : -1);
      } else if (auto e = dyn_cast<tt::ExpandDimsOp>(op)) {
        for (unsigned i = 0; i < dst.size(); ++i)
          oldOf.push_back(i < e.getAxis() ? (int)i : i == e.getAxis() ? -1 : (int)i - 1);
      } else {
        for (int32_t o : cast<tt::TransOp>(op).getOrder())
          oldOf.push_back(o);
      }
      syms[op->getResult(0)] = remapAxes(*src, oldOf, dst);
      return success();
    }
    if (auto rs = dyn_cast<tt::ReshapeOp>(op)) {
      const Sym *src = get(rs.getSrc());
      if (!src)
        return success(); // data
      SmallVector<int64_t> dst = shapeOf(rs.getResult().getType());
      if (auto oldOf = unitReshape(src->shape, dst)) {
        syms[rs.getResult()] = remapAxes(*src, *oldOf, dst);
        return success();
      }
      if (!src->data.empty())
        return fail(op, "furiosa: reshape of a data-dependent index");
      // Each old axis is a mixed-radix digit of the new axes' row-major linear index.
      AffineExpr lin = cst(0);
      for (int64_t i = 0, stride = 1, n = dst.size(); i < n; stride *= dst[n - 1 - i], ++i)
        lin = lin + getAffineDimExpr(n - 1 - i, ctx) * stride;
      SmallVector<AffineExpr> repl(src->shape.size());
      for (int64_t k = src->shape.size() - 1, stride = 1; k >= 0; stride *= src->shape[k], --k)
        repl[k] = lin.floorDiv(stride) % src->shape[k];
      Sym out = *src;
      out.shape.assign(dst.begin(), dst.end());
      if (out.expr)
        out.expr = out.expr.replaceDims(repl);
      for (AffineExpr &c : out.conds)
        c = c.replaceDims(repl);
      syms[rs.getResult()] = out;
      return success();
    }
    if (auto a = dyn_cast<tt::AddPtrOp>(op)) {
      const Sym *ptr = get(a.getPtr());
      if (!ptr)
        return fail(op, "furiosa: address of an unknown pointer");
      Sym out = *ptr;
      out.shape = shapeOf(a.getType());
      if (const Sym *off = get(a.getOffset())) {
        out.expr = ptr->expr + off->expr;
        out.data.append(off->data.begin(), off->data.end());
      } else {
        out.data.push_back({a.getOffset(), identityAxes(out.shape.size()), 1});
      }
      syms[a] = out;
      return success();
    }
    if (isa<arith::AddIOp, arith::SubIOp>(op)) {
      const Sym *l = get(op->getOperand(0)), *r = get(op->getOperand(1));
      if (!l && !r)
        return success(); // integer data
      int sign = isa<arith::SubIOp>(op) ? -1 : 1;
      Sym out;
      out.shape = shapeOf(op->getResult(0).getType());
      SmallVector<int> ident = identityAxes(out.shape.size());
      auto addSide = [&](Value v, const Sym *s, int sgn) {
        if (!s) {
          out.data.push_back({v, ident, sgn});
          return;
        }
        out.expr = out.expr ? out.expr + s->expr * sgn : s->expr * sgn;
        for (DataTerm t : s->data) {
          t.sign *= sgn;
          out.data.push_back(t);
        }
      };
      addSide(op->getOperand(0), l, 1);
      addSide(op->getOperand(1), r, sign);
      if (!out.expr)
        out.expr = cst(0);
      syms[op->getResult(0)] = out;
      return success();
    }
    if (isa<arith::MulIOp, arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::MinSIOp,
            arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp>(op)) {
      const Sym *l = get(op->getOperand(0)), *r = get(op->getOperand(1));
      if (!l || !r || !l->data.empty() || !r->data.empty())
        return success(); // integer data
      AffineExpr a = l->expr, b = r->expr;
      auto ca = dyn_cast<AffineConstantExpr>(a), cb = dyn_cast<AffineConstantExpr>(b);
      AffineExpr res;
      if (isa<arith::MulIOp>(op)) {
        if (!ca && !cb)
          return fail(op, "furiosa: product of two non-constant indices");
        res = a * b;
      } else if (isa<arith::DivSIOp, arith::DivUIOp>(op)) {
        if (!cb || cb.getValue() <= 0)
          return fail(op, "furiosa: index division by a non-constant or non-positive value");
        res = normalize(a.floorDiv(cb.getValue()));
      } else if (isa<arith::RemSIOp, arith::RemUIOp>(op)) {
        if (!cb || cb.getValue() <= 0)
          return fail(op, "furiosa: index remainder by a non-constant or non-positive value");
        res = normalize(a % cb.getValue());
      } else {
        // Decided by the ranges of the axes, loop counters and program ids when not by constants
        // (a grouped tile order's min(num_pid_m - first_pid_m, GROUP_SIZE_M)).
        auto [ra, ia] = ranged(a, l->shape);
        auto [rb, ib] = ranged(b, r->shape);
        bool isMin = isa<arith::MinSIOp, arith::MinUIOp>(op);
        if (ia.hi <= ib.lo)
          res = isMin ? ra : rb;
        else if (ib.hi <= ia.lo)
          res = isMin ? rb : ra;
        else // ranges that overlap (a pooling window's in-image taps): data, computed where it is read
          return success();
      }
      setInt(op->getResult(0), res);
      return success();
    }
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp>(op)) {
      // A mask cast to an integer is 0 or 1: data, not a predicate.
      if (const Sym *s = get(op->getOperand(0)); s && s->kind != Sym::Mask) {
        Sym out = *s;
        out.shape = shapeOf(op->getResult(0).getType());
        syms[op->getResult(0)] = out;
      }
      return success();
    }
    if (auto c = dyn_cast<arith::CmpIOp>(op)) {
      const Sym *l = get(c.getLhs()), *r = get(c.getRhs());
      if (!l || !r || !l->data.empty() || !r->data.empty() || l->kind != Sym::Int || r->kind != Sym::Int)
        return success(); // data comparison
      AffineExpr a = l->expr, b = r->expr;
      Sym out;
      out.kind = Sym::Mask;
      out.shape = shapeOf(c.getType());
      switch (c.getPredicate()) {
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        out.conds.push_back(b - a - 1);
        break;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        out.conds.push_back(b - a);
        break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        out.conds.push_back(a - b - 1);
        break;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        out.conds.push_back(a - b);
        break;
      case arith::CmpIPredicate::eq:
        out.conds.push_back(a - b);
        out.conds.push_back(b - a);
        break;
      default:
        return success(); // ne: left to the data path
      }
      SmallVector<AffineExpr> kept;
      for (AffineExpr e : out.conds) {
        if (auto k = dyn_cast<AffineConstantExpr>(e)) {
          out.never |= k.getValue() < 0;
          continue;
        }
        kept.push_back(e);
      }
      out.conds = kept;
      syms[c] = out;
      return success();
    }
    if (auto a = dyn_cast<arith::AndIOp>(op)) {
      const Sym *l = get(a.getLhs()), *r = get(a.getRhs());
      if (!l || !r || l->kind != Sym::Mask || r->kind != Sym::Mask)
        return success();
      Sym out;
      out.kind = Sym::Mask;
      out.shape = shapeOf(a.getType());
      out.conds = l->conds;
      out.conds.append(r->conds.begin(), r->conds.end());
      out.never = l->never || r->never;
      syms[a] = out;
      return success();
    }
    if (auto s = dyn_cast<arith::SelectOp>(op)) {
      const Sym *c = get(s.getCondition()), *t = get(s.getTrueValue()), *f = get(s.getFalseValue());
      if (!t || !f || !c || c->kind != Sym::Mask)
        return success();
      // A condition the ranges decide (tl.where(offs < M, offs, 0) with offs below M in every tile).
      bool always = !c->never && llvm::all_of(c->conds, [&](AffineExpr e) { return ranged(e, c->shape).second.lo >= 0; });
      bool never = c->never || llvm::any_of(c->conds, [&](AffineExpr e) { return ranged(e, c->shape).second.hi < 0; });
      if (always || never)
        syms[s] = always ? *t : *f;
      return success();
    }
    if (auto f = dyn_cast<scf::ForOp>(op))
      return evalFor(f);
    if (auto i = dyn_cast<scf::IfOp>(op))
      return evalIf(i);
    if (isa<tt::StoreOp, tt::AtomicRMWOp, tt::AtomicCASOp>(op))
      executed.insert(op);
    return success();
  }

  /// A branch whose condition the program id decides: only the taken region runs in this instance.
  LogicalResult evalIf(scf::IfOp i) {
    const Sym *c = get(i.getCondition());
    if (!c || c->kind != Sym::Mask || !c->conds.empty())
      return fail(i.getOperation(), "furiosa: a branch whose condition varies inside a program instance");
    bool then = !c->never;
    taken[i] = then;
    Region &region = then ? i.getThenRegion() : i.getElseRegion();
    if (region.empty())
      return success();
    if (failed(evalBlock(region.front())))
      return failure();
    auto yield = cast<scf::YieldOp>(region.front().getTerminator());
    for (auto [r, y] : llvm::zip(i.getResults(), yield.getOperands()))
      if (const Sym *s = get(y))
        syms[r] = *s;
    return success();
  }

  LogicalResult evalFor(scf::ForOp f) {
    // Bounds may depend on the program id; in one instance they must be constants.
    auto bound = [&](Value v) -> std::optional<int64_t> {
      if (const Sym *s = get(v); s && s->kind == Sym::Int && s->data.empty())
        if (auto c = dyn_cast<AffineConstantExpr>(s->expr))
          return c.getValue();
      return getConstantIntValue(v);
    };
    auto lb = bound(f.getLowerBound()), ub = bound(f.getUpperBound()), step = bound(f.getStep());
    if (symbolic && step && *step > 0 && (!lb || !ub)) {
      // Program-dependent bounds: keep the counter's expression, the trip count is not needed here.
      const Sym *l = get(f.getLowerBound());
      if (!l || l->kind != Sym::Int || !l->data.empty())
        return success();
      unsigned id = loopIds.lookup(f);
      setInt(f.getInductionVar(), l->expr + getAffineSymbolExpr(id, ctx) * *step);
      return evalBody(f, id, std::nullopt);
    }
    if (!lb || !ub || !step || *step <= 0)
      return fail(f.getOperation(), "furiosa: loop bounds vary inside a program instance");
    unsigned id = loopIds.lookup(f);
    int64_t trip = std::max<int64_t>(0, (*ub - *lb + *step - 1) / *step);
    trips[id] = trip;
    setInt(f.getInductionVar(), cst(*lb) + getAffineSymbolExpr(id, ctx) * *step);
    return evalBody(f, id, trip);
  }

  /// `e` (over the axes of a value of `shape`) simplified with the ranges of its axes, of the loop
  /// counters this instance knows the trips of, and of symbolic program ids; and its range.
  std::pair<AffineExpr, Interval> ranged(AffineExpr e, ArrayRef<int64_t> shape) {
    const int64_t wide = int64_t(1) << 40;
    SmallVector<Interval> dims, syms;
    for (int64_t d : shape)
      dims.push_back({0, d - 1});
    unsigned ns = symbolic ? pidSymbol + 3 : loopIds.size();
    for (unsigned k = 0; k < ns; ++k) {
      Interval in{-wide, wide};
      if (auto it = trips.find(k); it != trips.end() && k < loopIds.size())
        in = {0, std::max<int64_t>(it->second - 1, 0)};
      if (symbolic && k >= pidSymbol)
        in = {0, grid[k - pidSymbol] - 1};
      syms.push_back(in);
    }
    AffineExpr r = simplify(resolveDigits(simplify(e, dims.size(), ns), dims, syms), dims.size(), ns);
    return {r, bounds(r, dims, syms)};
  }

  /// The body of loop `f` (counter symbol `id`): carried indices in closed form, then every op.
  LogicalResult evalBody(scf::ForOp f, unsigned id, std::optional<int64_t> trip) {
    AffineExpr t = getAffineSymbolExpr(id, ctx);
    // Index-typed iter args (pointer increments): closed form init + t * delta, with delta read off a
    // first evaluation of the body at the initial value.
    SmallVector<unsigned> carried;
    for (auto [i, init] : llvm::enumerate(f.getInitArgs()))
      if (const Sym *s = get(init)) {
        syms[f.getRegionIterArgs()[i]] = *s;
        carried.push_back(i);
      }
    auto yield = cast<scf::YieldOp>(f.getBody()->getTerminator());
    if (!carried.empty()) {
      if (failed(evalBlock(*f.getBody())))
        return failure();
      SmallVector<unsigned> affine;
      for (unsigned i : carried) {
        const Sym *init = get(f.getInitArgs()[i]);
        const Sym *next = get(yield.getOperand(i));
        // A carried integer that does not advance by a loop-invariant amount is data, not an index.
        if (!next || next->kind != init->kind || next->data.size() != init->data.size() ||
            (next->expr - init->expr).isFunctionOfSymbol(id)) {
          if (init->kind == Sym::Ptr)
            return fail(f.getOperation(), "furiosa: loop-carried pointer does not advance by a loop-invariant amount");
          continue;
        }
        Sym s = *init;
        s.expr = init->expr + t * (next->expr - init->expr);
        syms[f.getRegionIterArgs()[i]] = s;
        affine.push_back(i);
      }
      for (unsigned i : carried)
        if (!llvm::is_contained(affine, i))
          syms.erase(f.getRegionIterArgs()[i]);
      // Forget what the first pass derived; the second pass recomputes it from the closed forms.
      f.getBody()->walk([&](Operation *op) {
        for (Value r : op->getResults())
          syms.erase(r);
      });
      carried = affine;
    }
    if (failed(evalBlock(*f.getBody())))
      return failure();
    if (trip)
      for (unsigned i : carried) {
        Sym s = *get(f.getRegionIterArgs()[i]);
        s.expr = s.expr.replace(t, cst(*trip));
        syms[f.getResult(i)] = s;
      }
    return success();
  }
};

// ---------------------------------------------------------------------------------------------------
// The raiser
// ---------------------------------------------------------------------------------------------------

struct BufferInfo {
  SmallVector<int64_t> shape;
  Type elem;
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n;
  }
};

/// The coordinate space a store defines. Each stored axis has a global index (the axis plus its tile
/// origin in that instance) that is an expression of the view dims: one view dim for a linear axis, or
/// the mixed-radix recombination of the view dims of its digits for an address built with div/mod.
/// What writes a buffer: a store, an atomic read-modify-write (it stores old (op) val), or a compare
/// and swap (val where old == cmp).
struct Write {
  Operation *op;
  // An atomic every instance applies to the same elements: one write of old (op) each instance's value,
  // reduced over the program ids in program order (raiseStore).
  bool overPrograms = false;
  Value getPtr() const {
    if (auto cas = dyn_cast<tt::AtomicCASOp>(op))
      return cas.getPtr();
    return isa<tt::StoreOp>(op) ? cast<tt::StoreOp>(op).getPtr() : cast<tt::AtomicRMWOp>(op).getPtr();
  }
  Value getValue() const {
    if (auto cas = dyn_cast<tt::AtomicCASOp>(op))
      return cas.getVal();
    return isa<tt::StoreOp>(op) ? cast<tt::StoreOp>(op).getValue() : cast<tt::AtomicRMWOp>(op).getVal();
  }
  Value getMask() const {
    if (isa<tt::AtomicCASOp>(op))
      return Value();
    return isa<tt::StoreOp>(op) ? cast<tt::StoreOp>(op).getMask() : cast<tt::AtomicRMWOp>(op).getMask();
  }
  InFlightDiagnostic emitError(const Twine &message) const { return op->emitError(message); }
  Operation *getOperation() const { return op; }
  Operation *operator->() const { return op; }
};

/// What reads a buffer: a load, or the old value an atomic read-modify-write returns.
struct Read {
  Operation *op;
  Value getPtr() const {
    if (auto cas = dyn_cast<tt::AtomicCASOp>(op))
      return cas.getPtr();
    return isa<tt::LoadOp>(op) ? cast<tt::LoadOp>(op).getPtr() : cast<tt::AtomicRMWOp>(op).getPtr();
  }
  Value getMask() const {
    if (isa<tt::AtomicCASOp>(op))
      return Value();
    return isa<tt::LoadOp>(op) ? cast<tt::LoadOp>(op).getMask() : cast<tt::AtomicRMWOp>(op).getMask();
  }
  Value getOther() const { return isa<tt::LoadOp>(op) ? cast<tt::LoadOp>(op).getOther() : Value(); }
  Type getType() const { return op->getResult(0).getType(); }
  InFlightDiagnostic emitError(const Twine &message) const { return op->emitError(message); }
  operator Operation *() const { return op; }
};

struct StoreSpace {
  int arg = -1;
  SmallVector<int64_t> viewShape, viewStrides; // row-major view of the buffer
  SmallVector<int64_t> boxLo, boxHi;           // written box (inclusive) per view dim
  bool full = false;
  SmallVector<AffineExpr> global;              // per stored axis, over view dims (null: unit axis)
  SmallVector<int64_t> extent, phase;          // per stored axis: tile extent and origin phase (-1: unaligned)
  // Per stored axis: its step along its view dim (coordinate = origin + step * axis); origins of a
  // stepped axis lie in [0, step), so axis = G floordiv step and origin = G mod step.
  SmallVector<int64_t> step;
  SmallVector<unsigned> loops;                 // ids of the loops around the store: its leading axes
  SmallVector<unsigned> freeDims;              // view dims no axis moves along
  // Per instance: the origin of each stored axis, then the coordinate of each free dim.
  std::vector<SmallVector<int64_t>> origin;
  SmallVector<Value> dataMask; // mask terms that depend on data: the write keeps the old value where one fails
  unsigned programDims = 0;    // trailing view dims that are program ids (free dims), reduced over
};

/// A write mask taken apart over its `and`s: the index predicates (conds, never) decide where it can
/// write; the terms that depend on data are left in `data` for the write to select on.
template <typename Get>
void splitMask(Value mask, Get get, SmallVector<AffineExpr> &conds, bool &never, SmallVector<Value> &data) {
  if (const Sym *m = get(mask); m && m->kind == Sym::Mask) {
    conds.append(m->conds.begin(), m->conds.end());
    never |= m->never;
  } else if (auto a = mask.getDefiningOp<arith::AndIOp>()) {
    splitMask(a.getLhs(), get, conds, never, data);
    splitMask(a.getRhs(), get, conds, never, data);
  } else {
    data.push_back(mask);
  }
}

/// What an axis of a value means inside the generic being built.
struct Axis {
  enum Kind { Store, Iter, Unit, Expr } kind = Unit;
  int dim = 0;          // Store: stored axis; Iter: generic dim
  AffineExpr expr;      // Expr: a function of the axes `of`, as its dims
  std::vector<Axis> of; // Expr: the axes it reads
};

/// The generic being built: its dims are the store-view dims it keeps, then its own dims.
struct Gen {
  SmallVector<int> coordPos;      // store-view dim -> generic dim (-1: absent)
  SmallVector<int64_t> lo, extent; // per generic dim: offset of linalg.index, and extent
  DenseMap<unsigned, int> loopPos; // loop id -> generic dim
  DenseMap<unsigned, Value> loopIv; // loop id -> its counter, for loops kept as scf.for
  // Iter args of kept loops -> their state tensor and its dims (generic dims of this generic).
  std::map<void *, std::pair<Value, SmallVector<int>>> state;
  OpBuilder *ob = nullptr;
  Operation *anchor = nullptr;     // insertion point for what this generic reads
  SmallVector<Value> index;        // per generic dim, lazily created absolute index
  std::map<std::pair<void *, std::string>, Value> memo;
};

std::string key(ArrayRef<Axis> env) {
  std::string s;
  for (const Axis &a : env) {
    s += std::to_string((int)a.kind) + ":" + std::to_string(a.dim);
    if (a.kind == Axis::Expr) {
      llvm::raw_string_ostream os(s);
      os << "[" << a.expr << "|" << key(a.of) << "]";
    }
    s += ",";
  }
  return s;
}

/// The iter args of `loop` that its result `result` depends on: those its yielded value reads,
/// directly or through ops of the body (nested regions included), and so on through their yields.
SmallVector<bool> stateClosure(scf::ForOp loop, unsigned result) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<bool> needed(loop.getNumRegionIterArgs(), false);
  DenseSet<Operation *> seen;
  SmallVector<Value> work{yield.getOperand(result)};
  needed[result] = true;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() == loop.getBody()) {
      unsigned j = arg.getArgNumber() - loop.getNumInductionVars();
      if (arg.getArgNumber() >= loop.getNumInductionVars() && !needed[j]) {
        needed[j] = true;
        work.push_back(yield.getOperand(j));
      }
      continue;
    }
    Operation *def = v.getDefiningOp();
    if (!def || !loop->isProperAncestor(def) || !seen.insert(def).second)
      continue;
    def->walk([&](Operation *inner) { work.append(inner->operand_begin(), inner->operand_end()); });
  }
  return needed;
}

/// An env over a parent's dims, in a generic built with `remap` (parent dim -> its dim).
void remapAxis(Axis &a, ArrayRef<int> remap) {
  if (a.kind == Axis::Iter)
    a.dim = remap[a.dim];
  for (Axis &x : a.of)
    remapAxis(x, remap);
}

SmallVector<Axis> remapEnv(ArrayRef<Axis> env, ArrayRef<int> remap) {
  SmallVector<Axis> out(env.begin(), env.end());
  for (Axis &a : out)
    remapAxis(a, remap);
  return out;
}

/// The source axes of a row-major reshape to `src`, given the result axes `env` (shape `dst`): each
/// source axis is a mixed-radix digit of the result's linear index, resolved over the result axes'
/// ranges, so a source axis that is one result axis stays that axis.
SmallVector<Axis> reshapeEnv(ArrayRef<Axis> env, ArrayRef<int64_t> dst, ArrayRef<int64_t> src, MLIRContext *ctx) {
  std::vector<Axis> of; // the result axes that vary, as dims
  SmallVector<Interval> ranges;
  AffineExpr lin = getAffineConstantExpr(0, ctx);
  int64_t stride = 1;
  for (int k = dst.size() - 1; k >= 0; stride *= dst[k], --k)
    if (env[k].kind != Axis::Unit) { // Unit: index 0
      lin = lin + getAffineDimExpr(of.size(), ctx) * stride;
      of.push_back(env[k]);
      ranges.push_back({0, dst[k] - 1});
    }
  SmallVector<Axis> out(src.size());
  stride = 1;
  for (int k = src.size() - 1; k >= 0; stride *= src[k], --k) {
    AffineExpr e = simplify(resolveDigits(simplify(lin.floorDiv(stride) % src[k], of.size()), ranges, {}), of.size());
    if (auto c = dyn_cast<AffineConstantExpr>(e); c && c.getValue() == 0)
      continue;
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      out[k] = of[d.getPosition()];
      continue;
    }
    // Only the axes it reads, renumbered.
    SmallVector<AffineExpr> renum(of.size(), getAffineConstantExpr(0, ctx));
    out[k].kind = Axis::Expr;
    for (unsigned i = 0; i < of.size(); ++i)
      if (e.isFunctionOfDim(i)) {
        renum[i] = getAffineDimExpr(out[k].of.size(), ctx);
        out[k].of.push_back(of[i]);
      }
    out[k].expr = e.replaceDims(renum);
  }
  return out;
}

/// The identity of a single-operand combiner that is one known op on its two arguments, else null.
TypedAttr identityOf(Block &combiner) {
  if (combiner.getNumArguments() != 2 || !llvm::hasSingleElement(combiner.without_terminator()))
    return {};
  Operation *comb = &combiner.front();
  Value a = combiner.getArgument(0), x = combiner.getArgument(1);
  if (comb->getNumOperands() != 2 || combiner.getTerminator()->getOperands() != comb->getResults() ||
      !((comb->getOperand(0) == a && comb->getOperand(1) == x) || (comb->getOperand(0) == x && comb->getOperand(1) == a)))
    return {};
  Builder b(comb->getContext());
  Type et = a.getType();
  if (isa<arith::AddFOp>(comb))
    return b.getFloatAttr(et, 0.0);
  if (isa<arith::MulFOp>(comb))
    return b.getFloatAttr(et, 1.0);
  if (isa<arith::MaxNumFOp, arith::MaximumFOp>(comb))
    return b.getFloatAttr(et, -std::numeric_limits<double>::infinity());
  if (isa<arith::MinNumFOp, arith::MinimumFOp>(comb))
    return b.getFloatAttr(et, std::numeric_limits<double>::infinity());
  if (isa<arith::AddIOp, arith::XOrIOp, arith::OrIOp, arith::MaxUIOp>(comb))
    return b.getIntegerAttr(et, 0);
  if (isa<arith::MulIOp>(comb))
    return b.getIntegerAttr(et, 1);
  if (isa<arith::AndIOp, arith::MinUIOp>(comb))
    return b.getIntegerAttr(et, APInt::getAllOnes(et.getIntOrFloatBitWidth()));
  if (isa<arith::MaxSIOp>(comb))
    return b.getIntegerAttr(et, APInt::getSignedMinValue(et.getIntOrFloatBitWidth()));
  if (isa<arith::MinSIOp>(comb))
    return b.getIntegerAttr(et, APInt::getSignedMaxValue(et.getIntOrFloatBitWidth()));
  return {};
}

class Raiser {
public:
  Raiser(ModuleOp module, tt::FuncOp kernel)
      : module(module), kernel(kernel), ctx(module.getContext()), b(ctx), loc(kernel.getLoc()) {}

  LogicalResult run();
  void lowerScans();

private:
  ModuleOp module;
  tt::FuncOp kernel;
  MLIRContext *ctx;
  OpBuilder b;
  Location loc;
  SmallVector<int64_t, 3> grid;
  std::map<int, BufferInfo> buffers;
  DenseMap<Value, int> argIndex;
  DenseMap<Operation *, unsigned> loopIds;
  SmallVector<int64_t> loopTrip;
  SmallVector<bool> varTrip;
  std::vector<std::unique_ptr<Evaluator>> evals;
  // The instances that run the store being raised, with one control path and one store base.
  SmallVector<Evaluator *> act;
  // One evaluation with the program ids as symbols (null where something is not affine in them).
  std::unique_ptr<Evaluator> symEval;
  std::map<int, Value> current;
  std::map<void *, Value> flat;                        // buffer version -> its flat view
  SmallVector<std::tuple<Operation *, int, Value>> stores; // raised stores: op, buffer, version after it
  SmallVector<Value> args;                              // the function's buffer arguments
  const StoreSpace *space = nullptr;
  // Stored axes whose tile origin (not just their global index) an expression of this store used.
  std::set<unsigned> originUses;
  // (view dim, divisor) pairs an expression of this store took apart with floordiv/mod.
  std::set<std::pair<unsigned, int64_t>> digitUses;

  LogicalResult parseLaunch();
  LogicalResult evaluate();
  FailureOr<StoreSpace> storeSpace(Write store, ArrayRef<int64_t> extraStrides = {});
  LogicalResult raiseStoreIn(Write store, const StoreSpace &sp);
  LogicalResult raiseStore(Write store);

  AffineExpr originExpr(const Gen &g, unsigned viewDim);
  AffineExpr tidy(const Gen &g, AffineExpr e);
  FailureOr<AffineExpr> rewrite(Operation *at, const Gen &g, ArrayRef<Axis> env,
                                std::function<AffineExpr(const Sym &)> select, Value v);
  FailureOr<AffineExpr> fitOrigins(Operation *at, const Gen &g, ArrayRef<int64_t> consts);
  FailureOr<SmallVector<AffineExpr>> tripBounds(Gen &g, unsigned id);
  FailureOr<Value> tripGuard(Gen &g, unsigned id, Value t);
  FailureOr<AffineExpr> rewriteSymbolic(Operation *at, const Gen &g, ArrayRef<Axis> env,
                                        std::function<AffineExpr(const Sym &)> select, Value v);
  Value index(Gen &g, unsigned dim);
  Value exprValue(Gen &g, AffineExpr e);
  Value flatOf(Value version, int arg);
  Value versionAt(Operation *load, int arg);
  FailureOr<Value> emit(Gen &g, Value v, ArrayRef<Axis> env);
  FailureOr<Value> emitLoad(Gen &g, Read load, ArrayRef<Axis> env);
  FailureOr<Value> axisValue(Gen &g, Operation *at, ArrayRef<Axis> env, unsigned k);
  FailureOr<Value> emitMapElementwise(Gen &g, tt::MapElementwiseOp op, ArrayRef<Axis> env);
  FailureOr<Value> emitSym(Gen &g, Value v, const Sym &s0, ArrayRef<Axis> env);
  FailureOr<Value> emitDataTerms(Gen &g, const Sym &s0, ArrayRef<Axis> env);
  FailureOr<Value> emitElementwise(Gen &g, Operation *op, ArrayRef<Axis> env);
  FailureOr<Value> materialize(Gen &g, Value v, ArrayRef<Axis> env);
  int64_t liveExtent(tt::DotOp dot);
  int64_t maskLive(Value mask, unsigned axis, int64_t full);
  int64_t liveAlong(Value v, unsigned axis, TypedAttr identity);
  FailureOr<Value> materializeState(Gen &g, scf::ForOp loop, unsigned result, ArrayRef<Axis> env);
  FailureOr<SmallVector<SmallVector<Axis>>> stateEnvs(scf::ForOp loop, unsigned result, ArrayRef<Axis> env,
                                                     SmallVector<bool> *skipped = nullptr);
  FailureOr<Value> parallelGeneric(Gen &parent, Type elem, std::function<FailureOr<Value>(Gen &)> fn);
  // (loop, consumer anchor) -> its results and, per result, the consumer dims its state spans.
  std::map<std::pair<Operation *, Operation *>, std::pair<SmallVector<Value>, SmallVector<SmallVector<unsigned>>>>
      keptLoops;
  // (reduce, result) -> the reduction in its own index space and the dims it keeps (materializeOwn).
  std::map<std::pair<Operation *, unsigned>, std::pair<Value, SmallVector<unsigned>>> ownReductions;
  unsigned ownDepth = 0; // > 0 while building one
  FailureOr<std::pair<Value, SmallVector<unsigned>>> reduceTensor(Gen &g, tt::ReduceOp red, unsigned result,
                                                                  ArrayRef<Axis> env);
  FailureOr<Value> materializeOwn(Gen &g, tt::ReduceOp red, unsigned result, ArrayRef<Axis> env);
  /// Erases a placeholder anchor and forgets the loops kept for it (a later op may reuse its address).
  void retire(Operation *anchor) {
    for (auto it = keptLoops.begin(); it != keptLoops.end();)
      it = it->first.second == anchor ? keptLoops.erase(it) : std::next(it);
    anchor->erase();
  }

  /// Builds one generic whose first dims are `parentDims` of `parent` (a subset, in order) followed by
  /// `ownExtents`; `body` yields the value at a point given the generic, the out value (accumulator)
  /// and the parent-dim remap. Returns the result tensor and the parent dims it keeps.
  using BodyFn = std::function<FailureOr<Value>(Gen &, Value acc, ArrayRef<int> remap)>;
  using InitFn = std::function<FailureOr<Value>(Gen &, ArrayRef<int> remap)>;
  FailureOr<std::pair<Value, SmallVector<unsigned>>> buildGeneric(Gen &parent, ArrayRef<unsigned> parentDims,
                                                                  ArrayRef<int64_t> ownExtents,
                                                                  ArrayRef<std::pair<unsigned, unsigned>> loopDims,
                                                                  Type elem, InitFn init, BodyFn body);
  /// buildGeneric with one result per `elems` (each body yields one value per accumulator), its own dims
  /// starting at `ownLo`.
  using InitsFn = std::function<FailureOr<SmallVector<Value>>(Gen &, ArrayRef<int> remap)>;
  using BodiesFn = std::function<FailureOr<SmallVector<Value>>(Gen &, ValueRange accs, ArrayRef<int> remap)>;
  FailureOr<std::pair<SmallVector<Value>, SmallVector<unsigned>>>
  buildGenerics(Gen &parent, ArrayRef<unsigned> parentDims, ArrayRef<int64_t> ownExtents,
                ArrayRef<std::pair<unsigned, unsigned>> loopDims, ArrayRef<Type> elems, InitsFn init, BodiesFn body,
                int64_t ownLo);
  Value reshape(Value t, ArrayRef<int64_t> shape);
  Value constantOf(Type t, Attribute a);
};

LogicalResult Raiser::parseLaunch() {
  auto attr = module->getAttrOfType<StringAttr>("furiosa.launch");
  if (!attr)
    return module.emitError("furiosa: missing furiosa.launch (compile through furiosa_compiler)");
  auto json = llvm::json::parse(attr.getValue());
  if (!json)
    return module.emitError("furiosa: bad furiosa.launch JSON");
  auto *obj = json->getAsObject();
  for (const llvm::json::Value &g : *obj->getArray("grid"))
    grid.push_back(*g.getAsInteger());
  for (const llvm::json::Value &v : *obj->getArray("buffers")) {
    const auto *o = v.getAsObject();
    BufferInfo info;
    for (const llvm::json::Value &d : *o->getArray("shape"))
      info.shape.push_back(*d.getAsInteger());
    buffers[*o->getInteger("arg")] = info;
  }
  // Float scalars arrive as fp32 arguments, as Triton's JIT passes them, so arithmetic on them rounds to fp32 at each
  // step (a constexpr folds in Python's double): bind each to its launch value, in order, and fold what that fixes.
  if (const llvm::json::Array *values = obj->getArray("scalars"); values && !values->empty()) {
    OpBuilder ob = OpBuilder::atBlockBegin(&kernel.getBody().front());
    llvm::BitVector bound(kernel.getNumArguments());
    auto value = values->begin();
    for (auto [i, arg] : llvm::enumerate(kernel.getArguments())) {
      if (isa<tt::PointerType>(arg.getType()))
        continue;
      auto ft = dyn_cast<FloatType>(arg.getType());
      if (!ft || value == values->end() || !value->getAsNumber())
        return kernel.emitError("furiosa: argument ") << i << " is not a buffer or a float scalar of the launch";
      arg.replaceAllUsesWith(arith::ConstantOp::create(ob, loc, FloatAttr::get(ft, *(value++)->getAsNumber())));
      bound.set(i);
    }
    GreedyRewriteConfig foldOnly;
    foldOnly.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(kernel.eraseArguments(bound)) ||
        failed(applyPatternsGreedily(kernel, FrozenRewritePatternSet(), foldOnly)))
      return kernel.emitError("furiosa: binding the float scalars failed");
  }
  for (auto [i, arg] : llvm::enumerate(kernel.getArguments())) {
    auto ptr = dyn_cast<tt::PointerType>(arg.getType());
    if (!ptr || !buffers.count(i))
      return kernel.emitError("furiosa: argument ") << i << " is not a buffer (scalars must be constexpr)";
    buffers[i].elem = ptr.getPointeeType();
    argIndex[arg] = i;
  }
  return success();
}

LogicalResult Raiser::evaluate() {
  kernel.walk([&](scf::ForOp f) {
    loopIds[f] = loopTrip.size();
    loopTrip.push_back(0);
  });
  for (int64_t z = 0; z < grid[2]; ++z)
    for (int64_t y = 0; y < grid[1]; ++y)
      for (int64_t x = 0; x < grid[0]; ++x) {
        evals.push_back(std::make_unique<Evaluator>(ctx, ArrayRef<int64_t>{x, y, z}, grid, argIndex, loopIds));
        if (failed(evals.back()->run(kernel)))
          return failure();
      }
  symEval = std::make_unique<Evaluator>(ctx, (unsigned)loopTrip.size(), grid, argIndex, loopIds);
  if (failed(symEval->run(kernel)))
    symEval.reset();
  // A loop's reduction extent is its longest trip; instances with shorter trips mask the rest.
  varTrip.assign(loopTrip.size(), false);
  for (unsigned id = 0; id < loopTrip.size(); ++id) {
    int64_t lo = std::numeric_limits<int64_t>::max(), hi = 0;
    for (auto &ev : evals) {
      int64_t t = ev->trips.lookup(id);
      lo = std::min(lo, t);
      hi = std::max(hi, t);
    }
    loopTrip[id] = hi;
    varTrip[id] = lo != hi;
  }
  return success();
}

namespace {
/// A term of a store address: `stride * u`, where u is a linear axis or a digit (G floordiv a) mod b of
/// a global axis index (b = 0: unbounded).
struct Component {
  int64_t stride;
  int axis;
  bool digit;
  int64_t a, b;
};

/// Collects the inner linear forms of every floordiv/mod node of `e`.
void atoms(AffineExpr e, SmallVectorImpl<AffineExpr> &out) {
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
    if (e.getKind() == AffineExprKind::FloorDiv || e.getKind() == AffineExprKind::Mod) {
      if (!isa<AffineBinaryOpExpr>(bin.getLHS()) || bin.getLHS().getKind() == AffineExprKind::Add ||
          bin.getLHS().getKind() == AffineExprKind::Mul || isa<AffineDimExpr>(bin.getLHS()))
        out.push_back(bin.getLHS());
    }
    atoms(bin.getLHS(), out);
    atoms(bin.getRHS(), out);
  }
}

/// Splits `e` (a sum of constant multiples of dims and of div/mod atoms of single dims) into components.
bool components(AffineExpr e, int64_t scale, SmallVectorImpl<Component> &out) {
  if (isa<AffineConstantExpr>(e))
    return true; // constants are origins, handled separately
  if (auto d = dyn_cast<AffineDimExpr>(e)) {
    out.push_back({scale, (int)d.getPosition(), false, 1, 0});
    return true;
  }
  auto bin = cast<AffineBinaryOpExpr>(e);
  switch (e.getKind()) {
  case AffineExprKind::Add:
    return components(bin.getLHS(), scale, out) && components(bin.getRHS(), scale, out);
  case AffineExprKind::Mul:
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
      return components(bin.getLHS(), scale * c.getValue(), out);
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS()))
      return components(bin.getRHS(), scale * c.getValue(), out);
    return false;
  case AffineExprKind::FloorDiv:
  case AffineExprKind::Mod: {
    int64_t a = 1, b = 0;
    AffineExpr x = e;
    if (x.getKind() == AffineExprKind::Mod) {
      b = cast<AffineConstantExpr>(cast<AffineBinaryOpExpr>(x).getRHS()).getValue();
      x = cast<AffineBinaryOpExpr>(x).getLHS();
    }
    if (x.getKind() == AffineExprKind::FloorDiv) {
      a = cast<AffineConstantExpr>(cast<AffineBinaryOpExpr>(x).getRHS()).getValue();
      x = cast<AffineBinaryOpExpr>(x).getLHS();
    }
    auto d = dyn_cast<AffineDimExpr>(x);
    if (!d)
      return false;
    out.push_back({scale, (int)d.getPosition(), true, a, b});
    return true;
  }
  default:
    return false;
  }
}
} // namespace

FailureOr<StoreSpace> Raiser::storeSpace(Write store, ArrayRef<int64_t> extraStrides) {
  // The loops around the store become its leading axes: each iteration is one more stored tile.
  SmallVector<unsigned> encl;
  for (Operation *op = store->getParentOp(); op && op != kernel.getOperation(); op = op->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(op))
      encl.insert(encl.begin(), loopIds.lookup(f));
  unsigned nl = encl.size();
  SmallVector<int64_t> vshape = shapeOf(store.getValue().getType());
  SmallVector<int64_t> shape;
  for (unsigned id : encl)
    shape.push_back(loopTrip[id]);
  shape.append(vshape.begin(), vshape.end());
  unsigned rank = shape.size();
  const Sym *p0 = act[0]->get(store.getPtr());
  if (!p0 || p0->kind != Sym::Ptr || !p0->data.empty())
    return store.emitError("furiosa: store address is not an affine function of program ids and axes"), failure();
  int arg = p0->base;
  const BufferInfo &buf = buffers[arg];
  int64_t numel = buf.numel();
  // Per instance: the address and mask over the store's axes, and the extent of each axis there.
  SmallVector<AffineExpr> toDims;
  for (unsigned i = 0; i < vshape.size(); ++i)
    toDims.push_back(getAffineDimExpr(nl + i, ctx));
  SmallVector<AffineExpr> toSyms;
  for (unsigned l = 0; l < loopTrip.size(); ++l) {
    auto it = llvm::find(encl, l);
    toSyms.push_back(it != encl.end() ? getAffineDimExpr(it - encl.begin(), ctx) : getAffineConstantExpr(0, ctx));
  }
  std::vector<AffineExpr> address(act.size()), unresolved(act.size());
  SmallVector<Value> dataMask;
  std::vector<SmallVector<AffineExpr>> maskConds(act.size());
  std::vector<bool> maskNever(act.size(), false);
  std::vector<SmallVector<int64_t>> extentsOf(act.size());
  for (auto [n, ev] : llvm::enumerate(act)) {
    const Sym *p = ev->get(store.getPtr());
    if (!p || p->kind != Sym::Ptr || p->base != arg || !p->data.empty())
      return store.emitError("furiosa: store base differs between program instances"), failure();
    for (unsigned id : encl)
      extentsOf[n].push_back(ev->trips.lookup(id));
    extentsOf[n].append(vshape.begin(), vshape.end());
    // Over this instance's ranges of the loop counters and axes, digits resolve ((pid + 4 t) mod 32
    // floordiv 4 = t when the address stays below 32).
    SmallVector<Interval> ranges;
    for (int64_t e : extentsOf[n])
      ranges.push_back({0, std::max<int64_t>(e - 1, 0)});
    unresolved[n] = simplify(p->expr.replaceDimsAndSymbols(toDims, toSyms), rank, 0);
    address[n] = simplify(resolveDigits(unresolved[n], ranges, {}), rank, 0);
    if (store.getMask()) {
      SmallVector<AffineExpr> conds;
      bool never = false;
      SmallVector<Value> data;
      splitMask(store.getMask(), [&](Value v) { return ev->get(v); }, conds, never, data);
      for (AffineExpr c : conds)
        maskConds[n].push_back(simplify(c.replaceDimsAndSymbols(toDims, toSyms), rank, 0));
      maskNever[n] = never;
      if (n == 0)
        dataMask = data;
    }
  }

  // Axes the address takes apart with div/mod get a per-instance origin read off the atoms
  // (`origin + axis` inside each), so the address becomes one expression of their global indices.
  std::vector<SmallVector<int64_t>> digitOrigin(act.size(), SmallVector<int64_t>(rank, 0));
  SmallVector<bool> isDigit(rank, false);
  SmallVector<AffineExpr> rest(act.size());
  SmallVector<int64_t> constant(act.size());
  auto explain = [&]() -> const char * {
    for (auto [n, ev] : llvm::enumerate(act)) {
      SmallVector<AffineExpr> inner;
      atoms(address[n], inner);
      SmallVector<bool> seen(rank, false);
      for (AffineExpr form : inner) {
        auto lin = linearize(form, rank);
        if (!lin)
          return "a store address digit is not linear in one axis";
        int axis = -1;
        for (unsigned i = 0; i < rank; ++i)
          if (lin->first[i]) {
            if (axis >= 0 || lin->first[i] != 1)
              return "a store address digit mixes axes or scales one";
            axis = i;
          }
        if (axis < 0)
          continue;
        if (seen[axis] && digitOrigin[n][axis] != lin->second)
          return "a store axis has inconsistent digit origins";
        seen[axis] = true;
        digitOrigin[n][axis] = lin->second;
      }
      if (n == 0)
        isDigit = seen;
      else if (seen != isDigit)
        return "store addresses differ between program instances";
      SmallVector<AffineExpr> sub;
      for (unsigned i = 0; i < rank; ++i)
        sub.push_back(getAffineDimExpr(i, ctx) - (isDigit[i] ? digitOrigin[n][i] : 0));
      auto [r, c] = splitConstant(simplify(address[n].replaceDims(sub), rank, 0));
      if (n && r != rest[0])
        return "store addresses differ between program instances";
      rest[n] = r;
      constant[n] = c;
    }
    return nullptr;
  };
  // A tile that crosses a divisor in some instances only (a 64-row block of 196 patch rows per image)
  // keeps digits there that the other instances' ranges resolve: then every instance keeps them all.
  if (const char *why = explain()) {
    address = unresolved;
    if (explain())
      return store.emitError("furiosa: ") << why, failure();
  }
  SmallVector<Component> terms;
  if (!components(rest[0], 1, terms))
    return store.emitError("furiosa: store address is not a sum of axis terms and digits: ") << show(rest[0]), failure();
  // Rewrite the terms of each div/mod axis over its mixed-radix digits u_k = (G floordiv a_k) mod
  // (a_k+1 / a_k), where 1 = a_0 | a_1 | ... are the divisors its terms use; a term (G floordiv a) mod b
  // is sum_{a <= a_k < ab} u_k * a_k / a. The coefficient a digit collects is its address stride.
  SmallVector<Component> comps;
  for (const Component &t : terms)
    if (!isDigit[t.axis])
      comps.push_back(t);
  for (unsigned i = 0; i < rank; ++i) {
    if (!isDigit[i])
      continue;
    std::set<int64_t> chainSet{1};
    for (const Component &t : terms)
      if (t.axis == (int)i) {
        chainSet.insert(t.a);
        if (t.b)
          chainSet.insert(t.a * t.b);
      }
    SmallVector<int64_t> chain(chainSet.begin(), chainSet.end());
    for (unsigned k = 1; k < chain.size(); ++k)
      if (chain[k] % chain[k - 1])
        return store.emitError("furiosa: the digits of a store axis do not nest: ") << show(rest[0]), failure();
    SmallVector<int64_t> coeff(chain.size(), 0);
    for (const Component &t : terms)
      if (t.axis == (int)i)
        for (unsigned k = 0; k < chain.size(); ++k)
          if (chain[k] >= t.a && (!t.b || chain[k] < t.a * t.b))
            coeff[k] += t.stride * (chain[k] / t.a);
    for (unsigned k = 0; k < chain.size(); ++k)
      if (coeff[k])
        comps.push_back({coeff[k], (int)i, true, chain[k], k + 1 < chain.size() ? chain[k + 1] / chain[k] : 0});
  }
  for (const Component &c : comps)
    if (c.stride <= 0 || (shape[c.axis] == 1 && !c.digit))
      return store.emitError("furiosa: a stored axis does not advance the address"), failure();

  // View candidates: with the buffer's own dim strides (nicer views), then without.
  SmallVector<int64_t> bufStrides(buf.shape.size(), 1);
  for (int i = (int)buf.shape.size() - 2; i >= 0; --i)
    bufStrides[i] = bufStrides[i + 1] * buf.shape[i + 1];
  // View candidates, in order: the address strides plus the buffer's own dim strides (nicest), the
  // address strides alone, then both again refined by the spacing of the tile origins (when tiles
  // cover a strided subset of a dim). Each is checked by enumeration.
  std::string why = "store strides do not nest: " + show(rest[0]);
  SmallVector<int64_t> probeStrides;
  std::vector<SmallVector<int64_t>> probeOrigins;
  // A linear term whose stride does not divide the buffer (a persistent loop striding rows by the grid
  // size) cannot be a view dim; it moves along the view dim of the largest stride dividing it.
  SmallVector<bool> stepped;
  for (const Component &c : comps)
    stepped.push_back(!c.digit && numel % c.stride != 0);
  // One view (strides `st`): each address term moves along the view dim of its stride and, for a
  // linear term, the `span` - 1 dims above it, whose recombination it is (a flattened index tiled across
  // dims, such as a pixel index over [H, W], takes those dims as its digits). Checked by enumerating
  // every stored point; on failure sets `why`, and `crossed` to the axis whose tile left its dims.
  auto tryView = [&](ArrayRef<int64_t> st, ArrayRef<int> dimOf, ArrayRef<unsigned> span, int &crossed,
                     std::vector<SmallVector<int64_t>> *viewOrigins) -> std::optional<StoreSpace> {
    unsigned nd = st.size();
    StoreSpace sp;
    sp.arg = arg;
    sp.dataMask = dataMask;
    sp.viewStrides.assign(st.begin(), st.end());
    sp.viewShape.push_back(numel / st[0]);
    for (unsigned j = 1; j < nd; ++j)
      sp.viewShape.push_back(st[j - 1] / st[j]);
    SmallVector<bool> used(nd, false);
    sp.step.assign(rank, 1);
    sp.global.assign(rank, AffineExpr());
    // A term stepped along a view dim by no more than the extent of a term that moves along that dim alone (a
    // loop over column tiles: cols = c0 + arange) shares the dim with it: the two axes are the dim's digits
    // under the step, tile then lane.
    SmallVector<int> lanes(comps.size(), -1);
    for (auto [k, c] : llvm::enumerate(comps))
      for (auto [l, d] : llvm::enumerate(comps))
        if (stepped[k] && !stepped[l] && !d.digit && dimOf[l] == dimOf[k] && span[l] == 1 &&
            d.stride == st[dimOf[k]] && shape[d.axis] <= c.stride / st[dimOf[k]])
          lanes[k] = l;
    for (auto [k, c] : llvm::enumerate(comps)) {
      if (llvm::is_contained(lanes, (int)k))
        continue; // global set with its tile's
      if (stepped[k])
        sp.step[c.axis] = c.stride / st[dimOf[k]];
      AffineExpr term = getAffineConstantExpr(0, ctx);
      for (unsigned t = 0; t < span[k]; ++t) {
        int j = dimOf[k] - t;
        if (used[j]) {
          why = "two store address terms share a view dim";
          return std::nullopt;
        }
        used[j] = true;
        term = term + getAffineDimExpr(j, ctx) * (c.a * (st[j] / st[dimOf[k]]));
      }
      if (lanes[k] >= 0) {
        sp.global[comps[lanes[k]].axis] = term % sp.step[c.axis];
        term = term.floorDiv(sp.step[c.axis]);
        sp.step[c.axis] = 1;
      }
      sp.global[c.axis] = sp.global[c.axis] ? sp.global[c.axis] + term : term;
    }
    for (unsigned j = 0; j < nd; ++j)
      if (!used[j])
        sp.freeDims.push_back(j);
    // Linear axes and free dims take their origins from the constant; digits carry their own. The
    // constant's share on a digit's view dim shifts the dim by instance 0's share in every instance
    // (digits stored past a leading row, a class token's); the rest moves the instance's digit origin
    // (simplification folds (12 + i) floordiv 6 into 2 + i floordiv 6).
    bool ok = true;
    SmallVector<int64_t> shift;
    for (auto [n, ev] : llvm::enumerate(act)) {
      SmallVector<int64_t> viewOrigin(nd, 0);
      int64_t c = constant[n];
      for (unsigned j = 0; j < nd; ++j) {
        viewOrigin[j] = floorDiv(c, st[j]);
        c -= viewOrigin[j] * st[j];
      }
      if (n == 0)
        shift = viewOrigin;
      ok &= (c == 0);
      SmallVector<int64_t> origin(rank, 0);
      for (auto [k, cmp] : llvm::enumerate(comps))
        if (cmp.digit)
          origin[cmp.axis] += (viewOrigin[dimOf[k]] - shift[dimOf[k]]) * cmp.a;
      for (unsigned i = 0; i < rank; ++i) {
        if (isDigit[i])
          origin[i] += digitOrigin[n][i];
        else
          for (auto [k, cmp] : llvm::enumerate(comps))
            if (cmp.axis == (int)i && !cmp.digit)
              for (unsigned t = 0; t < span[k]; ++t)
                origin[i] += viewOrigin[dimOf[k] - t] * (st[dimOf[k] - t] / st[dimOf[k]]);
      }
      for (unsigned f : sp.freeDims)
        origin.push_back(viewOrigin[f]);
      sp.origin.push_back(origin);
      if (viewOrigins)
        viewOrigins->push_back(viewOrigin);
    }
    if (!ok) {
      why = "store origins do not fit the view";
      return std::nullopt;
    }
    for (auto [k, cmp] : llvm::enumerate(comps))
      if (cmp.digit)
        sp.global[cmp.axis] = sp.global[cmp.axis] - shift[dimOf[k]] * cmp.a;

    // Enumerate every point: it must land where the view says, once.
    std::vector<uint8_t> covered(numel, 0);
    sp.boxLo.assign(nd, std::numeric_limits<int64_t>::max());
    sp.boxHi.assign(nd, std::numeric_limits<int64_t>::min());
    int64_t count = 0;
    SmallVector<int64_t> idx(rank), coord(nd), live(rank, 0); // live: per axis, past the last index written
    why.clear();
    for (auto [n, ev] : llvm::enumerate(act)) {
      if (maskNever[n])
        continue;
      ArrayRef<int64_t> ext = extentsOf[n];
      int64_t total = 1;
      for (int64_t d : ext)
        total *= d;
      for (int64_t f = 0; f < total && why.empty(); ++f) {
        int64_t r = f;
        for (int i = rank - 1; i >= 0; --i) {
          idx[i] = r % ext[i];
          r /= ext[i];
        }
        if (llvm::any_of(maskConds[n], [&](AffineExpr c) { return evalAt(c, idx) < 0; }))
          continue;
        int64_t addr = evalAt(address[n], idx);
        if (addr < 0 || addr >= numel) {
          why = "a store lands outside its buffer";
          break;
        }
        int64_t a = addr;
        for (unsigned j = 0; j < nd; ++j) {
          coord[j] = a / st[j];
          a %= st[j];
        }
        for (unsigned i = 0; i < rank && why.empty(); ++i)
          if (sp.global[i] && evalAt(sp.global[i], coord) - sp.origin[n][i] != idx[i] * sp.step[i]) {
            why = "a store tile crosses a view boundary";
            crossed = i;
          }
        for (auto [k, fdim] : llvm::enumerate(sp.freeDims))
          if (coord[fdim] != sp.origin[n][rank + k])
            why = "a store tile crosses a view boundary";
        if (!why.empty())
          break;
        if (covered[addr]++) {
          why = "two stores of the grid write the same element";
          break;
        }
        ++count;
        for (unsigned i = 0; i < rank; ++i)
          live[i] = std::max(live[i], idx[i] + 1);
        for (unsigned j = 0; j < nd; ++j) {
          sp.boxLo[j] = std::min(sp.boxLo[j], coord[j]);
          sp.boxHi[j] = std::max(sp.boxHi[j], coord[j]);
        }
      }
      if (!why.empty())
        return std::nullopt;
    }
    if (!count) {
      why = "the store never writes";
      return std::nullopt;
    }
    int64_t volume = 1;
    for (unsigned j = 0; j < nd; ++j)
      volume *= sp.boxHi[j] - sp.boxLo[j] + 1;
    if (volume != count) {
      why = "the grid writes a non-rectangular region";
      return std::nullopt;
    }
    sp.full = (count == numel);
    // Tile phase per axis, for recovering origins from global indices.
    sp.extent.assign(rank, 0);
    sp.phase.assign(rank, -1);
    sp.loops = encl;
    for (unsigned i = 0; i < rank; ++i)
      if (sp.step[i] > 1 && !llvm::all_of(sp.origin, [&](auto &o) { return o[i] >= 0 && o[i] < sp.step[i]; })) {
        why = "a stepped store axis starts outside its first step";
        return std::nullopt;
      }
    for (unsigned i = 0; i < rank; ++i) {
      if (!sp.global[i] || sp.step[i] > 1)
        continue;
      auto phaseFor = [&](int64_t e) -> int64_t {
        int64_t ph = ((sp.origin[0][i] % e) + e) % e;
        bool aligned = llvm::all_of(sp.origin, [&](auto &o) { return ((o[i] % e) + e) % e == ph; });
        return aligned ? ph : -1;
      };
      // Tiles masked to a prefix (a group's channels in a wider block) may step by the prefix instead.
      sp.extent[i] = shape[i];
      sp.phase[i] = phaseFor(shape[i]);
      if (sp.phase[i] < 0 && live[i] < shape[i] && phaseFor(live[i]) >= 0) {
        sp.extent[i] = live[i];
        sp.phase[i] = phaseFor(live[i]);
      }
    }
    return sp;
  };

  // Failing every view, a term whose stride is no buffer dim's (a loop over column tiles of a row the tile does not
  // divide) steps along the dim of the largest stride dividing it instead, as one whose stride does not divide the
  // buffer.
  std::string first;
  for (int round = 0; round < 2; ++round) {
    if (round) {
      first = why;
      bool more = false;
      for (auto [k, c] : llvm::enumerate(comps))
        if (!c.digit && !stepped[k] && !llvm::is_contained(bufStrides, c.stride))
          stepped[k] = more = true;
      if (!more)
        break;
      probeStrides.clear();
      probeOrigins.clear();
    }
    for (int attempt = 0; attempt < 4; ++attempt) {
      std::set<int64_t, std::greater<int64_t>> strides{1};
      for (auto [k, c] : llvm::enumerate(comps))
        if (!stepped[k])
          strides.insert(c.stride);
      if (attempt == 0 || attempt == 2) {
        strides.insert(bufStrides.begin(), bufStrides.end());
        strides.insert(extraStrides.begin(), extraStrides.end());
      }
      if (attempt >= 2) {
        size_t before = strides.size();
        for (unsigned j = 0; j < probeStrides.size() && !probeOrigins.empty(); ++j) {
          int64_t step = 0;
          for (auto &o : probeOrigins)
            step = std::gcd(step, std::abs(o[j] - probeOrigins[0][j]));
          if (step > 1)
            strides.insert(probeStrides[j] * step);
        }
        if (strides.size() == before)
          break;
      }
      SmallVector<int64_t> st(strides.begin(), strides.end());
      bool nests = numel % st[0] == 0;
      for (unsigned j = 1; j < st.size(); ++j)
        nests &= st[j - 1] % st[j] == 0;
      if (!nests)
        continue;
      SmallVector<int> dimOf;
      for (auto [k, c] : llvm::enumerate(comps)) {
        if (stepped[k])
          dimOf.push_back(llvm::find_if(st, [&](int64_t v) { return v < c.stride && c.stride % v == 0; }) - st.begin());
        else
          dimOf.push_back(llvm::find(st, c.stride) - st.begin());
      }
      // Widen the linear term of an axis whose tiles cross a view dim boundary over the dim above it,
      // while no other term moves along that dim.
      SmallVector<unsigned> span(comps.size(), 1);
      for (;;) {
        int crossed = -1;
        if (attempt == 1)
          probeOrigins.clear();
        auto sp = tryView(st, dimOf, span, crossed, attempt == 1 ? &probeOrigins : nullptr);
        if (sp)
          return *sp;
        if (why == "two stores of the grid write the same element" || why == "the store never writes")
          return store.emitError("furiosa: ") << why, failure();
        int widen = -1;
        for (auto [k, c] : llvm::enumerate(comps))
          if (crossed >= 0 && c.axis == crossed && !c.digit && !stepped[k])
            widen = k;
        if (widen < 0 || dimOf[widen] < (int)span[widen])
          break;
        int above = dimOf[widen] - span[widen];
        bool taken = false;
        for (auto [k, c] : llvm::enumerate(comps))
          taken |= above <= dimOf[k] && above > dimOf[k] - (int)span[k];
        if (taken)
          break;
        ++span[widen];
      }
      if (attempt == 1)
        probeStrides = st;
    }
  }
  return store.emitError("furiosa: ") << (first.empty() ? why : first), failure();
}

/// The basis element k (a stored axis origin, or a free dim coordinate) as an expression of the
/// generic's dims; null when coordinates do not determine it.
AffineExpr Raiser::originExpr(const Gen &g, unsigned k) {
  unsigned rank = space->global.size(), nd = g.extent.size(), nv = space->viewShape.size();
  // View dims this generic does not iterate become placeholder dims past its own, ranging over the
  // written box: the origin is determined when it does not depend on them.
  SmallVector<AffineExpr> viewToGen;
  SmallVector<Interval> ranges;
  for (unsigned d = 0; d < nd; ++d)
    ranges.push_back({g.lo[d], g.lo[d] + g.extent[d] - 1});
  for (unsigned j = 0; j < nv; ++j) {
    viewToGen.push_back(getAffineDimExpr(g.coordPos[j] >= 0 ? g.coordPos[j] : nd + j, ctx));
    ranges.push_back({space->boxLo[j], space->boxHi[j]});
  }
  AffineExpr o;
  if (k >= rank) {
    o = viewToGen[space->freeDims[k - rank]];
  } else {
    if (!space->global[k])
      return AffineExpr();
    AffineExpr gl = space->global[k].replaceDims(viewToGen);
    if (space->step[k] > 1)
      o = gl % space->step[k];
    else if (space->phase[k] >= 0)
      o = gl - (gl - space->phase[k]) % space->extent[k];
    else
      return AffineExpr();
  }
  o = simplify(resolveDigits(simplify(o, nd + nv), ranges, {}), nd + nv);
  for (unsigned j = 0; j < nv; ++j)
    if (o.isFunctionOfDim(nd + j))
      return AffineExpr();
  return o;
}

/// `e` simplified with the ranges of the generic's dims and of the loop counters.
AffineExpr Raiser::tidy(const Gen &g, AffineExpr e) {
  unsigned nd = g.extent.size(), nl = loopTrip.size();
  SmallVector<Interval> dims, syms;
  for (unsigned d = 0; d < nd; ++d)
    dims.push_back({g.lo[d], g.lo[d] + g.extent[d] - 1});
  for (int64_t t : loopTrip)
    syms.push_back({0, std::max<int64_t>(t, 1) - 1});
  AffineExpr r = simplify(resolveDigits(simplify(e, nd, nl), dims, syms), nd, nl);
  // A store-view dim this generic iterates that the expression divides into digits.
  if (space)
    r.walk([&](AffineExpr x) {
      if (x.getKind() != AffineExprKind::FloorDiv && x.getKind() != AffineExprKind::Mod)
        return;
      auto bin = cast<AffineBinaryOpExpr>(x);
      auto d = dyn_cast<AffineDimExpr>(bin.getLHS());
      auto c = dyn_cast<AffineConstantExpr>(bin.getRHS());
      if (!d || !c)
        return;
      for (auto [j, pos] : llvm::enumerate(g.coordPos))
        if (pos == (int)d.getPosition() && c.getValue() > 1 && c.getValue() < space->viewShape[j] &&
            space->viewShape[j] % c.getValue() == 0)
          digitUses.insert({(unsigned)j, c.getValue()});
    });
  return r;
}

/// One expression of the generic's dims for a per-instance symbolic value: substitute each axis of `v`
/// (its global index minus that instance's origin, or a generic dim) and each loop counter; the
/// instances must then agree up to a constant that is an affine function of the store origins.
FailureOr<AffineExpr> Raiser::rewrite(Operation *at, const Gen &g, ArrayRef<Axis> env,
                                      std::function<AffineExpr(const Sym &)> select, Value v) {
  SmallVector<AffineExpr> viewToGen;
  for (int pos : g.coordPos)
    viewToGen.push_back(pos >= 0 ? getAffineDimExpr(pos, ctx) : getAffineConstantExpr(0, ctx));
  SmallVector<AffineExpr> per;
  static const Sym none; // v null: `select` reads the env alone
  for (auto [n, ev] : llvm::enumerate(act)) {
    const Sym *s = v ? ev->get(v) : &none;
    if (!s)
      return at->emitError("furiosa: value is not symbolic in every instance"), failure();
    bool outside = false;
    std::function<AffineExpr(const Axis &)> axisAt = [&](const Axis &a) -> AffineExpr {
      if (a.kind == Axis::Store) {
        AffineExpr gl = space->global[a.dim];
        for (unsigned j = 0; j < g.coordPos.size(); ++j)
          outside |= g.coordPos[j] < 0 && gl.isFunctionOfDim(j);
        return space->step[a.dim] > 1 ? gl.replaceDims(viewToGen).floorDiv(space->step[a.dim])
                                      : gl.replaceDims(viewToGen) - space->origin[n][a.dim];
      }
      if (a.kind == Axis::Iter)
        return getAffineDimExpr(a.dim, ctx);
      if (a.kind == Axis::Expr) {
        SmallVector<AffineExpr> args;
        for (const Axis &x : a.of)
          args.push_back(axisAt(x));
        return a.expr.replaceDims(args);
      }
      return getAffineConstantExpr(0, ctx);
    };
    SmallVector<AffineExpr> dims;
    for (const Axis &a : env)
      dims.push_back(axisAt(a));
    if (outside)
      return at->emitError("furiosa: internal: coordinate dim not in this generic"), failure();
    SmallVector<AffineExpr> syms;
    for (unsigned l = 0; l < loopTrip.size(); ++l) {
      auto it = g.loopPos.find(l);
      auto around = llvm::find(space->loops, l);
      if (it != g.loopPos.end())
        syms.push_back(getAffineDimExpr(it->second, ctx));
      else if (g.loopIv.count(l))
        syms.push_back(getAffineSymbolExpr(l, ctx));
      else if (around != space->loops.end()) {
        unsigned axis = around - space->loops.begin();
        AffineExpr gl = space->global[axis] ? space->global[axis].replaceDims(viewToGen) : getAffineConstantExpr(0, ctx);
        syms.push_back(space->step[axis] > 1 ? gl.floorDiv(space->step[axis]) : gl - space->origin[n][axis]);
      } else
        syms.push_back(getAffineConstantExpr(0, ctx));
    }
    AffineExpr e = select(*s);
    for (unsigned l = 0; l < loopTrip.size(); ++l)
      if (e.isFunctionOfSymbol(l) && !g.loopPos.count(l) && !g.loopIv.count(l) && !llvm::is_contained(space->loops, l))
        return at->emitError("furiosa: a loop counter is used outside its loop"), failure();
    per.push_back(tidy(g, e.replaceDimsAndSymbols(dims, syms)));
  }
  auto [rest0, c0] = splitConstant(per[0]);
  SmallVector<int64_t> consts;
  bool same = true;
  for (AffineExpr e : per) {
    auto [rest, c] = splitConstant(e);
    if (rest != rest0)
      return at->emitError("furiosa: an address differs structurally between program instances"), failure();
    consts.push_back(c);
    same &= (c == c0);
  }
  if (same)
    return per[0];

  {
    // Try quietly; the symbolic path below reports if both fail.
    ScopedDiagnosticHandler quiet(ctx, [](Diagnostic &) { return success(); });
    auto fitted = fitOrigins(at, g, consts);
    if (succeeded(fitted))
      return tidy(g, rest0 + *fitted);
  }
  auto sym = rewriteSymbolic(at, g, env, select, v);
  if (succeeded(sym))
    return sym;
  return fitOrigins(at, g, consts); // for its diagnostic
}

/// The same rewrite from the symbolic-pid evaluation: each program id is recovered from a tile origin
/// that depends on it alone (origin = a * pid + c, so pid = (origin - c) floordiv a), which handles
/// per-instance values that are not affine in the origins (pid floordiv 2, ...).
FailureOr<AffineExpr> Raiser::rewriteSymbolic(Operation *at, const Gen &g, ArrayRef<Axis> env,
                                              std::function<AffineExpr(const Sym &)> select, Value v) {
  if (!symEval || !v)
    return failure();
  const Sym *s = symEval->get(v), *s0 = act[0]->get(v);
  // Masks fold differently with constant program ids; only use shapes that line up.
  if (!s || !s0 || s->kind != s0->kind || s->conds.size() != s0->conds.size())
    return failure();
  unsigned nl = loopTrip.size(), nb = space->origin[0].size();
  SmallVector<SmallVector<int64_t>> pids;
  for (Evaluator *ev : act)
    pids.push_back(SmallVector<int64_t>(ev->programId()));
  // Each basis element (axis origin, free coordinate) as an affine function of the program ids.
  SmallVector<SmallVector<int64_t>> basisOf;
  for (unsigned k = 0; k < nb; ++k) {
    SmallVector<int64_t> y;
    for (auto &o : space->origin)
      y.push_back(o[k]);
    auto c = fitAffine(pids, y);
    if (!c)
      return failure();
    basisOf.push_back(*c);
  }
  // Each program id from a basis element that depends on it alone.
  SmallVector<AffineExpr> pidExpr(3);
  for (unsigned d = 0; d < 3; ++d) {
    if (llvm::all_of(pids, [&](auto &p) { return p[d] == pids[0][d]; })) {
      pidExpr[d] = getAffineConstantExpr(pids[0][d], ctx);
      continue;
    }
    for (unsigned k = 0; k < nb && !pidExpr[d]; ++k) {
      bool alone = basisOf[k][d] != 0;
      for (unsigned e = 0; e < 3; ++e)
        alone &= e == d || basisOf[k][e] == 0;
      AffineExpr o = alone ? originExpr(g, k) : AffineExpr();
      if (o) {
        pidExpr[d] = (o - basisOf[k][3]).floorDiv(basisOf[k][d]);
        if (k < space->global.size() && select(*s).isFunctionOfSymbol(nl + d))
          originUses.insert(k);
      }
    }
    if (!pidExpr[d])
      return failure();
  }
  SmallVector<AffineExpr> viewToGen;
  for (int pos : g.coordPos)
    viewToGen.push_back(pos >= 0 ? getAffineDimExpr(pos, ctx) : getAffineConstantExpr(0, ctx));
  std::function<AffineExpr(const Axis &)> axisAt = [&](const Axis &a) -> AffineExpr {
    if (a.kind == Axis::Store) {
      AffineExpr origin = getAffineConstantExpr(basisOf[a.dim][3], ctx);
      for (unsigned d = 0; d < 3; ++d)
        origin = origin + pidExpr[d] * basisOf[a.dim][d];
      AffineExpr gl = space->global[a.dim].replaceDims(viewToGen);
      return space->step[a.dim] > 1 ? gl.floorDiv(space->step[a.dim]) : gl - origin;
    }
    if (a.kind == Axis::Iter)
      return getAffineDimExpr(a.dim, ctx);
    if (a.kind == Axis::Expr) {
      SmallVector<AffineExpr> args;
      for (const Axis &x : a.of)
        args.push_back(axisAt(x));
      return a.expr.replaceDims(args);
    }
    return getAffineConstantExpr(0, ctx);
  };
  SmallVector<AffineExpr> dims;
  for (const Axis &a : env)
    dims.push_back(axisAt(a));
  SmallVector<AffineExpr> syms;
  for (unsigned l = 0; l < nl; ++l) {
    auto it = g.loopPos.find(l);
    auto around = llvm::find(space->loops, l);
    if (it != g.loopPos.end())
      syms.push_back(getAffineDimExpr(it->second, ctx));
    else if (g.loopIv.count(l))
      syms.push_back(getAffineSymbolExpr(l, ctx));
    else if (around != space->loops.end())
      syms.push_back(AffineExpr()); // filled below
    else
      syms.push_back(getAffineConstantExpr(0, ctx));
  }
  // Counters of the loops around the store: their store axis, as for an env axis.
  for (unsigned l = 0; l < nl; ++l) {
    auto around = llvm::find(space->loops, l);
    if (syms[l] || around == space->loops.end())
      continue;
    unsigned axis = around - space->loops.begin();
    AffineExpr origin = getAffineConstantExpr(basisOf[axis][3], ctx);
    for (unsigned d = 0; d < 3; ++d)
      origin = origin + pidExpr[d] * basisOf[axis][d];
    AffineExpr gl = space->global[axis] ? space->global[axis].replaceDims(viewToGen) : getAffineConstantExpr(0, ctx);
    syms[l] = space->step[axis] > 1 ? gl.floorDiv(space->step[axis]) : gl - origin;
  }
  for (unsigned d = 0; d < 3; ++d)
    syms.push_back(pidExpr[d]);
  return tidy(g, select(*s).replaceDimsAndSymbols(dims, syms));
}

/// consts[n] (one integer per instance) as beta + sum_k alpha_k * origin_k, with each origin rewritten
/// in the generic's coordinates (least squares, then checked exactly).
FailureOr<AffineExpr> Raiser::fitOrigins(Operation *at, const Gen &g, ArrayRef<int64_t> consts) {
  unsigned nb = space->origin[0].size();
  auto fit = fitAffine(space->origin, consts);
  // A value that is an origin over a tile size (a causal trip count, origin / BLOCK_M): fit it times the
  // origin's step between instances, then divide. Exact, as fitAffine checks every instance and an
  // origin expression always lands on an instance's origin.
  int64_t scale = 1;
  for (unsigned f = 0; f < nb && !fit; ++f) {
    int64_t step = 0;
    for (auto &o : space->origin)
      step = std::gcd(step, std::abs(o[f] - space->origin[0][f]));
    SmallVector<int64_t> scaled;
    for (int64_t c : consts)
      scaled.push_back(c * step);
    if (step > 1 && (fit = fitAffine(space->origin, scaled)))
      scale = step;
  }
  if (!fit) {
    std::string detail;
    llvm::raw_string_ostream os(detail);
    for (unsigned m = 0; m < std::min<unsigned>(act.size(), 4); ++m) {
      os << " [" << consts[m] << " @";
      for (int64_t o : space->origin[m])
        os << " " << o;
      os << "]";
    }
    return at->emitError("furiosa: an address varies between program instances in a way the store "
                         "tiles do not explain; constant @ origins:")
               << detail,
           failure();
  }
  ArrayRef<int64_t> alpha(*fit);
  int64_t beta = alpha[nb];
  AffineExpr out = getAffineConstantExpr(beta, ctx);
  for (unsigned k = 0; k < nb; ++k)
    if (alpha[k]) {
      AffineExpr o = originExpr(g, k);
      if (!o) {
        std::string view;
        for (int64_t d : space->viewShape)
          view += std::to_string(d) + " ";
        return at->emitError("furiosa: an address depends on a tile origin the coordinates do not "
                             "determine (basis ")
                   << k << " of " << nb << ", store view [ " << view << "])",
               failure();
      }
      out = out + o * alpha[k];
    }
  for (unsigned k = 0; k < std::min<unsigned>(nb, space->global.size()); ++k)
    if (alpha[k] && !isa<AffineConstantExpr>(originExpr(g, k)))
      originUses.insert(k);
  return tidy(g, scale > 1 ? out.floorDiv(scale) : out);
}

/// The bounds iteration t of loop `id` must stay below to run in the instance at the current point of `g`: its trip
/// count. A trip up to a min of bounds (rows from start to min(start + ROWS, M)) need not be affine in the tile
/// origins where the distance to each term of the min is: then one bound per term, (term - lower bound) ceildiv step.
FailureOr<SmallVector<AffineExpr>> Raiser::tripBounds(Gen &g, unsigned id) {
  scf::ForOp loop;
  for (auto [op, i] : loopIds)
    if (i == id)
      loop = cast<scf::ForOp>(op);
  SmallVector<int64_t> trip;
  for (Evaluator *ev : act)
    trip.push_back(ev->trips.lookup(id));
  SmallVector<Value> terms;
  std::function<void(Value)> split = [&](Value v) {
    if (auto m = v.getDefiningOp<arith::MinSIOp>()) {
      split(m.getLhs());
      split(m.getRhs());
    } else {
      terms.push_back(v);
    }
  };
  split(loop.getUpperBound());
  auto value = [](Evaluator *ev, Value v) -> std::optional<int64_t> {
    if (const Sym *s = ev->get(v); s && s->kind == Sym::Int && s->data.empty())
      if (auto c = dyn_cast<AffineConstantExpr>(s->expr))
        return c.getValue();
    return std::nullopt;
  };
  bool whole;
  {
    ScopedDiagnosticHandler quiet(ctx, [](Diagnostic &) { return success(); });
    whole = terms.size() == 1 || succeeded(fitOrigins(loop, g, trip));
  }
  std::vector<SmallVector<int64_t>> trips;
  if (whole)
    trips.push_back(trip);
  for (Value term : whole ? ArrayRef<Value>() : ArrayRef<Value>(terms)) {
    SmallVector<int64_t> distance;
    for (Evaluator *ev : act) {
      auto lb = value(ev, loop.getLowerBound()), ub = value(ev, term), step = value(ev, loop.getStep());
      if (!lb || !ub || !step)
        return loop.emitError("furiosa: a loop bound term is not an index in every instance"), failure();
      distance.push_back(floorDiv(*ub - *lb + *step - 1, *step));
    }
    trips.push_back(distance);
  }
  SmallVector<AffineExpr> out;
  for (ArrayRef<int64_t> bound : trips) {
    auto e = fitOrigins(loop, g, bound);
    if (failed(e))
      return failure();
    if (auto c = dyn_cast<AffineConstantExpr>(*e); !whole && c && c.getValue() >= loopTrip[id])
      continue; // every iteration is below it
    out.push_back(*e);
  }
  return out;
}

/// Whether the instance at the current point of `g` runs iteration `t` of loop `id`.
FailureOr<Value> Raiser::tripGuard(Gen &g, unsigned id, Value t) {
  auto bounds = tripBounds(g, id);
  if (failed(bounds))
    return failure();
  Value active;
  for (AffineExpr e : *bounds) {
    Value lt = arith::CmpIOp::create(*g.ob, loc, arith::CmpIPredicate::slt, t, exprValue(g, e));
    active = active ? arith::AndIOp::create(*g.ob, loc, active, lt).getResult() : lt;
  }
  return active;
}

Value Raiser::index(Gen &g, unsigned dim) {
  if (!g.index[dim]) {
    Value i = linalg::IndexOp::create(*g.ob, loc, dim);
    if (g.lo[dim])
      i = arith::AddIOp::create(*g.ob, loc, i, arith::ConstantIndexOp::create(*g.ob, loc, g.lo[dim]));
    g.index[dim] = i;
  }
  return g.index[dim];
}

Value Raiser::exprValue(Gen &g, AffineExpr e) {
  OpBuilder &ob = *g.ob;
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return arith::ConstantIndexOp::create(ob, loc, c.getValue());
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return index(g, d.getPosition());
  if (auto sy = dyn_cast<AffineSymbolExpr>(e))
    return g.loopIv.lookup(sy.getPosition());
  auto bin = cast<AffineBinaryOpExpr>(e);
  Value l = exprValue(g, bin.getLHS()), r = exprValue(g, bin.getRHS());
  switch (e.getKind()) {
  case AffineExprKind::Add:
    return arith::AddIOp::create(ob, loc, l, r);
  case AffineExprKind::Mul:
    return arith::MulIOp::create(ob, loc, l, r);
  case AffineExprKind::FloorDiv:
    return arith::FloorDivSIOp::create(ob, loc, l, r);
  case AffineExprKind::CeilDiv:
    return arith::CeilDivSIOp::create(ob, loc, l, r);
  case AffineExprKind::Mod: {
    Value m = arith::RemSIOp::create(ob, loc, l, r);
    Value neg = arith::CmpIOp::create(ob, loc, arith::CmpIPredicate::slt, m,
                                         arith::ConstantIndexOp::create(ob, loc, 0));
    return arith::SelectOp::create(ob, loc, neg, arith::AddIOp::create(ob, loc, m, r), m);
  }
  default:
    llvm_unreachable("affine kind");
  }
}

Value Raiser::reshape(Value t, ArrayRef<int64_t> shape) {
  auto type = cast<RankedTensorType>(t.getType());
  if (type.getShape() == shape)
    return t;
  int64_t n = type.getNumElements();
  if (type.getRank() != 1) {
    ReassociationIndices all;
    for (int64_t i = 0; i < type.getRank(); ++i)
      all.push_back(i);
    t = tensor::CollapseShapeOp::create(b, loc, RankedTensorType::get({n}, type.getElementType()), t,
                                          ArrayRef<ReassociationIndices>{all});
  }
  if (shape.size() == 1)
    return t;
  ReassociationIndices all;
  for (unsigned i = 0; i < shape.size(); ++i)
    all.push_back(i);
  return tensor::ExpandShapeOp::create(b, loc, RankedTensorType::get(shape, type.getElementType()), t,
                                         ArrayRef<ReassociationIndices>{all});
}

Value Raiser::flatOf(Value version, int arg) {
  Value &f = flat[version.getAsOpaquePointer()];
  if (!f) {
    OpBuilder::InsertionGuard guard(b);
    if (auto def = version.getDefiningOp())
      b.setInsertionPointAfter(def);
    else
      b.setInsertionPointToStart(version.getParentBlock());
    f = reshape(version, {buffers[arg].numel()});
  }
  return f;
}

/// The version of buffer `arg` a load reads: what the last store before it in program order left
/// (within one program, a load sees its own earlier stores and none of the later ones).
Value Raiser::versionAt(Operation *load, int arg) {
  auto before = [&](Operation *a, Operation *b) {
    for (Operation *x = a; x; x = x->getParentOp())
      if (Operation *y = x->getBlock() ? x->getBlock()->findAncestorOpInBlock(*b) : nullptr)
        return x != y && x->isBeforeInBlock(y);
    return false;
  };
  Value v = args[arg];
  for (auto &[st, a, after] : stores)
    if (a == arg && before(st, load))
      v = after;
  return v;
}

Value Raiser::constantOf(Type t, Attribute a) {
  if (auto d = dyn_cast_or_null<DenseElementsAttr>(a))
    a = d.getSplatValue<Attribute>();
  if (!a)
    a = b.getZeroAttr(t);
  return arith::ConstantOp::create(b, loc, cast<TypedAttr>(a));
}

FailureOr<Value> Raiser::emitDataTerms(Gen &g, const Sym &s0, ArrayRef<Axis> env) {
  Value sum;
  for (const DataTerm &t : s0.data) {
    SmallVector<int64_t> tshape = shapeOf(t.value.getType());
    SmallVector<Axis> tenv(tshape.size());
    for (auto [i, ta] : llvm::enumerate(t.axisOf))
      if (ta >= 0 && tshape[ta] > 1)
        tenv[ta] = env[i];
    auto v = emit(g, t.value, tenv);
    if (failed(v))
      return failure();
    Value iv = arith::IndexCastOp::create(*g.ob, loc, g.ob->getIndexType(), *v);
    if (t.sign < 0)
      iv = arith::SubIOp::create(*g.ob, loc, arith::ConstantIndexOp::create(*g.ob, loc, 0), iv);
    sum = sum ? arith::AddIOp::create(*g.ob, loc, sum, iv).getResult() : iv;
  }
  return sum;
}

FailureOr<Value> Raiser::emitSym(Gen &g, Value v, const Sym &s0, ArrayRef<Axis> env) {
  OpBuilder &ob = *g.ob;
  Operation *at = v.getDefiningOp() ? v.getDefiningOp() : kernel.getOperation();
  if (s0.kind == Sym::Mask) {
    SmallVector<AffineExpr> conds; // every one rewritten before an op is built: a failure leaves none behind
    for (unsigned c = 0; c < s0.conds.size(); ++c) {
      auto e = rewrite(at, g, env, [c](const Sym &s) { return s.conds[c]; }, v);
      if (failed(e))
        return failure();
      conds.push_back(*e);
    }
    Value on = arith::ConstantIntOp::create(ob, loc, ob.getI1Type(), s0.never ? 0 : 1);
    for (AffineExpr e : conds) {
      Value ok = arith::CmpIOp::create(ob, loc, arith::CmpIPredicate::sge, exprValue(g, e),
                                          arith::ConstantIndexOp::create(ob, loc, 0));
      on = arith::AndIOp::create(ob, loc, on, ok);
    }
    return on;
  }
  if (s0.kind != Sym::Int)
    return at->emitError("furiosa: a pointer is used as data"), failure();
  if (llvm::any_of(act, [&](Evaluator *ev) { return ev->get(v) && ev->get(v)->data != s0.data; }))
    return at->emitError("furiosa: an index adds different data in different program instances"), failure();
  auto e = rewrite(at, g, env, [](const Sym &s) { return s.expr; }, v);
  if (failed(e))
    return failure();
  Value iv = exprValue(g, *e);
  auto data = emitDataTerms(g, s0, env);
  if (failed(data))
    return failure();
  if (*data)
    iv = arith::AddIOp::create(ob, loc, iv, *data);
  return arith::IndexCastOp::create(ob, loc, getElementTypeOrSelf(v.getType()), iv).getResult();
}

FailureOr<Value> Raiser::emitLoad(Gen &g, Read load, ArrayRef<Axis> env) {
  OpBuilder &ob = *g.ob;
  const Sym *p0 = act[0]->get(load.getPtr());
  if (!p0 || p0->kind != Sym::Ptr)
    return load.emitError("furiosa: load address is not a pointer expression"), failure();
  int arg = p0->base;
  for (Evaluator *ev : act) {
    const Sym *p = ev->get(load.getPtr());
    if (!p || p->base != arg || p->data != p0->data)
      return load.emitError("furiosa: load base differs between program instances"), failure();
  }
  auto addr = rewrite(load, g, env, [](const Sym &s) { return s.expr; }, load.getPtr());
  if (failed(addr))
    return failure();
  SmallVector<Interval> ranges, symRanges;
  for (unsigned d = 0; d < g.extent.size(); ++d)
    ranges.push_back({g.lo[d], g.lo[d] + g.extent[d] - 1});
  for (int64_t trip : loopTrip)
    symRanges.push_back({0, std::max<int64_t>(trip - 1, 0)});

  Value on;
  if (load.getMask()) {
    const Sym *m0 = act[0]->get(load.getMask());
    if (!m0 || m0->kind != Sym::Mask)
      return load.emitError("furiosa: load mask is not an index predicate"), failure();
    for (unsigned c = 0; c < m0->conds.size(); ++c) {
      auto e = rewrite(load, g, env, [c](const Sym &s) { return s.conds[c]; }, load.getMask());
      if (failed(e))
        return failure();
      if (bounds(*e, ranges, symRanges).lo >= 0)
        continue; // the iteration domain already guarantees it
      Value ok = arith::CmpIOp::create(ob, loc, arith::CmpIPredicate::sge, exprValue(g, *e),
                                          arith::ConstantIndexOp::create(ob, loc, 0));
      on = on ? arith::AndIOp::create(ob, loc, on, ok).getResult() : ok;
    }
    if (m0->never)
      on = arith::ConstantIntOp::create(ob, loc, ob.getI1Type(), 0);
  }

  int64_t numel = buffers[arg].numel();
  Interval range = bounds(*addr, ranges, symRanges);
  if (range.lo < 0 || range.hi >= numel)
    range = exactBounds(*addr, ranges, symRanges);
  // A guarded load leaves the buffer only where its value is discarded: it reads the window of addresses it spans,
  // holding what the clamped address reads, at an affine address from the window's start (which furiosa-structure
  // promotes). A masked or data-dependent one is clamped in the body.
  bool padded = !on && p0->data.empty() && (range.lo < 0 || range.hi >= numel) && range.lo < numel && range.hi >= 0;
  Value i = exprValue(g, padded ? *addr - range.lo : *addr);
  auto data = emitDataTerms(g, *p0, env);
  if (failed(data))
    return failure();
  bool mayLeave = *data || range.lo < 0 || range.hi >= numel;
  // A load in an iteration that a shorter trip masks off may leave its buffer: its value is discarded.
  bool guarded = llvm::any_of(g.loopPos, [&](auto kv) { return varTrip[kv.first]; }) ||
                 llvm::any_of(g.loopIv, [&](auto kv) { return varTrip[kv.first]; });
  // A masked load that leaves its buffer in iterations a shorter trip masks off reads `other` there too (its value is
  // discarded): the mask then bounds every index (rows up to the last band's end), so furiosa-structure promotes it.
  if (on && guarded && mayLeave && !*data) {
    auto mask = [&](unsigned l, AffineExpr t) -> LogicalResult {
      auto bounds = tripBounds(g, l);
      if (failed(bounds))
        return failure();
      for (AffineExpr e : *bounds) {
        Value ok = arith::CmpIOp::create(ob, loc, arith::CmpIPredicate::sge, exprValue(g, tidy(g, e - t - 1)),
                                         arith::ConstantIndexOp::create(ob, loc, 0));
        on = arith::AndIOp::create(ob, loc, on, ok);
      }
      return success();
    };
    for (auto [l, pos] : g.loopPos)
      if (varTrip[l] && failed(mask(l, getAffineDimExpr(pos, ctx))))
        return failure();
    for (auto [l, iv] : g.loopIv)
      if (varTrip[l] && failed(mask(l, getAffineSymbolExpr(l, ctx))))
        return failure();
  }
  if (*data)
    i = arith::AddIOp::create(ob, loc, i, *data);
  if (mayLeave) {
    if (!on && !*data && !guarded)
      return load.emitError("furiosa: an unmasked load may read outside its buffer"), failure();
    if (!padded) {
      i = arith::MaxSIOp::create(ob, loc, i, arith::ConstantIndexOp::create(ob, loc, 0));
      i = arith::MinSIOp::create(ob, loc, i, arith::ConstantIndexOp::create(ob, loc, numel - 1));
    }
  }
  Value src = flatOf(versionAt(load, arg), arg);
  if (padded) {
    int64_t inLo = std::max<int64_t>(range.lo, 0), inHi = std::min<int64_t>(range.hi, numel - 1);
    OpBuilder::InsertionGuard guard(b);
    if (Operation *def = src.getDefiningOp())
      b.setInsertionPointAfter(def);
    else
      b.setInsertionPointToStart(src.getParentBlock());
    MLIRContext *ctx = b.getContext();
    Type et = cast<RankedTensorType>(src.getType()).getElementType();
    auto at = [&](int64_t v) { return getAsIndexOpFoldResult(ctx, {v}); };
    // The window holds what the clamped address would read: the buffer's part of it, its first and last element
    // repeated before and after (each a box copy or a broadcast).
    Value window = tensor::EmptyOp::create(b, loc, ArrayRef<int64_t>{range.hi - range.lo + 1}, et);
    auto put = [&](Value part, int64_t offset) {
      int64_t n = cast<RankedTensorType>(part.getType()).getDimSize(0);
      window = tensor::InsertSliceOp::create(b, loc, part, window, at(offset), at(n), at(1));
    };
    auto repeat = [&](int64_t element, int64_t n) {
      Value one = tensor::ExtractSliceOp::create(b, loc, src, at(element), at(1), at(1));
      Value out = tensor::EmptyOp::create(b, loc, ArrayRef<int64_t>{n}, et);
      return linalg::GenericOp::create(
                 b, loc, TypeRange{out.getType()}, ValueRange{one}, ValueRange{out},
                 ArrayRef<AffineMap>{AffineMap::get(1, 0, b.getAffineConstantExpr(0)), b.getMultiDimIdentityMap(1)},
                 SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
                 [](OpBuilder &nb, Location l, ValueRange args) { linalg::YieldOp::create(nb, l, args[0]); })
          .getResult(0);
    };
    if (range.lo < 0)
      put(repeat(0, -range.lo), 0);
    put(inLo > 0 || inHi < numel - 1
            ? tensor::ExtractSliceOp::create(b, loc, src, at(inLo), at(inHi - inLo + 1), at(1)).getResult()
            : src,
        inLo - range.lo);
    if (range.hi >= numel)
      put(repeat(numel - 1, range.hi - numel + 1), numel - range.lo);
    src = window;
  }
  Value elem = tensor::ExtractOp::create(ob, loc, src, ValueRange{i});
  if (Type t = getElementTypeOrSelf(load.getType()); t != elem.getType()) // through a pointer bitcast
    elem = arith::BitcastOp::create(ob, loc, t, elem);
  if (on) {
    Value other;
    if (Value o = load.getOther(); o && !o.getDefiningOp<arith::ConstantOp>()) {
      auto v = emit(g, o, env); // any value of the load's shape
      if (failed(v))
        return failure();
      other = *v;
    } else {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(ob.getInsertionBlock(), ob.getInsertionPoint());
      other = constantOf(elem.getType(), o ? o.getDefiningOp<arith::ConstantOp>().getValue() : Attribute());
    }
    elem = arith::SelectOp::create(ob, loc, on, elem, other);
  }
  return elem;
}

/// The math-dialect op a libdevice call means, by its symbol (__nv_asinf, __nv_pow, ...), or "".
static std::string libdeviceOp(tt::ExternElementwiseOp call) {
  StringRef name = call.getSymbol();
  if (!name.consume_front("__nv_"))
    return "";
  static const llvm::StringMap<StringRef> math = {
      {"asin", "asin"},   {"acos", "acos"},   {"atan", "atan"},   {"atan2", "atan2"}, {"sinh", "sinh"},
      {"cosh", "cosh"},   {"tanh", "tanh"},   {"asinh", "asinh"}, {"acosh", "acosh"}, {"atanh", "atanh"},
      {"sin", "sin"},     {"cos", "cos"},     {"tan", "tan"},     {"exp", "exp"},     {"exp2", "exp2"},
      {"expm1", "expm1"}, {"log", "log"},     {"log2", "log2"},   {"log10", "log10"}, {"log1p", "log1p"},
      {"sqrt", "sqrt"},   {"rsqrt", "rsqrt"}, {"cbrt", "cbrt"},   {"erf", "erf"},     {"erfc", "erfc"},
      {"pow", "powf"},    {"fabs", "absf"},   {"floor", "floor"}, {"ceil", "ceil"},   {"trunc", "trunc"},
      {"round", "round"}, {"rint", "roundeven"}, {"fma", "fma"}};
  auto it = math.find(name);
  if (it == math.end() && name.consume_back("f")) // the fp32 variant: __nv_asinf
    it = math.find(name);
  return it == math.end() ? "" : ("math." + it->second).str();
}

FailureOr<Value> Raiser::emitElementwise(Gen &g, Operation *op, ArrayRef<Axis> env) {
  SmallVector<Value> operands;
  for (Value o : op->getOperands()) {
    auto v = emit(g, o, env);
    if (failed(v))
      return failure();
    operands.push_back(*v);
  }
  OpBuilder &ob = *g.ob;
  Type rt = getElementTypeOrSelf(op->getResult(0).getType());
  if (isa<tt::PreciseDivFOp>(op))
    return arith::DivFOp::create(ob, loc, operands[0], operands[1]).getResult();
  if (isa<tt::PreciseSqrtOp>(op))
    return math::SqrtOp::create(ob, loc, operands[0]).getResult();
  if (isa<tt::FpToFpOp>(op)) {
    unsigned from = operands[0].getType().getIntOrFloatBitWidth(), to = rt.getIntOrFloatBitWidth();
    if (to > from)
      return arith::ExtFOp::create(ob, loc, rt, operands[0]).getResult();
    return arith::TruncFOp::create(ob, loc, rt, operands[0]).getResult();
  }
  if (isa<tt::BitcastOp>(op))
    return arith::BitcastOp::create(ob, loc, rt, operands[0]).getResult();
  if (isa<tt::MulhiUIOp>(op)) // the high half of the unsigned product (math.umulhi)
    return arith::MulUIExtendedOp::create(ob, loc, operands[0], operands[1]).getHigh();
  if (auto call = dyn_cast<tt::ExternElementwiseOp>(op)) { // a libdevice function
    OperationState state(loc, libdeviceOp(call));
    state.addOperands(operands);
    state.addTypes(rt);
    return ob.create(state)->getResult(0);
  }
  if (auto clamp = dyn_cast<tt::ClampFOp>(op)) {
    Value lo = arith::MaximumFOp::create(ob, loc, operands[0], operands[1]);
    return arith::MinimumFOp::create(ob, loc, lo, operands[2]).getResult();
  }
  OperationState state(loc, op->getName());
  state.addOperands(operands);
  state.addTypes(rt);
  state.addAttributes(op->getAttrs());
  return ob.create(state)->getResult(0);
}

static bool isElementwise(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumRegions() != 0)
    return false;
  if (isa<tt::PreciseDivFOp, tt::PreciseSqrtOp, tt::FpToFpOp, tt::BitcastOp, tt::ClampFOp, tt::MulhiUIOp>(op))
    return true;
  if (auto call = dyn_cast<tt::ExternElementwiseOp>(op))
    return !libdeviceOp(call).empty();
  Dialect *d = op->getDialect();
  return d && (isa<arith::ArithDialect>(d) || isa<math::MathDialect>(d)) && !isa<arith::ConstantOp>(op);
}

/// The index of axis `k` of `env` at the current point of `g`.
FailureOr<Value> Raiser::axisValue(Gen &g, Operation *at, ArrayRef<Axis> env, unsigned k) {
  auto e = rewrite(at, g, env, [&](const Sym &) { return getAffineDimExpr(k, ctx); }, Value());
  if (failed(e))
    return failure();
  return exprValue(g, *e);
}

/// tl.map_elementwise: the scalar function's blocks, all pure, evaluated in order (an acyclic CFG,
/// successors after their predecessors) with each block's arguments selected by the branch taken.
FailureOr<Value> Raiser::emitMapElementwise(Gen &g, tt::MapElementwiseOp op, ArrayRef<Axis> env) {
  if (op.getPackAttr().getInt() != 1)
    return op.emitError("furiosa: a packed map_elementwise is not raised yet"), failure();
  Region &region = op->getRegion(0);
  OpBuilder &ob = *g.ob;
  IRMapping map;
  for (auto [arg, src] : llvm::zip(region.front().getArguments(), op.getSrcs())) {
    auto x = emit(g, src, env);
    if (failed(x))
      return failure();
    map.map(arg, *x);
  }
  auto lookup = [&](Value v) -> FailureOr<Value> {
    if (Value m = map.lookupOrNull(v))
      return m;
    if (region.isAncestor(v.getParentRegion()))
      return op.emitError("furiosa: internal: map_elementwise value used before its block"), failure();
    return emit(g, v, {}); // a scalar from outside the function
  };
  DenseMap<Block *, unsigned> order;
  for (auto [i, blk] : llvm::enumerate(region))
    order[&blk] = i;
  DenseMap<Block *, SmallVector<std::pair<Value, SmallVector<Value>>>> incoming;
  DenseMap<Block *, Value> reach;
  reach[&region.front()] = arith::ConstantIntOp::create(ob, loc, ob.getI1Type(), 1);
  Value result;
  for (Block &blk : region) {
    if (&blk != &region.front()) {
      auto &in = incoming[&blk];
      if (in.empty())
        continue; // unreachable
      Value p = in[0].first;
      for (auto &[q, vals] : ArrayRef(in).drop_front())
        p = arith::OrIOp::create(ob, loc, p, q);
      reach[&blk] = p;
      for (auto [i, arg] : llvm::enumerate(blk.getArguments())) {
        Value x = in[0].second[i];
        for (auto &[q, vals] : ArrayRef(in).drop_front())
          x = arith::SelectOp::create(ob, loc, q, vals[i], x);
        map.map(arg, x);
      }
    }
    for (Operation &o : blk.without_terminator()) {
      if (o.getNumRegions() || !isMemoryEffectFree(&o))
        return o.emitError("furiosa: a map_elementwise function with side effects or regions is not raised"), failure();
      for (Value operand : o.getOperands())
        if (!map.contains(operand)) {
          auto x = lookup(operand);
          if (failed(x))
            return failure();
          map.map(operand, *x);
        }
      ob.clone(o, map);
    }
    Operation *term = blk.getTerminator();
    auto values = [&](ValueRange vs) -> FailureOr<SmallVector<Value>> {
      SmallVector<Value> out;
      for (Value x : vs) {
        auto y = lookup(x);
        if (failed(y))
          return failure();
        out.push_back(*y);
      }
      return out;
    };
    Value here = reach[&blk];
    auto edge = [&](Block *dest, Value cond, ValueRange operands) -> LogicalResult {
      if (order[dest] <= order[&blk])
        return op.emitError("furiosa: a map_elementwise function with a loop is not raised"), failure();
      auto vals = values(operands);
      if (failed(vals))
        return failure();
      incoming[dest].push_back({cond, *vals});
      return success();
    };
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      if (failed(edge(br.getDest(), here, br.getDestOperands())))
        return failure();
    } else if (auto cb = dyn_cast<cf::CondBranchOp>(term)) {
      auto c = lookup(cb.getCondition());
      if (failed(c))
        return failure();
      Value notC = arith::XOrIOp::create(ob, loc, *c, arith::ConstantIntOp::create(ob, loc, ob.getI1Type(), 1));
      if (failed(edge(cb.getTrueDest(), arith::AndIOp::create(ob, loc, here, *c), cb.getTrueDestOperands())) ||
          failed(edge(cb.getFalseDest(), arith::AndIOp::create(ob, loc, here, notC), cb.getFalseDestOperands())))
        return failure();
    } else if (isa<tt::MapElementwiseReturnOp>(term) && term->getNumOperands() == 1) {
      auto r = lookup(term->getOperand(0));
      if (failed(r))
        return failure();
      result = result ? arith::SelectOp::create(ob, loc, here, *r, result).getResult() : *r;
    } else {
      return term->emitError("furiosa: a map_elementwise terminator not raised"), failure();
    }
  }
  if (!result)
    return op.emitError("furiosa: a map_elementwise function that never returns"), failure();
  return result;
}

FailureOr<Value> Raiser::emit(Gen &g, Value v, ArrayRef<Axis> env) {
  auto memoKey = std::make_pair(v.getAsOpaquePointer(), key(env));
  if (auto it = g.memo.find(memoKey); it != g.memo.end())
    return it->second;
  FailureOr<Value> out = failure();
  Operation *def = v.getDefiningOp();
  const Sym *s0 = act[0]->get(v);
  // An index value that is no one expression of the coordinates in every instance (a min whose operand ranges
  // overlap in some instances only, a program's first key block max(start - WINDOW + 1, 0) // BLOCK * BLOCK) is
  // computed from its operands, as data is.
  if (s0 && s0->kind != Sym::Ptr && def &&
      (isElementwise(def) || isa<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp, tt::TransOp, tt::ReshapeOp>(def))) {
    if (llvm::all_of(act, [&](Evaluator *ev) { return ev->get(v) && ev->get(v)->data == s0->data; })) {
      auto origins = originUses;
      auto digits = digitUses;
      {
        ScopedDiagnosticHandler quiet(ctx, [](Diagnostic &) { return success(); });
        out = emitSym(g, v, *s0, env);
      }
      if (succeeded(out))
        return g.memo[memoKey] = *out;
      originUses = origins;
      digitUses = digits;
    }
    s0 = nullptr;
  }
  if (s0) {
    out = emitSym(g, v, *s0, env);
  } else if (auto it = g.state.find(v.getAsOpaquePointer()); it != g.state.end()) {
    SmallVector<Value> idx;
    for (int d : it->second.second) {
      if (d < 0)
        return kernel.emitError("furiosa: internal: loop state read without its dims"), failure();
      idx.push_back(linalg::IndexOp::create(*g.ob, loc, d));
    }
    out = tensor::ExtractOp::create(*g.ob, loc, it->second.first, idx).getResult();
  } else if (!def && !isa<scf::ForOp>(cast<BlockArgument>(v).getOwner()->getParentOp())) {
    return kernel.emitError("furiosa: a loop-carried value is used outside its loop"), failure();
  } else if (!def) {
    out = materialize(g, v, env);
  } else if (auto c = dyn_cast<arith::ConstantOp>(def)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(g.ob->getInsertionBlock(), g.ob->getInsertionPoint());
    out = constantOf(getElementTypeOrSelf(c.getType()), c.getValue());
  } else if (auto s = dyn_cast<tt::SplatOp>(def)) {
    out = emit(g, s.getSrc(), {});
  } else if (auto rs = dyn_cast<tt::ReshapeOp>(def)) {
    SmallVector<int64_t> src = shapeOf(rs.getSrc().getType());
    SmallVector<int64_t> dst = shapeOf(rs.getResult().getType());
    SmallVector<Axis> senv(src.size());
    if (auto oldOf = unitReshape(src, dst)) {
      for (auto [i, o] : llvm::enumerate(*oldOf))
        if (o >= 0)
          senv[o] = env[i];
    } else {
      senv = reshapeEnv(env, dst, src, ctx);
    }
    out = emit(g, rs.getSrc(), senv);
  } else if (auto br = dyn_cast<tt::BroadcastOp>(def)) {
    SmallVector<int64_t> src = shapeOf(br.getSrc().getType());
    SmallVector<Axis> senv;
    for (unsigned i = 0; i < src.size(); ++i)
      senv.push_back(src[i] == 1 ? Axis{} : env[i]);
    out = emit(g, br.getSrc(), senv);
  } else if (auto e = dyn_cast<tt::ExpandDimsOp>(def)) {
    SmallVector<Axis> senv(env.begin(), env.end());
    senv.erase(senv.begin() + e.getAxis());
    out = emit(g, e.getSrc(), senv);
  } else if (auto t = dyn_cast<tt::TransOp>(def)) {
    SmallVector<Axis> senv(env.size());
    for (auto [i, o] : llvm::enumerate(t.getOrder()))
      senv[o] = env[i];
    out = emit(g, t.getSrc(), senv);
  } else if (auto load = dyn_cast<tt::LoadOp>(def)) {
    out = emitLoad(g, Read{load}, env);
  } else if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(def)) { // its old value: the buffer before it writes
    out = emitLoad(g, Read{def}, env);
  } else if (auto j = dyn_cast<tt::JoinOp>(def)) { // a new last axis of 2: lhs at 0, rhs at 1
    SmallVector<Axis> senv(env.begin(), env.end() - 1);
    auto l = emit(g, j.getLhs(), senv), r = emit(g, j.getRhs(), senv);
    auto k = axisValue(g, def, env, env.size() - 1);
    if (failed(l) || failed(r) || failed(k))
      return failure();
    Value first = arith::CmpIOp::create(*g.ob, loc, arith::CmpIPredicate::eq, *k,
                                        arith::ConstantIndexOp::create(*g.ob, loc, 0));
    out = arith::SelectOp::create(*g.ob, loc, first, *l, *r).getResult();
  } else if (auto sp = dyn_cast<tt::SplitOp>(def)) { // the source's last axis of 2 at 0 or 1
    SmallVector<Axis> senv(env.begin(), env.end());
    Axis last;
    if (cast<OpResult>(v).getResultNumber()) {
      last.kind = Axis::Expr;
      last.expr = getAffineConstantExpr(1, ctx);
    }
    senv.push_back(last);
    out = emit(g, sp.getSrc(), senv);
  } else if (auto cat = dyn_cast<tt::CatOp>(def)) { // lhs then rhs along axis 0
    int64_t n = shapeOf(cat.getLhs().getType())[0];
    SmallVector<Axis> senv(env.begin(), env.end());
    if (env[0].kind != Axis::Unit) {
      senv[0] = Axis{};
      senv[0].kind = Axis::Expr;
      senv[0].expr = getAffineDimExpr(0, ctx) % n;
      senv[0].of = {env[0]};
    }
    auto l = emit(g, cat.getLhs(), senv), r = emit(g, cat.getRhs(), senv);
    auto i = axisValue(g, def, env, 0);
    if (failed(l) || failed(r) || failed(i))
      return failure();
    Value left = arith::CmpIOp::create(*g.ob, loc, arith::CmpIPredicate::slt, *i,
                                       arith::ConstantIndexOp::create(*g.ob, loc, n));
    out = arith::SelectOp::create(*g.ob, loc, left, *l, *r).getResult();
  } else if (auto ga = dyn_cast<tt::GatherOp>(def)) { // src with the gathered axis at the index
    const Sym *ix = act[0]->get(ga.getIndices());
    bool same = ix && llvm::all_of(act, [&](Evaluator *ev) {
      const Sym *s = ev->get(ga.getIndices());
      return s && s->expr == ix->expr;
    });
    bool symbolic = false; // a program id or loop counter: not a function of the axes alone
    if (same)
      ix->expr.walk([&](AffineExpr e) { symbolic |= isa<AffineSymbolExpr>(e); });
    unsigned axis = ga.getAxisAttr().getInt();
    if (same && !symbolic && ix->kind == Sym::Int && ix->data.empty()) {
      SmallVector<Axis> senv(env.begin(), env.end());
      senv[axis] = Axis{};
      senv[axis].kind = Axis::Expr;
      senv[axis].expr = ix->expr;
      senv[axis].of.assign(env.begin(), env.end());
      out = emit(g, ga.getSrc(), senv);
    } else {
      // Indices that are data: the source built once over its own axes, read at the index.
      SmallVector<int64_t> shape = shapeOf(ga.getSrc().getType());
      Gen own;
      own.coordPos.assign(g.coordPos.size(), -1);
      own.lo.assign(shape.size(), 0);
      own.extent.assign(shape.begin(), shape.end());
      own.anchor = g.anchor;
      auto src = parallelGeneric(own, getElementTypeOrSelf(ga.getSrc().getType()), [&](Gen &gg) {
        SmallVector<Axis> ownEnv(shape.size());
        for (auto [d, a] : llvm::enumerate(ownEnv)) {
          a.kind = Axis::Iter;
          a.dim = d;
        }
        return emit(gg, ga.getSrc(), ownEnv);
      });
      auto index = emit(g, ga.getIndices(), env);
      if (failed(src) || failed(index))
        return failure();
      SmallVector<Value> at;
      for (unsigned k = 0; k < env.size(); ++k) {
        if (k != axis) {
          auto i = axisValue(g, def, env, k);
          if (failed(i))
            return failure();
          at.push_back(*i);
          continue;
        }
        Value i = arith::IndexCastOp::create(*g.ob, loc, g.ob->getIndexType(), *index); // out of range: clamped
        i = arith::MaxSIOp::create(*g.ob, loc, i, arith::ConstantIndexOp::create(*g.ob, loc, 0));
        at.push_back(arith::MinSIOp::create(*g.ob, loc, i, arith::ConstantIndexOp::create(*g.ob, loc, shape[k] - 1)));
      }
      out = tensor::ExtractOp::create(*g.ob, loc, *src, at).getResult();
    }
  } else if (auto me = dyn_cast<tt::MapElementwiseOp>(def)) {
    out = emitMapElementwise(g, me, env);
  } else if (auto branch = dyn_cast<scf::IfOp>(def)) {
    // The partition's instances all take the same branch: follow its yield.
    bool then = act[0]->taken.lookup(branch);
    if (llvm::any_of(act, [&](Evaluator *ev) { return ev->taken.lookup(branch) != then; }))
      return branch.emitError("furiosa: internal: instances of one partition branch differently"), failure();
    Region &region = then ? branch.getThenRegion() : branch.getElseRegion();
    auto yield = cast<scf::YieldOp>(region.front().getTerminator());
    out = emit(g, yield.getOperand(cast<OpResult>(v).getResultNumber()), env);
  } else if (isa<tt::ReduceOp, tt::DotOp, scf::ForOp>(def)) {
    out = materialize(g, v, env);
  } else if (isElementwise(def)) {
    out = emitElementwise(g, def, env);
  } else {
    return def->emitError("furiosa: op not supported by the raising pass"), failure();
  }
  if (succeeded(out))
    g.memo[memoKey] = *out;
  return out;
}

FailureOr<std::pair<Value, SmallVector<unsigned>>>
Raiser::buildGeneric(Gen &parent, ArrayRef<unsigned> parentDims, ArrayRef<int64_t> ownExtents,
                     ArrayRef<std::pair<unsigned, unsigned>> loopDims, Type elem, InitFn init, BodyFn body) {
  auto one = [](FailureOr<Value> v) -> FailureOr<SmallVector<Value>> {
    if (failed(v))
      return failure();
    return SmallVector<Value>{*v};
  };
  auto built = buildGenerics(
      parent, parentDims, ownExtents, loopDims, ArrayRef<Type>(elem),
      [&](Gen &g, ArrayRef<int> remap) { return one(init(g, remap)); },
      [&](Gen &g, ValueRange accs, ArrayRef<int> remap) { return one(body(g, accs[0], remap)); }, 0);
  if (failed(built))
    return failure();
  return std::make_pair(built->first[0], built->second);
}

FailureOr<std::pair<SmallVector<Value>, SmallVector<unsigned>>>
Raiser::buildGenerics(Gen &parent, ArrayRef<unsigned> parentDims, ArrayRef<int64_t> ownExtents,
                      ArrayRef<std::pair<unsigned, unsigned>> loopDims, ArrayRef<Type> elems, InitsFn init,
                      BodiesFn body, int64_t ownLo) {
  unsigned nr = elems.size();
  auto attempt = [&](ArrayRef<unsigned> keep, bool &pruneable,
                     SmallVector<unsigned> &used) -> FailureOr<SmallVector<Value>> {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(parent.anchor);
    // What the bodies below read gets built before this placeholder, ahead of both generics.
    Operation *anchor = arith::ConstantIndexOp::create(b, loc, 0);
    llvm::scope_exit dropAnchor([&] { retire(anchor); });
    // remap: parent dim -> this generic's dim (-1: dropped)
    SmallVector<int> remap(parent.extent.size(), -1);
    Gen g;
    g.anchor = anchor;
    for (auto [i, d] : llvm::enumerate(keep)) {
      remap[d] = i;
      g.lo.push_back(parent.lo[d]);
      g.extent.push_back(parent.extent[d]);
    }
    unsigned np = keep.size();
    for (int64_t e : ownExtents) {
      g.lo.push_back(ownLo);
      g.extent.push_back(e);
    }
    g.coordPos.assign(parent.coordPos.size(), -1);
    for (auto [j, pos] : llvm::enumerate(parent.coordPos))
      if (pos >= 0 && remap[pos] >= 0)
        g.coordPos[j] = remap[pos];
    for (auto [l, pos] : parent.loopPos)
      if (remap[pos] >= 0)
        g.loopPos[l] = remap[pos];
    for (auto [l, own] : loopDims)
      g.loopPos[l] = np + own;
    g.loopIv = parent.loopIv;
    for (auto &[k, st] : parent.state) {
      SmallVector<int> dims;
      for (int d : st.second)
        dims.push_back(d >= 0 ? remap[d] : -1);
      g.state[k] = {st.first, dims};
    }

    SmallVector<int64_t> outShape;
    for (unsigned d = 0; d < np; ++d)
      outShape.push_back(g.extent[d]);
    // The init: a parallel generic over the kept dims (it may read other buffers).
    Gen gi = g;
    gi.extent.resize(np);
    gi.lo.resize(np);
    SmallVector<unsigned> ownLoops; // the init has no own dims: their loops are not its dims
    for (auto [l, pos] : gi.loopPos)
      if (pos >= (int)np)
        ownLoops.push_back(l);
    for (unsigned l : ownLoops)
      gi.loopPos.erase(l);
    SmallVector<int> remapParallel = remap;
    SmallVector<Value> empties;
    for (Type elem : elems)
      empties.push_back(tensor::EmptyOp::create(b, loc, outShape, elem));
    SmallVector<AffineMap> imaps(nr, AffineMap::getMultiDimIdentityMap(np, ctx));
    SmallVector<utils::IteratorType> iters(np, utils::IteratorType::parallel);
    LogicalResult status = success();
    auto initOp = linalg::GenericOp::create(b, 
        loc, ValueRange(empties).getTypes(), ValueRange{}, empties, imaps, iters,
        [&](OpBuilder &ob, Location, ValueRange) {
          gi.ob = &ob;
          gi.index.assign(np, Value());
          auto v = init(gi, remapParallel);
          if (failed(v)) {
            status = failure();
            SmallVector<Value> zeros;
            for (Type elem : elems)
              zeros.push_back(arith::ConstantOp::create(ob, loc, ob.getZeroAttr(elem)));
            linalg::YieldOp::create(ob, loc, zeros);
            return;
          }
          linalg::YieldOp::create(ob, loc, *v);
        });
    if (failed(status))
      return failure();
    if (ownExtents.empty())
      ; // parallel-only: the init generic already computes the value through `body` below
    unsigned nd = g.extent.size();
    SmallVector<Value> inputs;
    SmallVector<AffineMap> maps;
    if (!ownExtents.empty()) {
      // Carrier of the reduction extents (never read).
      inputs.push_back(tensor::EmptyOp::create(b, loc, ownExtents, b.getI8Type()));
      SmallVector<AffineExpr> own;
      for (unsigned d = np; d < nd; ++d)
        own.push_back(getAffineDimExpr(d, ctx));
      maps.push_back(AffineMap::get(nd, 0, own, ctx));
    }
    SmallVector<AffineExpr> outExprs;
    for (unsigned d = 0; d < np; ++d)
      outExprs.push_back(getAffineDimExpr(d, ctx));
    maps.append(nr, AffineMap::get(nd, 0, outExprs, ctx));
    SmallVector<utils::IteratorType> its(np, utils::IteratorType::parallel);
    its.append(nd - np, utils::IteratorType::reduction);
    SmallVector<Value> result(initOp.getResults());
    if (!ownExtents.empty()) {
      auto op = linalg::GenericOp::create(b, 
          loc, initOp.getResultTypes(), inputs, result, maps, its,
          [&](OpBuilder &ob, Location, ValueRange args) {
            g.ob = &ob;
            g.index.assign(nd, Value());
            auto v = body(g, args.take_back(nr), remap);
            if (failed(v)) {
              status = failure();
              linalg::YieldOp::create(ob, loc, args.take_back(nr));
              return;
            }
            linalg::YieldOp::create(ob, loc, *v);
          });
      if (failed(status))
        return failure();
      result.assign(op.getResults().begin(), op.getResults().end());
      // Which kept dims does either generic read?
      std::set<unsigned> readDims;
      for (Operation *o : {initOp.getOperation(), op.getOperation()})
        o->walk([&](linalg::IndexOp ix) {
          if (!ix->use_empty())
            readDims.insert(ix.getDim());
        });
      for (unsigned d = 0; d < np; ++d)
        if (readDims.count(d))
          used.push_back(keep[d]);
      pruneable = used.size() < keep.size();
    } else {
      std::set<unsigned> readDims;
      initOp->walk([&](linalg::IndexOp ix) {
        if (!ix->use_empty())
          readDims.insert(ix.getDim());
      });
      for (unsigned d = 0; d < np; ++d)
        if (readDims.count(d))
          used.push_back(keep[d]);
      pruneable = used.size() < keep.size();
    }
    return result;
  };

  bool pruneable = false;
  SmallVector<unsigned> used;
  auto first = attempt(parentDims, pruneable, used);
  if (failed(first))
    return failure();
  if (!pruneable)
    return std::make_pair(*first, SmallVector<unsigned>(parentDims));
  // Rebuild over the dims actually read, then drop the first attempt.
  SmallVector<Operation *> dead;
  for (Value v = (*first)[0]; v;) {
    Operation *op = v.getDefiningOp();
    dead.push_back(op);
    auto generic = dyn_cast<linalg::GenericOp>(op);
    v = generic ? generic.getDpsInits()[0] : Value();
    if (v && !isa<linalg::GenericOp>(v.getDefiningOp()))
      v = Value();
  }
  bool again = false;
  SmallVector<unsigned> used2;
  auto second = attempt(used, again, used2);
  if (failed(second))
    return failure();
  for (Operation *op : dead)
    if (op->use_empty())
      op->erase();
  return std::make_pair(*second, used);
}

/// The part of a dot's K axis that can contribute. Past it, both operands are loads whose masks
/// exclude k (by conditions on k and loop counters alone) in every instance and iteration, and whose
/// `other` is zero, so the products there add nothing: a padded reduction tile (BLOCK_K > K, masked)
/// reduces over its true extent.
/// How far along `axis` a value masked by `mask` can be live in any instance (and iteration): past it the mask's
/// conditions on that axis alone fail. `full` when no condition reads the axis alone.
int64_t Raiser::maskLive(Value mask, unsigned axis, int64_t full) {
  unsigned rank = shapeOf(mask.getType()).size(), nl = loopTrip.size();
  int64_t live = 0;
  std::set<std::string> seen;
  for (Evaluator *ev : act) {
    const Sym *m = ev->get(mask);
    if (!m || m->kind != Sym::Mask)
      return full;
    if (m->never)
      continue;
    SmallVector<AffineExpr> conds;
    for (AffineExpr c : m->conds) {
      bool others = false;
      for (unsigned d = 0; d < rank; ++d)
        others |= d != axis && c.isFunctionOfDim(d);
      if (!others && c.isFunctionOfDim(axis))
        conds.push_back(c);
    }
    if (conds.empty())
      return full;
    SmallVector<unsigned> used;
    for (unsigned l = 0; l < nl; ++l)
      if (llvm::any_of(conds, [&](AffineExpr c) { return c.isFunctionOfSymbol(l); }))
        used.push_back(l);
    std::string key;
    for (AffineExpr c : conds)
      key += show(c) + ";";
    int64_t total = 1;
    for (unsigned l : used) {
      key += std::to_string(ev->trips.lookup(l)) + ",";
      total *= ev->trips.lookup(l);
    }
    if (!seen.insert(key).second)
      continue;
    if (total > (1 << 16))
      return full;
    SmallVector<AffineExpr> symVals(nl, getAffineConstantExpr(0, ctx));
    for (int64_t f = 0; f < total && live < full; ++f) {
      int64_t r = f;
      for (unsigned l : used) {
        symVals[l] = getAffineConstantExpr(r % ev->trips.lookup(l), ctx);
        r /= ev->trips.lookup(l);
      }
      SmallVector<AffineExpr> at;
      for (AffineExpr c : conds)
        at.push_back(c.replaceSymbols(symVals));
      SmallVector<int64_t> point(rank, 0);
      for (int64_t kk = full - 1; kk >= live; --kk) {
        point[axis] = kk;
        if (llvm::all_of(at, [&](AffineExpr c) { return evalAt(c, point) >= 0; })) {
          live = kk + 1;
          break;
        }
      }
    }
  }
  return live;
}

int64_t Raiser::liveExtent(tt::DotOp dot) {
  int64_t k = shapeOf(dot.getA().getType())[1];
  auto isZero = [](Attribute a) {
    if (auto d = dyn_cast_or_null<DenseElementsAttr>(a); d && d.isSplat())
      a = d.getSplatValue<Attribute>();
    if (auto f = dyn_cast_or_null<FloatAttr>(a))
      return f.getValue().isZero();
    if (auto i = dyn_cast_or_null<IntegerAttr>(a))
      return i.getValue().isZero();
    return false;
  };
  auto liveOf = [&](Value v, unsigned axis) -> int64_t {
    while (Operation *def = v.getDefiningOp()) {
      if (!isa<arith::ExtFOp, arith::TruncFOp, tt::FpToFpOp>(def))
        break;
      v = def->getOperand(0);
    }
    auto load = v.getDefiningOp<tt::LoadOp>();
    if (!load || !load.getMask())
      return k;
    if (Value other = load.getOther()) {
      auto c = other.getDefiningOp<arith::ConstantOp>();
      if (!c || !isZero(c.getValue()))
        return k;
    }
    return maskLive(load.getMask(), axis, k);
  };
  return std::max<int64_t>(1, std::max(liveOf(dot.getA(), 1), liveOf(dot.getB(), 0)));
}

/// How far along `axis` `v` can differ from `identity` in any instance: past it every element is the identity. It is
/// so past the prefix a load's mask keeps (with the identity as `other`) or a select's condition keeps (with it as
/// the false value), through conversions, broadcasts and reductions over another axis with the same identity, and for
/// zero through products (any zero factor) and sums (every term zero).
int64_t Raiser::liveAlong(Value v, unsigned axis, TypedAttr identity) {
  int64_t full = shapeOf(v.getType())[axis];
  auto isIdentity = [&](Value c) {
    Attribute a;
    if (!c || !matchPattern(c, m_Constant(&a)))
      return false;
    if (auto d = dyn_cast<SplatElementsAttr>(a))
      a = d.getSplatValue<Attribute>();
    if (auto f = dyn_cast<FloatAttr>(a); f && isa<FloatAttr>(identity))
      return f.getValueAsDouble() == cast<FloatAttr>(identity).getValueAsDouble();
    if (auto i = dyn_cast<IntegerAttr>(a); i && isa<IntegerAttr>(identity))
      return i.getValue().getSExtValue() == cast<IntegerAttr>(identity).getValue().getSExtValue();
    return false;
  };
  bool zero = isa<FloatAttr>(identity) ? cast<FloatAttr>(identity).getValueAsDouble() == 0.0
                                       : cast<IntegerAttr>(identity).getValue().isZero();
  Operation *def = v.getDefiningOp();
  if (!def)
    return full;
  if (isa<arith::ExtFOp, arith::TruncFOp, tt::FpToFpOp>(def))
    return liveAlong(def->getOperand(0), axis, identity);
  if (auto load = dyn_cast<tt::LoadOp>(def))
    return load.getMask() && (!load.getOther() || isIdentity(load.getOther())) ? maskLive(load.getMask(), axis, full)
                                                                                : full;
  if (auto sel = dyn_cast<arith::SelectOp>(def); sel && isIdentity(sel.getFalseValue()) &&
                                                   isa<RankedTensorType>(sel.getCondition().getType()))
    return std::min(maskLive(sel.getCondition(), axis, full), liveAlong(sel.getTrueValue(), axis, identity));
  if (zero && isa<arith::MulFOp, arith::MulIOp>(def))
    return std::min(liveAlong(def->getOperand(0), axis, identity), liveAlong(def->getOperand(1), axis, identity));
  if (zero && isa<arith::AddFOp, arith::SubFOp, arith::AddIOp, arith::SubIOp>(def))
    return std::max(liveAlong(def->getOperand(0), axis, identity), liveAlong(def->getOperand(1), axis, identity));
  if (auto ed = dyn_cast<tt::ExpandDimsOp>(def))
    return axis == ed.getAxis() ? full : liveAlong(ed.getSrc(), axis - (axis > ed.getAxis()), identity);
  if (auto bc = dyn_cast<tt::BroadcastOp>(def))
    return shapeOf(bc.getSrc().getType())[axis] == full ? liveAlong(bc.getSrc(), axis, identity) : full;
  if (auto red = dyn_cast<tt::ReduceOp>(def); red && red.getNumOperands() == 1 &&
                                              identityOf(red.getCombineOp().front()) == identity)
    return liveAlong(red.getOperand(0), axis + (axis >= red.getAxis()), identity);
  return full;
}

FailureOr<Value> Raiser::materialize(Gen &g, Value v, ArrayRef<Axis> env) {
  Operation *def = v.getDefiningOp();
  Type elem = getElementTypeOrSelf(v.getType());
  SmallVector<unsigned> parentDims;
  for (unsigned d = 0; d < g.extent.size(); ++d)
    parentDims.push_back(d);
  // Remaps an env over the parent's dims into a generic built with `remap`.
  FailureOr<std::pair<Value, SmallVector<unsigned>>> built = failure();

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    // An iter arg read inside its loop (a running carry stored per tile) at the current iteration: the
    // accumulation of the iterations before it, when the loop updates it as arg (op) term by an
    // associative op and term reads no data state of the loop.
    auto loop = cast<scf::ForOp>(arg.getOwner()->getParentOp());
    unsigned i = arg.getArgNumber() - loop.getNumInductionVars(), id = loopIds.lookup(loop);
    Operation *upd = cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(i).getDefiningOp();
    if (!upd || upd->getParentOp() != loop.getOperation() || upd->getNumOperands() != 2 ||
        !llvm::is_contained(upd->getOperands(), v) ||
        !isa<arith::AddFOp, arith::MulFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp, arith::MinimumFOp,
             arith::AddIOp, arith::MulIOp, arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp,
             arith::XOrIOp, arith::OrIOp, arith::AndIOp>(upd))
      return loop.emitError("furiosa: a loop-carried value read inside its loop is not an accumulation"), failure();
    Value term = upd->getOperand(0) == v ? upd->getOperand(1) : upd->getOperand(0);
    SmallVector<Value> work{term};
    DenseSet<Operation *> seen;
    while (!work.empty()) {
      Value x = work.pop_back_val();
      if (auto a = dyn_cast<BlockArgument>(x); a && a.getOwner() == loop.getBody() && a.getArgNumber() &&
                                                 !act[0]->get(a))
        return loop.emitError("furiosa: a loop-carried accumulation reads the loop's state"), failure();
      Operation *d = x.getDefiningOp();
      if (d && loop->isProperAncestor(d) && seen.insert(d).second)
        d->walk([&](Operation *in) { work.append(in->operand_begin(), in->operand_end()); });
    }
    built = buildGeneric(
        g, parentDims, {loopTrip[id]}, {{id, 0}}, elem,
        [&](Gen &gi, ArrayRef<int> remap) { return emit(gi, loop.getInitArgs()[i], remapEnv(env, remap)); },
        [&](Gen &gg, Value acc, ArrayRef<int> remap) -> FailureOr<Value> {
          // The current iteration: the loop counter as the parent has it.
          Gen outer = gg;
          outer.memo.clear();
          outer.loopPos.erase(id);
          if (auto it = g.loopPos.find(id); it != g.loopPos.end()) {
            if (remap[it->second] < 0)
              return failure();
            outer.loopPos[id] = remap[it->second];
          }
          auto cur = emit(outer, loop.getInductionVar(), {});
          auto own = emit(gg, loop.getInductionVar(), {});
          auto x = emit(gg, term, remapEnv(env, remap));
          if (failed(cur) || failed(own) || failed(x))
            return failure();
          OperationState state(loc, upd->getName());
          state.addOperands({acc, *x});
          state.addTypes(elem);
          state.addAttributes(upd->getAttrs());
          Value before = arith::CmpIOp::create(*gg.ob, loc, arith::CmpIPredicate::slt, *own, *cur);
          return arith::SelectOp::create(*gg.ob, loc, before, gg.ob->create(state)->getResult(0), acc).getResult();
        });
  } else if (auto red = dyn_cast<tt::ReduceOp>(def)) {
    unsigned result = cast<OpResult>(v).getResultNumber();
    // Through a data-moving reshape (Expr axes) reductions nest in every context that reads them, and
    // gather form would rebuild them per context, exponentially in the nesting (tl.sort, tl.flip): build
    // those, and every reduction they read, once each.
    if (ownDepth || llvm::any_of(env, [](const Axis &a) { return a.kind == Axis::Expr; }))
      if (auto own = materializeOwn(g, red, result, env); succeeded(own))
        return own;
    built = reduceTensor(g, red, result, env);
  } else if (auto dot = dyn_cast<tt::DotOp>(def)) {
    int64_t k = liveExtent(dot);
    built = buildGeneric(
        g, parentDims, {k}, {}, elem,
        [&](Gen &gi, ArrayRef<int> remap) { return emit(gi, dot.getC(), remapEnv(env, remap)); },
        [&](Gen &gg, Value acc, ArrayRef<int> remap) -> FailureOr<Value> {
          SmallVector<Axis> e = remapEnv(env, remap);
          Axis kk;
          kk.kind = Axis::Iter;
          kk.dim = gg.extent.size() - 1;
          auto a = emit(gg, dot.getA(), {e[0], kk});
          auto bb = emit(gg, dot.getB(), {kk, e[1]});
          if (failed(a) || failed(bb))
            return failure();
          Value x = *a, y = *bb;
          if (x.getType() != elem)
            x = isa<FloatType>(elem) ? arith::ExtFOp::create(*gg.ob, loc, elem, x).getResult()
                                     : arith::ExtSIOp::create(*gg.ob, loc, elem, x).getResult();
          if (y.getType() != elem)
            y = isa<FloatType>(elem) ? arith::ExtFOp::create(*gg.ob, loc, elem, y).getResult()
                                     : arith::ExtSIOp::create(*gg.ob, loc, elem, y).getResult();
          if (isa<FloatType>(elem))
            return arith::AddFOp::create(*gg.ob, loc, acc, arith::MulFOp::create(*gg.ob, loc, x, y)).getResult();
          return arith::AddIOp::create(*gg.ob, loc, acc, arith::MulIOp::create(*gg.ob, loc, x, y)).getResult();
        });
  } else if (auto loop = dyn_cast<scf::ForOp>(def)) {
    // Follow the accumulator from this loop inward: through each nested loop it initializes, down to
    // the op that updates it. Every loop on the way contributes one reduction dim.
    unsigned i = cast<OpResult>(v).getResultNumber();
    Value init = loop.getInitArgs()[i];
    SmallVector<int64_t> extents;
    SmallVector<std::pair<unsigned, unsigned>> loopDims;
    scf::ForOp cur = loop;
    unsigned idx = i;
    Value iter;
    Operation *upd = nullptr;
    while (true) {
      iter = cur.getRegionIterArgs()[idx];
      unsigned id = loopIds.lookup(cur);
      loopDims.push_back({id, (unsigned)extents.size()});
      extents.push_back(loopTrip[id]);
      Value next = cast<scf::YieldOp>(cur.getBody()->getTerminator()).getOperand(idx);
      auto inner = next.getDefiningOp<scf::ForOp>();
      if (inner && inner->getParentOp() == cur.getOperation()) {
        unsigned r = cast<OpResult>(next).getResultNumber();
        if (inner.getInitArgs()[r] != iter || !iter.hasOneUse())
          return materializeState(g, loop, i, env);
        cur = inner;
        idx = r;
        continue;
      }
      upd = next.getDefiningOp();
      if (!upd || upd->getParentOp() != cur.getOperation() || !iter.hasOneUse() ||
          iter.getUses().begin()->getOwner() != upd)
        return materializeState(g, loop, i, env);
      break;
    }
    auto initFn = [&](Gen &gi, ArrayRef<int> remap) { return emit(gi, init, remapEnv(env, remap)); };
    // Iterations past an instance's own trip count leave the accumulator unchanged.
    auto guard = [&](Gen &gg, Value acc, Value next) -> FailureOr<Value> {
      Value active;
      for (auto [id, own] : loopDims) {
        if (!varTrip[id])
          continue;
        auto lt = tripGuard(gg, id, index(gg, gg.loopPos[id]));
        if (failed(lt))
          return failure();
        active = active ? arith::AndIOp::create(*gg.ob, loc, active, *lt).getResult() : *lt;
      }
      if (!active)
        return next;
      return arith::SelectOp::create(*gg.ob, loc, active, next, acc).getResult();
    };
    if (auto d = dyn_cast<tt::DotOp>(upd); d && d.getC() == iter) {
      extents.push_back(liveExtent(d));
      built = buildGeneric(
          g, parentDims, extents, loopDims, elem, initFn,
          [&](Gen &gg, Value acc, ArrayRef<int> remap) -> FailureOr<Value> {
            SmallVector<Axis> e = remapEnv(env, remap);
            Axis kk;
            kk.kind = Axis::Iter;
            kk.dim = gg.extent.size() - 1;
            auto a = emit(gg, d.getA(), {e[0], kk});
            auto bb = emit(gg, d.getB(), {kk, e[1]});
            if (failed(a) || failed(bb))
              return failure();
            Value x = *a, y = *bb;
            if (x.getType() != elem)
              x = arith::ExtFOp::create(*gg.ob, loc, elem, x);
            if (y.getType() != elem)
              y = arith::ExtFOp::create(*gg.ob, loc, elem, y);
            return guard(gg, acc, arith::AddFOp::create(*gg.ob, loc, acc, arith::MulFOp::create(*gg.ob, loc, x, y)));
          });
    } else if (upd->getNumOperands() == 2 &&
               isa<arith::AddFOp, arith::MulFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp,
                   arith::MinimumFOp, arith::AddIOp, arith::MulIOp, arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp,
                   arith::MinUIOp, arith::XOrIOp, arith::OrIOp, arith::AndIOp>(upd)) {
      Value term = upd->getOperand(0) == iter ? upd->getOperand(1) : upd->getOperand(0);
      built = buildGeneric(
          g, parentDims, extents, loopDims, elem, initFn,
          [&](Gen &gg, Value acc, ArrayRef<int> remap) -> FailureOr<Value> {
            auto x = emit(gg, term, remapEnv(env, remap));
            if (failed(x))
              return failure();
            OperationState state(loc, upd->getName());
            state.addOperands({acc, *x});
            state.addTypes(elem);
            state.addAttributes(upd->getAttrs());
            return guard(gg, acc, gg.ob->create(state)->getResult(0));
          });
    } else {
      return materializeState(g, loop, i, env);
    }
  }
  if (failed(built))
    return failure();
  auto [tensor, kept] = *built;
  SmallVector<Value> idx;
  for (unsigned d : kept) {
    Value i = linalg::IndexOp::create(*g.ob, loc, d);
    idx.push_back(i);
  }
  return tensor::ExtractOp::create(*g.ob, loc, tensor, idx).getResult();
}

/// A reduction's result over the dims of `g` it keeps (env: its axes in `g`).
FailureOr<std::pair<Value, SmallVector<unsigned>>> Raiser::reduceTensor(Gen &g, tt::ReduceOp red, unsigned result,
                                                                        ArrayRef<Axis> env) {
  SmallVector<unsigned> parentDims;
  for (unsigned d = 0; d < g.extent.size(); ++d)
    parentDims.push_back(d);
  Block &combiner = red.getCombineOp().front();
  unsigned n = red.getNumOperands();
  TypedAttr identity = identityOf(combiner);
  unsigned axis = red.getAxis();
  int64_t extent = shapeOf(red.getOperand(0).getType())[axis];
  // A padded tile (BLOCK > N, masked to the identity) reduces over its live prefix.
  if (identity && n == 1)
    extent = std::max<int64_t>(1, liveAlong(red.getOperand(0), axis, identity));
  // Every operand at reduction index `r`.
  auto operandsAt = [&](Gen &gg, ArrayRef<int> remap, Axis r) -> FailureOr<SmallVector<Value>> {
    SmallVector<Axis> in = remapEnv(env, remap);
    in.insert(in.begin() + axis, r);
    SmallVector<Value> xs;
    for (Value o : red.getOperands()) {
      auto x = emit(gg, o, in);
      if (failed(x))
        return failure();
      xs.push_back(*x);
    }
    return xs;
  };
  // Without a known identity (any other combiner, or several operands), element 0 seeds the
  // accumulators and the reduction runs over elements 1.. (own dims start at 1).
  SmallVector<int64_t> own;
  if (identity || extent > 1)
    own.push_back(identity ? extent : extent - 1);
  SmallVector<Type> elems;
  for (Value o : red.getOperands())
    elems.push_back(getElementTypeOrSelf(o.getType()));
  auto multi = buildGenerics(
      g, parentDims, own, {}, elems,
      [&](Gen &gi, ArrayRef<int> remap) -> FailureOr<SmallVector<Value>> {
        if (!identity)
          return operandsAt(gi, remap, Axis{}); // Unit: index 0
        return SmallVector<Value>{arith::ConstantOp::create(*gi.ob, loc, cast<TypedAttr>(identity)).getResult()};
      },
      [&](Gen &gg, ValueRange accs, ArrayRef<int> remap) -> FailureOr<SmallVector<Value>> {
        Axis r;
        r.kind = Axis::Iter;
        r.dim = gg.extent.size() - 1;
        auto xs = operandsAt(gg, remap, r);
        if (failed(xs))
          return failure();
        IRMapping map;
        for (unsigned j = 0; j < n; ++j) {
          map.map(combiner.getArgument(j), accs[j]);
          map.map(combiner.getArgument(n + j), (*xs)[j]);
        }
        for (Operation &o : combiner.without_terminator())
          gg.ob->clone(o, map);
        SmallVector<Value> outs;
        for (Value y : combiner.getTerminator()->getOperands())
          outs.push_back(map.lookupOrDefault(y));
        return outs;
      },
      identity ? 0 : 1);
  if (failed(multi))
    return failure();
  return std::make_pair(multi->first[result], multi->second);
}

/// A reduction built once in its own index space (dims: its result's axes) before `g`, and read at
/// `env`. Fails (quietly) where its value depends on more than its own axes: a loop counter, or a
/// program id that only a store coordinate determines; the caller then builds it in `g`.
FailureOr<Value> Raiser::materializeOwn(Gen &g, tt::ReduceOp red, unsigned result, ArrayRef<Axis> env) {
  if (red->getParentOfType<scf::ForOp>())
    return failure();
  auto &cached = ownReductions[{red.getOperation(), result}];
  // Reuse it where it is built ahead of `g` (anchors of nested generics come first).
  if (!cached.first || !cached.first.getDefiningOp()->isBeforeInBlock(g.anchor)) {
    SmallVector<int64_t> shape = shapeOf(red->getResult(result).getType());
    Gen own;
    own.coordPos.assign(g.coordPos.size(), -1);
    own.lo.assign(shape.size(), 0);
    own.extent.assign(shape.begin(), shape.end());
    own.anchor = g.anchor;
    SmallVector<Axis> ownEnv(shape.size());
    for (auto [d, a] : llvm::enumerate(ownEnv)) {
      a.kind = Axis::Iter;
      a.dim = d;
    }
    ScopedDiagnosticHandler quiet(ctx, [](Diagnostic &) { return success(); });
    ++ownDepth;
    auto built = reduceTensor(own, red, result, ownEnv);
    --ownDepth;
    if (failed(built))
      return failure();
    cached = *built;
  }
  SmallVector<Value> idx;
  for (unsigned d : cached.second) {
    auto e = rewrite(red, g, env, [&](const Sym &) { return getAffineDimExpr(d, ctx); }, Value());
    if (failed(e))
      return failure();
    idx.push_back(exprValue(g, *e));
  }
  return tensor::ExtractOp::create(*g.ob, loc, cached.first, idx).getResult();
}

/// A tensor over every dim of `parent` (all parallel), computed by `fn` at each point.
FailureOr<Value> Raiser::parallelGeneric(Gen &parent, Type elem, std::function<FailureOr<Value>(Gen &)> fn) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(parent.anchor);
  Operation *anchor = arith::ConstantIndexOp::create(b, loc, 0);
  llvm::scope_exit dropAnchor([&] { retire(anchor); });
  Gen g;
  g.coordPos = parent.coordPos;
  g.lo = parent.lo;
  g.extent = parent.extent;
  g.loopPos = parent.loopPos;
  g.loopIv = parent.loopIv;
  g.state = parent.state;
  g.anchor = anchor;
  unsigned nd = g.extent.size();
  Value empty = tensor::EmptyOp::create(b, loc, g.extent, elem);
  LogicalResult status = success();
  auto op = linalg::GenericOp::create(
      b, loc, TypeRange{empty.getType()}, ValueRange{}, ValueRange{empty},
      ArrayRef<AffineMap>{AffineMap::getMultiDimIdentityMap(nd, ctx)},
      SmallVector<utils::IteratorType>(nd, utils::IteratorType::parallel), [&](OpBuilder &ob, Location, ValueRange) {
        g.ob = &ob;
        g.index.assign(nd, Value());
        auto v = fn(g);
        if (failed(v)) {
          status = failure();
          linalg::YieldOp::create(ob, loc, ValueRange{arith::ConstantOp::create(ob, loc, ob.getZeroAttr(elem))});
          return;
        }
        linalg::YieldOp::create(ob, loc, ValueRange{*v});
      });
  if (failed(status))
    return failure();
  return op.getResult(0);
}

/// Bindings of every data iter arg of `loop`, from the binding `env` of result `result`: axes are
/// unified through the body (elementwise, broadcast, expand_dims, trans, reduce, dot) and through
/// init = iter arg = yield = result.
FailureOr<SmallVector<SmallVector<Axis>>> Raiser::stateEnvs(scf::ForOp loop, unsigned result, ArrayRef<Axis> env,
                                                             SmallVector<bool> *skipped) {
  std::map<std::pair<void *, unsigned>, int> ids;
  SmallVector<int> parent;
  SmallVector<std::optional<Axis>> bound;
  auto node = [&](Value v, unsigned axis) {
    auto key = std::make_pair(v.getAsOpaquePointer(), axis);
    auto [it, fresh] = ids.insert({key, (int)parent.size()});
    if (fresh) {
      parent.push_back(parent.size());
      bound.push_back(std::nullopt);
    }
    return it->second;
  };
  std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
  bool conflict = false;
  auto unite = [&](Value a, unsigned ax, Value c, unsigned cx) {
    int x = find(node(a, ax)), y = find(node(c, cx));
    if (x == y)
      return;
    if (bound[x] && bound[y] && (bound[x]->kind != bound[y]->kind || bound[x]->dim != bound[y]->dim))
      conflict = true;
    parent[y] = x;
    if (!bound[x])
      bound[x] = bound[y];
  };
  auto rank = [](Value v) { return (unsigned)shapeOf(v.getType()).size(); };
  auto dim = [](Value v, unsigned k) { return shapeOf(v.getType())[k]; };
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (unsigned j = 0; j < loop.getNumRegionIterArgs(); ++j)
    for (unsigned k = 0; k < rank(loop.getRegionIterArgs()[j]); ++k) {
      unite(loop.getRegionIterArgs()[j], k, yield.getOperand(j), k);
      unite(loop.getRegionIterArgs()[j], k, loop.getResult(j), k);
    }
  loop.getBody()->walk([&](Operation *op) {
    if (op->getNumResults() != 1 || !isa<RankedTensorType>(op->getResult(0).getType()))
      return;
    Value r = op->getResult(0);
    if (auto br = dyn_cast<tt::BroadcastOp>(op)) {
      for (unsigned k = 0; k < rank(r); ++k)
        if (dim(br.getSrc(), k) == dim(r, k))
          unite(r, k, br.getSrc(), k);
    } else if (auto e = dyn_cast<tt::ExpandDimsOp>(op)) {
      for (unsigned k = 0; k < rank(r); ++k)
        if (k != e.getAxis())
          unite(r, k, e.getSrc(), k < e.getAxis() ? k : k - 1);
    } else if (auto t = dyn_cast<tt::TransOp>(op)) {
      for (auto [k, o] : llvm::enumerate(t.getOrder()))
        unite(r, k, t.getSrc(), o);
    } else if (auto rs = dyn_cast<tt::ReshapeOp>(op)) {
      if (auto oldOf = unitReshape(shapeOf(rs.getSrc().getType()), shapeOf(r.getType())))
        for (auto [k, o] : llvm::enumerate(*oldOf))
          if (o >= 0)
            unite(r, k, rs.getSrc(), o);
    } else if (auto red = dyn_cast<tt::ReduceOp>(op)) {
      for (unsigned k = 0; k < rank(r); ++k)
        unite(r, k, red.getOperand(0), k < red.getAxis() ? k : k + 1);
    } else if (auto d = dyn_cast<tt::DotOp>(op)) {
      unite(r, 0, d.getA(), 0);
      unite(r, 1, d.getB(), 1);
      unite(r, 0, d.getC(), 0);
      unite(r, 1, d.getC(), 1);
      unite(d.getA(), 1, d.getB(), 0);
    } else if (isElementwise(op) || isa<arith::SelectOp, arith::CmpFOp, arith::CmpIOp>(op)) {
      for (Value o : op->getOperands())
        if (rank(o) == rank(r))
          for (unsigned k = 0; k < rank(r); ++k)
            if (dim(o, k) == dim(r, k))
              unite(r, k, o, k);
    }
  });
  for (auto [k, a] : llvm::enumerate(env))
    if (a.kind == Axis::Expr)
      return loop.emitError("furiosa: a loop state read through a data reshape is not raised yet"), failure();
    else if (a.kind != Axis::Unit) {
      int x = find(node(loop.getResult(result), k));
      if (bound[x] && (bound[x]->kind != a.kind || bound[x]->dim != a.dim))
        conflict = true;
      bound[x] = a;
    }
  if (conflict)
    return loop.emitError("furiosa: loop state axes bind to two different output coordinates"), failure();
  SmallVector<SmallVector<Axis>> envs;
  SmallVector<bool> needed = stateClosure(loop, result);
  if (skipped)
    skipped->assign(loop.getNumRegionIterArgs(), false);
  for (auto [j, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    SmallVector<Axis> e;
    for (unsigned k = 0; k < rank(arg); ++k) {
      if (dim(arg, k) == 1) {
        e.push_back(Axis{});
        continue;
      }
      auto b = bound[find(node(arg, k))];
      if (!b && !needed[j] && skipped) { // unrelated to this result's coordinates and not read by it
        (*skipped)[j] = true;
        b = Axis{};
      }
      if (!b)
        return loop.emitError("furiosa: a loop state axis does not relate to the output coordinates"), failure();
      e.push_back(*b);
    }
    envs.push_back(e);
  }
  return envs;
}

/// A loop whose iter args carry state other than an accumulation, kept as scf.for over its trip count:
/// each state is a tensor over the dims of the consumer it depends on, and each iteration computes the
/// next states as parallel generics. Instances with a shorter trip keep their state unchanged.
FailureOr<Value> Raiser::materializeState(Gen &g, scf::ForOp loop, unsigned result, ArrayRef<Axis> env) {
  auto extract = [&](Value t, ArrayRef<unsigned> dims) {
    SmallVector<Value> idx;
    for (unsigned d : dims)
      idx.push_back(linalg::IndexOp::create(*g.ob, loc, d));
    return tensor::ExtractOp::create(*g.ob, loc, t, idx).getResult();
  };
  auto key = std::make_pair(loop.getOperation(), g.anchor);
  if (auto it = keptLoops.find(key); it != keptLoops.end() && it->second.first[result])
    return extract(it->second.first[result], it->second.second[result]);
  SmallVector<bool> skipped; // states this result neither reads nor relates to: left out of the loop
  auto envs = stateEnvs(loop, result, env, &skipped);
  if (failed(envs))
    return failure();
  unsigned id = loopIds.lookup(loop);
  SmallVector<unsigned> dataArgs;
  for (auto [j, arg] : llvm::enumerate(loop.getRegionIterArgs()))
    if (!act[0]->get(arg) && !skipped[j])
      dataArgs.push_back(j);
  if (!llvm::is_contained(dataArgs, result))
    return loop.emitError("furiosa: internal: an index loop result used as data"), failure();
  unsigned ns = dataArgs.size();

  // The loop with state k over keeps[k] (dims of g, in order). Per state, `direct` collects the kept dims
  // its generics read other than to read states, and `deps` the states they read (what nested generics
  // read reaches these generics as extract indices).
  SmallVector<Operation *> built;
  auto build = [&](ArrayRef<SmallVector<unsigned>> keeps, SmallVector<std::set<unsigned>> &direct,
                   SmallVector<std::set<unsigned>> &deps) -> FailureOr<scf::ForOp> {
    built.clear();
    direct.assign(ns, {});
    deps.assign(ns, {});
    std::vector<Gen> gks(ns);
    SmallVector<SmallVector<int>> remaps(ns);
    SmallVector<SmallVector<Axis>> kenvs(ns);
    for (unsigned k = 0; k < ns; ++k) {
      SmallVector<int> &remap = remaps[k];
      remap.assign(g.extent.size(), -1);
      Gen &gk = gks[k];
      for (auto [i, d] : llvm::enumerate(keeps[k])) {
        remap[d] = i;
        gk.lo.push_back(g.lo[d]);
        gk.extent.push_back(g.extent[d]);
      }
      gk.coordPos.assign(g.coordPos.size(), -1);
      for (auto [j, pos] : llvm::enumerate(g.coordPos))
        if (pos >= 0)
          gk.coordPos[j] = remap[pos];
      for (auto [l, pos] : g.loopPos)
        if (remap[pos] >= 0)
          gk.loopPos[l] = remap[pos];
      gk.loopIv = g.loopIv;
      for (auto &[v, st] : g.state) {
        SmallVector<int> dims;
        for (int d : st.second)
          dims.push_back(d >= 0 ? remap[d] : -1);
        gk.state[v] = {st.first, dims};
      }
      gk.anchor = g.anchor;
      kenvs[k] = (*envs)[dataArgs[k]];
      for (Axis &a : kenvs[k])
        if (a.kind == Axis::Iter) {
          if (remap[a.dim] < 0)
            return loop.emitError("furiosa: internal: a loop state axis binds to a pruned dim"), failure();
          a.dim = remap[a.dim];
        }
    }
    SmallVector<Value> inits;
    SmallVector<Operation *> initOps, nextOps(ns, nullptr);
    for (auto [k, j] : llvm::enumerate(dataArgs)) {
      auto init = parallelGeneric(gks[k], getElementTypeOrSelf(loop.getInitArgs()[j].getType()),
                                  [&](Gen &gg) { return emit(gg, loop.getInitArgs()[j], kenvs[k]); });
      if (failed(init))
        return failure();
      inits.push_back(*init);
      initOps.push_back(init->getDefiningOp());
      built.push_back(init->getDefiningOp());
    }
    // Bodies share one builder with `b`, so the loop is built under a guard that ends before the
    // consumer reads it.
    LogicalResult status = success();
    scf::ForOp forOp;
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(g.anchor);
      Value lb = arith::ConstantIndexOp::create(b, loc, 0), ub = arith::ConstantIndexOp::create(b, loc, loopTrip[id]),
            one = arith::ConstantIndexOp::create(b, loc, 1);
      forOp = scf::ForOp::create(b, loc, lb, ub, one, inits, [&](OpBuilder &nb, Location, Value iv, ValueRange args) {
        Operation *anchor = arith::ConstantIndexOp::create(nb, loc, 0);
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        SmallVector<Value> next;
        for (auto [k, j] : llvm::enumerate(dataArgs)) {
          Gen gl = gks[k];
          gl.loopIv[id] = iv;
          for (auto [m, jm] : llvm::enumerate(dataArgs)) {
            SmallVector<int> dims;
            for (unsigned d : keeps[m])
              dims.push_back(remaps[k][d]);
            gl.state[loop.getRegionIterArgs()[jm].getAsOpaquePointer()] = {args[m], dims};
          }
          gl.anchor = anchor;
          auto v = parallelGeneric(gl, getElementTypeOrSelf(yield.getOperand(j).getType()), [&](Gen &gg) -> FailureOr<Value> {
            auto n = emit(gg, yield.getOperand(j), kenvs[k]);
            if (failed(n) || !varTrip[id])
              return n;
            auto active = tripGuard(gg, id, iv);
            auto old = emit(gg, loop.getRegionIterArgs()[j], kenvs[k]);
            if (failed(active) || failed(old))
              return failure();
            return arith::SelectOp::create(*gg.ob, loc, *active, *n, *old).getResult();
          });
          if (failed(v)) {
            status = failure();
            next.push_back(args[k]);
            continue;
          }
          next.push_back(*v);
          nextOps[k] = v->getDefiningOp();
        }
        scf::YieldOp::create(nb, loc, next);
        retire(anchor);
      });
    }
    built.push_back(forOp);
    if (failed(status))
      return failure();
    auto stateOf = [&](Value t) -> int {
      auto arg = dyn_cast<BlockArgument>(t);
      return arg && arg.getOwner() == forOp.getBody() && arg.getArgNumber() > 0 ? (int)arg.getArgNumber() - 1 : -1;
    };
    for (unsigned k = 0; k < ns; ++k)
      for (Operation *op : {initOps[k], nextOps[k]}) {
        op->walk([&](tensor::ExtractOp ex) {
          if (int m = stateOf(ex.getTensor()); m >= 0)
            deps[k].insert(m);
        });
        op->walk([&](linalg::IndexOp ix) {
          for (Operation *user : ix->getUsers()) {
            auto ex = dyn_cast<tensor::ExtractOp>(user);
            if (!ex || stateOf(ex.getTensor()) < 0)
              direct[k].insert(keeps[k][ix.getDim()]);
          }
        });
      }
    return forOp;
  };

  SmallVector<unsigned> all;
  for (unsigned d = 0; d < g.extent.size(); ++d)
    all.push_back(d);
  SmallVector<SmallVector<unsigned>> keeps(ns, all);
  SmallVector<std::set<unsigned>> direct, deps;
  auto forOp = build(keeps, direct, deps);
  if (failed(forOp))
    return failure();
  // A state spans the dims it reads, its binding's dims, and the dims of every state it reads. Nested
  // generics see the states at their current span, so prune until nothing shrinks.
  while (true) {
    SmallVector<std::set<unsigned>> need(direct);
    for (unsigned k = 0; k < ns; ++k)
      for (const Axis &a : (*envs)[dataArgs[k]])
        if (a.kind == Axis::Iter)
          need[k].insert(a.dim);
    for (bool changed = true; changed;) {
      changed = false;
      for (unsigned k = 0; k < ns; ++k)
        for (unsigned m : deps[k])
          for (unsigned d : SmallVector<unsigned>(need[m].begin(), need[m].end()))
            changed |= need[k].insert(d).second;
    }
    SmallVector<SmallVector<unsigned>> pruned;
    for (auto &n : need)
      pruned.push_back(SmallVector<unsigned>(n.begin(), n.end()));
    if (pruned == keeps)
      break;
    // Rebuild with the pruned states, then drop the previous build.
    SmallVector<Operation *> previous = built;
    forOp = build(pruned, direct, deps);
    if (failed(forOp))
      return failure();
    for (Operation *op : llvm::reverse(previous))
      if (op->use_empty())
        op->erase();
    keeps = pruned;
  }
  auto &[results, dimsOf] = keptLoops[key]; // other results of the loop may be built already
  results.resize(loop.getNumResults());
  dimsOf.resize(loop.getNumResults());
  for (auto [k, j] : llvm::enumerate(dataArgs)) {
    results[j] = forOp->getResult(k);
    dimsOf[j] = keeps[k];
  }
  return extract(results[result], dimsOf[result]);
}

LogicalResult Raiser::raiseStore(Write store) {
  if (store.overPrograms) {
    // The space one instance writes, plus a free view dim per program id the instances vary along (z, y,
    // then x: x innermost, so the reduction runs in program order).
    SmallVector<Evaluator *> all = act;
    act = {all[0]};
    auto sp = storeSpace(store);
    act = all;
    if (failed(sp))
      return failure();
    SmallVector<int64_t> origin0 = sp->origin[0];
    sp->origin.assign(all.size(), origin0);
    for (int d = 2; d >= 0; --d) {
      int64_t lo = all[0]->programId()[d], hi = lo;
      for (Evaluator *ev : all) {
        lo = std::min(lo, ev->programId()[d]);
        hi = std::max(hi, ev->programId()[d]);
      }
      if (lo == hi)
        continue;
      sp->freeDims.push_back(sp->viewShape.size());
      sp->viewShape.push_back(hi - lo + 1);
      sp->boxLo.push_back(lo);
      sp->boxHi.push_back(hi);
      ++sp->programDims;
      for (auto [n, ev] : llvm::enumerate(all))
        sp->origin[n].push_back(ev->programId()[d]);
    }
    return raiseStoreIn(store, *sp);
  }
  auto sp = storeSpace(store);
  if (failed(sp))
    return failure();
  Block *block = b.getInsertionBlock();
  Operation *last = block->empty() ? nullptr : &block->back();
  auto currentBefore = current;
  auto storesBefore = stores;
  originUses.clear();
  digitUses.clear();
  if (failed(raiseStoreIn(store, *sp)))
    return failure();
  // When an expression used a stored axis's tile origin, the computation depends on that tile index
  // apart from the axis (a head index of a [T, H*D] row, say). Raise again over a view where the tile
  // index is a dim of its own, so values that depend on it alone span it alone; keep that if it works.
  SmallVector<int64_t> extra;
  for (unsigned k : originUses)
    if (auto d = dyn_cast_or_null<AffineDimExpr>(sp->global[k]); d && sp->step[k] == 1 && sp->phase[k] >= 0) {
      unsigned j = d.getPosition();
      if (sp->extent[k] < sp->viewShape[j] && sp->viewShape[j] % sp->extent[k] == 0)
        extra.push_back(sp->viewStrides[j] * sp->extent[k]);
    }
  // A view dim the computation divides into digits (a head index read as head // 2 by grouped-query
  // attention) splits into them, so those reads become projections of the split dims.
  for (auto [j, c] : digitUses)
    extra.push_back(sp->viewStrides[j] * c);
  if (extra.empty())
    return success();
  ScopedDiagnosticHandler quiet(ctx, [](Diagnostic &) { return success(); });
  auto refined = storeSpace(store, extra);
  if (failed(refined) || refined->viewShape == sp->viewShape)
    return success();
  Operation *mid = &block->back();
  auto currentFirst = current;
  auto storesFirst = stores;
  current = currentBefore;
  stores = storesBefore;
  // Ops of [from, to], users first; the final sweep drops whatever is still used.
  auto eraseRange = [&](Operation *from, Operation *to) {
    SmallVector<Operation *> ops;
    for (Operation *op = from; op; op = op->getNextNode()) {
      ops.push_back(op);
      if (op == to)
        break;
    }
    for (Operation *op : llvm::reverse(ops))
      if (op->use_empty())
        op->erase();
  };
  if (failed(raiseStoreIn(store, *refined))) {
    if (mid->getNextNode())
      eraseRange(mid->getNextNode(), &block->back());
    current = currentFirst;
    stores = storesFirst;
    return success();
  }
  eraseRange(last ? last->getNextNode() : &block->front(), mid);
  return success();
}

/// Raises `store` over the coordinate space `sp`.
LogicalResult Raiser::raiseStoreIn(Write store, const StoreSpace &spRef) {
  const StoreSpace *sp = &spRef;
  space = sp;
  int arg = sp->arg;
  unsigned nd = sp->viewShape.size(), nv = nd - sp->programDims; // view dims of the buffer, then program ids
  SmallVector<int64_t> viewShape(sp->viewShape.begin(), sp->viewShape.begin() + nv);
  SmallVector<int64_t> boxLo(sp->boxLo.begin(), sp->boxLo.begin() + nv);
  SmallVector<Axis> env;
  for (unsigned i = sp->loops.size(); i < sp->global.size(); ++i) {
    Axis a;
    if (sp->global[i]) {
      a.kind = Axis::Store;
      a.dim = i;
    }
    env.push_back(a);
  }
  // A root context whose dims are the view dims over the written box.
  Gen root;
  for (unsigned j = 0; j < nd; ++j) {
    root.coordPos.push_back(j);
    root.lo.push_back(sp->boxLo[j]);
    root.extent.push_back(sp->boxHi[j] - sp->boxLo[j] + 1);
  }
  // Anchor everything this store reads before a placeholder at the current end of the function.
  Operation *anchor = arith::ConstantIndexOp::create(b, loc, 0);
  root.anchor = anchor;
  Type elem = buffers[arg].elem;
  // The stored value, as a parallel generic over every view dim (no pruning: it fills the box). With program
  // dims, that is each instance's value, and one reduction then combines them into the old value in program order.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(anchor);
  SmallVector<int64_t> box(root.extent.begin(), root.extent.begin() + nv);
  auto rmw = dyn_cast<tt::AtomicRMWOp>(store.getOperation());
  auto cas = dyn_cast<tt::AtomicCASOp>(store.getOperation());
  // An atomic, or a write whose mask depends on data, starts from what the box holds.
  bool keepsOld = rmw || cas || !sp->dataMask.empty();
  Value init = !keepsOld ? tensor::EmptyOp::create(b, loc, box, elem).getResult()
               : sp->full ? reshape(current[arg], viewShape)
                          : tensor::ExtractSliceOp::create(b, loc, reshape(current[arg], viewShape),
                                                           getAsIndexOpFoldResult(ctx, boxLo),
                                                           getAsIndexOpFoldResult(ctx, box),
                                                           getAsIndexOpFoldResult(ctx, SmallVector<int64_t>(nv, 1)))
                                .getResult();
  // old (op) x, the atomic's combine.
  auto combine = [&](OpBuilder &ob, Value old, Value x) -> Value {
    bool fp = isa<FloatType>(x.getType());
    switch (rmw.getAtomicRmwOp()) {
    case tt::RMWOp::ADD:
    case tt::RMWOp::FADD:
      return fp ? arith::AddFOp::create(ob, loc, old, x).getResult() : arith::AddIOp::create(ob, loc, old, x).getResult();
    case tt::RMWOp::MAX:
      return fp ? arith::MaxNumFOp::create(ob, loc, old, x).getResult() : arith::MaxSIOp::create(ob, loc, old, x).getResult();
    case tt::RMWOp::MIN:
      return fp ? arith::MinNumFOp::create(ob, loc, old, x).getResult() : arith::MinSIOp::create(ob, loc, old, x).getResult();
    case tt::RMWOp::UMAX:
      return arith::MaxUIOp::create(ob, loc, old, x);
    case tt::RMWOp::UMIN:
      return arith::MinUIOp::create(ob, loc, old, x);
    case tt::RMWOp::AND:
      return arith::AndIOp::create(ob, loc, old, x);
    case tt::RMWOp::OR:
      return arith::OrIOp::create(ob, loc, old, x);
    case tt::RMWOp::XOR:
      return arith::XOrIOp::create(ob, loc, old, x);
    case tt::RMWOp::XCHG:
      return x;
    }
    llvm_unreachable("an atomic read-modify-write kind");
  };
  Value values = sp->programDims ? tensor::EmptyOp::create(b, loc, root.extent, elem).getResult() : init;
  LogicalResult status = success();
  Gen g = root;
  auto op = linalg::GenericOp::create(b,
      loc, TypeRange{values.getType()}, ValueRange{}, ValueRange{values},
      ArrayRef<AffineMap>{AffineMap::getMultiDimIdentityMap(nd, ctx)},
      SmallVector<utils::IteratorType>(nd, utils::IteratorType::parallel), [&](OpBuilder &ob, Location, ValueRange args) {
        g.ob = &ob;
        g.index.assign(nd, Value());
        auto v = emit(g, store.getValue(), env);
        if (failed(v)) {
          status = failure();
          linalg::YieldOp::create(ob, loc, ValueRange{arith::ConstantOp::create(ob, loc, ob.getZeroAttr(elem))});
          return;
        }
        Value out = *v;
        if (sp->programDims) {
          linalg::YieldOp::create(ob, loc, ValueRange{out});
          return;
        }
        Type vt = out.getType(); // the buffer's type, or one of its width through a pointer bitcast
        Value old = args[0];
        if (vt != elem)
          old = arith::BitcastOp::create(ob, loc, vt, old);
        if (rmw)
          out = combine(ob, old, out);
        if (cas) { // val where the old value's bits equal cmp's
          auto cmp = emit(g, cas.getCmp(), env);
          if (failed(cmp)) {
            status = failure();
          } else {
            Type bits = ob.getIntegerType(vt.getIntOrFloatBitWidth());
            auto asBits = [&](Value x) -> Value { return vt.isInteger() ? x : arith::BitcastOp::create(ob, loc, bits, x).getResult(); };
            Value same = arith::CmpIOp::create(ob, loc, arith::CmpIPredicate::eq, asBits(old), asBits(*cmp));
            out = arith::SelectOp::create(ob, loc, same, out, old);
          }
        }
        for (Value term : sp->dataMask) {
          auto on = emit(g, term, env);
          if (failed(on)) {
            status = failure();
            break;
          }
          out = arith::SelectOp::create(ob, loc, *on, out, old);
        }
        if (vt != elem)
          out = arith::BitcastOp::create(ob, loc, elem, out);
        linalg::YieldOp::create(ob, loc, ValueRange{out});
      });
  g.anchor = op;
  if (failed(status)) {
    retire(anchor);
    space = nullptr;
    return failure();
  }
  b.setInsertionPoint(anchor);
  Value result = op.getResult(0);
  if (sp->programDims) {
    SmallVector<utils::IteratorType> iters(nv, utils::IteratorType::parallel);
    iters.append(sp->programDims, utils::IteratorType::reduction);
    AffineMap all = AffineMap::getMultiDimIdentityMap(nd, ctx);
    result = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ValueRange{result}, ValueRange{init},
                                       ArrayRef<AffineMap>{all, all.getMajorSubMap(nv)}, iters,
                                       [&](OpBuilder &ob, Location, ValueRange args) {
                                         linalg::YieldOp::create(ob, loc, combine(ob, args[1], args[0]));
                                       })
                 .getResult(0);
  }
  Value view = reshape(current[arg], viewShape);
  Value updated = sp->full ? result
                           : tensor::InsertSliceOp::create(b, loc, result, view,
                                                             getAsIndexOpFoldResult(ctx, boxLo),
                                                             getAsIndexOpFoldResult(ctx, box),
                                                             getAsIndexOpFoldResult(ctx, SmallVector<int64_t>(nv, 1)))
                                 .getResult();
  current[arg] = reshape(updated, buffers[arg].shape);
  stores.push_back({store.getOperation(), arg, current[arg]});
  retire(anchor);
  space = nullptr;
  return success();
}

/// A scan as a reduction: out[i] combines the elements j <= i (j >= i reversed) of the scanned axis.
/// The reduction runs over a new j axis next to i: with a known identity, of x_j where j is in range and
/// the identity elsewhere; otherwise with j's condition as one more operand, and a combiner that keeps
/// the accumulator where it fails.
void Raiser::lowerScans() {
  // A counted while loop (cmpi on a counter it steps) is a for loop: upstream's uplift.
  SmallVector<scf::WhileOp> whiles;
  kernel.walk([&](scf::WhileOp w) { whiles.push_back(w); });
  IRRewriter rw(ctx);
  for (scf::WhileOp w : whiles) {
    rw.setInsertionPoint(w);
    (void)scf::upliftWhileToForLoop(rw, w);
  }
  SmallVector<tt::ScanOp> scans;
  kernel.walk([&](tt::ScanOp s) { scans.push_back(s); });
  for (tt::ScanOp scan : scans) {
    OpBuilder ob(scan);
    unsigned a = scan.getAxis(), n = scan.getNumOperands();
    SmallVector<int64_t> wide = shapeOf(scan.getOperand(0).getType());
    int64_t c = wide[a];
    wide.insert(wide.begin() + a, c); // [..., i, j, ...]
    auto bcast = [&](Value v) -> Value {
      return tt::BroadcastOp::create(ob, loc, RankedTensorType::get(wide, getElementTypeOrSelf(v.getType())), v);
    };
    auto along = [&](unsigned at) { // the scanned axis' index, running along dim `at` of `wide`
      Value r = tt::MakeRangeOp::create(ob, loc, RankedTensorType::get({c}, ob.getI32Type()), 0, c);
      for (unsigned d = 0; d < wide.size(); ++d)
        if (d != at)
          r = tt::ExpandDimsOp::create(ob, loc, r, d);
      return bcast(r);
    };
    SmallVector<Value> srcs;
    for (Value x : scan.getOperands())
      srcs.push_back(bcast(tt::ExpandDimsOp::create(ob, loc, x, a)));
    Value in = arith::CmpIOp::create(ob, loc, scan.getReverse() ? arith::CmpIPredicate::sge : arith::CmpIPredicate::sle,
                                     along(a + 1), along(a));
    Block &old = scan.getCombineOp().front();
    if (TypedAttr e = identityOf(old)) { // the identity where j is out of range, the combiner as is
      auto t = cast<RankedTensorType>(srcs[0].getType());
      Value fill = arith::ConstantOp::create(ob, loc, DenseElementsAttr::get(t, e));
      auto red = tt::ReduceOp::create(ob, loc, ValueRange{arith::SelectOp::create(ob, loc, in, srcs[0], fill)}, a + 1);
      red.getCombineOp().takeBody(scan.getCombineOp());
      Block &body = red.getCombineOp().front();
      OpBuilder rb(body.getTerminator());
      tt::ReduceReturnOp::create(rb, loc, body.getTerminator()->getOperands());
      body.getTerminator()->erase();
      scan->getResult(0).replaceAllUsesWith(red->getResult(0));
      scan.erase();
      continue;
    }
    srcs.push_back(in);
    auto red = tt::ReduceOp::create(ob, loc, srcs, a + 1);
    Block *blk = new Block();
    red.getCombineOp().push_back(blk);
    SmallVector<Value> accs, xs;
    for (unsigned k = 0; k < n; ++k)
      accs.push_back(blk->addArgument(old.getArgument(k).getType(), loc));
    Value accOk = blk->addArgument(ob.getI1Type(), loc);
    for (unsigned k = 0; k < n; ++k)
      xs.push_back(blk->addArgument(old.getArgument(n + k).getType(), loc));
    Value xOk = blk->addArgument(ob.getI1Type(), loc);
    OpBuilder rb = OpBuilder::atBlockEnd(blk);
    IRMapping map;
    for (unsigned k = 0; k < n; ++k) {
      map.map(old.getArgument(k), accs[k]);
      map.map(old.getArgument(n + k), xs[k]);
    }
    for (Operation &o : old.without_terminator())
      rb.clone(o, map);
    SmallVector<Value> outs;
    for (auto [k, y] : llvm::enumerate(old.getTerminator()->getOperands())) {
      Value v = arith::SelectOp::create(rb, loc, accOk, map.lookupOrDefault(y), xs[k]);
      outs.push_back(arith::SelectOp::create(rb, loc, xOk, v, accs[k]));
    }
    outs.push_back(arith::OrIOp::create(rb, loc, accOk, xOk));
    tt::ReduceReturnOp::create(rb, loc, outs);
    for (unsigned k = 0; k < n; ++k)
      scan->getResult(k).replaceAllUsesWith(red->getResult(k));
    scan.erase();
  }
  // A histogram as a reduction: count[b] sums (x_i == b) over i, masked lanes left out.
  SmallVector<tt::HistogramOp> hists;
  kernel.walk([&](tt::HistogramOp h) { hists.push_back(h); });
  for (tt::HistogramOp h : hists) {
    OpBuilder ob(h);
    int64_t n = shapeOf(h.getSrc().getType())[0], bins = shapeOf(h.getType())[0];
    SmallVector<int64_t> wide{bins, n};
    Type src = getElementTypeOrSelf(h.getSrc().getType()), count = getElementTypeOrSelf(h.getType());
    auto bcast = [&](Value x, unsigned axis) -> Value {
      return tt::BroadcastOp::create(ob, loc, RankedTensorType::get(wide, getElementTypeOrSelf(x.getType())),
                                     tt::ExpandDimsOp::create(ob, loc, x, axis));
    };
    Value b = tt::MakeRangeOp::create(ob, loc, RankedTensorType::get({bins}, ob.getI32Type()), 0, bins);
    if (src.getIntOrFloatBitWidth() > 32)
      b = arith::ExtSIOp::create(ob, loc, RankedTensorType::get({bins}, src), b);
    else if (src.getIntOrFloatBitWidth() < 32)
      b = arith::TruncIOp::create(ob, loc, RankedTensorType::get({bins}, src), b);
    Value hit = arith::CmpIOp::create(ob, loc, arith::CmpIPredicate::eq, bcast(h.getSrc(), 0), bcast(b, 1));
    if (Value m = h.getMask())
      hit = arith::AndIOp::create(ob, loc, hit, bcast(m, 0));
    auto t = RankedTensorType::get(wide, count);
    Value one = arith::SelectOp::create(ob, loc, hit, arith::ConstantOp::create(ob, loc, DenseElementsAttr::get(t, ob.getIntegerAttr(count, 1))),
                                        arith::ConstantOp::create(ob, loc, DenseElementsAttr::get(t, ob.getIntegerAttr(count, 0))));
    auto red = tt::ReduceOp::create(ob, loc, ValueRange{one}, 1);
    Block *blk = new Block();
    red.getCombineOp().push_back(blk);
    Value acc = blk->addArgument(count, loc), x = blk->addArgument(count, loc);
    OpBuilder rb = OpBuilder::atBlockEnd(blk);
    tt::ReduceReturnOp::create(rb, loc, ValueRange{arith::AddIOp::create(rb, loc, acc, x)});
    h.getResult().replaceAllUsesWith(red->getResult(0));
    h.erase();
  }
}

LogicalResult Raiser::run() {
  if (failed(parseLaunch()))
    return failure();
  lowerScans();
  if (failed(evaluate()))
    return failure();
  b.setInsertionPointToEnd(module.getBody());
  SmallVector<Type> argTypes;
  for (auto &[i, buf] : buffers)
    argTypes.push_back(RankedTensorType::get(buf.shape, buf.elem));
  // Stores, in program order, decide the results. The instances that run a store are split by their
  // control path and the buffer they write; each part is raised on its own.
  SmallVector<Write> stores;
  WalkResult walked = kernel.walk([&](Operation *op) {
    if (isa<tt::StoreOp>(op))
      stores.push_back({op});
    if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op) && !op->getParentOfType<scf::ForOp>())
      stores.push_back({op});
    else if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op))
      return op->emitError("furiosa: only an atomic read-modify-write outside loops is raised"), WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();
  SmallVector<Operation *> branches;
  kernel.walk([&](scf::IfOp i) { branches.push_back(i); });
  std::vector<std::pair<Write, SmallVector<Evaluator *>>> parts;
  std::set<int> writtenSet;
  for (Write st : stores) {
    std::map<std::pair<int, std::vector<int>>, SmallVector<Evaluator *>> byKey;
    for (auto &ev : evals) {
      if (!ev->executed.count(st.getOperation()))
        continue;
      const Sym *p = ev->get(st.getPtr());
      if (!p || p->kind != Sym::Ptr)
        return st.emitError("furiosa: store address is not a pointer expression");
      std::vector<int> path;
      for (Operation *br : branches)
        path.push_back(ev->taken.count(br) ? (int)ev->taken.lookup(br) : -1);
      byKey[{p->base, path}].push_back(ev.get());
      writtenSet.insert(p->base);
    }
    for (auto &[key, instances] : byKey) {
      if (!isa<tt::AtomicRMWOp, tt::AtomicCASOp>(st.getOperation())) {
        parts.push_back({st, instances});
        continue;
      }
      // Instances whose atomics touch an element in common apply one after the other: split them into
      // rounds that write disjoint elements, each raised as a store of old (op) val.
      std::vector<std::set<int64_t>> roundAddrs;
      std::vector<SmallVector<Evaluator *>> rounds;
      SmallVector<int64_t> shape = shapeOf(st.getValue().getType());
      std::set<int64_t> firstAddrs;
      // Every instance writes the same elements through the same address, of the buffer's own type, with no
      // mask on data (Triton's float max/min through an i32 pointer and a sign mask stays in rounds).
      const Sym *p0 = instances[0]->get(st.getPtr());
      bool sameElements = getElementTypeOrSelf(st.getValue().getType()) == buffers[p0->base].elem;
      for (Evaluator *ev : instances) {
        const Sym *p = ev->get(st.getPtr());
        if (!p->data.empty())
          return st.emitError("furiosa: an atomic address depends on data");
        SmallVector<AffineExpr> conds; // where it may write: the data terms of its mask only narrow that
        bool never = false;
        SmallVector<Value> data;
        if (st.getMask())
          splitMask(st.getMask(), [&](Value v) { return ev->get(v); }, conds, never, data);
        sameElements &= data.empty();
        std::set<int64_t> addrs;
        SmallVector<int64_t> at(shape.size(), 0);
        for (bool more = !never; more;) {
          if (llvm::all_of(conds, [&](AffineExpr c) { return evalAt(c, at) >= 0; }))
            addrs.insert(evalAt(p->expr, at));
          more = false;
          for (int k = shape.size() - 1; k >= 0 && !more; --k)
            more = ++at[k] < shape[k] || (at[k] = 0, false);
        }
        if (ev == instances[0])
          firstAddrs = addrs;
        else
          sameElements &= addrs == firstAddrs && p->expr == p0->expr;
        unsigned r = 0;
        while (r < rounds.size() && llvm::any_of(addrs, [&](int64_t a) { return roundAddrs[r].count(a); }))
          ++r;
        if (r == rounds.size()) {
          rounds.emplace_back();
          roundAddrs.emplace_back();
        }
        rounds[r].push_back(ev);
        roundAddrs[r].insert(addrs.begin(), addrs.end());
      }
      // When every instance combines into the same elements and the old value is unused, the rounds are one
      // write: old (op) each instance's value in program order, a reduction over the program ids (they must
      // fill a box) of a value computed once for all of them.
      auto rmw = dyn_cast<tt::AtomicRMWOp>(st.getOperation());
      int64_t box = 1;
      for (unsigned d = 0; d < 3; ++d) {
        auto [lo, hi] = std::minmax_element(instances.begin(), instances.end(), [&](Evaluator *x, Evaluator *y) {
          return x->programId()[d] < y->programId()[d];
        });
        box *= (*hi)->programId()[d] - (*lo)->programId()[d] + 1;
      }
      if (rounds.size() > 1 && sameElements && rmw && rmw.getAtomicRmwOp() != tt::RMWOp::XCHG && st->use_empty() &&
          box == (int64_t)instances.size()) {
        Write once = st;
        once.overPrograms = true;
        parts.push_back({once, instances});
        continue;
      }
      // Its old value is the buffer before it writes: one value only if every instance writes alone.
      if (rounds.size() > 1 && !st->use_empty())
        return st.emitError("furiosa: an atomic whose old value is used, by instances that write an element in "
                            "common, is not raised");
      for (auto &round : rounds)
        parts.push_back({st, round});
    }
  }
  SmallVector<Type> resultTypes;
  for (int w : writtenSet)
    resultTypes.push_back(argTypes[w]);
  auto func = func::FuncOp::create(b, loc, kernel.getName(), b.getFunctionType(argTypes, resultTypes));
  Block *entry = func.addEntryBlock();
  b.setInsertionPointToStart(entry);
  for (auto [i, arg] : llvm::enumerate(entry->getArguments())) {
    arg.setLoc(kernel.getArgument(i).getLoc()); // the Triton parameter name, as a NameLoc
    current[i] = arg;
    args.push_back(arg);
  }
  for (auto &[st, instances] : parts) {
    act = instances;
    if (failed(raiseStore(st))) {
      func.erase();
      return failure();
    }
  }
  SmallVector<Value> results;
  SmallVector<int64_t> written;
  for (int w : writtenSet) {
    results.push_back(current[w]);
    written.push_back(w);
  }
  func::ReturnOp::create(b, loc, results);
  func->setAttr("furiosa.written", b.getI64ArrayAttr(written));
  // Drop what the first attempts of pruned generics and kept loops left behind.
  for (bool erased = true; erased;) {
    erased = false;
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(entry->getOperations())))
      if (isOpTriviallyDead(&op)) {
        op.erase();
        erased = true;
      }
  }
  std::string why;
  bool invalid;
  {
    // Collect the verifier's messages into the one diagnostic this pass reports.
    ScopedDiagnosticHandler capture(ctx, [&](Diagnostic &d) {
      why += " | " + d.str();
      return success();
    });
    invalid = failed(verify(func));
  }
  if (invalid) {
    if (std::getenv("FURIOSA_DEBUG"))
      func.print(llvm::errs());
    return kernel.emitError("furiosa: internal error: the raised function does not verify") << why;
  }
  kernel.erase();
  return success();
}

struct RaiseToLinalgPass : PassWrapper<RaiseToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaiseToLinalgPass)

  StringRef getArgument() const override { return "furiosa-raise-to-linalg"; }
  StringRef getDescription() const override {
    return "Raise a launch-specialized Triton kernel to whole-tensor linalg";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registerDialects(registry); }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<tt::FuncOp> kernels(module.getOps<tt::FuncOp>());
    if (kernels.size() != 1) {
      module.emitError("furiosa: expected exactly one kernel");
      return signalPassFailure();
    }
    Raiser raiser(module, kernels[0]);
    if (failed(raiser.run()))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRaiseToLinalgPass() { return std::make_unique<RaiseToLinalgPass>(); }

void registerDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect, math::MathDialect,
                  scf::SCFDialect, tensor::TensorDialect>();
  func::registerInlinerExtension(registry);
}

} // namespace mlir::furiosa

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
#include "mlir/IR/Verifier.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/JSON.h"

#include <cmath>
#include <functional>
#include <limits>
#include <map>
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

Interval bounds(AffineExpr e, ArrayRef<Interval> dims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return {c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return dims[d.getPosition()];
  auto bin = cast<AffineBinaryOpExpr>(e);
  Interval l = bounds(bin.getLHS(), dims), r = bounds(bin.getRHS(), dims);
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

private:
  MLIRContext *ctx;
  SmallVector<int64_t, 3> pid, grid;
  const DenseMap<Value, int> &argIndex;
  const DenseMap<Operation *, unsigned> &loopIds;
  DenseMap<Value, Sym> syms;

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
      return setInt(p, cst(pid[p.getAxisAsInt()])), success();
    if (auto p = dyn_cast<tt::GetNumProgramsOp>(op))
      return setInt(p, cst(grid[p.getAxisAsInt()])), success();
    if (auto r = dyn_cast<tt::MakeRangeOp>(op))
      return setInt(r, getAffineDimExpr(0, ctx) + cst(r.getStart())), success();
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
    if (auto a = dyn_cast<tt::AddPtrOp>(op)) {
      const Sym *ptr = get(a.getPtr());
      if (!ptr)
        return op->emitError("furiosa: address of an unknown pointer");
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
          return op->emitError("furiosa: product of two non-constant indices");
        res = a * b;
      } else if (isa<arith::DivSIOp, arith::DivUIOp>(op)) {
        if (!cb || cb.getValue() <= 0)
          return op->emitError("furiosa: index division by a non-constant or non-positive value");
        res = a.floorDiv(cb.getValue());
      } else if (isa<arith::RemSIOp, arith::RemUIOp>(op)) {
        if (!cb || cb.getValue() <= 0)
          return op->emitError("furiosa: index remainder by a non-constant or non-positive value");
        res = a % cb.getValue();
      } else {
        if (!ca || !cb)
          return op->emitError("furiosa: min/max of non-constant indices");
        bool isMin = isa<arith::MinSIOp, arith::MinUIOp>(op);
        res = cst(isMin ? std::min(ca.getValue(), cb.getValue()) : std::max(ca.getValue(), cb.getValue()));
      }
      setInt(op->getResult(0), res);
      return success();
    }
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp>(op)) {
      if (const Sym *s = get(op->getOperand(0))) {
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
      if (t && f && c && c->kind == Sym::Mask && c->conds.empty())
        syms[s] = c->never ? *f : *t;
      return success();
    }
    if (auto f = dyn_cast<scf::ForOp>(op))
      return evalFor(f);
    return success();
  }

  LogicalResult evalFor(scf::ForOp f) {
    auto lb = getConstantIntValue(f.getLowerBound()), ub = getConstantIntValue(f.getUpperBound()),
         step = getConstantIntValue(f.getStep());
    if (!lb || !ub || !step || *step <= 0)
      return f.emitError("furiosa: loop bounds are not constant after launch specialization");
    unsigned id = loopIds.lookup(f);
    int64_t trip = std::max<int64_t>(0, (*ub - *lb + *step - 1) / *step);
    AffineExpr t = getAffineSymbolExpr(id, ctx);
    setInt(f.getInductionVar(), cst(*lb) + t * *step);

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
      for (unsigned i : carried) {
        const Sym *init = get(f.getInitArgs()[i]);
        const Sym *next = get(yield.getOperand(i));
        if (!next || next->data.size() != init->data.size())
          return f.emitError("furiosa: loop-carried index is not affine");
        AffineExpr delta = next->expr - init->expr;
        if (delta.isFunctionOfSymbol(id))
          return f.emitError("furiosa: loop-carried index does not advance by a loop-invariant amount");
        Sym s = *init;
        s.expr = init->expr + t * delta;
        syms[f.getRegionIterArgs()[i]] = s;
      }
    }
    if (failed(evalBlock(*f.getBody())))
      return failure();
    for (unsigned i : carried) {
      Sym s = *get(f.getRegionIterArgs()[i]);
      s.expr = s.expr.replace(t, cst(trip));
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

/// The coordinate space a store defines.
struct StoreSpace {
  int arg = -1;
  SmallVector<int64_t> viewShape, viewStrides; // row-major view of the buffer
  SmallVector<int64_t> boxLo, boxHi;           // written box (inclusive) per view dim
  bool full = false;
  SmallVector<int> axisDim;                    // per stored axis: its view dim (-1: unit axis)
  SmallVector<int64_t> axisExtent, phase;      // per view dim: extent of its axis (0: none), origin phase
  std::vector<SmallVector<int64_t>> origin;    // per instance, per view dim
};

/// What an axis of a value means inside the generic being built.
struct Axis {
  enum Kind { Coord, Iter, Unit } kind = Unit;
  int dim = 0; // Coord: store-view dim; Iter: generic dim
};

/// The generic being built: its dims are the store-view dims it keeps, then its own dims.
struct Gen {
  SmallVector<int> coordPos;      // store-view dim -> generic dim (-1: absent)
  SmallVector<int64_t> lo, extent; // per generic dim: offset of linalg.index, and extent
  DenseMap<unsigned, int> loopPos; // loop id -> generic dim
  OpBuilder *ob = nullptr;
  Operation *anchor = nullptr;     // insertion point for what this generic reads
  SmallVector<Value> index;        // per generic dim, lazily created absolute index
  std::map<std::pair<void *, std::string>, Value> memo;
};

std::string key(ArrayRef<Axis> env) {
  std::string s;
  for (const Axis &a : env)
    s += std::to_string((int)a.kind) + ":" + std::to_string(a.dim) + ",";
  return s;
}

class Raiser {
public:
  Raiser(ModuleOp module, tt::FuncOp kernel)
      : module(module), kernel(kernel), ctx(module.getContext()), b(ctx), loc(kernel.getLoc()) {}

  LogicalResult run();

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
  std::vector<std::unique_ptr<Evaluator>> evals;
  std::map<int, Value> current;
  std::map<int, Value> flat;
  const StoreSpace *space = nullptr;

  LogicalResult parseLaunch();
  LogicalResult evaluate();
  FailureOr<StoreSpace> storeSpace(tt::StoreOp store);
  LogicalResult raiseStore(tt::StoreOp store);

  AffineExpr originExpr(const Gen &g, unsigned viewDim);
  FailureOr<AffineExpr> rewrite(Operation *at, const Gen &g, ArrayRef<Axis> env,
                                std::function<AffineExpr(const Sym &)> select, Value v);
  Value index(Gen &g, unsigned dim);
  Value exprValue(Gen &g, AffineExpr e);
  Value flatOf(int arg);
  FailureOr<Value> emit(Gen &g, Value v, ArrayRef<Axis> env);
  FailureOr<Value> emitLoad(Gen &g, tt::LoadOp load, ArrayRef<Axis> env);
  FailureOr<Value> emitSym(Gen &g, Value v, const Sym &s0, ArrayRef<Axis> env);
  FailureOr<Value> emitDataTerms(Gen &g, const Sym &s0, ArrayRef<Axis> env);
  FailureOr<Value> emitElementwise(Gen &g, Operation *op, ArrayRef<Axis> env);
  FailureOr<Value> materialize(Gen &g, Value v, ArrayRef<Axis> env);

  /// Builds one generic whose first dims are `parentDims` of `parent` (a subset, in order) followed by
  /// `ownExtents`; `body` yields the value at a point given the generic, the out value (accumulator)
  /// and the parent-dim remap. Returns the result tensor and the parent dims it keeps.
  using BodyFn = std::function<FailureOr<Value>(Gen &, Value acc, ArrayRef<int> remap)>;
  using InitFn = std::function<FailureOr<Value>(Gen &, ArrayRef<int> remap)>;
  FailureOr<std::pair<Value, SmallVector<unsigned>>> buildGeneric(Gen &parent, ArrayRef<unsigned> parentDims,
                                                                  ArrayRef<int64_t> ownExtents,
                                                                  ArrayRef<std::pair<unsigned, unsigned>> loopDims,
                                                                  Type elem, InitFn init, BodyFn body);
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
    auto lb = getConstantIntValue(f.getLowerBound()), ub = getConstantIntValue(f.getUpperBound()),
         step = getConstantIntValue(f.getStep());
    loopTrip.push_back((lb && ub && step && *step > 0) ? std::max<int64_t>(0, (*ub - *lb + *step - 1) / *step) : 0);
  });
  for (int64_t z = 0; z < grid[2]; ++z)
    for (int64_t y = 0; y < grid[1]; ++y)
      for (int64_t x = 0; x < grid[0]; ++x) {
        evals.push_back(std::make_unique<Evaluator>(ctx, ArrayRef<int64_t>{x, y, z}, grid, argIndex, loopIds));
        if (failed(evals.back()->run(kernel)))
          return failure();
      }
  return success();
}

FailureOr<StoreSpace> Raiser::storeSpace(tt::StoreOp store) {
  if (store->getParentOp() != kernel.getOperation())
    return store.emitError("furiosa: stores inside loops or branches are not raised yet"), failure();
  SmallVector<int64_t> shape = shapeOf(store.getValue().getType());
  unsigned rank = shape.size();
  StoreSpace sp;
  const Sym *p0 = evals[0]->get(store.getPtr());
  if (!p0 || p0->kind != Sym::Ptr || !p0->data.empty())
    return store.emitError("furiosa: store address is not an affine function of program ids and axes"), failure();
  sp.arg = p0->base;
  const BufferInfo &buf = buffers[sp.arg];
  auto lin0 = linearize(p0->expr, rank);
  if (!lin0)
    return store.emitError("furiosa: store address is not linear in the stored axes"), failure();
  for (auto &ev : evals) {
    const Sym *p = ev->get(store.getPtr());
    auto lin = linearize(p->expr, rank);
    if (!lin || lin->first != lin0->first || p->base != sp.arg || !p->data.empty())
      return store.emitError("furiosa: store strides differ between program instances"), failure();
  }
  const SmallVector<int64_t> &coef = lin0->first;

  // View strides: every axis stride, plus each buffer dim stride no axis spans across.
  SmallVector<int64_t> bufStrides(buf.shape.size(), 1);
  for (int i = (int)buf.shape.size() - 2; i >= 0; --i)
    bufStrides[i] = bufStrides[i + 1] * buf.shape[i + 1];
  std::set<int64_t, std::greater<int64_t>> strides{1};
  for (unsigned i = 0; i < rank; ++i)
    if (shape[i] > 1) {
      if (coef[i] <= 0)
        return store.emitError("furiosa: a stored axis does not advance the address"), failure();
      strides.insert(coef[i]);
    }
  for (int64_t s : bufStrides) {
    bool spanned = false;
    for (unsigned i = 0; i < rank; ++i)
      spanned |= shape[i] > 1 && coef[i] < s && s < coef[i] * shape[i];
    if (!spanned)
      strides.insert(s);
  }
  SmallVector<int64_t> st(strides.begin(), strides.end());
  for (unsigned j = 1; j < st.size(); ++j)
    if (st[j - 1] % st[j])
      return store.emitError("furiosa: store strides do not nest"), failure();
  if (buf.numel() % st[0])
    return store.emitError("furiosa: store strides do not divide the buffer"), failure();
  unsigned nd = st.size();
  sp.viewStrides = st;
  sp.viewShape.push_back(buf.numel() / st[0]);
  for (unsigned j = 1; j < nd; ++j)
    sp.viewShape.push_back(st[j - 1] / st[j]);
  sp.axisDim.assign(rank, -1);
  sp.axisExtent.assign(nd, 0);
  for (unsigned i = 0; i < rank; ++i)
    if (shape[i] > 1) {
      int j = llvm::find(st, coef[i]) - st.begin();
      if (sp.axisExtent[j])
        return store.emitError("furiosa: two stored axes share a stride"), failure();
      sp.axisDim[i] = j;
      sp.axisExtent[j] = shape[i];
    }

  // Origins and the written box, by enumerating every instance and point.
  std::vector<uint8_t> covered(buf.numel(), 0);
  sp.boxLo.assign(nd, std::numeric_limits<int64_t>::max());
  sp.boxHi.assign(nd, std::numeric_limits<int64_t>::min());
  int64_t count = 0, total = 1;
  for (int64_t d : shape)
    total *= d;
  SmallVector<int64_t> idx(rank);
  for (auto &ev : evals) {
    const Sym *p = ev->get(store.getPtr());
    const Sym *m = store.getMask() ? ev->get(store.getMask()) : nullptr;
    if (store.getMask() && (!m || m->kind != Sym::Mask))
      return store.emitError("furiosa: store mask is not an index predicate"), failure();
    int64_t base = linearize(p->expr, rank)->second;
    SmallVector<int64_t> origin(nd);
    int64_t rest = base;
    for (unsigned j = 0; j < nd; ++j) {
      origin[j] = floorDiv(rest, st[j]);
      rest -= origin[j] * st[j];
    }
    sp.origin.push_back(origin);
    if (m && m->never)
      continue;
    for (int64_t f = 0; f < total; ++f) {
      int64_t r = f;
      for (int i = rank - 1; i >= 0; --i) {
        idx[i] = r % shape[i];
        r /= shape[i];
      }
      if (m && llvm::any_of(m->conds, [&](AffineExpr c) { return evalAt(c, idx) < 0; }))
        continue;
      int64_t addr = evalAt(p->expr, idx);
      if (addr < 0 || addr >= buf.numel())
        return store.emitError("furiosa: a store lands outside its buffer"), failure();
      if (covered[addr]++)
        return store.emitError("furiosa: two stores of the grid write the same element"), failure();
      ++count;
      for (unsigned j = 0; j < nd; ++j) {
        int64_t c = addr / st[j];
        addr %= st[j];
        sp.boxLo[j] = std::min(sp.boxLo[j], c);
        sp.boxHi[j] = std::max(sp.boxHi[j], c);
      }
    }
  }
  if (!count)
    return store.emitError("furiosa: the store never writes"), failure();
  int64_t volume = 1;
  for (unsigned j = 0; j < nd; ++j)
    volume *= sp.boxHi[j] - sp.boxLo[j] + 1;
  if (volume != count)
    return store.emitError("furiosa: the grid writes a non-rectangular region"), failure();
  sp.full = (count == buf.numel());
  sp.phase.assign(nd, 0);
  for (unsigned j = 0; j < nd; ++j) {
    if (!sp.axisExtent[j])
      continue;
    int64_t e = sp.axisExtent[j], ph = ((sp.origin[0][j] % e) + e) % e;
    for (auto &o : sp.origin)
      if (((o[j] % e) + e) % e != ph)
        return store.emitError("furiosa: store tiles are not aligned along view dim ") << j, failure();
    sp.phase[j] = ph;
  }
  return sp;
}

AffineExpr Raiser::originExpr(const Gen &g, unsigned j) {
  AffineExpr c = getAffineDimExpr(g.coordPos[j], ctx);
  if (!space->axisExtent[j])
    return c;
  return c - (c - space->phase[j]) % space->axisExtent[j];
}

/// One expression of the generic's dims for a per-instance symbolic value: substitute each axis of `v`
/// (coordinate minus that instance's origin, or a generic dim) and each loop counter; the instances
/// must then agree up to a constant that is an affine function of the store origins.
FailureOr<AffineExpr> Raiser::rewrite(Operation *at, const Gen &g, ArrayRef<Axis> env,
                                      std::function<AffineExpr(const Sym &)> select, Value v) {
  unsigned nd = g.extent.size();
  SmallVector<AffineExpr> per;
  for (auto [n, ev] : llvm::enumerate(evals)) {
    const Sym *s = ev->get(v);
    if (!s)
      return at->emitError("furiosa: value is not symbolic in every instance"), failure();
    SmallVector<AffineExpr> dims;
    for (const Axis &a : env) {
      if (a.kind == Axis::Coord) {
        if (g.coordPos[a.dim] < 0)
          return at->emitError("furiosa: internal: coordinate dim not in this generic"), failure();
        dims.push_back(getAffineDimExpr(g.coordPos[a.dim], ctx) - space->origin[n][a.dim]);
      } else if (a.kind == Axis::Iter) {
        dims.push_back(getAffineDimExpr(a.dim, ctx));
      } else {
        dims.push_back(getAffineConstantExpr(0, ctx));
      }
    }
    SmallVector<AffineExpr> syms;
    for (unsigned l = 0; l < loopTrip.size(); ++l) {
      auto it = g.loopPos.find(l);
      syms.push_back(it == g.loopPos.end() ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(it->second, ctx));
    }
    AffineExpr e = select(*s);
    for (unsigned l = 0; l < loopTrip.size(); ++l)
      if (e.isFunctionOfSymbol(l) && !g.loopPos.count(l))
        return at->emitError("furiosa: a loop counter is used outside its loop"), failure();
    per.push_back(simplifyAffineExpr(e.replaceDimsAndSymbols(dims, syms), nd, 0));
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

  // Fit consts[n] = beta + sum_j alpha_j * origin[n][j] (least squares, then checked exactly).
  unsigned nv = space->viewShape.size() + 1;
  std::vector<std::vector<double>> a(nv, std::vector<double>(nv, 0.0));
  std::vector<double> rhs(nv, 0.0);
  for (unsigned n = 0; n < evals.size(); ++n) {
    SmallVector<double> row;
    for (int64_t o : space->origin[n])
      row.push_back((double)o);
    row.push_back(1.0);
    for (unsigned i = 0; i < nv; ++i) {
      rhs[i] += row[i] * consts[n];
      for (unsigned j = 0; j < nv; ++j)
        a[i][j] += row[i] * row[j];
    }
  }
  std::vector<double> x(nv, 0.0);
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
      x[col] = rhs[pivot[col]] / a[pivot[col]][col];
  SmallVector<int64_t> alpha;
  for (unsigned j = 0; j + 1 < nv; ++j)
    alpha.push_back(std::llround(x[j]));
  int64_t beta = std::llround(x[nv - 1]);
  for (unsigned n = 0; n < evals.size(); ++n) {
    int64_t v = beta;
    for (unsigned j = 0; j + 1 < nv; ++j)
      v += alpha[j] * space->origin[n][j];
    if (v != consts[n])
      return at->emitError("furiosa: an address varies between program instances in a way the store "
                           "tiles do not explain"),
             failure();
  }
  AffineExpr out = rest0 + beta;
  for (unsigned j = 0; j + 1 < nv; ++j)
    if (alpha[j]) {
      if (g.coordPos[j] < 0)
        return at->emitError("furiosa: internal: origin of a coordinate dim not in this generic"), failure();
      out = out + originExpr(g, j) * alpha[j];
    }
  return simplifyAffineExpr(out, nd, 0);
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

Value Raiser::flatOf(int arg) {
  if (!flat.count(arg)) {
    OpBuilder::InsertionGuard guard(b);
    if (auto def = current[arg].getDefiningOp())
      b.setInsertionPointAfter(def);
    else
      b.setInsertionPointToStart(current[arg].getParentBlock());
    flat[arg] = reshape(current[arg], {buffers[arg].numel()});
  }
  return flat[arg];
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
    Value on = arith::ConstantIntOp::create(ob, loc, ob.getI1Type(), s0.never ? 0 : 1);
    for (unsigned c = 0; c < s0.conds.size(); ++c) {
      auto e = rewrite(at, g, env, [c](const Sym &s) { return s.conds[c]; }, v);
      if (failed(e))
        return failure();
      Value ok = arith::CmpIOp::create(ob, loc, arith::CmpIPredicate::sge, exprValue(g, *e),
                                          arith::ConstantIndexOp::create(ob, loc, 0));
      on = arith::AndIOp::create(ob, loc, on, ok);
    }
    return on;
  }
  if (s0.kind != Sym::Int)
    return at->emitError("furiosa: a pointer is used as data"), failure();
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

FailureOr<Value> Raiser::emitLoad(Gen &g, tt::LoadOp load, ArrayRef<Axis> env) {
  OpBuilder &ob = *g.ob;
  const Sym *p0 = evals[0]->get(load.getPtr());
  if (!p0 || p0->kind != Sym::Ptr)
    return load.emitError("furiosa: load address is not a pointer expression"), failure();
  int arg = p0->base;
  for (auto &ev : evals) {
    const Sym *p = ev->get(load.getPtr());
    if (!p || p->base != arg || p->data.size() != p0->data.size())
      return load.emitError("furiosa: load base differs between program instances"), failure();
  }
  auto addr = rewrite(load, g, env, [](const Sym &s) { return s.expr; }, load.getPtr());
  if (failed(addr))
    return failure();
  SmallVector<Interval> ranges;
  for (unsigned d = 0; d < g.extent.size(); ++d)
    ranges.push_back({g.lo[d], g.lo[d] + g.extent[d] - 1});

  Value on;
  if (load.getMask()) {
    const Sym *m0 = evals[0]->get(load.getMask());
    if (!m0 || m0->kind != Sym::Mask)
      return load.emitError("furiosa: load mask is not an index predicate"), failure();
    for (unsigned c = 0; c < m0->conds.size(); ++c) {
      auto e = rewrite(load, g, env, [c](const Sym &s) { return s.conds[c]; }, load.getMask());
      if (failed(e))
        return failure();
      if (bounds(*e, ranges).lo >= 0)
        continue; // the iteration domain already guarantees it
      Value ok = arith::CmpIOp::create(ob, loc, arith::CmpIPredicate::sge, exprValue(g, *e),
                                          arith::ConstantIndexOp::create(ob, loc, 0));
      on = on ? arith::AndIOp::create(ob, loc, on, ok).getResult() : ok;
    }
    if (m0->never)
      on = arith::ConstantIntOp::create(ob, loc, ob.getI1Type(), 0);
  }

  int64_t numel = buffers[arg].numel();
  Value i = exprValue(g, *addr);
  auto data = emitDataTerms(g, *p0, env);
  if (failed(data))
    return failure();
  Interval range = bounds(*addr, ranges);
  bool mayLeave = *data || range.lo < 0 || range.hi >= numel;
  if (*data)
    i = arith::AddIOp::create(ob, loc, i, *data);
  if (mayLeave) {
    if (!on && !*data)
      return load.emitError("furiosa: an unmasked load may read outside its buffer"), failure();
    i = arith::MaxSIOp::create(ob, loc, i, arith::ConstantIndexOp::create(ob, loc, 0));
    i = arith::MinSIOp::create(ob, loc, i, arith::ConstantIndexOp::create(ob, loc, numel - 1));
  }
  Value src = flatOf(arg);
  Value elem = tensor::ExtractOp::create(ob, loc, src, ValueRange{i});
  if (on) {
    Attribute otherAttr;
    if (Value other = load.getOther()) {
      auto c = other.getDefiningOp<arith::ConstantOp>();
      if (!c)
        return load.emitError("furiosa: load `other` is not a constant"), failure();
      otherAttr = c.getValue();
    }
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(ob.getInsertionBlock(), ob.getInsertionPoint());
    Value other = constantOf(elem.getType(), otherAttr);
    elem = arith::SelectOp::create(ob, loc, on, elem, other);
  }
  return elem;
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
  if (isa<tt::PreciseDivFOp, tt::PreciseSqrtOp, tt::FpToFpOp, tt::BitcastOp, tt::ClampFOp>(op))
    return true;
  Dialect *d = op->getDialect();
  return d && (isa<arith::ArithDialect>(d) || isa<math::MathDialect>(d)) && !isa<arith::ConstantOp>(op);
}

FailureOr<Value> Raiser::emit(Gen &g, Value v, ArrayRef<Axis> env) {
  auto memoKey = std::make_pair(v.getAsOpaquePointer(), key(env));
  if (auto it = g.memo.find(memoKey); it != g.memo.end())
    return it->second;
  FailureOr<Value> out = failure();
  Operation *def = v.getDefiningOp();
  if (const Sym *s = evals[0]->get(v)) {
    out = emitSym(g, v, *s, env);
  } else if (!def) {
    return kernel.emitError("furiosa: a loop-carried value is used outside an accumulation"), failure();
  } else if (auto c = dyn_cast<arith::ConstantOp>(def)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(g.ob->getInsertionBlock(), g.ob->getInsertionPoint());
    out = constantOf(getElementTypeOrSelf(c.getType()), c.getValue());
  } else if (auto s = dyn_cast<tt::SplatOp>(def)) {
    out = emit(g, s.getSrc(), {});
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
    out = emitLoad(g, load, env);
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
  auto attempt = [&](ArrayRef<unsigned> keep, bool &pruneable,
                     SmallVector<unsigned> &used) -> FailureOr<Value> {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(parent.anchor);
    // What the bodies below read gets built before this placeholder, ahead of both generics.
    Operation *anchor = arith::ConstantIndexOp::create(b, loc, 0);
    llvm::scope_exit dropAnchor([&] { anchor->erase(); });
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
      g.lo.push_back(0);
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

    SmallVector<int64_t> outShape;
    for (unsigned d = 0; d < np; ++d)
      outShape.push_back(g.extent[d]);
    // The init: a parallel generic over the kept dims (it may read other buffers).
    Gen gi = g;
    gi.extent.resize(np);
    gi.lo.resize(np);
    SmallVector<int> remapParallel = remap;
    Value empty = tensor::EmptyOp::create(b, loc, outShape, elem);
    SmallVector<AffineMap> imaps{AffineMap::getMultiDimIdentityMap(np, ctx)};
    SmallVector<utils::IteratorType> iters(np, utils::IteratorType::parallel);
    LogicalResult status = success();
    auto initOp = linalg::GenericOp::create(b, 
        loc, TypeRange{empty.getType()}, ValueRange{}, ValueRange{empty}, imaps, iters,
        [&](OpBuilder &ob, Location, ValueRange) {
          gi.ob = &ob;
          gi.index.assign(np, Value());
          auto v = init(gi, remapParallel);
          if (failed(v)) {
            status = failure();
            linalg::YieldOp::create(ob, loc, ValueRange{arith::ConstantOp::create(ob, loc, ob.getZeroAttr(elem))});
            return;
          }
          linalg::YieldOp::create(ob, loc, ValueRange{*v});
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
    maps.push_back(AffineMap::get(nd, 0, outExprs, ctx));
    SmallVector<utils::IteratorType> its(np, utils::IteratorType::parallel);
    its.append(nd - np, utils::IteratorType::reduction);
    Value result = initOp.getResult(0);
    if (!ownExtents.empty()) {
      auto op = linalg::GenericOp::create(b, 
          loc, TypeRange{result.getType()}, inputs, ValueRange{result}, maps, its,
          [&](OpBuilder &ob, Location, ValueRange args) {
            g.ob = &ob;
            g.index.assign(nd, Value());
            auto v = body(g, args.back(), remap);
            if (failed(v)) {
              status = failure();
              linalg::YieldOp::create(ob, loc, ValueRange{args.back()});
              return;
            }
            linalg::YieldOp::create(ob, loc, ValueRange{*v});
          });
      if (failed(status))
        return failure();
      result = op.getResult(0);
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
  for (Value v = *first; v;) {
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

FailureOr<Value> Raiser::materialize(Gen &g, Value v, ArrayRef<Axis> env) {
  Operation *def = v.getDefiningOp();
  Type elem = getElementTypeOrSelf(v.getType());
  SmallVector<unsigned> parentDims;
  for (unsigned d = 0; d < g.extent.size(); ++d)
    parentDims.push_back(d);
  // Remaps an env over the parent's dims into a generic built with `remap`.
  auto remapEnv = [](ArrayRef<Axis> e, ArrayRef<int> remap) {
    SmallVector<Axis> out(e.begin(), e.end());
    for (Axis &a : out)
      if (a.kind == Axis::Iter)
        a.dim = remap[a.dim];
    return out;
  };
  FailureOr<std::pair<Value, SmallVector<unsigned>>> built = failure();

  if (auto red = dyn_cast<tt::ReduceOp>(def)) {
    if (red.getNumOperands() != 1)
      return red.emitError("furiosa: multi-operand reductions are not raised yet"), failure();
    Region &combiner = red.getCombineOp();
    Operation *comb = &combiner.front().front();
    Attribute identity;
    Type et = elem;
    if (isa<arith::AddFOp>(comb))
      identity = b.getFloatAttr(et, 0.0);
    else if (isa<arith::MulFOp>(comb))
      identity = b.getFloatAttr(et, 1.0);
    else if (isa<arith::MaxNumFOp, arith::MaximumFOp>(comb))
      identity = b.getFloatAttr(et, -std::numeric_limits<double>::infinity());
    else if (isa<arith::MinNumFOp, arith::MinimumFOp>(comb))
      identity = b.getFloatAttr(et, std::numeric_limits<double>::infinity());
    else if (isa<arith::AddIOp>(comb))
      identity = b.getIntegerAttr(et, 0);
    else if (isa<arith::MaxSIOp>(comb))
      identity = b.getIntegerAttr(et, APInt::getSignedMinValue(et.getIntOrFloatBitWidth()));
    else if (isa<arith::MinSIOp>(comb))
      identity = b.getIntegerAttr(et, APInt::getSignedMaxValue(et.getIntOrFloatBitWidth()));
    else
      return red.emitError("furiosa: reduction combiner without a known identity"), failure();
    unsigned axis = red.getAxis();
    int64_t extent = shapeOf(red.getOperand(0).getType())[axis];
    built = buildGeneric(
        g, parentDims, {extent}, {}, elem,
        [&](Gen &gi, ArrayRef<int>) -> FailureOr<Value> { return arith::ConstantOp::create(*gi.ob, loc, cast<TypedAttr>(identity)).getResult(); },
        [&](Gen &gg, Value acc, ArrayRef<int> remap) -> FailureOr<Value> {
          SmallVector<Axis> in = remapEnv(env, remap);
          Axis r;
          r.kind = Axis::Iter;
          r.dim = gg.extent.size() - 1;
          in.insert(in.begin() + axis, r);
          auto x = emit(gg, red.getOperand(0), in);
          if (failed(x))
            return failure();
          IRMapping map;
          map.map(combiner.front().getArgument(0), acc);
          map.map(combiner.front().getArgument(1), *x);
          for (Operation &o : combiner.front().without_terminator())
            gg.ob->clone(o, map);
          return map.lookup(combiner.front().getTerminator()->getOperand(0));
        });
  } else if (auto dot = dyn_cast<tt::DotOp>(def)) {
    int64_t k = shapeOf(dot.getA().getType())[1];
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
    unsigned i = cast<OpResult>(v).getResultNumber();
    unsigned id = loopIds.lookup(loop);
    Value iter = loop.getRegionIterArgs()[i];
    Value next = cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(i);
    Operation *upd = next.getDefiningOp();
    if (!upd || !iter.hasOneUse() || iter.getUses().begin()->getOwner() != upd)
      return loop.emitError("furiosa: the loop carries state other than an accumulation"), failure();
    int64_t trip = loopTrip[id];
    if (auto d = dyn_cast<tt::DotOp>(upd); d && d.getC() == iter) {
      int64_t k = shapeOf(d.getA().getType())[1];
      built = buildGeneric(
          g, parentDims, {trip, k}, {{id, 0}}, elem,
          [&](Gen &gi, ArrayRef<int> remap) { return emit(gi, loop.getInitArgs()[i], remapEnv(env, remap)); },
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
            return arith::AddFOp::create(*gg.ob, loc, acc, arith::MulFOp::create(*gg.ob, loc, x, y)).getResult();
          });
    } else if (upd->getNumOperands() == 2 &&
               isa<arith::AddFOp, arith::MulFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp,
                   arith::MinimumFOp, arith::AddIOp, arith::MaxSIOp, arith::MinSIOp>(upd)) {
      Value term = upd->getOperand(0) == iter ? upd->getOperand(1) : upd->getOperand(0);
      built = buildGeneric(
          g, parentDims, {trip}, {{id, 0}}, elem,
          [&](Gen &gi, ArrayRef<int> remap) { return emit(gi, loop.getInitArgs()[i], remapEnv(env, remap)); },
          [&](Gen &gg, Value acc, ArrayRef<int> remap) -> FailureOr<Value> {
            auto x = emit(gg, term, remapEnv(env, remap));
            if (failed(x))
              return failure();
            OperationState state(loc, upd->getName());
            state.addOperands({acc, *x});
            state.addTypes(elem);
            state.addAttributes(upd->getAttrs());
            return gg.ob->create(state)->getResult(0);
          });
    } else {
      return loop.emitError("furiosa: the loop carries state other than an accumulation"), failure();
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

LogicalResult Raiser::raiseStore(tt::StoreOp store) {
  auto sp = storeSpace(store);
  if (failed(sp))
    return failure();
  space = &*sp;
  int arg = sp->arg;
  unsigned nd = sp->viewShape.size();
  SmallVector<Axis> env;
  for (int j : sp->axisDim) {
    Axis a;
    if (j >= 0) {
      a.kind = Axis::Coord;
      a.dim = j;
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
  SmallVector<unsigned> all;
  for (unsigned j = 0; j < nd; ++j)
    all.push_back(j);
  // The stored value, as a parallel generic over every view dim (no pruning: it fills the box).
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(anchor);
  SmallVector<int64_t> box(root.extent);
  Value empty = tensor::EmptyOp::create(b, loc, box, elem);
  LogicalResult status = success();
  Gen g = root;
  auto op = linalg::GenericOp::create(b, 
      loc, TypeRange{empty.getType()}, ValueRange{}, ValueRange{empty},
      ArrayRef<AffineMap>{AffineMap::getMultiDimIdentityMap(nd, ctx)},
      SmallVector<utils::IteratorType>(nd, utils::IteratorType::parallel), [&](OpBuilder &ob, Location, ValueRange) {
        g.ob = &ob;
        g.index.assign(nd, Value());
        auto v = emit(g, store.getValue(), env);
        if (failed(v)) {
          status = failure();
          linalg::YieldOp::create(ob, loc, ValueRange{arith::ConstantOp::create(ob, loc, ob.getZeroAttr(elem))});
          return;
        }
        linalg::YieldOp::create(ob, loc, ValueRange{*v});
      });
  g.anchor = op;
  if (failed(status))
    return failure();
  b.setInsertionPoint(anchor);
  Value result = op.getResult(0);
  Value view = reshape(current[arg], sp->viewShape);
  Value updated = sp->full ? result
                           : tensor::InsertSliceOp::create(b, loc, result, view,
                                                             getAsIndexOpFoldResult(ctx, sp->boxLo),
                                                             getAsIndexOpFoldResult(ctx, box),
                                                             getAsIndexOpFoldResult(ctx, SmallVector<int64_t>(nd, 1)))
                                 .getResult();
  current[arg] = reshape(updated, buffers[arg].shape);
  flat.erase(arg);
  anchor->erase();
  space = nullptr;
  return success();
}

LogicalResult Raiser::run() {
  if (failed(parseLaunch()) || failed(evaluate()))
    return failure();
  b.setInsertionPointToEnd(module.getBody());
  SmallVector<Type> argTypes;
  for (auto &[i, buf] : buffers)
    argTypes.push_back(RankedTensorType::get(buf.shape, buf.elem));
  // Stores, in program order, decide the results.
  SmallVector<tt::StoreOp> stores;
  for (Operation &op : kernel.getBody().front())
    if (auto s = dyn_cast<tt::StoreOp>(op))
      stores.push_back(s);
  bool nested = false;
  kernel.walk([&](tt::StoreOp s) { nested |= s->getParentOp() != kernel.getOperation(); });
  if (nested)
    return kernel.emitError("furiosa: stores inside loops or branches are not raised yet");
  kernel.walk([&](Operation *op) {
    if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op))
      nested = true;
  });
  if (nested)
    return kernel.emitError("furiosa: atomics are not raised");
  std::set<int> writtenSet;
  for (tt::StoreOp s : stores) {
    const Sym *p = evals[0]->get(s.getPtr());
    if (!p || p->kind != Sym::Ptr)
      return s.emitError("furiosa: store address is not a pointer expression");
    writtenSet.insert(p->base);
  }
  SmallVector<Type> resultTypes;
  for (int w : writtenSet)
    resultTypes.push_back(argTypes[w]);
  auto func = func::FuncOp::create(b, loc, kernel.getName(), b.getFunctionType(argTypes, resultTypes));
  Block *entry = func.addEntryBlock();
  b.setInsertionPointToStart(entry);
  for (auto [i, arg] : llvm::enumerate(entry->getArguments()))
    current[i] = arg;
  for (tt::StoreOp s : stores)
    if (failed(raiseStore(s))) {
      func.erase();
      return failure();
    }
  SmallVector<Value> results;
  SmallVector<int64_t> written;
  for (int w : writtenSet) {
    results.push_back(current[w]);
    written.push_back(w);
  }
  func::ReturnOp::create(b, loc, results);
  func->setAttr("furiosa.written", b.getI64ArrayAttr(written));
  if (failed(verify(func)))
    return kernel.emitError("furiosa: internal error: the raised function does not verify");
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

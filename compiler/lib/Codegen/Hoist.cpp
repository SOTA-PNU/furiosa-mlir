// Hoisting index-only computations: a value in a linalg.generic body that depends on nothing but the
// iteration indices (and constants) is the same for every launch, since every scalar of the launch is a
// constant. It is evaluated at compile time over the dims it depends on and read as a constant tensor
// instead: RoPE angle tables, causal masks, iotas. The NPU's engines then see data, and none of the integer
// and bit tricks such tables are built with. Values whose evaluation needs an op the interpreter below does
// not know stay in the body.

#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/TypeSwitch.h"

#include <bit>
#include <cmath>
#include <map>

using namespace mlir;

namespace {

constexpr int64_t kMaxElements = 1 << 20;

/// A scalar during evaluation: integers (and index) as int64, floats as double holding the value of their
/// type exactly (f32 arithmetic is done in float).
struct Scalar {
  int64_t i = 0;
  double f = 0;
};

bool isIndexOnly(Value v, Block &body, llvm::DenseMap<Value, bool> &memo) {
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  bool only = false;
  if (Operation *op = v.getDefiningOp()) {
    if (op->getBlock() != &body)
      only = op->hasTrait<OpTrait::ConstantLike>(); // a constant captured from outside the generic
    else if (isa<linalg::IndexOp>(op) || op->hasTrait<OpTrait::ConstantLike>())
      only = true;
    else if (op->getNumRegions() == 0 && op->getNumResults() >= 1 && isPure(op))
      only = llvm::all_of(op->getOperands(), [&](Value o) { return isIndexOnly(o, body, memo); });
  }
  return memo[v] = only;
}

double roundTo(Type type, double x) {
  if (type.isF32())
    return float(x);
  if (type.isBF16() || type.isF16()) {
    bool lost;
    llvm::APFloat a{float(x)};
    a.convert(type.isBF16() ? llvm::APFloat::BFloat() : llvm::APFloat::IEEEhalf(), llvm::APFloat::rmNearestTiesToEven,
              &lost);
    return a.convertToFloat();
  }
  return x;
}

int64_t wrap(Type type, int64_t x) {
  unsigned width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  if (width >= 64)
    return x;
  uint64_t mask = (uint64_t(1) << width) - 1;
  uint64_t u = uint64_t(x) & mask;
  if (width > 1 && (u >> (width - 1)))
    u |= ~mask; // sign-extend
  return int64_t(u);
}

/// Evaluates an index-only value at `indices` (one per loop dim); fails on an op it does not know.
class Interpreter {
public:
  Interpreter(Block &body) : body(body) {}

  FailureOr<Scalar> eval(Value v, ArrayRef<int64_t> indices) {
    values.clear();
    this->indices = indices;
    return get(v);
  }

private:
  Block &body;
  ArrayRef<int64_t> indices;
  llvm::DenseMap<Value, Scalar> values;

  FailureOr<Scalar> get(Value v) {
    if (auto it = values.find(v); it != values.end())
      return it->second;
    Operation *op = v.getDefiningOp();
    if (!op)
      return failure();
    SmallVector<Scalar> in;
    if (!op->hasTrait<OpTrait::ConstantLike>())
      for (Value o : op->getOperands()) {
        FailureOr<Scalar> x = get(o);
        if (failed(x))
          return failure();
        in.push_back(*x);
      }
    Type t = v.getType();
    auto ty = [&](unsigned i) { return op->getOperand(i).getType(); };
    auto F = [&](double x) {
      Scalar s;
      s.f = roundTo(t, x);
      return FailureOr<Scalar>(s);
    };
    auto I = [&](int64_t x) {
      Scalar s;
      s.i = wrap(t, x);
      return FailureOr<Scalar>(s);
    };
    auto f32 = [&](unsigned i) { return float(in[i].f); };
    bool single = t.isF32();
    FailureOr<Scalar> r =
        llvm::TypeSwitch<Operation *, FailureOr<Scalar>>(op)
            .Case([&](arith::ConstantOp c) -> FailureOr<Scalar> {
              if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
                return I(i.getValue().getSExtValue());
              if (auto f = dyn_cast<FloatAttr>(c.getValue()))
                return F(f.getValueAsDouble());
              return failure();
            })
            .Case([&](linalg::IndexOp x) { return I(indices[x.getDim()]); })
            .Case([&](arith::AddIOp) { return I(in[0].i + in[1].i); })
            .Case([&](arith::SubIOp) { return I(in[0].i - in[1].i); })
            .Case([&](arith::MulIOp) { return I(in[0].i * in[1].i); })
            .Case([&](arith::MulUIExtendedOp) -> FailureOr<Scalar> {
              // The 2w-bit unsigned product: the low w bits, or (the second result) the high ones.
              unsigned w = ty(0).getIntOrFloatBitWidth();
              if (w > 32)
                return failure();
              uint64_t mask = w == 32 ? 0xffffffffull : (1ull << w) - 1;
              uint64_t product = (uint64_t(in[0].i) & mask) * (uint64_t(in[1].i) & mask);
              return I(int64_t(cast<OpResult>(v).getResultNumber() ? product >> w : product));
            })
            .Case([&](arith::DivSIOp) -> FailureOr<Scalar> {
              if (!in[1].i)
                return failure();
              return I(in[0].i / in[1].i);
            })
            .Case([&](arith::RemSIOp) -> FailureOr<Scalar> {
              if (!in[1].i)
                return failure();
              return I(in[0].i % in[1].i);
            })
            .Case([&](arith::FloorDivSIOp) -> FailureOr<Scalar> {
              if (!in[1].i)
                return failure();
              int64_t q = in[0].i / in[1].i;
              if ((in[0].i % in[1].i) && ((in[0].i < 0) != (in[1].i < 0)))
                --q;
              return I(q);
            })
            .Case([&](arith::AndIOp) { return I(in[0].i & in[1].i); })
            .Case([&](arith::OrIOp) { return I(in[0].i | in[1].i); })
            .Case([&](arith::XOrIOp) { return I(in[0].i ^ in[1].i); })
            .Case([&](arith::ShLIOp) { return I(int64_t(uint64_t(in[0].i) << in[1].i)); })
            .Case([&](arith::ShRSIOp) { return I(in[0].i >> in[1].i); })
            .Case([&](arith::ShRUIOp) {
              unsigned w = ty(0).isIndex() ? 64 : ty(0).getIntOrFloatBitWidth();
              uint64_t bits = w >= 64 ? uint64_t(in[0].i) : uint64_t(in[0].i) & ((uint64_t(1) << w) - 1);
              return I(int64_t(bits >> in[1].i));
            })
            .Case([&](arith::MaxSIOp) { return I(std::max(in[0].i, in[1].i)); })
            .Case([&](arith::MinSIOp) { return I(std::min(in[0].i, in[1].i)); })
            .Case<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp>([&](Operation *) { return I(in[0].i); })
            .Case([&](arith::ExtUIOp) {
              unsigned w = ty(0).getIntOrFloatBitWidth();
              return I(w >= 64 ? in[0].i : int64_t(uint64_t(in[0].i) & ((uint64_t(1) << w) - 1)));
            })
            .Case([&](arith::SIToFPOp) { return F(double(in[0].i)); })
            .Case([&](arith::FPToSIOp) { return I(int64_t(std::trunc(in[0].f))); })
            .Case([&](arith::AddFOp) { return F(single ? double(f32(0) + f32(1)) : in[0].f + in[1].f); })
            .Case([&](arith::SubFOp) { return F(single ? double(f32(0) - f32(1)) : in[0].f - in[1].f); })
            .Case([&](arith::MulFOp) { return F(single ? double(f32(0) * f32(1)) : in[0].f * in[1].f); })
            .Case([&](arith::DivFOp) { return F(single ? double(f32(0) / f32(1)) : in[0].f / in[1].f); })
            .Case([&](arith::NegFOp) { return F(-in[0].f); })
            .Case<arith::MaximumFOp, arith::MaxNumFOp>([&](Operation *) { return F(std::fmax(in[0].f, in[1].f)); })
            .Case<arith::MinimumFOp, arith::MinNumFOp>([&](Operation *) { return F(std::fmin(in[0].f, in[1].f)); })
            .Case<arith::ExtFOp, arith::TruncFOp>([&](Operation *) { return F(in[0].f); })
            .Case([&](arith::BitcastOp) -> FailureOr<Scalar> {
              if (t.isF32() && ty(0).isInteger(32))
                return F(std::bit_cast<float>(uint32_t(in[0].i)));
              if (t.isInteger(32) && ty(0).isF32())
                return I(int64_t(std::bit_cast<int32_t>(float(in[0].f))));
              return failure();
            })
            .Case([&](arith::CmpIOp c) -> FailureOr<Scalar> {
              int64_t a = in[0].i, b = in[1].i;
              uint64_t ua = uint64_t(a), ub = uint64_t(b);
              switch (c.getPredicate()) {
              case arith::CmpIPredicate::eq: return I(a == b);
              case arith::CmpIPredicate::ne: return I(a != b);
              case arith::CmpIPredicate::slt: return I(a < b);
              case arith::CmpIPredicate::sle: return I(a <= b);
              case arith::CmpIPredicate::sgt: return I(a > b);
              case arith::CmpIPredicate::sge: return I(a >= b);
              case arith::CmpIPredicate::ult: return I(ua < ub);
              case arith::CmpIPredicate::ule: return I(ua <= ub);
              case arith::CmpIPredicate::ugt: return I(ua > ub);
              case arith::CmpIPredicate::uge: return I(ua >= ub);
              }
              return failure();
            })
            .Case([&](arith::CmpFOp c) -> FailureOr<Scalar> {
              double a = in[0].f, b = in[1].f;
              bool unordered = std::isnan(a) || std::isnan(b);
              switch (c.getPredicate()) {
              case arith::CmpFPredicate::OEQ: return I(!unordered && a == b);
              case arith::CmpFPredicate::ONE: return I(!unordered && a != b);
              case arith::CmpFPredicate::OLT: return I(!unordered && a < b);
              case arith::CmpFPredicate::OLE: return I(!unordered && a <= b);
              case arith::CmpFPredicate::OGT: return I(!unordered && a > b);
              case arith::CmpFPredicate::OGE: return I(!unordered && a >= b);
              case arith::CmpFPredicate::UEQ: return I(unordered || a == b);
              case arith::CmpFPredicate::UNE: return I(unordered || a != b);
              case arith::CmpFPredicate::ULT: return I(unordered || a < b);
              case arith::CmpFPredicate::ULE: return I(unordered || a <= b);
              case arith::CmpFPredicate::UGT: return I(unordered || a > b);
              case arith::CmpFPredicate::UGE: return I(unordered || a >= b);
              default: return failure();
              }
            })
            .Case([&](arith::SelectOp) { return in[0].i ? FailureOr<Scalar>(in[1]) : FailureOr<Scalar>(in[2]); })
            .Case([&](math::FloorOp) { return F(std::floor(in[0].f)); })
            .Case([&](math::CeilOp) { return F(std::ceil(in[0].f)); })
            .Case([&](math::RoundOp) { return F(std::round(in[0].f)); })
            .Case([&](math::TruncOp) { return F(std::trunc(in[0].f)); })
            .Case([&](math::AbsFOp) { return F(std::fabs(in[0].f)); })
            .Case([&](math::FmaOp) {
              return F(single ? double(std::fmaf(f32(0), f32(1), f32(2))) : std::fma(in[0].f, in[1].f, in[2].f));
            })
            .Case([&](math::SqrtOp) { return F(single ? double(std::sqrt(f32(0))) : std::sqrt(in[0].f)); })
            .Case([&](math::RsqrtOp) { return F(single ? double(1.0f / std::sqrt(f32(0))) : 1 / std::sqrt(in[0].f)); })
            .Case([&](math::ExpOp) { return F(single ? double(std::exp(f32(0))) : std::exp(in[0].f)); })
            .Case([&](math::Exp2Op) { return F(single ? double(std::exp2(f32(0))) : std::exp2(in[0].f)); })
            .Case([&](math::LogOp) { return F(single ? double(std::log(f32(0))) : std::log(in[0].f)); })
            .Case([&](math::Log2Op) { return F(single ? double(std::log2(f32(0))) : std::log2(in[0].f)); })
            .Case([&](math::SinOp) { return F(single ? double(std::sin(f32(0))) : std::sin(in[0].f)); })
            .Case([&](math::CosOp) { return F(single ? double(std::cos(f32(0))) : std::cos(in[0].f)); })
            .Case([&](math::TanhOp) { return F(single ? double(std::tanh(f32(0))) : std::tanh(in[0].f)); })
            .Default([](Operation *) { return failure(); });
    if (failed(r))
      return failure();
    return values[v] = *r;
  }
};

/// The loop dims an index-only value reads, in order.
SmallVector<unsigned> dimsOf(Value v, Block &body) {
  llvm::SmallSetVector<unsigned, 4> dims;
  SmallVector<Value> stack{v};
  llvm::DenseSet<Value> seen;
  while (!stack.empty()) {
    Value x = stack.pop_back_val();
    if (!seen.insert(x).second)
      continue;
    Operation *op = x.getDefiningOp();
    if (!op || op->getBlock() != &body)
      continue;
    if (auto index = dyn_cast<linalg::IndexOp>(op))
      dims.insert(index.getDim());
    llvm::append_range(stack, op->getOperands());
  }
  SmallVector<unsigned> sorted(dims.begin(), dims.end());
  llvm::sort(sorted);
  return sorted;
}

/// A select of data by an index-only condition against a constant becomes arithmetic with a table the
/// hoisting below evaluates: `select(p, x, -inf)` is x + (p ? 0 : -inf), a mask; `select(p, x, 0)` is
/// x * (p ? 1 : 0). Exact for finite x (the Vector Engine streams arithmetic; it has no per-element
/// condition from a table).
void selectsToArithmetic(Block &body) {
  llvm::DenseMap<Value, bool> memo;
  SmallVector<arith::SelectOp> selects(body.getOps<arith::SelectOp>());
  for (arith::SelectOp sel : selects) {
    if (!isa<FloatType>(sel.getType()) || !isIndexOnly(sel.getCondition(), body, memo))
      continue;
    for (int side = 0; side < 2; ++side) {
      Value x = side ? sel.getFalseValue() : sel.getTrueValue(), other = side ? sel.getTrueValue() : sel.getFalseValue();
      llvm::APFloat c(0.0);
      if (isIndexOnly(x, body, memo) || !matchPattern(other, m_ConstantFloat(&c)))
        continue;
      OpBuilder b(sel);
      Location loc = sel.getLoc();
      Type t = sel.getType();
      auto constant = [&](double v) { return arith::ConstantOp::create(b, loc, b.getFloatAttr(t, v)); };
      Value result;
      if (c.isInfinity()) {
        Value zero = constant(0.0), inf = arith::ConstantOp::create(b, loc, b.getFloatAttr(t, c));
        Value bias = side ? arith::SelectOp::create(b, loc, sel.getCondition(), inf, zero)
                          : arith::SelectOp::create(b, loc, sel.getCondition(), zero, inf);
        result = arith::AddFOp::create(b, loc, x, bias);
      } else if (c.isZero()) {
        Value one = constant(1.0), zero = constant(0.0);
        Value mask = side ? arith::SelectOp::create(b, loc, sel.getCondition(), zero, one)
                          : arith::SelectOp::create(b, loc, sel.getCondition(), one, zero);
        result = arith::MulFOp::create(b, loc, x, mask);
      } else {
        continue;
      }
      sel.replaceAllUsesWith(result);
      sel.erase();
      break;
    }
  }
}

/// Whether an index-typed value only computes data (it reaches no tensor address): an index value feeding
/// arithmetic that is cast back to an integer, like tl.arange(..) + x on i32s, or compared (with loaded bounds).
bool onlyData(Value v) {
  return llvm::all_of(v.getUsers(), [](Operation *user) {
    if (isa<arith::CmpIOp>(user))
      return true;
    if (isa<arith::IndexCastOp>(user))
      return !user->getResult(0).getType().isIndex() || onlyData(user->getResult(0));
    return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::MaxSIOp, arith::MinSIOp, arith::AndIOp>(user) &&
           onlyData(user->getResult(0));
  });
}

void eraseDead(Block &body) {
  for (bool erased = true; erased;) {
    erased = false;
    SmallVector<Operation *> ops;
    for (Operation &op : body.without_terminator())
      ops.push_back(&op);
    for (Operation *op : llvm::reverse(ops))
      if (op->use_empty() && isPure(op)) {
        op->erase();
        erased = true;
      }
  }
}

/// Hoists the index-only values of one generic; returns the generic that replaces it (or the same one).
linalg::GenericOp hoist(linalg::GenericOp generic) {
  Block &body = generic.getRegion().front();
  selectsToArithmetic(body);
  llvm::DenseMap<Value, bool> memo;
  // Frontier values: index-only, of a scalar element type, read by something that is not, and computed
  // (not a constant, not an index unless it only computes data: an index addresses).
  SmallVector<Value> frontier;
  for (Operation &op : body) {
    if (op.hasTrait<OpTrait::ConstantLike>() || isa<linalg::IndexOp>(op))
      continue;
    for (Value v : op.getResults()) {
      Type t = v.getType();
      if ((t.isIndex() && !onlyData(v)) || !t.isIntOrIndexOrFloat() || !isIndexOnly(v, body, memo))
        continue;
      if (llvm::any_of(v.getUsers(), [&](Operation *user) {
            return user->getNumResults() == 0 || !isIndexOnly(user->getResult(0), body, memo);
          }))
        frontier.push_back(v);
    }
  }
  if (frontier.empty())
    return generic;

  SmallVector<int64_t> ranges = generic.getStaticLoopRanges();
  if (llvm::any_of(ranges, ShapedType::isDynamic))
    return generic;
  OpBuilder b(generic);
  Interpreter interp(body);
  SmallVector<Value> constants;
  SmallVector<AffineMap> maps;
  SmallVector<Value> replaced;
  bool folds = false;
  for (Value v : frontier) {
    SmallVector<unsigned> dims = dimsOf(v, body);
    SmallVector<int64_t> shape;
    int64_t count = 1;
    for (unsigned d : dims) {
      shape.push_back(ranges[d]);
      count *= ranges[d];
    }
    if (count > kMaxElements)
      continue;
    Type t = v.getType();
    // A table holds bytes: a predicate is an i32 table compared with 0, an index an i32 table cast to index.
    bool widened = t.isInteger(1) || t.isIndex();
    Type tt = widened ? b.getI32Type() : t;
    SmallVector<Attribute> elements;
    SmallVector<int64_t> point(ranges.size(), 0);
    bool ok = true;
    for (int64_t flat = 0; flat < count && ok; ++flat) {
      int64_t rest = flat;
      for (int k = dims.size() - 1; k >= 0; --k) {
        point[dims[k]] = rest % ranges[dims[k]];
        rest /= ranges[dims[k]];
      }
      FailureOr<Scalar> x = interp.eval(v, point);
      if (failed(x)) {
        ok = false;
        break;
      }
      if (isa<FloatType>(t))
        elements.push_back(b.getFloatAttr(t, x->f));
      else if (widened && x->i != int32_t(x->i))
        ok = false;
      else
        elements.push_back(b.getIntegerAttr(tt, x->i));
    }
    if (!ok)
      continue;
    if (llvm::all_equal(elements)) {
      // No dim, or the same value at every point (a mask true over the whole range): an ordinary constant, and
      // what it makes trivial (x * 1, a select on it) folds.
      OpBuilder inner = OpBuilder::atBlockBegin(&body);
      Attribute c = widened ? b.getIntegerAttr(t, cast<IntegerAttr>(elements.front()).getInt()) : elements.front();
      Value k = arith::ConstantOp::create(inner, v.getLoc(), cast<TypedAttr>(c));
      v.replaceAllUsesWith(k);
      if (dims.empty())
        continue;
      for (Operation *user : llvm::to_vector(k.getUsers())) {
        SmallVector<OpFoldResult> folded;
        if (user->getNumResults() == 1 && !isIndexOnly(user->getResult(0), body, memo) &&
            succeeded(user->fold(folded)) && folded.size() == 1 && isa<Value>(folded[0])) {
          user->getResult(0).replaceAllUsesWith(cast<Value>(folded[0]));
          user->erase();
          folds = true;
        }
      }
      continue;
    }
    auto type = RankedTensorType::get(shape, tt);
    constants.push_back(arith::ConstantOp::create(b, v.getLoc(), DenseElementsAttr::get(type, elements)));
    SmallVector<AffineExpr> results;
    for (unsigned d : dims)
      results.push_back(b.getAffineDimExpr(d));
    maps.push_back(AffineMap::get(ranges.size(), 0, results, b.getContext()));
    replaced.push_back(v);
  }
  if (constants.empty()) {
    if (folds)
      eraseDead(body);
    return generic;
  }

  // The generic again, reading the tables as extra inputs.
  SmallVector<Value> inputs(generic.getDpsInputs());
  llvm::append_range(inputs, constants);
  SmallVector<AffineMap> allMaps;
  for (OpOperand *in : generic.getDpsInputOperands())
    allMaps.push_back(generic.getMatchingIndexingMap(in));
  llvm::append_range(allMaps, maps);
  for (OpOperand &out : generic.getDpsInitsMutable())
    allMaps.push_back(generic.getMatchingIndexingMap(&out));
  auto replacement = linalg::GenericOp::create(b, generic.getLoc(), generic.getResultTypes(), inputs,
                                               generic.getDpsInits(), allMaps, generic.getIteratorTypesArray());
  replacement->setDiscardableAttrs(generic->getDiscardableAttrDictionary());
  replacement.getRegion().takeBody(generic.getRegion());
  Block &moved = replacement.getRegion().front();
  unsigned at = generic.getNumDpsInputs();
  for (auto [k, v] : llvm::enumerate(replaced)) {
    Type tt = cast<RankedTensorType>(constants[k].getType()).getElementType();
    BlockArgument arg = moved.insertArgument(at + k, tt, v.getLoc());
    Value read = arg;
    OpBuilder inner = OpBuilder::atBlockBegin(&moved);
    if (v.getType().isInteger(1))
      read = arith::CmpIOp::create(inner, v.getLoc(), arith::CmpIPredicate::ne, arg,
                                   arith::ConstantOp::create(inner, v.getLoc(), inner.getI32IntegerAttr(0)));
    else if (v.getType().isIndex())
      read = arith::IndexCastOp::create(inner, v.getLoc(), v.getType(), arg);
    v.replaceAllUsesWith(read);
  }
  generic.replaceAllUsesWith(replacement.getResults());
  generic.erase();
  eraseDead(moved); // the computations that only fed the tables
  return replacement;
}

} // namespace

std::optional<std::pair<unsigned, SmallVector<bool>>> mlir::furiosa::truthOverOneDim(Value cond, Block &body,
                                                                                    ArrayRef<int64_t> ranges) {
  llvm::DenseMap<Value, bool> memo;
  if (!cond.getType().isInteger(1) || !isIndexOnly(cond, body, memo))
    return std::nullopt;
  SmallVector<unsigned> dims = dimsOf(cond, body);
  if (dims.size() != 1)
    return std::nullopt;
  Interpreter interp(body);
  SmallVector<int64_t> point(ranges.size(), 0);
  SmallVector<bool> truth;
  for (int64_t i = 0; i < ranges[dims[0]]; ++i) {
    point[dims[0]] = i;
    FailureOr<Scalar> x = interp.eval(cond, point);
    if (failed(x))
      return std::nullopt;
    truth.push_back(x->i != 0);
  }
  return std::make_pair(dims[0], truth);
}

void mlir::furiosa::hoistIndexOnly(ModuleOp module) {
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp g) { generics.push_back(g); });
  for (linalg::GenericOp g : generics)
    hoist(g);
}

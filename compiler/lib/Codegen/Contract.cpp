// Normalizing contractions for the op classes, which map one 2-D contraction y[m, n] = sum_k x[m, k] w[n, k]
// from a zero fill:
//  - a factor the body computes (softmax probabilities in attention) becomes its own generic over the dims it
//    depends on, read as an input;
//  - an accumulator that does not start from zero starts from zero, and the start is added after;
//  - batch dims (in every operand) are taken apart into one contraction per batch index, on slices;
//  - several m (or n, k) dims are collapsed into one, and a w laid out [k, n] is transposed first.

#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {

constexpr int64_t kMaxBatches = 64;

/// A generic computing acc + a * b over one reduction loop: the factors (extf stripped) in its body.
struct Contraction {
  Value a, b;
  OpOperand *extA = nullptr, *extB = nullptr; // the mulf operands (the extf results, or a / b themselves)
};

std::optional<Contraction> asContraction(linalg::GenericOp g) {
  if (g.getNumDpsInits() != 1 || g.getNumReductionLoops() != 1)
    return std::nullopt;
  Block &body = g.getRegion().front();
  Value acc = body.getArgument(g.getNumDpsInputs());
  auto add = cast<linalg::YieldOp>(body.getTerminator()).getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add)
    return std::nullopt;
  Value prod = add.getLhs() == acc ? add.getRhs() : add.getRhs() == acc ? add.getLhs() : Value();
  auto mul = prod ? prod.getDefiningOp<arith::MulFOp>() : arith::MulFOp();
  if (!mul || !mul->hasOneUse())
    return std::nullopt;
  Contraction c;
  c.extA = &mul->getOpOperand(0);
  c.extB = &mul->getOpOperand(1);
  auto strip = [](Value v) {
    if (auto ext = v.getDefiningOp<arith::ExtFOp>())
      return ext.getIn();
    return v;
  };
  c.a = strip(mul.getLhs());
  c.b = strip(mul.getRhs());
  // A square (x * x, say a sum of squares) is a reduction of one operand, not a contraction of two.
  auto read = [&](Value v) -> std::pair<Value, AffineMap> {
    auto arg = dyn_cast<BlockArgument>(v);
    if (!arg || arg.getOwner() != &body || arg.getArgNumber() >= g.getNumDpsInputs())
      return {v, AffineMap()};
    OpOperand *in = g.getDpsInputOperand(arg.getArgNumber());
    return {in->get(), g.getMatchingIndexingMap(in)};
  };
  if (read(c.a) == read(c.b))
    return std::nullopt;
  return c;
}

/// A copy of `g` reading `extra` as further inputs (through `maps`); returns it and its new block arguments.
std::pair<linalg::GenericOp, SmallVector<BlockArgument>> withInputs(linalg::GenericOp g, ArrayRef<Value> extra,
                                                                    ArrayRef<AffineMap> maps) {
  OpBuilder b(g);
  SmallVector<Value> inputs(g.getDpsInputs());
  llvm::append_range(inputs, extra);
  SmallVector<AffineMap> allMaps;
  for (OpOperand *in : g.getDpsInputOperands())
    allMaps.push_back(g.getMatchingIndexingMap(in));
  llvm::append_range(allMaps, maps);
  for (OpOperand &out : g.getDpsInitsMutable())
    allMaps.push_back(g.getMatchingIndexingMap(&out));
  auto replacement = linalg::GenericOp::create(b, g.getLoc(), g.getResultTypes(), inputs, g.getDpsInits(), allMaps,
                                               g.getIteratorTypesArray());
  replacement.getRegion().takeBody(g.getRegion());
  Block &body = replacement.getRegion().front();
  SmallVector<BlockArgument> args;
  for (auto [k, v] : llvm::enumerate(extra))
    args.push_back(body.insertArgument(g.getNumDpsInputs() + k, cast<RankedTensorType>(v.getType()).getElementType(),
                                       g.getLoc()));
  g.replaceAllUsesWith(replacement.getResults());
  g.erase();
  return {replacement, args};
}

void eraseDead(Block &body) {
  for (bool erased = true; erased;) {
    erased = false;
    SmallVector<Operation *> ops;
    for (Operation &op : body.without_terminator())
      ops.push_back(&op);
    for (Operation *op : llvm::reverse(ops))
      if (isOpTriviallyDead(op)) {
        op->erase();
        erased = true;
      }
  }
}

/// A factor the body computes becomes a generic over the dims it reads, an input of `g`.
linalg::GenericOp materializeFactor(linalg::GenericOp g, Value factor) {
  Block &body = g.getRegion().front();
  if (isa<BlockArgument>(factor))
    return g;
  // The body ops computing the factor, and the inputs and dims they read.
  llvm::SetVector<Operation *> slice;
  llvm::SmallSetVector<unsigned, 4> dims;
  llvm::SmallSetVector<unsigned, 4> inputs;
  BlockArgument acc = body.getArgument(g.getNumDpsInputs());
  SmallVector<Value> stack{factor};
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner() != &body)
        continue;
      if (arg == acc)
        return g;
      inputs.insert(arg.getArgNumber());
      for (AffineExpr e : g.getMatchingIndexingMap(g.getDpsInputOperand(arg.getArgNumber())).getResults()) {
        auto d = dyn_cast<AffineDimExpr>(e);
        if (!d && !isa<AffineConstantExpr>(e))
          return g;
        if (d)
          dims.insert(d.getPosition());
      }
      continue;
    }
    Operation *op = v.getDefiningOp();
    if (!op || op->getBlock() != &body || !slice.insert(op))
      continue;
    if (op->getNumRegions())
      return g;
    if (auto index = dyn_cast<linalg::IndexOp>(op))
      dims.insert(index.getDim());
    llvm::append_range(stack, op->getOperands());
  }
  SmallVector<unsigned> order(dims.begin(), dims.end());
  llvm::sort(order);
  if (order.size() == g.getNumLoops())
    return g; // it varies along every loop: nothing to take out
  auto position = [&](unsigned d) { return unsigned(llvm::find(order, d) - order.begin()); };
  SmallVector<int64_t> ranges = g.getStaticLoopRanges(), shape;
  for (unsigned d : order)
    shape.push_back(ranges[d]);
  MLIRContext *ctx = g.getContext();
  // The new generic: the inputs it reads, re-indexed over its own loops.
  OpBuilder b(g);
  SmallVector<Value> ins;
  SmallVector<AffineMap> maps;
  SmallVector<unsigned> inputOrder(inputs.begin(), inputs.end());
  llvm::sort(inputOrder);
  for (unsigned i : inputOrder) {
    AffineMap m = g.getMatchingIndexingMap(g.getDpsInputOperand(i));
    SmallVector<AffineExpr> results;
    for (AffineExpr e : m.getResults())
      results.push_back(isa<AffineDimExpr>(e) ? getAffineDimExpr(position(cast<AffineDimExpr>(e).getPosition()), ctx) : e);
    ins.push_back(g.getDpsInputs()[i]);
    maps.push_back(AffineMap::get(order.size(), 0, results, ctx));
  }
  Value init = tensor::EmptyOp::create(b, g.getLoc(), shape, factor.getType());
  maps.push_back(AffineMap::getMultiDimIdentityMap(order.size(), ctx));
  SmallVector<utils::IteratorType> iters(order.size(), utils::IteratorType::parallel);
  auto producer = linalg::GenericOp::create(
      b, g.getLoc(), TypeRange{init.getType()}, ins, ValueRange{init}, maps, iters,
      [&](OpBuilder &nb, Location loc, ValueRange args) {
        IRMapping map;
        for (auto [k, i] : llvm::enumerate(inputOrder))
          map.map(body.getArgument(i), args[k]);
        SmallVector<Operation *> ops(slice.begin(), slice.end());
        llvm::sort(ops, [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
        for (Operation *op : ops) {
          if (auto index = dyn_cast<linalg::IndexOp>(op)) {
            map.map(index.getResult(), linalg::IndexOp::create(nb, loc, position(index.getDim())).getResult());
            continue;
          }
          nb.clone(*op, map);
        }
        linalg::YieldOp::create(nb, loc, map.lookup(factor));
      });
  SmallVector<AffineExpr> read;
  for (unsigned d : order)
    read.push_back(getAffineDimExpr(d, ctx));
  auto [replacement, args] =
      withInputs(g, {producer.getResult(0)}, {AffineMap::get(g.getNumLoops(), 0, read, ctx)});
  factor.replaceAllUsesWith(Value(args.front()));
  eraseDead(replacement.getRegion().front());
  return replacement;
}

/// An accumulator that does not start from a zero fill starts from one; its start is added after.
linalg::GenericOp splitInit(linalg::GenericOp g) {
  Value init = g.getDpsInits()[0];
  if (auto fill = init.getDefiningOp<linalg::FillOp>(); fill && matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
    return g;
  auto type = cast<RankedTensorType>(init.getType());
  if (!type.getElementType().isF32())
    return g;
  OpBuilder b(g);
  Location loc = g.getLoc();
  Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0));
  Value empty = tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
  g.getDpsInitsMutable()[0].set(linalg::FillOp::create(b, loc, zero, empty).getResult(0));
  b.setInsertionPointAfter(g);
  AffineMap id = b.getMultiDimIdentityMap(type.getRank());
  SmallVector<utils::IteratorType> iters(type.getRank(), utils::IteratorType::parallel);
  Value out = tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
  auto sum = linalg::GenericOp::create(b, loc, TypeRange{type}, ValueRange{init, g.getResult(0)}, ValueRange{out},
                                       ArrayRef<AffineMap>{id, id, id}, iters,
                                       [&](OpBuilder &nb, Location l, ValueRange args) {
                                         linalg::YieldOp::create(nb, l, arith::AddFOp::create(nb, l, args[0], args[1]).getResult());
                                       });
  g.getResult(0).replaceAllUsesExcept(sum.getResult(0), sum);
  return g;
}

/// The loop dims that are exactly a result of every input's and the output's map: batch dims.
SmallVector<unsigned> batchDims(linalg::GenericOp g) {
  SmallVector<unsigned> batch;
  auto iters = g.getIteratorTypesArray();
  for (unsigned d = 0; d < g.getNumLoops(); ++d) {
    if (iters[d] != utils::IteratorType::parallel)
      continue;
    bool everywhere = llvm::all_of(g.getIndexingMapsArray(), [&](AffineMap m) {
      return llvm::any_of(m.getResults(), [&](AffineExpr e) { return e == getAffineDimExpr(d, g.getContext()); });
    });
    if (everywhere)
      batch.push_back(d);
  }
  return batch;
}

/// One contraction per batch index, on slices (unit dims kept, then collapsed away by collapse()).
void unbatch(linalg::GenericOp g) {
  SmallVector<unsigned> batch = batchDims(g);
  if (batch.empty())
    return;
  SmallVector<int64_t> ranges = g.getStaticLoopRanges();
  int64_t count = 1;
  for (unsigned d : batch)
    count *= ranges[d];
  if (count > kMaxBatches)
    return;
  OpBuilder b(g);
  Location loc = g.getLoc();
  MLIRContext *ctx = g.getContext();
  auto folds = [&](ArrayRef<int64_t> xs) {
    SmallVector<OpFoldResult> r;
    for (int64_t x : xs)
      r.push_back(b.getIndexAttr(x));
    return r;
  };
  // A slice of an operand at a batch point: size 1 along each batch dim its map reads.
  auto sliceAt = [&](Value v, AffineMap m, ArrayRef<int64_t> point) {
    auto t = cast<RankedTensorType>(v.getType());
    SmallVector<int64_t> offsets(t.getRank(), 0), sizes(t.getShape()), strides(t.getRank(), 1);
    for (auto [k, e] : llvm::enumerate(m.getResults()))
      if (auto d = dyn_cast<AffineDimExpr>(e))
        if (auto it = llvm::find(batch, d.getPosition()); it != batch.end()) {
          offsets[k] = point[it - batch.begin()];
          sizes[k] = 1;
        }
    return std::make_tuple(offsets, sizes, strides);
  };
  Value result = g.getDpsInits()[0];
  SmallVector<int64_t> point(batch.size(), 0);
  for (int64_t flat = 0; flat < count; ++flat) {
    int64_t rest = flat;
    for (int k = batch.size() - 1; k >= 0; --k) {
      point[k] = rest % ranges[batch[k]];
      rest /= ranges[batch[k]];
    }
    SmallVector<Value> operands;
    for (OpOperand &operand : g->getOpOperands()) {
      auto [offsets, sizes, strides] = sliceAt(operand.get(), g.getMatchingIndexingMap(&operand), point);
      // A slice of a fill is a smaller fill.
      if (auto fill = operand.get().getDefiningOp<linalg::FillOp>()) {
        Value empty = tensor::EmptyOp::create(b, loc, sizes, cast<RankedTensorType>(operand.get().getType()).getElementType());
        operands.push_back(linalg::FillOp::create(b, loc, fill.getInputs()[0], empty).getResult(0));
        continue;
      }
      operands.push_back(tensor::ExtractSliceOp::create(b, loc, operand.get(), folds(offsets), folds(sizes), folds(strides)));
    }
    // Same loops and maps; each batch dim now spans one index.
    auto sliced = linalg::GenericOp::create(b, loc, TypeRange{operands.back().getType()},
                                            ArrayRef<Value>(operands).drop_back(), ValueRange{operands.back()},
                                            g.getIndexingMapsArray(), g.getIteratorTypesArray());
    IRMapping map;
    g.getRegion().cloneInto(&sliced.getRegion(), map);
    // linalg.index of a batch dim is the batch index now.
    sliced.getRegion().front().walk([&](linalg::IndexOp index) {
      if (auto it = llvm::find(batch, index.getDim()); it != batch.end()) {
        OpBuilder ib(index);
        index.replaceAllUsesWith(arith::ConstantIndexOp::create(ib, index.getLoc(), point[it - batch.begin()]).getResult());
        index.erase();
      }
    });
    auto [offsets, sizes, strides] = sliceAt(result, g.getMatchingIndexingMap(g.getDpsInitOperand(0)), point);
    result = tensor::InsertSliceOp::create(b, loc, sliced.getResult(0), result, folds(offsets), folds(sizes), folds(strides));
  }
  (void)ctx;
  g.getResult(0).replaceAllUsesWith(result);
  g.erase();
}

/// The contraction as y[m, n] = sum_k x[m, k] w[n, k] on collapsed operands, when its dims group so.
void collapse(linalg::GenericOp g) {
  std::optional<Contraction> c = asContraction(g);
  if (!c || g.getNumDpsInputs() != 2)
    return;
  Block &body = g.getRegion().front();
  eraseDead(body);
  if (!body.getOps<linalg::IndexOp>().empty())
    return;
  auto argA = dyn_cast<BlockArgument>(c->a), argB = dyn_cast<BlockArgument>(c->b);
  if (!argA || !argB || argA.getOwner() != &body || argB.getOwner() != &body || argA == argB)
    return;
  auto iters = g.getIteratorTypesArray();
  SmallVector<int64_t> ranges = g.getStaticLoopRanges();
  AffineMap out = g.getMatchingIndexingMap(g.getDpsInitOperand(0));
  // A constant 0 into a unit dim of the operand rides along like a unit loop dim.
  auto dimsOf = [](AffineMap m, Value v) {
    ArrayRef<int64_t> shape = cast<RankedTensorType>(v.getType()).getShape();
    SmallVector<unsigned> ds;
    for (auto [i, e] : llvm::enumerate(m.getResults())) {
      if (auto c = dyn_cast<AffineConstantExpr>(e); c && c.getValue() == 0 && shape[i] == 1)
        continue;
      auto d = dyn_cast<AffineDimExpr>(e);
      if (!d)
        return SmallVector<unsigned>{~0u};
      ds.push_back(d.getPosition());
    }
    return ds;
  };
  // m: the dims only the output and operand 0 share; n: those of operand 1.
  unsigned xi = 0;
  SmallVector<unsigned> a = dimsOf(g.getIndexingMapsArray()[0], g.getDpsInputs()[0]),
                        w = dimsOf(g.getIndexingMapsArray()[1], g.getDpsInputs()[1]),
                        y = dimsOf(out, g.getDpsInits()[0]);
  if (llvm::is_contained(a, ~0u) || llvm::is_contained(w, ~0u) || llvm::is_contained(y, ~0u))
    return;
  SmallVector<unsigned> m, n, k;
  for (unsigned d = 0; d < g.getNumLoops(); ++d) {
    bool inA = llvm::is_contained(a, d), inW = llvm::is_contained(w, d), inY = llvm::is_contained(y, d);
    if (iters[d] == utils::IteratorType::reduction && inA && inW && !inY)
      k.push_back(d);
    else if (inA && inY && !inW)
      m.push_back(d);
    else if (inW && inY && !inA)
      n.push_back(d);
    else if (ranges[d] != 1)
      return; // a batch dim or a broadcast: not a plain contraction
  }
  // Unit dims anywhere ride along in whichever group holds them in each operand.
  auto ordered = [&](ArrayRef<unsigned> operand, ArrayRef<unsigned> first, ArrayRef<unsigned> second) {
    SmallVector<unsigned> live;
    for (unsigned d : operand)
      if (ranges[d] != 1)
        live.push_back(d);
    SmallVector<unsigned> want;
    for (unsigned d : first)
      if (ranges[d] != 1)
        want.push_back(d);
    for (unsigned d : second)
      if (ranges[d] != 1)
        want.push_back(d);
    return live == want;
  };
  if (m.empty() || n.empty() || k.empty())
    return;
  // x is the operand whose dims lead the output.
  if (!ordered(y, m, n)) {
    if (!ordered(y, n, m))
      return;
    std::swap(m, n);
    std::swap(a, w);
    xi = 1;
  }
  if (!ordered(a, m, k))
    return;
  bool transposeW = !ordered(w, n, k);
  if (transposeW && !ordered(w, k, n))
    return;
  // Already y[m, n] = sum_k x[m, k] w[n, k] over three loops: nothing to do.
  if (g.getNumLoops() == 3 && xi == 0 && !transposeW && m.size() == 1 && n.size() == 1 && k.size() == 1 &&
      g.getIndexingMapsArray() == SmallVector<AffineMap>{AffineMap::get(3, 0, {getAffineDimExpr(m[0], g.getContext()), getAffineDimExpr(k[0], g.getContext())}, g.getContext()),
                                                        AffineMap::get(3, 0, {getAffineDimExpr(n[0], g.getContext()), getAffineDimExpr(k[0], g.getContext())}, g.getContext()),
                                                        AffineMap::get(3, 0, {getAffineDimExpr(m[0], g.getContext()), getAffineDimExpr(n[0], g.getContext())}, g.getContext())} &&
      m[0] == 0 && n[0] == 1 && k[0] == 2)
    return;
  OpBuilder b(g);
  Location loc = g.getLoc();
  auto product = [&](ArrayRef<unsigned> ds) {
    int64_t p = 1;
    for (unsigned d : ds)
      p *= ranges[d];
    return p;
  };
  int64_t M = product(m), N = product(n), K = product(k);
  auto flat2 = [&](Value v, int64_t rows, int64_t cols) -> Value {
    auto t = cast<RankedTensorType>(v.getType());
    auto to = RankedTensorType::get({rows, cols}, t.getElementType());
    if (t == to)
      return v;
    SmallVector<ReassociationIndices> all{ReassociationIndices{}};
    for (int64_t d = 0; d < t.getRank(); ++d)
      all[0].push_back(d);
    Value flat = tensor::CollapseShapeOp::create(b, loc, RankedTensorType::get({t.getNumElements()}, t.getElementType()), v, all);
    return tensor::ExpandShapeOp::create(b, loc, to, flat, SmallVector<ReassociationIndices>{{0, 1}});
  };
  Value x = flat2(g.getDpsInputs()[xi], M, K);
  Value wv = g.getDpsInputs()[1 - xi];
  if (transposeW) {
    Value w2 = flat2(wv, K, N);
    auto wt = cast<RankedTensorType>(w2.getType());
    Value empty = tensor::EmptyOp::create(b, loc, ArrayRef<int64_t>{N, K}, wt.getElementType());
    wv = linalg::TransposeOp::create(b, loc, w2, empty, ArrayRef<int64_t>{1, 0}).getResult()[0];
  } else {
    wv = flat2(wv, N, K);
  }
  Value init = flat2(g.getDpsInits()[0], M, N);
  MLIRContext *ctx = g.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx), d2 = getAffineDimExpr(2, ctx);
  SmallVector<AffineMap> maps{AffineMap::get(3, 0, {d0, d2}, ctx), AffineMap::get(3, 0, {d1, d2}, ctx),
                              AffineMap::get(3, 0, {d0, d1}, ctx)};
  SmallVector<utils::IteratorType> iters3{utils::IteratorType::parallel, utils::IteratorType::parallel,
                                         utils::IteratorType::reduction};
  auto flatG = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ValueRange{x, wv}, ValueRange{init}, maps,
                                         iters3);
  IRMapping map;
  g.getRegion().cloneInto(&flatG.getRegion(), map);
  if (xi == 1) {
    // The body read (w, x); the new generic reads (x, w).
    Block &nb = flatG.getRegion().front();
    BlockArgument first = nb.getArgument(0), second = nb.getArgument(1);
    Value tmp = nb.insertArgument(2u, first.getType(), loc);
    first.replaceAllUsesWith(tmp);
    second.replaceAllUsesWith(first);
    tmp.replaceAllUsesWith(second);
    nb.eraseArgument(2);
  }
  auto outType = cast<RankedTensorType>(g.getResult(0).getType());
  SmallVector<ReassociationIndices> all{ReassociationIndices{0, 1}};
  Value back = tensor::CollapseShapeOp::create(
      b, loc, RankedTensorType::get({outType.getNumElements()}, outType.getElementType()), flatG.getResult(0), all);
  if (outType.getRank() != 1) {
    ReassociationIndices group;
    for (int64_t d = 0; d < outType.getRank(); ++d)
      group.push_back(d);
    back = tensor::ExpandShapeOp::create(b, loc, outType, back, SmallVector<ReassociationIndices>{group});
  }
  g.getResult(0).replaceAllUsesWith(back);
  g.erase();
}

/// An int8 contraction (i32 sums of int8 products: Triton's int8 tl.dot) runs on the f32 path where that is exact:
/// int8 values are exact in the bf16 the Contraction Engine reads, their products (at most 2^14) in f32, and a sum of
/// K of them while K * 2^14 stays within 2^24. The operands convert to f32 and the sums back to i32 in elementwise
/// generics of their own.
/// shortcut: K <= 1024 only; longer K needs the Contraction Engine's native int8 path (i8 operands, i32 sums).
void int8ContractionsAsFloat(ModuleOp module) {
  SmallVector<linalg::GenericOp> found;
  module.walk([&](linalg::GenericOp g) {
    if (g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1 || !g.getNumReductionLoops())
      return;
    auto fill = g.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    if (!fill || !matchPattern(fill.getInputs()[0], m_Zero()))
      return;
    int64_t k = 1;
    for (auto [it, n] : llvm::zip(g.getIteratorTypesArray(), g.getStaticLoopRanges()))
      if (it == utils::IteratorType::reduction)
        k *= n;
    Block &body = g.getRegion().front();
    auto add = dyn_cast_or_null<arith::AddIOp>(body.getTerminator()->getOperand(0).getDefiningOp());
    auto mul = add ? add.getOperand(1).getDefiningOp<arith::MulIOp>() : arith::MulIOp();
    auto extended = [&](Value v) {
      auto ext = v.getDefiningOp<arith::ExtSIOp>();
      return ext && ext.getIn().getType().isInteger(8) && isa<BlockArgument>(ext.getIn());
    };
    if (k <= 1024 && add && add.getOperand(0) == body.getArgument(2) && add.getType().isInteger(32) && mul &&
        extended(mul.getLhs()) && extended(mul.getRhs()) && body.getOperations().size() == 5)
      found.push_back(g);
  });
  for (linalg::GenericOp g : found) {
    OpBuilder b(g);
    Location loc = g.getLoc();
    Type f32 = b.getF32Type();
    auto elementwise = [&](Value in, Type to, function_ref<Value(OpBuilder &, Location, Value)> fn) {
      auto t = cast<RankedTensorType>(in.getType());
      Value empty = tensor::EmptyOp::create(b, loc, t.getShape(), to);
      AffineMap id = b.getMultiDimIdentityMap(t.getRank());
      return linalg::GenericOp::create(b, loc, TypeRange{t.clone(to)}, ValueRange{in}, ValueRange{empty},
                                       ArrayRef<AffineMap>{id, id},
                                       SmallVector<utils::IteratorType>(t.getRank(), utils::IteratorType::parallel),
                                       [&](OpBuilder &nb, Location l, ValueRange args) {
                                         linalg::YieldOp::create(nb, l, fn(nb, l, args[0]));
                                       })
          .getResult(0);
    };
    auto toFloat = [&](OpBuilder &nb, Location l, Value v) { return arith::SIToFPOp::create(nb, l, f32, v).getResult(); };
    Value x = elementwise(g.getDpsInputs()[0], f32, toFloat), w = elementwise(g.getDpsInputs()[1], f32, toFloat);
    auto outType = cast<RankedTensorType>(g.getResult(0).getType());
    Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0));
    Value acc = linalg::FillOp::create(b, loc, zero, tensor::EmptyOp::create(b, loc, outType.getShape(), f32).getResult())
                    .getResult(0);
    Value sums = linalg::GenericOp::create(b, loc, TypeRange{outType.clone(f32)}, ValueRange{x, w}, ValueRange{acc},
                                           g.getIndexingMapsArray(), g.getIteratorTypesArray(),
                                           [](OpBuilder &nb, Location l, ValueRange args) {
                                             Value m = arith::MulFOp::create(nb, l, args[0], args[1]);
                                             linalg::YieldOp::create(nb, l, arith::AddFOp::create(nb, l, args[2], m).getResult());
                                           })
                     .getResult(0);
    // The sums are integers: rounding them first makes the conversion exact however it rounds.
    Value back = elementwise(sums, outType.getElementType(), [&](OpBuilder &nb, Location l, Value v) {
      return arith::FPToSIOp::create(nb, l, outType.getElementType(), math::RoundOp::create(nb, l, v)).getResult();
    });
    g.getResult(0).replaceAllUsesWith(back);
    g.erase();
  }
}

} // namespace

void mlir::furiosa::normalizeContractions(ModuleOp module, function_ref<bool(linalg::GenericOp)> mapsAlone) {
  int8ContractionsAsFloat(module);
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp g) {
    if (asContraction(g))
      generics.push_back(g);
  });
  for (linalg::GenericOp g : generics) {
    std::optional<Contraction> c = asContraction(g);
    g = materializeFactor(g, c->a);
    c = asContraction(g);
    if (c)
      g = materializeFactor(g, c->b);
    g = splitInit(g);
    // A batched contraction some op class maps as a step of its own keeps its batch dims.
    if (!mapsAlone || !mapsAlone(g))
      unbatch(g);
  }
  // The inputs a materialized factor no longer reads go (upstream patterns), leaving x and w.
  RewritePatternSet patterns(module.getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
  (void)applyPatternsGreedily(module, std::move(patterns));
  generics.clear();
  module.walk([&](linalg::GenericOp g) {
    if (asContraction(g))
      generics.push_back(g);
  });
  for (linalg::GenericOp g : generics)
    collapse(g);
}

// Kept loops: an scf.for the raise left at a kernel's top level (a loop-carried state, such as an argmax's
// running maximum and its index) runs as a host launch sequence.
//
// Rolled: each generic of the body is lifted over the iterations (a leading loop j whose row j is the
// generic's value in iteration j, read back with a slice at the iteration), so furiosa-structure promotes the
// reads the induction variable addressed as it would any affine read; after the index-only values are hoisted,
// sliceKeptLoops tiles each lifted generic back to its iteration's slice (upstream TilingInterface), whose
// operands are slices of padded or hoisted tensors at the iteration. A body the lift does not cover unrolls
// into its iterations instead (upstream SCF utils).

#include "furiosa/Codegen/Codegen.h"
#include "furiosa/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

using namespace mlir;

// shortcut: a launch per iteration step, so trip counts are capped at kMaxTrips; device loops (fopt.for)
// would take longer loops.
static constexpr int64_t kMaxTrips = 64;

namespace {

/// The constant an index value is, if it is one.
std::optional<int64_t> constantIndex(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  return std::nullopt;
}

/// Whether every generic of the loop body has a fresh output (an empty or a fill of one), so it lifts.
bool liftable(scf::ForOp loop) {
  if (!constantIndex(loop.getLowerBound()) || !constantIndex(loop.getStep()))
    return false;
  bool any = false;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto g = dyn_cast<linalg::GenericOp>(op);
    if (!g) {
      if (op.getNumRegions() || llvm::any_of(op.getResultTypes(), [](Type t) { return isa<ShapedType>(t); }))
        return false; // only generics compute tensors in a lifted body
      continue;
    }
    any = true;
    Value init = g.getNumDpsInits() == 1 ? g.getDpsInits()[0] : Value();
    auto fill = init ? init.getDefiningOp<linalg::FillOp>() : linalg::FillOp();
    if (!init || !(init.getDefiningOp<tensor::EmptyOp>() || (fill && fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())))
      return false;
  }
  return any;
}

/// Lifts `g` over the loop's `trips` iterations: a leading loop j whose row j is g's value in iteration j, the
/// induction variable lb + step * j, and replaces g by the row of the current iteration.
void lift(linalg::GenericOp g, scf::ForOp loop, int64_t trips) {
  OpBuilder b(g);
  Location loc = g.getLoc();
  MLIRContext *ctx = g.getContext();
  int64_t lb = *constantIndex(loop.getLowerBound()), step = *constantIndex(loop.getStep());
  Value init = g.getDpsInits()[0];
  auto t = cast<RankedTensorType>(init.getType());
  SmallVector<int64_t> shape{trips};
  llvm::append_range(shape, t.getShape());
  Value fresh = tensor::EmptyOp::create(b, loc, shape, t.getElementType());
  if (auto fill = init.getDefiningOp<linalg::FillOp>())
    fresh = linalg::FillOp::create(b, loc, fill.getInputs()[0], fresh).getResult(0);
  SmallVector<AffineMap> maps;
  for (OpOperand *in : g.getDpsInputOperands())
    maps.push_back(g.getMatchingIndexingMap(in).shiftDims(1));
  AffineMap out = g.getMatchingIndexingMap(g.getDpsInitOperand(0)).shiftDims(1);
  SmallVector<AffineExpr> outResults{getAffineDimExpr(0, ctx)};
  llvm::append_range(outResults, out.getResults());
  maps.push_back(AffineMap::get(out.getNumDims(), 0, outResults, ctx));
  SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel};
  llvm::append_range(iters, g.getIteratorTypesArray());
  auto lifted = linalg::GenericOp::create(b, loc, TypeRange{fresh.getType()}, g.getDpsInputs(), ValueRange{fresh},
                                          maps, iters);
  IRMapping map;
  g.getRegion().cloneInto(&lifted.getRegion(), map);
  Block &body = lifted.getRegion().front();
  SmallVector<linalg::IndexOp> indices(body.getOps<linalg::IndexOp>());
  for (linalg::IndexOp index : indices) {
    OpBuilder ib(index);
    index.replaceAllUsesWith(linalg::IndexOp::create(ib, index.getLoc(), index.getDim() + 1).getResult());
    index.erase();
  }
  OpBuilder bb = OpBuilder::atBlockBegin(&body);
  Value j = linalg::IndexOp::create(bb, loc, 0);
  Value iv = arith::AddIOp::create(bb, loc, arith::ConstantIndexOp::create(bb, loc, lb),
                                   arith::MulIOp::create(bb, loc, j, arith::ConstantIndexOp::create(bb, loc, step)));
  loop.getInductionVar().replaceUsesWithIf(iv, [&](OpOperand &use) { return lifted->isAncestor(use.getOwner()); });
  // The row of this iteration: j = (iv - lb) / step.
  Value k = arith::DivUIOp::create(
      b, loc, arith::SubIOp::create(b, loc, loop.getInductionVar(), arith::ConstantIndexOp::create(b, loc, lb)),
      arith::ConstantIndexOp::create(b, loc, step));
  SmallVector<OpFoldResult> offsets{k}, sizes{b.getIndexAttr(1)}, strides(shape.size(), b.getIndexAttr(1));
  for (int64_t d : t.getShape()) {
    offsets.push_back(b.getIndexAttr(0));
    sizes.push_back(b.getIndexAttr(d));
  }
  Value row = tensor::ExtractSliceOp::create(b, loc, t, lifted.getResult(0), offsets, sizes, strides);
  g.getResult(0).replaceAllUsesWith(row);
  g.erase();
}

} // namespace

LogicalResult mlir::furiosa::unrollKeptLoops(ModuleOp module) {
  SmallVector<scf::ForOp> loops;
  for (auto func : module.getOps<func::FuncOp>())
    for (auto loop : func.getBody().front().getOps<scf::ForOp>())
      if (std::optional<APInt> trips = loop.getStaticTripCount(); trips && trips->getSExtValue() <= kMaxTrips)
        loops.push_back(loop);
  if (loops.empty())
    return success();
  for (scf::ForOp loop : loops) {
    int64_t trips = loop.getStaticTripCount()->getSExtValue();
    if (liftable(loop)) {
      SmallVector<linalg::GenericOp> generics(loop.getBody()->getOps<linalg::GenericOp>());
      for (linalg::GenericOp g : generics)
        lift(g, loop, trips);
    } else if (failed(loopUnrollFull(loop))) {
      return loop.emitError("cannot unroll a kept loop");
    }
  }
  PassManager pm(module.getContext());
  pm.addPass(createStructurePass());
  return pm.run(module);
}

void mlir::furiosa::sliceKeptLoops(ModuleOp module) {
  SmallVector<tensor::ExtractSliceOp> slices;
  module.walk([&](scf::ForOp loop) {
    for (auto slice : loop.getBody()->getOps<tensor::ExtractSliceOp>())
      if (slice.getSource().getDefiningOp<linalg::GenericOp>())
        slices.push_back(slice);
  });
  for (tensor::ExtractSliceOp slice : slices) {
    auto lifted = slice.getSource().getDefiningOp<linalg::GenericOp>();
    OpBuilder b(slice);
    FailureOr<TilingResult> tiled =
        tensor::replaceExtractSliceWithTiledProducer(b, slice, cast<OpResult>(slice.getSource()));
    if (failed(tiled))
      continue;
    Value v = tiled->tiledValues[0];
    if (v.getType() != slice.getType())
      v = reshapeTo(b, slice.getLoc(), v, slice.getType());
    slice.replaceAllUsesWith(v);
    slice.erase();
    if (lifted->use_empty())
      lifted.erase();
  }
  // The unit dims the tiling left go, slices of empties fold (upstream patterns), and what does not change per
  // iteration (padding, reshapes, tables) leaves the loop; a padded source only boxed becomes one insert.
  module.walk([&](scf::ForOp loop) {
    RewritePatternSet patterns(module.getContext());
    linalg::ControlDropUnitDims options;
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    SmallVector<Operation *> ops;
    loop.getBody()->walk([&](Operation *op) { ops.push_back(op); });
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    (void)applyOpPatternsGreedily(ops, std::move(patterns), config);
    // A whole, rank-reducing slice of a value's unit-dim expansion is the value.
    SmallVector<tensor::ExtractSliceOp> whole;
    loop.getBody()->walk([&](tensor::ExtractSliceOp s) {
      auto expand = s.getSource().getDefiningOp<tensor::ExpandShapeOp>();
      if (expand && expand.getSrc().getType() == s.getType() &&
          s.getSourceType().getNumElements() == s.getType().getNumElements())
        whole.push_back(s);
    });
    for (tensor::ExtractSliceOp s : whole) {
      s.replaceAllUsesWith(s.getSource().getDefiningOp<tensor::ExpandShapeOp>().getSrc());
      s.erase();
    }
    moveLoopInvariantCode(cast<LoopLikeOpInterface>(loop.getOperation()));
  });
  padsToInserts(module);
}

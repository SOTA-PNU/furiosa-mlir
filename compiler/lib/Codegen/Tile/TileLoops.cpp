// Reading furiosa-tile's loops back (TileLoops.h).

#include "TileLoops.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;
using namespace mlir::furiosa::tile;

namespace {

/// A static loop 0 to trips * step by step with one iteration argument.
std::optional<TileLoop> staticLoop(scf::ForOp loop) {
  std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound()),
                         ub = getConstantIntValue(loop.getUpperBound()), step = getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step || *lb != 0 || *step <= 0 || *ub % *step || loop.getNumRegionIterArgs() != 1)
    return std::nullopt;
  TileLoop t;
  t.op = loop;
  t.iv = loop.getInductionVar();
  t.step = *step;
  t.trips = *ub / *step;
  t.iterArg = loop.getRegionIterArgs()[0];
  t.init = loop.getInitArgs()[0];
  t.yield = loop.getBody()->getTerminator()->getOperand(0);
  if (auto d = loop->getAttrOfType<IntegerAttr>("furiosa.tile_dim"))
    t.dim = d.getInt();
  return t;
}

} // namespace

FailureOr<AffineExpr> mlir::furiosa::tile::offsetExpr(OpFoldResult index, ArrayRef<Value> ivs, MLIRContext *ctx) {
  if (std::optional<int64_t> c = getConstantIntValue(index))
    return getAffineConstantExpr(*c, ctx);
  Value v = cast<Value>(index);
  for (auto [i, iv] : llvm::enumerate(ivs))
    if (v == iv)
      return getAffineDimExpr(i, ctx);
  if (auto apply = v.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<AffineExpr> operands;
    for (Value o : apply.getMapOperands()) {
      auto e = offsetExpr(o, ivs, ctx);
      if (failed(e))
        return failure();
      operands.push_back(*e);
    }
    AffineMap m = apply.getAffineMap();
    return m.getResult(0).replaceDimsAndSymbols(ArrayRef(operands).take_front(m.getNumDims()),
                                                ArrayRef(operands).drop_front(m.getNumDims()));
  }
  return failure();
}

std::optional<TileLoop> mlir::furiosa::tile::kLoop(Value v) {
  auto loop = v.getDefiningOp<scf::ForOp>();
  if (!loop || !loop->hasAttr("furiosa.tile_dim"))
    return std::nullopt;
  return staticLoop(loop);
}

FailureOr<std::optional<TileNest>> mlir::furiosa::tile::tileNest(Value stored, std::string &why) {
  auto outer = stored.getDefiningOp<scf::ForOp>();
  if (!outer || outer->hasAttr("furiosa.tile_dim"))
    return std::optional<TileNest>();
  TileNest nest;
  for (scf::ForOp loop = outer; loop;) {
    std::optional<TileLoop> t = staticLoop(loop);
    if (!t || (!nest.loops.empty() && t->init != nest.loops.back().iterArg)) {
      why = "a tile loop that is not a static nest over its stored value";
      return failure();
    }
    nest.loops.push_back(*t);
    loop = t->yield.getDefiningOp<scf::ForOp>();
    if (!loop) {
      nest.insert = t->yield.getDefiningOp<tensor::InsertSliceOp>();
      if (!nest.insert || nest.insert.getDest() != t->iterArg) {
        why = "a tile loop that does not insert a tile of its stored value";
        return failure();
      }
    }
  }
  tensor::InsertSliceOp insert = nest.insert;
  if (llvm::is_contained(insert.getStaticSizes(), ShapedType::kDynamic) ||
      llvm::any_of(insert.getStaticStrides(), [](int64_t x) { return x != 1; }) ||
      insert.getSourceType().getRank() != insert.getDestType().getRank()) {
    why = "a tile insert that is not a static box at the tile loops' offsets";
    return failure();
  }
  SmallVector<Value> ivs;
  for (const TileLoop &l : nest.loops)
    ivs.push_back(l.iv);
  auto shape = insert.getDestType().getShape();
  nest.loopOf.resize(shape.size());
  for (auto [j, offset] : llvm::enumerate(insert.getMixedOffsets())) {
    auto e = offsetExpr(offset, ivs, stored.getContext());
    if (failed(e)) {
      why = "a tile insert offset that is not affine in the tile loops";
      return failure();
    }
    nest.offsets.push_back(*e);
    int64_t size = insert.getStaticSizes()[j];
    if (auto c = dyn_cast<AffineConstantExpr>(*e)) {
      if (c.getValue() != 0 || size != shape[j]) {
        why = "a tile insert at a fixed offset";
        return failure();
      }
      continue;
    }
    auto d = dyn_cast<AffineDimExpr>(*e);
    const TileLoop *l = d ? &nest.loops[d.getPosition()] : nullptr;
    if (!l || size != l->step || l->step * l->trips != shape[j]) {
      why = "a tile insert whose offset is not a whole tile of a loop";
      return failure();
    }
    nest.loopOf[j] = d.getPosition();
  }
  nest.tile = insert.getSource();
  return std::optional<TileNest>(nest);
}

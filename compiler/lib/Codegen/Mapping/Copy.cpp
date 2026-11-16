// Op class: a copy of one buffer into a box of another, `tensor.insert_slice %src into %dst` with static
// offsets and unit strides, written in place: an HBM-to-HBM DMA from the source into the destination's tile
// view (one tile per dimension the box does not span). HBM-to-HBM moves have no alignment rule.

#include "Mapping.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;
using namespace mlir::furiosa::mapping;

namespace {

tensor::InsertSliceOp find(RaisedKernel &kernel) {
  func::FuncOp func = kernel.func;
  Block &block = func.getBody().front();
  auto ret = cast<func::ReturnOp>(block.getTerminator());
  if (kernel.written.size() != 1 || ret.getNumOperands() != 1 || block.getOperations().size() != 2)
    return {};
  auto insert = ret.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  if (!insert)
    return {};
  auto isArg = [&](Value v) { return isa<BlockArgument>(v) && cast<BlockArgument>(v).getOwner() == &block; };
  if (!isArg(insert.getSource()) || !isArg(insert.getDest()) ||
      cast<BlockArgument>(insert.getDest()).getArgNumber() != kernel.written[0] ||
      insert.getSourceType().getRank() != insert.getDestType().getRank() || !insert.hasUnitStride() ||
      ShapedType::isDynamicShape(insert.getStaticOffsets()) || ShapedType::isDynamicShape(insert.getStaticSizes()))
    return {};
  return insert;
}

bool match(RaisedKernel &kernel) { return bool(find(kernel)); }

LogicalResult map(RaisedKernel &kernel) {
  tensor::InsertSliceOp insert = find(kernel);
  if (!insert)
    return kernel.func.emitError("copy: not an insert_slice of a buffer into a buffer");
  KernelBuilder kb(kernel);
  MLIRContext *ctx = kb.getContext();
  RankedTensorType dstType = insert.getDestType();
  Type et = dstType.getElementType();
  // One axis per destination dimension; the source spans each one's box, resized.
  SmallVector<MapAttr> dstAxes, srcAxes;
  for (int64_t d = 0; d < dstType.getRank(); ++d) {
    MapAttr axis = kb.axis("D" + std::to_string(d), dstType.getDimSize(d));
    int64_t size = insert.getStaticSizes()[d];
    dstAxes.push_back(axis);
    srcAxes.push_back(size == dstType.getDimSize(d) ? axis : axis.resize(size));
  }
  MapAttr srcMap = MapAttr::pair(ctx, srcAxes), dstMap = MapAttr::pair(ctx, dstAxes);
  unsigned src = kb.input(cast<BlockArgument>(insert.getSource()).getArgNumber(), et, srcMap);
  unsigned dst = kb.inout(kernel.written[0], et, dstMap);
  kb.beginBody();
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  Value from = ViewOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), srcMap, false), kb.param(src));
  Value to = ViewOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), dstMap, true), kb.param(dst));
  SmallVector<MapAttr> tiled = dstAxes;
  for (int64_t d = 0; d < dstType.getRank(); ++d) {
    int64_t size = insert.getStaticSizes()[d];
    if (size == dstType.getDimSize(d))
      continue;
    tiled[d] = dstAxes[d].resize(size).padTo(dstType.getDimSize(d), PadKind::Bottom);
    to = TileOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), MapAttr::pair(ctx, tiled), true), to,
                        dstAxes[d], b.getI64IntegerAttr(size), b.getI64IntegerAttr(insert.getStaticOffsets()[d]));
  }
  HbmToHbmViewOp::create(b, loc, from, to);
  kb.finish({});
  return success();
}

} // namespace

OpClass mlir::furiosa::mapping::copyClass() { return {"copy", match, map}; }

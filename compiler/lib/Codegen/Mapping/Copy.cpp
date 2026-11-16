// Op class: a box copy between buffers, as one HBM-to-HBM DMA (no alignment rule). Either
// `tensor.insert_slice %src into %dst`, written in place into the destination's tile view, or
// `tensor.extract_slice %src`, the source's tile view copied whole into the written buffer. Static offsets,
// unit strides; reshapes around the buffers move no data, so the box is described in the slice's own shape.
// An insert into a zero fill (a padded read) writes the box into the fresh written buffer, which starts as
// zeros.

#include "Mapping.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;
using namespace mlir::furiosa::mapping;

namespace {

/// The function argument a value is (through reshapes), or null.
BlockArgument argOf(Value v, Block &block) {
  while (v.getDefiningOp() && isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(v.getDefiningOp()))
    v = v.getDefiningOp()->getOperand(0);
  auto arg = dyn_cast<BlockArgument>(v);
  return arg && arg.getOwner() == &block ? arg : BlockArgument();
}

struct Copy {
  OffsetSizeAndStrideOpInterface slice;
  RankedTensorType whole; // the sliced (inserted-into or extracted-from) tensor, in the slice's shape
  BlockArgument from, to; // source and destination buffers
  bool insert;
};

FailureOr<Copy> find(RaisedKernel &kernel) {
  func::FuncOp func = kernel.func;
  Block &block = func.getBody().front();
  auto ret = cast<func::ReturnOp>(block.getTerminator());
  if (kernel.written.size() != 1 || ret.getNumOperands() != 1)
    return failure();
  Copy c;
  Value result = ret.getOperand(0);
  while (result.getDefiningOp() && isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(result.getDefiningOp()))
    result = result.getDefiningOp()->getOperand(0);
  if (auto insert = result.getDefiningOp<tensor::InsertSliceOp>()) {
    c = {insert, insert.getDestType(), argOf(insert.getSource(), block), argOf(insert.getDest(), block), true};
    // shortcut: the zero fill relies on the crate allocating a temporary as zeros when a step first writes
    // it in place; a non-zero fill needs a memset.
    auto fill = insert.getDest().getDefiningOp<linalg::FillOp>();
    if (!c.to && fill && matchPattern(fill.getInputs()[0], m_AnyZeroFloat()) &&
        fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
      c.to = func.getArgument(kernel.written[0]);
    if (insert.getSourceType().getRank() != insert.getDestType().getRank())
      return failure();
  } else if (auto extract = result.getDefiningOp<tensor::ExtractSliceOp>()) {
    c = {extract, extract.getSourceType(), argOf(extract.getSource(), block), func.getArgument(kernel.written[0]),
         false};
    if (extract.getSourceType().getRank() != extract.getType().getRank())
      return failure();
  } else {
    return failure();
  }
  if (!c.from || !c.to || c.to.getArgNumber() != kernel.written[0] || c.from == c.to ||
      llvm::any_of(c.slice.getStaticStrides(), [](int64_t s) { return s != 1; }) ||
      ShapedType::isDynamicShape(c.slice.getStaticOffsets()) || ShapedType::isDynamicShape(c.slice.getStaticSizes()))
    return failure();
  // Nothing but the copy and reshapes.
  for (Operation &op : block.without_terminator())
    if (&op != c.slice.getOperation() &&
        !isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::EmptyOp, linalg::FillOp, arith::ConstantOp>(op))
      return failure();
  return c;
}

bool match(RaisedKernel &kernel) { return succeeded(find(kernel)); }

LogicalResult map(RaisedKernel &kernel) {
  FailureOr<Copy> found = find(kernel);
  if (failed(found))
    return kernel.func.emitError("copy: not a box copy between buffers");
  Copy c = *found;
  KernelBuilder kb(kernel);
  MLIRContext *ctx = kb.getContext();
  Type et = c.whole.getElementType();
  // One axis per dimension of the sliced tensor; the box resizes the ones it does not span.
  SmallVector<MapAttr> wholeAxes, boxAxes, tiled;
  for (int64_t d = 0; d < c.whole.getRank(); ++d) {
    MapAttr axis = kb.axis("D" + std::to_string(d), c.whole.getDimSize(d));
    int64_t size = c.slice.getStaticSizes()[d];
    wholeAxes.push_back(axis);
    boxAxes.push_back(size == c.whole.getDimSize(d) ? axis : axis.resize(size));
  }
  MapAttr wholeMap = MapAttr::pair(ctx, wholeAxes), boxMap = MapAttr::pair(ctx, boxAxes);
  // insert: the source is the box, the destination the whole; extract: the other way round.
  MapAttr fromMap = c.insert ? boxMap : wholeMap, toMap = c.insert ? wholeMap : boxMap;
  unsigned from = kb.input(c.from.getArgNumber(), et, fromMap);
  unsigned to = kb.inout(c.to.getArgNumber(), et, toMap);
  kb.beginBody();
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  Value src = ViewOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), fromMap, false), kb.param(from));
  Value dst = ViewOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), toMap, true), kb.param(to));
  // The whole side is tiled down to the box, one dimension at a time.
  Value &side = c.insert ? dst : src;
  tiled = wholeAxes;
  for (int64_t d = 0; d < c.whole.getRank(); ++d) {
    int64_t size = c.slice.getStaticSizes()[d], full = c.whole.getDimSize(d);
    if (size == full)
      continue;
    tiled[d] = wholeAxes[d].resize(size).padTo(full, c.insert ? PadKind::Bottom : PadKind::Top);
    side = TileOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), MapAttr::pair(ctx, tiled), c.insert), side,
                          wholeAxes[d], b.getI64IntegerAttr(size), b.getI64IntegerAttr(c.slice.getStaticOffsets()[d]));
  }
  HbmToHbmViewOp::create(b, loc, src, dst);
  kb.finish({});
  return success();
}

} // namespace

OpClass mlir::furiosa::mapping::copyClass() { return {"copy", match, map}; }

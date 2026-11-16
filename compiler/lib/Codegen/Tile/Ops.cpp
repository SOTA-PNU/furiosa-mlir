// Verifiers of the tiled-kernel fopt ops (FoptTileOps.td).

#include "furiosa/Codegen/Fopt.h"

using namespace mlir;
using namespace mlir::furiosa::fopt;

LogicalResult ReshapeHbmOp::verify() {
  auto src = getSource().getType(), out = getResult().getType();
  if (src.getElementType() != out.getElementType() || src.getChip() != out.getChip())
    return emitOpError("a reshape keeps the element type and the chip mapping");
  if (src.getElement().size() != out.getElement().size())
    return emitOpError("a reshape keeps the size: ") << src.getElement().size() << " vs " << out.getElement().size();
  // Rust moves the source into the reshape: every other use must come before it.
  for (Operation *user : getSource().getUsers())
    if (user != getOperation() && (user->getBlock() != getOperation()->getBlock() || !user->isBeforeInBlock(*this)))
      return emitOpError("consumes its source, which must have no use after it");
  return success();
}

LogicalResult ViewReshapeOp::verify() {
  auto src = getSource().getType(), out = getResult().getType();
  if (src.getElementType() != out.getElementType() || src.getChip() != out.getChip())
    return emitOpError("a reshape keeps the element type and the chip mapping");
  if (src.getIsMut() || out.getIsMut())
    return emitOpError("only read views are reshaped (a reshaped mutable view miscompiles once tiled)");
  if (src.getElement().size() != out.getElement().size())
    return emitOpError("a reshape keeps the size: ") << src.getElement().size() << " vs " << out.getElement().size();
  if (!getSource().getDefiningOp<ViewOp>())
    return emitOpError("reshapes a whole view");
  return success();
}

LogicalResult VectorAluOp::verify() {
  if (!getInput().getType().getElementType().isF32() || !getResult().getType().getElementType().isF32())
    return emitOpError("Vector Engine ALU calls of tiled kernels run on f32");
  int operands = (getImm() ? 1 : 0) + (getVrf() && !getBranches() ? 1 : 0) + (getStash() && !getBranches() ? 1 : 0) +
                 (getBranches() ? 1 : 0);
  if (operands > 1)
    return emitOpError("takes at most one operand form");
  return success();
}

LogicalResult VectorTagCmpOp::verify() {
  if (getCmps().size() != 4)
    return emitOpError("a comparison tag has four Cmp cases");
  return success();
}

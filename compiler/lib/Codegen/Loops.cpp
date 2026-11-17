// Kept loops: an scf.for the raise left at a kernel's top level (a loop-carried state, such as an argmax's
// running maximum and its index) runs as a host launch sequence. The loop unrolls into its iterations
// (upstream SCF utils), each a few generics the op classes map, and furiosa-structure then promotes the reads
// the induction variable addressed, now constants.

#include "furiosa/Codegen/Codegen.h"
#include "furiosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

// shortcut: a launch per iteration step, so trip counts are capped at kMaxTrips; device loops (fopt.for)
// would take longer loops.
static constexpr int64_t kMaxTrips = 64;

LogicalResult mlir::furiosa::unrollKeptLoops(ModuleOp module) {
  SmallVector<scf::ForOp> loops;
  for (auto func : module.getOps<func::FuncOp>())
    for (auto loop : func.getBody().front().getOps<scf::ForOp>())
      if (std::optional<APInt> trips = loop.getStaticTripCount(); trips && trips->getSExtValue() <= kMaxTrips)
        loops.push_back(loop);
  if (loops.empty())
    return success();
  for (scf::ForOp loop : loops)
    if (failed(loopUnrollFull(loop)))
      return loop.emitError("cannot unroll a kept loop");
  PassManager pm(module.getContext());
  pm.addPass(createStructurePass());
  return pm.run(module);
}

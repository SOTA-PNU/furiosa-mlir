// Pass furiosa-tile: before an op class maps a kernel, makes the loops the class's tiling hook asks for
// (mapping::Tiling): row tile loops over the root generic (upstream TilingInterface tile and fuse: the generics and
// fills it reads compute per tile, pads and buffers are sliced) and K loops over one generic's reduction loops
// (tileUsingSCF, each marked furiosa.tile_dim). The class is the one mapKernel picks; it maps the tiled kernel,
// reading the loops back (TileLoops.h). The tiled class's hook tiles by its plan (Plan.h).

#include "../Mapping/Mapping.h"

#include "furiosa/Codegen/Codegen.h"
#include "furiosa/Codegen/Fopt.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::furiosa;

namespace {

/// Tiles `g`'s reduction loops by `sizes` into K loops, each marked with the loop it tiles.
LogicalResult tileReduction(IRRewriter &rewriter, linalg::GenericOp g, ArrayRef<int64_t> sizes) {
  rewriter.setInsertionPoint(g);
  scf::SCFTilingOptions options;
  options.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(), sizes));
  FailureOr<scf::SCFTilingResult> result = scf::tileUsingSCF(rewriter, cast<TilingInterface>(g.getOperation()), options);
  if (failed(result))
    return failure();
  SmallVector<unsigned> tiledDims;
  for (auto [d, size] : llvm::enumerate(sizes))
    if (size)
      tiledDims.push_back(d);
  for (auto [loop, d] : llvm::zip(result->loops, tiledDims))
    loop->setAttr("furiosa.tile_dim", rewriter.getI64IntegerAttr(d));
  rewriter.replaceOp(g, result->replacements);
  return success();
}

/// Makes `t`'s loops in `func`; failure leaves the kernel as it was where upstream tiling refuses.
LogicalResult applyTiling(func::FuncOp func, mapping::Tiling t) {
  IRRewriter rewriter(func.getContext());
  linalg::GenericOp reduce = t.reduce;
  if (llvm::any_of(t.rowSizes, [](int64_t s) { return s; })) {
    rewriter.setInsertionPoint(t.root);
    scf::SCFTilingOptions rowTiling;
    rowTiling.setTileSizes(getAsIndexOpFoldResult(func.getContext(), t.rowSizes));
    scf::SCFTileAndFuseOptions options;
    options.setTilingOptions(rowTiling);
    // Producers computed per tile: generics and fills. Pads and buffers are sliced as they are.
    options.setFusionControlFn([](tensor::ExtractSliceOp, OpResult producer,
                                  bool) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      if (isa<linalg::GenericOp, linalg::FillOp>(producer.getOwner()))
        return scf::SCFTileAndFuseOptions::ControlFnResult{};
      return std::nullopt;
    });
    FailureOr<scf::SCFTileAndFuseResult> result =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, cast<TilingInterface>(t.root.getOperation()), options);
    if (failed(result))
      return failure();
    for (Value r : t.root->getResults())
      rewriter.replaceAllUsesWith(r, result->replacements.lookup(r));
    // The reduction's tile: the fused copy with the same loops and maps.
    if (reduce) {
      linalg::GenericOp tiled;
      for (Operation *op : result->tiledAndFusedOps)
        if (auto g = dyn_cast<linalg::GenericOp>(op); g &&
            g.getIteratorTypesArray() == reduce.getIteratorTypesArray() &&
            g.getIndexingMapsArray() == reduce.getIndexingMapsArray())
          tiled = g;
      reduce = tiled;
    }
  }
  if (reduce && failed(tileReduction(rewriter, reduce, t.reduceSizes)))
    return failure();
  // The untiled root, the producers only it read, and the slices of them fusion left unused are dead now.
  (void)applyPatternsGreedily(func, RewritePatternSet(func.getContext()));
  return success();
}

struct TilePass : PassWrapper<TilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilePass)

  StringRef getArgument() const override { return "furiosa-tile"; }
  StringRef getDescription() const override {
    return "Make the loops the op class mapping each kernel tiles it by (row tiles, K tiles)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerCodegenDialects(registry);
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    for (func::FuncOp func : llvm::to_vector(getOperation().getOps<func::FuncOp>())) {
      if (func.isExternal() || func->hasAttr(fopt::kChipAttr) || !func->hasAttr("furiosa.written"))
        continue;
      // The class mapKernel picks: the first that matches and maps the kernel.
      mapping::RaisedKernel raised = mapping::raisedKernel(func);
      const mapping::OpClass *cls = nullptr;
      for (const mapping::OpClass &c : mapping::opClasses())
        if (c.match(raised) && mapping::mapsOnCopy(func, c)) {
          cls = &c;
          break;
        }
      if (!cls || !cls->tiling)
        continue;
      if (std::optional<mapping::Tiling> t = cls->tiling(raised))
        (void)applyTiling(func, *t);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createTilePass() { return std::make_unique<TilePass>(); }

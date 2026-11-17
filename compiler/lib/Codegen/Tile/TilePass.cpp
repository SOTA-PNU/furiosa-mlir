// Pass furiosa-tile: tiles each kernel by the tiled lowering's plan (Plan.h), its cost model. The plan's tiles
// of the output rows become a nest of scf.for over the kernel's root generic (upstream TilingInterface tile and
// fuse: the generics and fills it reads are computed per tile, pads and buffers are sliced), so the loops carry
// the tiling decision and the tiled class maps one tile (Analysis.cpp folds the loops back into the plan's tile
// digits). Kernels the plan cannot lower, or whose tiles are not whole ranges of one loop, stay as they are.

#include "Plan.h"

#include "../Mapping/Mapping.h"

#include "furiosa/Codegen/Codegen.h"
#include "furiosa/Codegen/Fopt.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::tile;

namespace {

/// The plan's tiles as tile sizes of the root generic's loops (0: not tiled), or nothing when there is nothing
/// to tile or a tile is not a whole range of one loop (a padded axis, an axis that is an inner digit of its dim).
std::optional<std::pair<linalg::GenericOp, SmallVector<int64_t>>> tileSizes(func::FuncOp func,
                                                                              ArrayRef<int64_t> written) {
  std::string why;
  FailureOr<Program> p = analyze(func, written, why);
  if (failed(p) || !p->rootOp || p->outLoop.empty())
    return std::nullopt;
  Plan plan(func, *p, /*quiet=*/true);
  if (failed(plan.plan()) || plan.tiles().empty())
    return std::nullopt;
  auto root = cast<linalg::GenericOp>(p->rootOp);
  SmallVector<int64_t> sizes(root.getNumLoops(), 0);
  for (const Digit &d : plan.tiles()) {
    auto [dim, place] = p->axisDim[d.axis];
    int64_t t = place * d.place;
    if (d.extent || dim >= p->outLoop.size() || t * d.size != p->dimExtent[dim] || sizes[p->outLoop[dim]]) {
      if (std::getenv("FURIOSA_TILED_DEBUG"))
        llvm::errs() << "tile: " << func.getName() << ": tile digit of axis " << d.axis << " (place " << d.place
                     << ", size " << d.size << ", padded " << d.extent << ") is no whole range of a loop\n";
      return std::nullopt;
    }
    sizes[p->outLoop[dim]] = t;
  }
  return std::make_pair(root, sizes);
}

struct TilePass : PassWrapper<TilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilePass)

  StringRef getArgument() const override { return "furiosa-tile"; }
  StringRef getDescription() const override {
    return "Tile kernels by the tiled lowering's plan: scf.for over output tiles (TilingInterface)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerCodegenDialects(registry);
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    for (func::FuncOp func : llvm::to_vector(getOperation().getOps<func::FuncOp>())) {
      auto attr = func->getAttrOfType<ArrayAttr>("furiosa.written");
      if (func.isExternal() || func->hasAttr(fopt::kChipAttr) || !attr)
        continue;
      SmallVector<int64_t> written;
      for (Attribute x : attr)
        written.push_back(cast<IntegerAttr>(x).getInt());
      // Only kernels the tiled class maps (no op class before it does) are tiled for it.
      mapping::RaisedKernel raised = mapping::raisedKernel(func);
      bool earlier = false;
      for (const mapping::OpClass &c : mapping::opClasses()) {
        if (StringRef(c.name) == "tiled")
          break;
        if ((earlier = c.match(raised) && mapping::mapsOnCopy(func, c)))
          break;
      }
      if (earlier)
        continue;
      auto found = tileSizes(func, written);
      if (!found)
        continue;
      auto [root, sizes] = *found;
      IRRewriter rewriter(ctx);
      rewriter.setInsertionPoint(root);
      scf::SCFTilingOptions tiling;
      tiling.setTileSizes(getAsIndexOpFoldResult(ctx, sizes));
      scf::SCFTileAndFuseOptions options;
      options.setTilingOptions(tiling);
      // Producers computed per tile: generics and fills. Pads and buffers are sliced as they are.
      options.setFusionControlFn([](tensor::ExtractSliceOp, OpResult producer,
                                    bool) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (isa<linalg::GenericOp, linalg::FillOp>(producer.getOwner()))
          return scf::SCFTileAndFuseOptions::ControlFnResult{};
        return std::nullopt;
      });
      FailureOr<scf::SCFTileAndFuseResult> result =
          scf::tileConsumerAndFuseProducersUsingSCF(rewriter, cast<TilingInterface>(root.getOperation()), options);
      if (failed(result))
        continue;
      for (Value r : root->getResults())
        rewriter.replaceAllUsesWith(r, result->replacements.lookup(r));
      // The untiled root, the producers only it read, and the slices of them fusion left unused are dead now.
      (void)applyPatternsGreedily(func, RewritePatternSet(ctx));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createTilePass() { return std::make_unique<TilePass>(); }

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

/// What the plan tiles: the root generic's loops (row tiles; 0 for a loop not tiled) and the contraction
/// generic's reduction loops (K tiles). Empty when the plan cannot lower the kernel or a tile is no whole range of
/// one loop (a padded axis, an axis that is an inner digit of its dim).
struct Tiling {
  linalg::GenericOp root, contraction;
  SmallVector<int64_t> rowSizes, kSizes;
  bool rows() const { return llvm::any_of(rowSizes, [](int64_t t) { return t; }); }
};

std::optional<Tiling> planTiling(func::FuncOp func, ArrayRef<int64_t> written) {
  std::string why;
  FailureOr<Program> p = analyze(func, written, why);
  if (failed(p) || !p->rootOp || p->outLoop.empty())
    return std::nullopt;
  Plan plan(func, *p, /*quiet=*/true);
  if (failed(plan.plan()))
    return std::nullopt;
  auto debug = [&](const Twine &msg) {
    if (std::getenv("FURIOSA_TILED_DEBUG"))
      llvm::errs() << "tile: " << func.getName() << ": " << msg << "\n";
    return std::nullopt;
  };
  Tiling t;
  t.root = cast<linalg::GenericOp>(p->rootOp);
  t.rowSizes.assign(t.root.getNumLoops(), 0);
  for (const Digit &d : plan.tiles()) {
    auto [dim, place] = p->axisDim[d.axis];
    int64_t size = place * d.place;
    if (d.extent || dim >= p->outLoop.size() || size * d.size != p->dimExtent[dim] || t.rowSizes[p->outLoop[dim]])
      return debug("tile digit of axis " + Twine(d.axis) + " is no whole range of a loop");
    t.rowSizes[p->outLoop[dim]] = size;
  }
  if (auto k = plan.kAxisTiles()) {
    for (auto [a, size] : *k) {
      auto [dim, place] = p->axisDim[a];
      auto it = p->reductionLoop.find(dim);
      if (place != 1 || p->axes[a].size != p->dimExtent[dim] || it == p->reductionLoop.end() ||
          (t.contraction && t.contraction != it->second.first))
        return debug("K tile of axis " + Twine(a) + " is no whole reduction loop of the contraction");
      t.contraction = cast<linalg::GenericOp>(it->second.first);
      t.kSizes.resize(t.contraction.getNumLoops(), 0);
      t.kSizes[it->second.second] = size;
    }
  }
  if (!t.rows() && !t.contraction)
    return std::nullopt;
  return t;
}

/// Tiles `g`'s reduction loops by `sizes` into K loops, each marked with the loop it tiles.
LogicalResult tileK(IRRewriter &rewriter, linalg::GenericOp g, ArrayRef<int64_t> sizes) {
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
      std::optional<Tiling> tiling = planTiling(func, written);
      if (!tiling)
        continue;
      IRRewriter rewriter(ctx);
      linalg::GenericOp contraction = tiling->contraction;
      if (tiling->rows()) {
        rewriter.setInsertionPoint(tiling->root);
        scf::SCFTilingOptions rowTiling;
        rowTiling.setTileSizes(getAsIndexOpFoldResult(ctx, tiling->rowSizes));
        scf::SCFTileAndFuseOptions options;
        options.setTilingOptions(rowTiling);
        // Producers computed per tile: generics and fills. Pads and buffers are sliced as they are.
        options.setFusionControlFn([](tensor::ExtractSliceOp, OpResult producer,
                                      bool) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          if (isa<linalg::GenericOp, linalg::FillOp>(producer.getOwner()))
            return scf::SCFTileAndFuseOptions::ControlFnResult{};
          return std::nullopt;
        });
        FailureOr<scf::SCFTileAndFuseResult> result = scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(tiling->root.getOperation()), options);
        if (failed(result))
          continue;
        for (Value r : tiling->root->getResults())
          rewriter.replaceAllUsesWith(r, result->replacements.lookup(r));
        // The contraction's tile: the fused copy with the same loops and maps.
        if (contraction) {
          linalg::GenericOp tiled;
          for (Operation *op : result->tiledAndFusedOps)
            if (auto g = dyn_cast<linalg::GenericOp>(op);
                g && g.getIteratorTypesArray() == contraction.getIteratorTypesArray() &&
                g.getIndexingMapsArray() == contraction.getIndexingMapsArray())
              tiled = g;
          contraction = tiled;
        }
      }
      if (contraction && failed(tileK(rewriter, contraction, tiling->kSizes)))
        continue;
      // The untiled root, the producers only it read, and the slices of them fusion left unused are dead now.
      (void)applyPatternsGreedily(func, RewritePatternSet(ctx));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createTilePass() { return std::make_unique<TilePass>(); }

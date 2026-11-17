// furiosa-mlir-opt: mlir-opt with the Furiosa passes and the dialects they read and write, for the
// per-pass lit tests in compiler/tests/lit.

#include "furiosa/Codegen/Codegen.h"
#include "furiosa/Transforms/Passes.h"

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect>();
  mlir::furiosa::registerDialects(registry);
  mlir::furiosa::registerCodegenDialects(registry);
  mlir::registerPass([] { return mlir::furiosa::createRaiseToLinalgPass(); });
  mlir::registerPass([] { return mlir::furiosa::createStructurePass(); });
  mlir::registerPass([] { return mlir::furiosa::createLinalgToFoptPass(); });
  mlir::registerPass([] { return mlir::furiosa::createSplitStepsPass(); });
  mlir::registerPass([] { return mlir::furiosa::createNormalizeKernelsPass(); });
  mlir::registerPass([] { return mlir::furiosa::createNormalizeContractionsPass(); });
  mlir::registerPass([] { return mlir::furiosa::createHoistIndexOnlyPass(); });
  mlir::registerPass([] { return mlir::furiosa::createKeptLoopsPass(); });
  mlir::registerPass([] { return mlir::furiosa::createSliceKeptLoopsPass(); });
  mlir::registerPass([] { return mlir::furiosa::createStepsPlanPass(); });
  mlir::registerPass([] { return mlir::furiosa::createTilePass(); });
  mlir::registerPass([] { return mlir::furiosa::createDmaPass(); });
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Furiosa passes", registry));
}

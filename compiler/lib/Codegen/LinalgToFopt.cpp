// linalg -> fopt: each raised kernel goes to the first op class (lib/Codegen/Mapping/) that matches it.

#include "Mapping/Mapping.h"
#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::mapping;

//===----------------------------------------------------------------------===//
// KernelBuilder
//===----------------------------------------------------------------------===//

KernelBuilder::KernelBuilder(RaisedKernel &raised)
    : raised(raised), ctx(raised.func.getContext()), b(raised.func.getContext()) {}

fopt::MapAttr KernelBuilder::axis(StringRef name, int64_t size) {
  for (auto &[existing, n] : axes)
    if (existing == name) {
      assert(n == size && "an axis redeclared with another size");
      return fopt::MapAttr::axis(ctx, name, size);
    }
  axes.emplace_back(name.str(), size);
  return fopt::MapAttr::axis(ctx, name, size);
}

unsigned KernelBuilder::input(int64_t ptrArg, Type elementType, fopt::MapAttr element) {
  assert(!kernel && "inputs come before beginBody");
  inputTypes.push_back(hbmType(elementType, element));
  inputArgs.push_back(ptrArg);
  return inputTypes.size() - 1;
}

void KernelBuilder::output(int64_t ptrArg, Type elementType, fopt::MapAttr element) {
  assert(!kernel && "outputs come before beginBody");
  outputTypes.push_back(hbmType(elementType, element));
  outputArgs.push_back(ptrArg);
}

void KernelBuilder::beginBody() {
  func::FuncOp src = raised.func;
  b.setInsertionPoint(src);
  kernel = func::FuncOp::create(b, src.getLoc(), src.getName(), b.getFunctionType(inputTypes, outputTypes));
  kernel->setAttr(fopt::kChipAttr, b.getI64IntegerAttr(1));
  kernel->setAttr(fopt::kPeAttr, b.getI64IntegerAttr(8));
  for (auto [i, arg] : llvm::enumerate(inputArgs)) {
    kernel.setArgAttr(i, fopt::kNameAttr, b.getStringAttr(raised.argNames[arg]));
    kernel.setArgAttr(i, "fopt.arg", b.getI64IntegerAttr(arg));
  }
  for (auto [i, arg] : llvm::enumerate(outputArgs)) {
    kernel.setResultAttr(i, fopt::kNameAttr, b.getStringAttr(raised.argNames[arg]));
    kernel.setResultAttr(i, "fopt.arg", b.getI64IntegerAttr(arg));
  }
  b.setInsertionPointToStart(kernel.addEntryBlock());
}

void KernelBuilder::finish(ArrayRef<Value> results) {
  SmallVector<Attribute> decls;
  for (auto &[name, size] : axes)
    decls.push_back(b.getArrayAttr({b.getStringAttr(name), b.getI64IntegerAttr(size)}));
  kernel->setAttr(fopt::kAxesAttr, b.getArrayAttr(decls));
  func::ReturnOp::create(b, raised.func.getLoc(), results);
  raised.func.erase();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace {

/// The pointer-argument names: the Triton parameter names when the raised function kept their locations.
SmallVector<std::string> argNames(func::FuncOp func) {
  SmallVector<std::string> names;
  for (BlockArgument arg : func.getArguments()) {
    std::string name = "arg" + std::to_string(arg.getArgNumber());
    Location loc = arg.getLoc();
    while (auto fused = dyn_cast<FusedLoc>(loc))
      loc = fused.getLocations().empty() ? loc : fused.getLocations().front();
    if (auto named = dyn_cast<NameLoc>(loc))
      name = named.getName().str();
    names.push_back(name);
  }
  return names;
}

SmallVector<int64_t> writtenArgs(func::FuncOp func) {
  SmallVector<int64_t> written;
  if (auto attr = func->getAttrOfType<ArrayAttr>("furiosa.written"))
    for (Attribute x : attr)
      written.push_back(cast<IntegerAttr>(x).getInt());
  return written;
}

struct LinalgToFoptPass : PassWrapper<LinalgToFoptPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToFoptPass)

  StringRef getArgument() const override { return "furiosa-linalg-to-fopt"; }
  StringRef getDescription() const override { return "Map raised linalg kernels to furiosa-opt kernels (fopt)"; }
  void getDependentDialects(DialectRegistry &registry) const override { registerCodegenDialects(registry); }

  void runOnOperation() override {
    static const OpClass classes[] = {elementwiseClass()};
    SmallVector<func::FuncOp> funcs(getOperation().getOps<func::FuncOp>());
    for (func::FuncOp func : funcs) {
      if (func->hasAttr(fopt::kChipAttr))
        continue;
      RaisedKernel kernel{func, argNames(func), writtenArgs(func)};
      const OpClass *chosen = nullptr;
      for (const OpClass &c : classes)
        if (c.match(kernel)) {
          chosen = &c;
          break;
        }
      if (!chosen) {
        func.emitError("no op class maps this kernel (classes: elementwise)");
        return signalPassFailure();
      }
      if (failed(chosen->map(kernel)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createLinalgToFoptPass() { return std::make_unique<LinalgToFoptPass>(); }

void mlir::furiosa::registerCodegenDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect, math::MathDialect,
                  scf::SCFDialect, tensor::TensorDialect, fopt::FoptDialect>();
  func::registerInlinerExtension(registry);
}

#include "furiosa/Runtime/Jit.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <mutex>

namespace mlir::furiosa {
namespace {

// Bufferize with tensor results turned into caller-allocated out-params, free every temporary (a
// model graph calls kernels hundreds of times), then lower to the LLVM dialect. Arguments are marked
// non-writable first, so bufferization copies instead of writing them.
// The pipeline text is relative to the module-anchored pass manager it is parsed into.
/// arith.extf from f8E4M3FN, which LLVM has no type for, decoded from its bits: sign, 4 exponent bits
/// (bias 7), 3 mantissa bits; exponent 0 is subnormal (mantissa * 2^-9); S.1111.111 is NaN, and there
/// are no infinities.
struct ExtFromF8E4M3FN : OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp op, PatternRewriter &rw) const override {
    if (!isa<Float8E4M3FNType>(op.getIn().getType()))
      return failure();
    Location loc = op.getLoc();
    Type i32 = rw.getI32Type(), f32 = rw.getF32Type();
    auto c = [&](int64_t v) -> Value { return arith::ConstantIntOp::create(rw, loc, i32, v); };
    Value bits = arith::ExtUIOp::create(rw, loc, i32, arith::BitcastOp::create(rw, loc, rw.getI8Type(), op.getIn()));
    Value sign = arith::ShLIOp::create(rw, loc, arith::ShRUIOp::create(rw, loc, bits, c(7)), c(31));
    Value exp = arith::AndIOp::create(rw, loc, arith::ShRUIOp::create(rw, loc, bits, c(3)), c(15));
    Value man = arith::AndIOp::create(rw, loc, bits, c(7));
    Value normal = arith::OrIOp::create(rw, loc, arith::ShLIOp::create(rw, loc, arith::AddIOp::create(rw, loc, exp, c(120)), c(23)),
                                        arith::ShLIOp::create(rw, loc, man, c(20)));
    Value sub = arith::BitcastOp::create(
        rw, loc, i32, arith::MulFOp::create(rw, loc, arith::UIToFPOp::create(rw, loc, f32, man),
                                            arith::ConstantFloatOp::create(rw, loc, cast<FloatType>(f32), APFloat(0x1p-9f))));
    Value isSub = arith::CmpIOp::create(rw, loc, arith::CmpIPredicate::eq, exp, c(0));
    Value isNan = arith::CmpIOp::create(rw, loc, arith::CmpIPredicate::eq, arith::AndIOp::create(rw, loc, bits, c(127)), c(127));
    Value mag = arith::SelectOp::create(rw, loc, isNan, c(0x7fc00000), arith::SelectOp::create(rw, loc, isSub, sub, normal));
    Value f = arith::BitcastOp::create(rw, loc, f32, arith::OrIOp::create(rw, loc, sign, mag));
    if (op.getType() != f32)
      f = op.getType().getIntOrFloatBitWidth() > 32 ? arith::ExtFOp::create(rw, loc, op.getType(), f).getResult()
                                                    : arith::TruncFOp::create(rw, loc, op.getType(), f).getResult();
    rw.replaceOp(op, f);
    return success();
  }
};

constexpr const char *kPipeline =
    "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map "
    "allow-return-allocs-from-loops},"
    "buffer-results-to-out-params{hoist-static-allocs modify-public-functions},"
    "ownership-based-buffer-deallocation,"
    "canonicalize,"
    "buffer-deallocation-simplification,"
    "bufferization-lower-deallocations,"
    "convert-bufferization-to-memref,"
    "cse,"
    "canonicalize,"
    "convert-linalg-to-loops,"
    "expand-strided-metadata,"
    "lower-affine,"
    "convert-scf-to-cf,"
    "convert-math-to-llvm,"
    "convert-math-to-libm,"
    "convert-arith-to-llvm,"
    "convert-index-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "convert-cf-to-llvm,"
    "convert-ub-to-llvm,"
    "reconcile-unrealized-casts";

// The runtime library memref.copy of non-contiguous memrefs calls into.
constexpr const char *kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;

void registerOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    bufferization::registerOneShotBufferizePass();
    bufferization::registerBufferResultsToOutParamsPass();
    bufferization::registerOwnershipBasedBufferDeallocationPass();
    bufferization::registerBufferDeallocationSimplificationPass();
    bufferization::registerLowerDeallocationsPass();
    registerConvertBufferizationToMemRefPass();
    registerCanonicalizerPass();
    registerCSEPass();
    registerConvertLinalgToLoopsPass();
    memref::registerExpandStridedMetadataPass();
    registerLowerAffinePass();
    registerSCFToControlFlowPass();
    registerConvertMathToLLVMPass();
    registerConvertMathToLibmPass();
    registerArithToLLVMConversionPass();
    registerConvertIndexToLLVMPass();
    registerFinalizeMemRefToLLVMConversionPass();
    registerConvertFuncToLLVMPass();
    registerConvertControlFlowToLLVMPass();
    registerUBToLLVMConversionPass();
    registerReconcileUnrealizedCastsPass();
  });
}

DialectRegistry makeRegistry() {
  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect, bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, func::FuncDialect, linalg::LinalgDialect, LLVM::LLVMDialect,
                  math::MathDialect, memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
                  ub::UBDialect>();
  // Interfaces these dialects promise but implement in separate libraries; a query of a promised,
  // unregistered interface aborts.
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  memref::registerValueBoundsOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerSubsetOpInterfaceExternalModels(registry);
  memref::registerAllocationOpInterfaceExternalModels(registry);
  memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  return registry;
}

} // namespace

JitModule::JitModule() = default;
JitModule::~JitModule() = default;

std::unique_ptr<JitModule> JitModule::compile(llvm::StringRef text, std::string &error) {
  registerOnce();
  std::unique_ptr<JitModule> jit(new JitModule());
  jit->context = std::make_unique<MLIRContext>(makeRegistry());
  MLIRContext &ctx = *jit->context;

  llvm::raw_string_ostream diagnostics(error);
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    diagnostics << diag.getLocation() << ": " << diag << "\n";
    return success();
  });

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, &ctx);
  if (!module) {
    diagnostics << "failed to parse the raised module\n";
    return nullptr;
  }
  for (auto func : module->getOps<func::FuncOp>()) {
    if (func.isPrivate() || func.isExternal())
      continue;
    func->setAttr("llvm.emit_c_interface", UnitAttr::get(&ctx));
    for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i)
      if (isa<TensorType>(func.getArgumentTypes()[i]))
        func.setArgAttr(i, "bufferization.writable", BoolAttr::get(&ctx, false));
  }

  RewritePatternSet f8(&ctx);
  f8.add<ExtFromF8E4M3FN>(&ctx);
  (void)applyPatternsGreedily(*module, std::move(f8));
  PassManager pm(&ctx);
  if (failed(parsePassPipeline(kPipeline, pm, diagnostics)) || failed(pm.run(*module))) {
    diagnostics << "lowering the raised module to LLVM failed\n";
    return nullptr;
  }
  llvm::raw_string_ostream lowered(jit->loweredText);
  module->print(lowered);

  // -O3 for the host: the loops convert-linalg-to-loops leaves are scalar, and whole models run here.
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) {
    diagnostics << "detecting the host target failed: " << llvm::toString(builder.takeError()) << "\n";
    return nullptr;
  }
  auto machine = builder->createTargetMachine();
  if (!machine) {
    diagnostics << "creating the host target machine failed: " << llvm::toString(machine.takeError()) << "\n";
    return nullptr;
  }
  std::function<llvm::Error(llvm::Module *)> optimize = makeOptimizingTransformer(3, 0, machine->get());

  llvm::StringRef libs[] = {kRunnerUtils};
  ExecutionEngineOptions options;
  options.sharedLibPaths = libs;
  options.transformer = optimize;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  options.enableGDBNotificationListener = false;
  options.enablePerfNotificationListener = false;
  auto engine = ExecutionEngine::create(*module, options, std::move(*machine));
  if (!engine) {
    diagnostics << "creating the execution engine failed: " << llvm::toString(engine.takeError())
                << "\nlowered module:\n" << llvm::StringRef(jit->loweredText).take_front(4000) << "\n";
    return nullptr;
  }
  jit->engine = std::move(*engine);
  jit->engine->initialize();
  return jit;
}

void *JitModule::lookup(llvm::StringRef name, std::string &error) const {
  auto address = engine->lookup(("_mlir_ciface_" + name).str());
  if (!address) {
    error = llvm::toString(address.takeError());
    return nullptr;
  }
  return *address;
}

} // namespace mlir::furiosa

// linalg -> fopt: each raised kernel goes to the first op class (lib/Codegen/Mapping/) that matches it.

#include "Mapping/Mapping.h"
#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

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
  inoutParams.push_back(false);
  return inputTypes.size() - 1;
}

unsigned KernelBuilder::inout(int64_t ptrArg, Type elementType, fopt::MapAttr element) {
  unsigned i = input(ptrArg, elementType, element);
  inoutParams[i] = true;
  return i;
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
    if (inoutParams[i])
      kernel.setArgAttr(i, fopt::kInoutAttr, b.getUnitAttr());
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

/// fp16 and fp8 buffers live as f32 on the device (furiosa-opt has no f16, every f16 and fp8 is exact in f32, and
/// an fp8 is exact in the bf16 a contraction casts f32 to; both convert at the graph boundary): the kernel's f16 and
/// fp8 tensors and scalars become f32, and the conversions between them vanish.
void widenF16(func::FuncOp func) {
  Type f32 = Float32Type::get(func.getContext());
  auto narrow = [](Type t) { return t.isF16() || isa<Float8E4M3FNType, Float8E5M2Type>(t); };
  auto widen = [&](Type t) -> Type {
    if (narrow(t))
      return f32;
    if (auto rt = dyn_cast<RankedTensorType>(t); rt && narrow(rt.getElementType()))
      return rt.clone(f32);
    return t;
  };
  bool any = false;
  auto retype = [&](Value v) {
    if (Type w = widen(v.getType()); w != v.getType()) {
      v.setType(w);
      any = true;
    }
  };
  func.walk([&](Operation *op) {
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (BlockArgument a : b.getArguments())
          retype(a);
    for (Value v : op->getResults())
      retype(v);
    if (auto c = dyn_cast<arith::ConstantOp>(op))
      if (auto fa = dyn_cast<FloatAttr>(c.getValue()); fa && narrow(fa.getType()))
        c.setValueAttr(FloatAttr::get(f32, fa.getValueAsDouble()));
  });
  if (!any)
    return;
  SmallVector<Operation *> casts;
  func.walk([&](Operation *op) {
    if (isa<arith::ExtFOp, arith::TruncFOp>(op) && op->getOperand(0).getType() == op->getResult(0).getType())
      casts.push_back(op);
  });
  for (Operation *op : casts) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    op->erase();
  }
  SmallVector<Type> results = llvm::map_to_vector(func.getResultTypes(), widen);
  func.setType(FunctionType::get(func.getContext(), func.getBody().getArgumentTypes(), results));
}

/// int8 and int16 values live as i32 on the device (the Vector Engine computes on 32-bit lanes; I8 converts at the
/// graph boundary): the kernel's i8 and i16 tensors, scalars and constants become i32 holding the same values. An op
/// whose w-bit result can leave its range wraps back into it, (v << (32 - w)) >> (32 - w); a sign extension from a
/// narrow integer vanishes, a truncation to one is the wrap, and unsigned reads (zero extension, unsigned compares
/// and shifts, uitofp) take its low w bits. A bf16's bits as an i16 (tl.flip's bitcast) are those of the f32 it
/// widens to, >> 16, and an i16's as a bf16 are << 16: exact, a bf16 being an f32's high half. (An f16 lives as
/// f32, whose bits are not the f16's: its bitcasts stay unmapped.)
void widenNarrowInts(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  Type i32 = IntegerType::get(ctx, 32);
  auto width = [](Type t) -> unsigned {
    t = getElementTypeOrSelf(t);
    return t.isInteger(8) || t.isInteger(16) ? t.getIntOrFloatBitWidth() : 0;
  };
  auto narrow = [&](Value v) { return width(v.getType()); };
  auto widen = [&](Type t) -> Type {
    if (!width(t))
      return t;
    if (auto rt = dyn_cast<RankedTensorType>(t))
      return rt.clone(i32);
    return i32;
  };
  // What to fix once retyped, decided on the narrow types (with each one's width).
  SmallVector<std::pair<Operation *, unsigned>> wraps, extends, lows;
  SmallVector<arith::BitcastOp> bitcasts;
  bool any = false;
  func.walk([&](Operation *op) {
    for (Value v : op->getResults())
      any |= narrow(v) != 0;
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (BlockArgument a : b.getArguments())
          any |= narrow(a) != 0;
    unsigned w = op->getNumResults() == 1 ? narrow(op->getResult(0)) : 0;
    if (w && isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp, math::AbsIOp, arith::DivSIOp,
                 arith::CeilDivSIOp, arith::FloorDivSIOp, arith::TruncIOp, arith::FPToSIOp, arith::ShRUIOp>(op))
      wraps.push_back({op, w});
    if (isa<arith::ExtSIOp, arith::ExtUIOp>(op) && narrow(op->getOperand(0)))
      extends.push_back({op, narrow(op->getOperand(0))});
    auto cmp = dyn_cast<arith::CmpIOp>(op);
    if (cmp && narrow(cmp.getLhs()) && cmp.getPredicate() >= arith::CmpIPredicate::ult)
      lows.push_back({op, narrow(cmp.getLhs())});
    if (isa<arith::ShRUIOp, arith::UIToFPOp>(op) && narrow(op->getOperand(0)))
      lows.push_back({op, narrow(op->getOperand(0))});
    if (auto c = dyn_cast<arith::BitcastOp>(op);
        c && ((c.getIn().getType().isBF16() && c.getType().isInteger(16)) ||
              (c.getIn().getType().isInteger(16) && c.getType().isBF16())))
      bitcasts.push_back(c);
  });
  for (BlockArgument a : func.getArguments())
    any |= narrow(a) != 0;
  if (!any)
    return;
  func.walk([&](Operation *op) {
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (BlockArgument a : b.getArguments())
          a.setType(widen(a.getType()));
    for (Value v : op->getResults())
      v.setType(widen(v.getType()));
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto ia = dyn_cast<IntegerAttr>(c.getValue()); ia && width(ia.getType()))
        c.setValueAttr(IntegerAttr::get(i32, ia.getValue().sext(32)));
      if (auto da = dyn_cast<DenseIntElementsAttr>(c.getValue()); da && width(da.getElementType()))
        c.setValueAttr(da.mapValues(i32, [](const APInt &v) { return v.sext(32); }));
    }
  });
  auto constant = [&](OpBuilder &b, Location loc, int64_t v) { return arith::ConstantIntOp::create(b, loc, i32, v); };
  // The low w bits of an operand, for the unsigned reads.
  for (auto [op, w] : lows) {
    OpBuilder b(op);
    unsigned n = isa<arith::CmpIOp>(op) ? 2 : 1;
    for (unsigned i = 0; i < n; ++i)
      op->setOperand(i, arith::AndIOp::create(b, op->getLoc(), op->getOperand(i),
                                              constant(b, op->getLoc(), (int64_t(1) << w) - 1)));
  }
  for (auto [op, w] : extends) {
    OpBuilder b(op);
    Value x = op->getOperand(0);
    if (isa<arith::ExtUIOp>(op))
      x = arith::AndIOp::create(b, op->getLoc(), x, constant(b, op->getLoc(), (int64_t(1) << w) - 1));
    Type to = op->getResult(0).getType();
    if (to != i32)
      x = isa<arith::ExtUIOp>(op) ? arith::ExtUIOp::create(b, op->getLoc(), to, x).getResult()
                                  : arith::ExtSIOp::create(b, op->getLoc(), to, x).getResult();
    op->getResult(0).replaceAllUsesWith(x);
    op->erase();
  }
  for (auto [op, w] : wraps) {
    OpBuilder b(op);
    Location loc = op->getLoc();
    Value v = op->getResult(0);
    if (auto t = dyn_cast<arith::TruncIOp>(op)) {
      // From wider than i32: truncate to i32 first; from i32 (a narrow source already widened): just wrap.
      b.setInsertionPoint(op);
      v = t.getIn().getType() == i32 ? t.getIn() : arith::TruncIOp::create(b, loc, i32, t.getIn()).getResult();
    } else {
      b.setInsertionPointAfter(op);
    }
    Value shift = constant(b, loc, 32 - w);
    Value wrapped = arith::ShRSIOp::create(b, loc, arith::ShLIOp::create(b, loc, v, shift), shift);
    if (isa<arith::TruncIOp>(op)) {
      op->getResult(0).replaceAllUsesWith(wrapped);
      op->erase();
    } else {
      v.replaceAllUsesExcept(wrapped, wrapped.getDefiningOp()->getOperand(0).getDefiningOp());
    }
  }
  for (arith::BitcastOp c : bitcasts) {
    OpBuilder b(c);
    Location loc = c.getLoc();
    Type f32 = Float32Type::get(ctx);
    Value sixteen = constant(b, loc, 16), r;
    if (c.getIn().getType().isBF16()) {
      Value bits = arith::BitcastOp::create(b, loc, i32, arith::ExtFOp::create(b, loc, f32, c.getIn()));
      r = arith::ShRSIOp::create(b, loc, bits, sixteen);
    } else {
      Value high = arith::BitcastOp::create(b, loc, f32, arith::ShLIOp::create(b, loc, c.getIn(), sixteen));
      r = arith::TruncFOp::create(b, loc, c.getType(), high);
    }
    c.getResult().replaceAllUsesWith(r);
    c.erase();
  }
  SmallVector<Type> results = llvm::map_to_vector(func.getResultTypes(), widen);
  func.setType(FunctionType::get(ctx, func.getBody().getArgumentTypes(), results));
}

} // namespace

ArrayRef<OpClass> mapping::opClasses() {
  static const OpClass classes[] = {gatherClass(), copyClass(), elementwiseClass(), contractionClass(),
                                    rowwiseClass(), tiledClass()};
  return classes;
}

const OpClass *mapping::matchClass(RaisedKernel &kernel) {
  for (const OpClass &c : opClasses())
    if (c.match(kernel))
      return &c;
  return nullptr;
}

RaisedKernel mapping::raisedKernel(func::FuncOp func) { return {func, argNames(func), writtenArgs(func)}; }

/// A launch passes the device and the kernel's parameters as one tuple, which furiosa-opt takes up to 24 long.
static constexpr unsigned kMaxParams = 23;

bool mapping::mapsOnCopy(func::FuncOp func, const OpClass &c) {
  OpBuilder b(func.getContext());
  OwningOpRef<ModuleOp> scratch = ModuleOp::create(b, func.getLoc());
  if (ModuleOp parent = func->getParentOfType<ModuleOp>())
    for (NamedAttribute attr : parent->getAttrs())
      (*scratch)->setAttr(attr.getName(), attr.getValue());
  auto copy = cast<func::FuncOp>(func->clone());
  scratch->push_back(copy);
  widenF16(copy);
  widenNarrowInts(copy);
  ScopedDiagnosticHandler quiet(func.getContext(), [](Diagnostic &) { return success(); });
  RaisedKernel raised = raisedKernel(copy);
  if (!c.match(raised) || failed(c.map(raised)))
    return false;
  // A class whose mapping does not verify does not map the kernel (codegen would fail on it).
  auto kernel = scratch->lookupSymbol<func::FuncOp>(func.getName());
  return kernel && kernel.getNumArguments() <= kMaxParams && succeeded(verify(kernel));
}

LogicalResult mapping::mapKernel(func::FuncOp func) {
  widenF16(func);
  widenNarrowInts(func);
  RaisedKernel kernel = raisedKernel(func);
  SmallVector<const OpClass *> matching;
  for (const OpClass &c : opClasses())
    if (c.match(kernel))
      matching.push_back(&c);
  if (matching.empty()) {
    bool gather = false;
    func.walk([&](tensor::ExtractOp) { gather = true; });
    if (gather)
      return func.emitError("the kernel reads buffers with tensor.extract (gather form); mapping needs structured "
                            "linalg (affine reads promoted to indexing-map inputs by furiosa-structure)");
    return func.emitError("no op class maps this kernel (classes: elementwise, contraction, row-wise)");
  }
  // A class's match is a quick filter: the first of several matching classes that maps the kernel (tried on a
  // copy) maps it; the last one tried explains a failure.
  ModuleOp module = func->getParentOfType<ModuleOp>();
  StringRef name = func.getName();
  const OpClass *chosen = matching.back();
  for (const OpClass *c : ArrayRef(matching).drop_back())
    if (mapsOnCopy(func, *c)) {
      chosen = c;
      break;
    }
  Location loc = func.getLoc();
  if (failed(chosen->map(kernel)))
    return failure();
  if (auto mapped = module.lookupSymbol<func::FuncOp>(name); mapped && mapped.getNumArguments() > kMaxParams)
    return emitError(loc) << chosen->name << ": " << mapped.getNumArguments() << " parameters, more than the "
                          << kMaxParams << " a launch passes";
  return success();
}

namespace {

struct LinalgToFoptPass : PassWrapper<LinalgToFoptPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToFoptPass)

  StringRef getArgument() const override { return "furiosa-linalg-to-fopt"; }
  StringRef getDescription() const override { return "Map raised linalg kernels to furiosa-opt kernels (fopt)"; }
  void getDependentDialects(DialectRegistry &registry) const override { registerCodegenDialects(registry); }

  void runOnOperation() override {
    SmallVector<func::FuncOp> funcs(getOperation().getOps<func::FuncOp>());
    for (func::FuncOp func : funcs) {
      if (func->hasAttr(fopt::kChipAttr))
        continue;
      if (failed(mapKernel(func)))
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
  // Kept loops tile lifted generics back to an iteration (lib/Codegen/Loops.cpp).
  linalg::registerTilingInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
}

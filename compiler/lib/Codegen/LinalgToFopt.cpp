// linalg -> fopt: each raised kernel goes to the first op class (lib/Codegen/Mapping/) that matches it.

#include "Mapping/Mapping.h"
#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::mapping;

//===----------------------------------------------------------------------===//
// KernelBuilder
//===----------------------------------------------------------------------===//

/// A kernel whose split pointer argument has no digit boundary at its split dim in the class's layout of it.
static constexpr StringLiteral kUnsplitAttr = "furiosa.unsplit";

int64_t mapping::chipCount(Operation *op) {
  ModuleOp module = isa<ModuleOp>(op) ? cast<ModuleOp>(op) : op->getParentOfType<ModuleOp>();
  auto n = module ? module->getAttrOfType<IntegerAttr>(kChipsAttr) : nullptr;
  return n ? n.getInt() : 1;
}

fopt::MapAttr mapping::chipMapping(Operation *op) {
  int64_t n = chipCount(op);
  return n > 1 ? fopt::MapAttr::axis(op->getContext(), kChipAxis, n) : fopt::MapAttr::broadcast(op->getContext(), 1);
}

KernelBuilder::KernelBuilder(RaisedKernel &raised)
    : raised(raised), ctx(raised.func.getContext()), b(raised.func.getContext()), chips(chipCount(raised.func)) {}

void KernelBuilder::copy(Value from, Value to) {
  auto view = [&](Value v, bool mut) {
    auto t = cast<fopt::HbmTensorType>(v.getType());
    return fopt::ViewOp::create(b, loc(), fopt::HbmViewType::get(ctx, t.getElementType(), t.getChip(), t.getElement(), mut),
                                v);
  };
  Value src = view(from, false), dst = view(to, true);
  fopt::HbmToHbmViewOp::create(b, loc(), src, dst);
}

fopt::HbmTensorType KernelBuilder::wholeType(fopt::HbmTensorType shard, RankedTensorType tensor, Attribute dim) {
  auto d = dyn_cast_or_null<IntegerAttr>(dim);
  if (!d)
    return {};
  // The whole buffer is chip-major within the split dim: its layout is the shard's with the chip axis before the
  // digits of that dim and the ones after it (a digit straddling that boundary splits there).
  fopt::MapAttr element = shard.getElement();
  SmallVector<fopt::MapAttr> parts;
  if (element.getKind() == fopt::MapKind::Pair)
    for (Attribute p : element.getOperands())
      parts.push_back(cast<fopt::MapAttr>(p));
  else
    parts.push_back(element);
  int64_t inner = 1;
  for (int64_t s : tensor.getShape().drop_front(d.getInt()))
    inner *= s;
  size_t at = parts.size();
  int64_t acc = 1;
  while (acc < inner) {
    if (inner % acc || !at)
      return {};
    fopt::MapAttr part = parts[--at];
    int64_t need = inner / acc;
    if (part.size() > need) {
      if (part.size() % need)
        return {};
      parts[at] = part.stride(need);
      parts.insert(parts.begin() + at + 1, part.modulo(need));
      ++at;
    }
    acc *= parts[at].size();
  }
  if (acc != inner)
    return {};
  parts.insert(parts.begin() + at, axis(kChipAxis, chips));
  // The same on every chip: a broadcast chip mapping.
  return fopt::HbmTensorType::get(ctx, shard.getElementType(), fopt::MapAttr::broadcast(ctx, chips),
                                  fopt::MapAttr::pair(ctx, parts));
}

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
  // A pointer argument furiosa-shard split over the chips: the kernel takes (returns) the whole buffer, the same on
  // every chip, and the class maps one chip's shard: copied out of it first (finish gathers results back).
  SmallVector<Type> ins(inputTypes), outs(outputTypes);
  bool unsplit = false;
  auto whole = [&](Type shard, int64_t arg, bool result) -> Type {
    unsigned r = llvm::find(raised.written, arg) - raised.written.begin();
    if (result && r == raised.written.size())
      return shard;
    Attribute dim = result ? src.getResultAttr(r, kShardAttr) : src.getArgAttr(arg, kShardAttr);
    auto tensor = dyn_cast<RankedTensorType>(result ? src.getResultTypes()[r] : src.getArgumentTypes()[arg]);
    fopt::HbmTensorType t = tensor ? wholeType(cast<fopt::HbmTensorType>(shard), tensor, dim) : nullptr;
    unsplit |= dim && !t;
    return t ? t : shard;
  };
  for (auto [i, arg] : llvm::enumerate(inputArgs))
    ins[i] = whole(inputTypes[i], arg, false);
  for (auto [i, arg] : llvm::enumerate(outputArgs))
    outs[i] = whole(outputTypes[i], arg, true);
  kernel = func::FuncOp::create(b, src.getLoc(), src.getName(), b.getFunctionType(ins, outs));
  if (unsplit) // no digit boundary of a parameter's layout at its split dim: mapKernel fails the kernel
    kernel->setAttr(kUnsplitAttr, b.getUnitAttr());
  kernel->setAttr(fopt::kChipAttr, b.getI64IntegerAttr(chips));
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
  params = llvm::to_vector_of<Value>(kernel.getArguments());
  Location loc = src.getLoc();
  for (auto [i, t] : llvm::enumerate(ins)) {
    if (t == inputTypes[i])
      continue;
    // Chip c copies block c of the split dim out of its own copy of the whole buffer.
    params[i] = fopt::AllocHbmOp::create(b, loc, inputTypes[i]);
    copy(kernel.getArgument(i), params[i]);
  }
}

void KernelBuilder::finish(ArrayRef<Value> results) {
  SmallVector<Value> returned(results);
  Location loc = raised.func.getLoc();
  for (auto [r, t] : llvm::enumerate(kernel.getResultTypes())) {
    if (t == outputTypes[r])
      continue;
    // All-gather: every chip's block into every chip's copy of the whole buffer (a DMA across chips).
    returned[r] = fopt::AllocHbmOp::create(b, loc, t);
    copy(results[r], returned[r]);
  }
  // A split buffer updated in place: its block back into every chip's copy.
  for (auto [i, t] : llvm::enumerate(kernel.getArgumentTypes()))
    if (t != inputTypes[i] && inoutParams[i])
      copy(params[i], kernel.getArgument(i));
  SmallVector<Attribute> decls;
  for (auto &[name, size] : axes)
    decls.push_back(b.getArrayAttr({b.getStringAttr(name), b.getI64IntegerAttr(size)}));
  kernel->setAttr(fopt::kAxesAttr, b.getArrayAttr(decls));
  func::ReturnOp::create(b, loc, returned);
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
/// fp8 tensors and scalars become f32, and the conversions between them vanish (a truncation to fp8 arrives exact:
/// roundToF8 rounded its operand before any kernel was mapped).
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
                                    rowwiseClass(), tiledClass(), rowwisePaddedClass()};
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

/// furiosa-opt's memory grows with the elements the reads of a Vector Engine reduction are staged to when they are
/// replicated (along output axes the reduction's result is broadcast over): the corpus's steps stage up to 40M and
/// build (26.6M took over 16 GB); a 155M step of attention at 197 tokens took over 50 GB. A mapping that replicates
/// more does not map the kernel (a step candidate splits instead).
static constexpr int64_t kMaxBroadcast = 64 << 20;

bool mapping::mapsOnCopy(func::FuncOp func, const OpClass &c, unsigned maxDmas) {
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
  unsigned dmas = 0;
  int64_t broadcast = 0;
  if (kernel)
    kernel.walk([&](Operation *op) {
      dmas += isa<fopt::StageOp, fopt::ReadOp, fopt::ToDmOp, fopt::ViewToDmOp, fopt::ToHbmOp, fopt::DmToHbmViewOp,
                  fopt::HbmToHbmViewOp, fopt::GatherOp>(op);
      // A read inside a Vector Engine reduction ("rf") staged larger than its buffer: replicated.
      if (auto stage = dyn_cast<fopt::StageOp>(op); stage && stage.getSpec().get("rf")) {
        int64_t staged = stage.getType().getElement().size();
        if (staged > stage.getSource().getType().getElement().size())
          broadcast += staged;
      }
    });
  return kernel && !kernel->hasAttr(kUnsplitAttr) && kernel.getNumArguments() <= kMaxParams && dmas <= maxDmas &&
         broadcast <= kMaxBroadcast && succeeded(verify(kernel));
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
  auto mapped = module.lookupSymbol<func::FuncOp>(name);
  if (mapped && mapped.getNumArguments() > kMaxParams)
    return emitError(loc) << chosen->name << ": " << mapped.getNumArguments() << " parameters, more than the "
                          << kMaxParams << " a launch passes";
  if (mapped && mapped->hasAttr(kUnsplitAttr))
    return emitError(loc) << chosen->name << ": a buffer's layout has no digit boundary at the dim split over chips";
  return success();
}

//===----------------------------------------------------------------------===//
// furiosa-shard: one parallel dim of a kernel split over the chips
//===----------------------------------------------------------------------===//

namespace {

/// A parallel dim split over the chips: the tensor values it reaches with the dim of each the chips split, and the
/// linalg ops computing them with the loop the chips split.
struct Split {
  DenseMap<Value, int64_t> dims;
  DenseMap<Operation *, unsigned> loops;
};

/// The dim of `slice`'s result that source dim `d` is, when the slice keeps that dim whole.
static std::optional<int64_t> sliceResultDim(tensor::ExtractSliceOp slice, int64_t d) {
  if (slice.getStaticOffsets()[d] != 0 || slice.getStaticStrides()[d] != 1 ||
      slice.getStaticSizes()[d] != slice.getSourceType().getDimSize(d))
    return std::nullopt;
  llvm::SmallBitVector dropped = slice.getDroppedDims();
  if (dropped.test(d))
    return std::nullopt;
  int64_t r = d;
  for (int64_t i = 0; i < d; ++i)
    r -= dropped.test(i);
  return r;
}

/// The source dim of `slice` that result dim `d` is, when the slice keeps it whole.
static std::optional<int64_t> sliceSourceDim(tensor::ExtractSliceOp slice, int64_t d) {
  for (int64_t i = 0, e = slice.getSourceType().getRank(); i < e; ++i)
    if (std::optional<int64_t> r = sliceResultDim(slice, i); r && *r == d)
      return i;
  return std::nullopt;
}

/// Whether `pad` pads nothing along dim `d`.
static bool unpadded(tensor::PadOp pad, int64_t d) { return !pad.getStaticLow()[d] && !pad.getStaticHigh()[d]; }

/// Splits dim `dim` of `seed` over `n` chips through `func`: each linalg op computing or reading a split value runs
/// that loop split, so every value the split reaches is computed block by block, one block per chip, from blocks of
/// its operands along the same loop (or their whole value where the loop does not index them). A slice that keeps
/// the dim whole (maxpool's window reads) or a pad that pads other dims is a block of its source's block. Fails
/// where a block would not be a box of one dim of every operand (a window, the loop's index used as data, another
/// op outside linalg), or would not divide.
LogicalResult propagate(func::FuncOp func, Value seed, int64_t dim, int64_t n, Split &s) {
  SmallVector<std::pair<Value, int64_t>> work;
  SmallVector<Value> whole; // read whole by a split op
  auto mark = [&](Value v, int64_t d) -> LogicalResult {
    auto [it, fresh] = s.dims.try_emplace(v, d);
    if (!fresh)
      return success(it->second == d);
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || !t.hasStaticShape() || t.getDimSize(d) % n)
      return failure();
    work.push_back({v, d});
    return success();
  };
  auto at = [](AffineMap map, unsigned l) -> std::optional<int64_t> {
    std::optional<int64_t> pos;
    for (auto [i, e] : llvm::enumerate(map.getResults())) {
      if (auto d = dyn_cast<AffineDimExpr>(e); d && d.getPosition() == l) {
        if (pos)
          return -1;
        pos = i;
      } else if (e.isFunctionOfDim(l)) {
        return -1;
      }
    }
    return pos;
  };
  auto splitOp = [&](linalg::LinalgOp op, unsigned l) -> LogicalResult {
    auto [it, fresh] = s.loops.try_emplace(op, l);
    if (!fresh)
      return success(it->second == l);
    if (op.getIteratorTypesArray()[l] != utils::IteratorType::parallel)
      return failure();
    bool local = !op->walk([&](Operation *inner) {
                      auto index = dyn_cast<linalg::IndexOp>(inner);
                      return (index && index.getDim() == l) || isa<tensor::ExtractOp>(inner) ? WalkResult::interrupt()
                                                                                              : WalkResult::advance();
                    }).wasInterrupted();
    if (!local)
      return failure();
    for (OpOperand &o : op->getOpOperands()) {
      if (!isa<RankedTensorType>(o.get().getType()))
        continue;
      std::optional<int64_t> p = at(op.getMatchingIndexingMap(&o), l);
      if (!p)
        whole.push_back(o.get());
      else if (*p < 0 || failed(mark(o.get(), *p)))
        return failure();
    }
    for (OpResult r : op->getResults()) {
      std::optional<int64_t> p = at(op.getIndexingMapMatchingResult(r), l);
      if (!p || *p < 0 || failed(mark(r, *p)))
        return failure();
    }
    return success();
  };
  auto loopOf = [](AffineMap map, int64_t d) -> std::optional<unsigned> {
    if (auto e = dyn_cast<AffineDimExpr>(map.getResult(d)))
      return e.getPosition();
    return std::nullopt;
  };
  if (failed(mark(seed, dim)))
    return failure();
  while (!work.empty()) {
    auto [v, d] = work.pop_back_val();
    // Its producer computes it split.
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner() != &func.getBody().front())
        return failure();
    } else if (auto op = v.getDefiningOp<linalg::LinalgOp>()) {
      std::optional<unsigned> l = loopOf(op.getIndexingMapMatchingResult(cast<OpResult>(v)), d);
      if (!l || failed(splitOp(op, *l)))
        return failure();
    } else if (auto slice = v.getDefiningOp<tensor::ExtractSliceOp>()) {
      std::optional<int64_t> sd = sliceSourceDim(slice, d);
      if (!sd || failed(mark(slice.getSource(), *sd)))
        return failure();
    } else if (auto pad = v.getDefiningOp<tensor::PadOp>()) {
      if (!unpadded(pad, d) || failed(mark(pad.getSource(), d)))
        return failure();
    } else if (!v.getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }
    // Its readers read it split.
    for (OpOperand &use : v.getUses()) {
      if (isa<func::ReturnOp>(use.getOwner()))
        continue;
      if (auto slice = dyn_cast<tensor::ExtractSliceOp>(use.getOwner())) {
        std::optional<int64_t> rd = sliceResultDim(slice, d);
        if (!rd || failed(mark(slice.getResult(), *rd)))
          return failure();
        continue;
      }
      if (auto pad = dyn_cast<tensor::PadOp>(use.getOwner())) {
        if (use.getOperandNumber() != 0 || !unpadded(pad, d) || failed(mark(pad.getResult(), d)))
          return failure();
        continue;
      }
      auto op = dyn_cast<linalg::LinalgOp>(use.getOwner());
      std::optional<unsigned> l = op ? loopOf(op.getMatchingIndexingMap(&use), d) : std::nullopt;
      if (!l || failed(splitOp(op, *l)))
        return failure();
    }
  }
  return success(llvm::none_of(whole, [&](Value v) { return s.dims.count(v); }));
}

/// Gives every value `s` reaches its block's type (and a written buffer its result's: the kernel does not read it)
/// and marks the split arguments and results with their dim.
void applySplit(func::FuncOp func, const Split &s, int64_t n) {
  Builder b(func.getContext());
  // A split slice's sizes are its result's (the source dim found before the types change).
  SmallVector<std::pair<tensor::ExtractSliceOp, int64_t>> slices;
  for (auto [v, d] : s.dims)
    if (auto slice = v.getDefiningOp<tensor::ExtractSliceOp>())
      slices.push_back({slice, *sliceSourceDim(slice, d)});
  for (auto [v, d] : s.dims) {
    auto t = cast<RankedTensorType>(v.getType());
    SmallVector<int64_t> shape(t.getShape());
    shape[d] /= n;
    v.setType(t.clone(shape));
  }
  for (auto [slice, sd] : slices) {
    SmallVector<int64_t> sizes(slice.getStaticSizes());
    sizes[sd] /= n;
    slice.setStaticSizes(sizes);
  }
  Operation *ret = func.getBody().front().getTerminator();
  for (BlockArgument arg : func.getArguments())
    if (auto it = s.dims.find(arg); it != s.dims.end())
      func.setArgAttr(arg.getArgNumber(), kShardAttr, b.getI64IntegerAttr(it->second));
  for (OpOperand &r : ret->getOpOperands())
    if (auto it = s.dims.find(r.get()); it != s.dims.end())
      func.setResultAttr(r.getOperandNumber(), kShardAttr, b.getI64IntegerAttr(it->second));
  // A written buffer is its result's block too (a class may update it in place).
  for (auto [r, a] : llvm::enumerate(writtenArgs(func))) {
    func.getArgument(a).setType(ret->getOperand(r).getType());
    if (Attribute dim = func.getResultAttr(r, kShardAttr))
      func.setArgAttr(a, kShardAttr, dim);
  }
  func.setType(FunctionType::get(func.getContext(), func.getBody().getArgumentTypes(), ret->getOperandTypes()));
}

/// Splits the outermost parallel dim of a result that reaches the kernel's other values as blocks and still maps
/// split (an op class maps the block kernel), trying the results in order; a kernel with none runs whole on every
/// chip. A kernel that reads a buffer it writes (in place) is not split.
void shardKernel(func::FuncOp func, int64_t n, function_ref<LogicalResult(ModuleOp)> maps) {
  for (int64_t a : writtenArgs(func))
    if (!func.getArgument(a).use_empty())
      return;
  Operation *ret = func.getBody().front().getTerminator();
  // FURIOSA_DEBUG=1 says why a dim does not split.
  auto why = [&](unsigned r, int64_t d, StringRef reason) {
    if (getenv("FURIOSA_DEBUG"))
      llvm::errs() << "shard: " << func.getName() << ": result " << r << " dim " << d << ": " << reason << "\n";
  };
  for (auto [i, r] : llvm::enumerate(ret->getOperands())) {
    auto t = dyn_cast<RankedTensorType>(r.getType());
    for (int64_t d = 0; t && d < t.getRank(); ++d) {
      Split s;
      if (failed(propagate(func, r, d, n, s))) {
        why(i, d, "not a box of one dim of every operand");
        continue;
      }
      // On a copy in a module like this one, nested in it while it maps: the block kernel must map all the way to
      // fopt (a class's shape rules hold for it too).
      ModuleOp parent = func->getParentOfType<ModuleOp>();
      OpBuilder b = OpBuilder::atBlockEnd(parent.getBody());
      auto scratch = ModuleOp::create(b, func.getLoc());
      for (NamedAttribute attr : parent->getAttrs())
        scratch->setAttr(attr.getName(), attr.getValue());
      IRMapping map;
      auto copy = cast<func::FuncOp>(func->clone(map));
      scratch.push_back(copy);
      Split c;
      for (auto [v, dv] : s.dims)
        c.dims[map.lookup(v)] = dv;
      applySplit(copy, c, n);
      LogicalResult mapped = failure();
      {
        ScopedDiagnosticHandler quiet(func.getContext(), [&](Diagnostic &diag) {
          if (getenv("FURIOSA_DEBUG"))
            llvm::errs() << "shard: " << func.getName() << ": " << diag << "\n";
          return success();
        });
        mapped = maps(scratch);
      }
      scratch.erase();
      if (failed(mapped)) {
        why(i, d, "the block does not map");
        continue;
      }
      applySplit(func, s, n);
      return;
    }
  }
}

struct ShardPass : PassWrapper<ShardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardPass)

  StringRef getArgument() const override { return "furiosa-shard"; }
  StringRef getDescription() const override {
    return "Split one parallel dim of each kernel over the module's furiosa.chips chips";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerCodegenDialects(registry);
    registry.insert<affine::AffineDialect>(); // furiosa-tile's, run on the trial copies
  }

  void runOnOperation() override {
    int64_t n = chipCount(getOperation());
    if (n < 2)
      return;
    auto maps = [&](ModuleOp m) {
      OpPassManager pm(ModuleOp::getOperationName());
      pm.addPass(createTilePass());
      pm.addPass(createLinalgToFoptPass());
      pm.addPass(createDmaPass());
      pm.addPass(createVeFusePass());
      return runPipeline(pm, m);
    };
    for (func::FuncOp func : llvm::to_vector(getOperation().getOps<func::FuncOp>()))
      if (!func.isExternal() && !func->hasAttr(fopt::kChipAttr) && func->hasAttr("furiosa.written"))
        shardKernel(func, n, maps);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createShardPass() { return std::make_unique<ShardPass>(); }

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

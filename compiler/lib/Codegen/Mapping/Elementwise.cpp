// Op class: an elementwise linalg.generic over f32 (identity maps, all parallel, one or two inputs), whose
// body is one Vector Engine op.
//
// Layout: the buffers, flattened to one axis A, split exactly into packets, slices of cluster 0 and time (see
// map()). Two inputs stream interleaved through one
// pass (axis I, innermost in time) and meet in the clip stage; one input meets an immediate in the clip
// stage, or a float op in the 4-wide float region between narrow_split and widen_concat.

#include "Mapping.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;
using namespace mlir::furiosa::mapping;

namespace {

enum class Kind { ClipZip, ClipImm, FpBinaryImm, FpUnary };

struct Body {
  Kind kind;
  std::string op;              // the furiosa-opt enum case
  float imm = 0;               // ClipImm, FpBinaryImm
  SmallVector<unsigned> inputs; // generic input indices, in operand order
};

/// The one linalg.generic whose results the raised function returns.
linalg::GenericOp theGeneric(func::FuncOp func) {
  linalg::GenericOp found;
  for (auto generic : func.getOps<linalg::GenericOp>()) {
    if (found)
      return {};
    found = generic;
  }
  return found;
}

bool isElementwise(linalg::GenericOp generic) {
  if (generic.getNumDpsInits() != 1 || generic.getNumDpsInputs() < 1 || generic.getNumDpsInputs() > 2)
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType t) { return t == utils::IteratorType::parallel; }))
    return false;
  if (!llvm::all_of(generic.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
    return false;
  for (Value operand : generic->getOperands())
    if (!cast<RankedTensorType>(operand.getType()).getElementType().isF32())
      return false;
  return true;
}

bool match(RaisedKernel &kernel) {
  linalg::GenericOp generic = theGeneric(kernel.func);
  return generic && isElementwise(generic);
}

FailureOr<Body> classify(linalg::GenericOp generic, std::string &why) {
  Block &block = generic.getRegion().front();
  auto yield = cast<linalg::YieldOp>(block.getTerminator());
  // Constants are immediates, not Vector Engine ops.
  SmallVector<Operation *> ops;
  for (Operation &op : block.without_terminator())
    if (!op.hasTrait<OpTrait::ConstantLike>())
      ops.push_back(&op);
  if (ops.size() != 1 || yield.getNumOperands() != 1) {
    why = "the body must be one op besides constants (Vector Engine pass splitting is not implemented yet)";
    return failure();
  }
  Operation *op = ops.front();
  if (yield.getOperand(0) != op->getResult(0)) {
    why = "the body must yield its op's result";
    return failure();
  }
  unsigned numInputs = generic.getNumDpsInputs();
  auto inputIndex = [&](Value v) -> int {
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && arg.getOwner() == &block && arg.getArgNumber() < numInputs ? int(arg.getArgNumber()) : -1;
  };
  auto constant = [](Value v) -> std::optional<float> {
    llvm::APFloat value(0.0f);
    if (matchPattern(v, m_ConstantFloat(&value))) {
      bool lost;
      value.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven, &lost);
      return value.convertToFloat();
    }
    return std::nullopt;
  };

  // Binary ops: (op name, clip stage?, commutative?).
  struct BinaryInfo {
    StringRef name;
    bool clip;
    bool commutative;
  };
  std::optional<BinaryInfo> binary = llvm::TypeSwitch<Operation *, std::optional<BinaryInfo>>(op)
                                         .Case<arith::AddFOp>([](auto) { return BinaryInfo{"Add", true, true}; })
                                         .Case<arith::MaximumFOp, arith::MaxNumFOp>(
                                             [](auto) { return BinaryInfo{"Max", true, true}; })
                                         .Case<arith::MinimumFOp, arith::MinNumFOp>(
                                             [](auto) { return BinaryInfo{"Min", true, true}; })
                                         .Case<arith::SubFOp>([](auto) { return BinaryInfo{"SubF", false, false}; })
                                         .Case<arith::MulFOp>(
                                             [](auto) { return BinaryInfo{"MulF(FpMulAlu::Mul0)", false, true}; })
                                         .Case<arith::DivFOp>([](auto) { return BinaryInfo{"DivF", false, false}; })
                                         .Default([](auto) { return std::nullopt; });
  if (binary) {
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    int l = inputIndex(lhs), r = inputIndex(rhs);
    if (l >= 0 && r >= 0 && l != r) {
      if (!binary->clip) {
        why = "a two-tensor " + binary->name.str() + " needs a VRF operand, which is not mapped yet";
        return failure();
      }
      return Body{Kind::ClipZip, binary->name.str(), 0, {unsigned(l), unsigned(r)}};
    }
    std::optional<float> c;
    int x = -1;
    if (l >= 0 && (c = constant(rhs)))
      x = l;
    else if (r >= 0 && (c = constant(lhs)) && binary->commutative)
      x = r;
    if (x >= 0)
      return Body{binary->clip ? Kind::ClipImm : Kind::FpBinaryImm, binary->name.str(), *c, {unsigned(x)}};
    why = "a binary body needs two inputs or an input and a constant (the input first unless commutative)";
    return failure();
  }

  std::optional<StringRef> unary = llvm::TypeSwitch<Operation *, std::optional<StringRef>>(op)
                                       .Case<math::ExpOp>([](auto) { return StringRef("Exp"); })
                                       .Case<math::SqrtOp>([](auto) { return StringRef("Sqrt"); })
                                       .Case<math::TanhOp>([](auto) { return StringRef("Tanh"); })
                                       .Case<math::LogOp>([](auto) { return StringRef("Log"); })
                                       .Case<math::ErfOp>([](auto) { return StringRef("Erf"); })
                                       .Case<math::SinOp>([](auto) { return StringRef("Sin"); })
                                       .Case<math::CosOp>([](auto) { return StringRef("Cos"); })
                                       .Default([](auto) { return std::nullopt; });
  if (unary && inputIndex(op->getOperand(0)) >= 0)
    return Body{Kind::FpUnary, unary->str(), 0, {unsigned(inputIndex(op->getOperand(0)))}};
  if (auto neg = dyn_cast<arith::NegFOp>(op); neg && inputIndex(neg.getOperand()) >= 0)
    return Body{Kind::FpBinaryImm, "MulF(FpMulAlu::Mul0)", -1.0f, {unsigned(inputIndex(neg.getOperand()))}};

  why = "no Vector Engine op for '" + op->getName().getStringRef().str() + "'";
  return failure();
}

/// The pointer-argument index behind a generic operand: it must be the raised function's argument.
FailureOr<int64_t> ptrArgOf(Value v, func::FuncOp func) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || arg.getOwner() != &func.getBody().front())
    return failure();
  return int64_t(arg.getArgNumber());
}

LogicalResult map(RaisedKernel &kernel) {
  func::FuncOp func = kernel.func;
  linalg::GenericOp generic = theGeneric(func);
  std::string why;
  FailureOr<Body> body = classify(generic, why);
  if (failed(body))
    return func.emitError("elementwise: ") << why;

  auto tensorType = cast<RankedTensorType>(generic.getDpsInits()[0].getType());
  if (tensorType.getRank() < 1)
    return func.emitError("elementwise: rank-0 tensors are not mapped");
  if (!generic.getRegion().front().getArgument(generic.getNumDpsInputs()).use_empty())
    return func.emitError("elementwise: the body reads the output (an in-place update)");
  // The init is the written buffer, or a tensor.empty when the generic overwrites all of it (identity
  // maps over the whole shape, which isElementwise guarantees).
  if (kernel.written.size() != 1)
    return func.emitError("elementwise: expected one written buffer, got ") << kernel.written.size();
  Value init = generic.getDpsInits()[0];
  FailureOr<int64_t> outArg = ptrArgOf(init, func);
  if (succeeded(outArg) ? *outArg != kernel.written[0] : !init.getDefiningOp<tensor::EmptyOp>())
    return func.emitError("elementwise: the generic must write the written buffer");
  outArg = kernel.written[0];
  if (cast<RankedTensorType>(func.getArgumentTypes()[*outArg]) != cast<RankedTensorType>(init.getType()))
    return func.emitError("elementwise: the written buffer's type differs from the generic's output");
  auto returned = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  if (returned.getNumOperands() != 1 || returned.getOperand(0) != generic.getResult(0))
    return func.emitError("elementwise: the function must return the generic's result");

  KernelBuilder kb(kernel);
  MLIRContext *ctx = kb.getContext();
  Type f32 = Float32Type::get(ctx);
  // Elementwise over identity maps reads every buffer in the same row-major order, so the kernel declares
  // each buffer as one flat axis of all its elements (the host relabels buffers by size, so the logical
  // shape does not reach the kernel); splitting a multi-axis pair across its rows is not expressible.
  int64_t n = tensorType.getNumElements();
  MapAttr a = kb.axis("A", n);

  // Each input is a distinct pointer argument, declared in generic operand order.
  SmallVector<int64_t> inputArgs;
  for (unsigned i : body->inputs) {
    FailureOr<int64_t> arg = ptrArgOf(generic.getDpsInputs()[i], func);
    if (failed(arg))
      return func.emitError("elementwise: input ") << i << " is not a buffer argument";
    inputArgs.push_back(*arg);
  }
  SmallVector<unsigned> params;
  for (int64_t arg : inputArgs)
    params.push_back(kb.input(arg, f32, a));
  kb.output(*outArg, f32, a);
  kb.beginBody();
  SmallVector<Value> hbms;
  for (unsigned p : params)
    hbms.push_back(kb.param(p));
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();

  // Layout, exact (DMA writes to DM must stay 8-byte aligned, so no element is padded): the packet is the
  // largest of 8 / 4 / 2 elements dividing n (collect pads it to one flit); the packets split into S slices,
  // the largest divisor of their count up to 256 (the slice digit padded to 256), and T time steps; each
  // slice holds one contiguous run of T packets.
  constexpr int64_t kFlit = 8, kSlices = 256;
  int64_t p = kFlit;
  while (p > 1 && n % p)
    p /= 2;
  if (p < 2)
    return func.emitError("elementwise: an odd number of f32 elements (") << n << ") breaks the 8-byte DMA alignment";
  int64_t q = n / p, slices = std::min(q, kSlices);
  while (q % slices)
    --slices;
  int64_t steps = q / slices;
  MapAttr slice = a.stride(steps * p);
  if (slices < kSlices)
    slice = slice.padTo(kSlices);
  MapAttr element = slices == 1 ? a : a.modulo(steps * p);
  MapAttr outer = steps == 1 ? kb.one() : element.stride(p);
  MapAttr packet = p == n ? a : a.modulo(p);
  MapAttr flit = p < kFlit ? packet.padTo(kFlit) : packet;
  DmTensorType dmType = kb.dmType(f32, slice, element);

  SmallVector<Value> dms;
  for (Value hbm : hbms)
    dms.push_back(ToDmOp::create(b, loc, dmType, hbm));

  auto stream = [&](MapAttr time, MapAttr pkt) { return kb.streamType(f32, slice, time, pkt); };
  Value s;
  if (body->kind == Kind::ClipZip) {
    MapAttr i = kb.axis("I", 2);
    MapAttr time = MapAttr::pair(ctx, {outer, i});
    s = BeginInterleavedOp::create(b, loc, stream(i, element), dms[0], dms[1], ContextKind::Main, i);
    s = FetchOp::create(b, loc, stream(time, packet), s);
    s = CollectOp::create(b, loc, stream(time, flit), s);
    s = VectorInitOp::create(b, loc, stream(time, flit), s);
    s = VectorIntraSliceUnzipOp::create(b, loc, stream(outer, flit), s, i);
    s = VectorClipZipOp::create(b, loc, stream(outer, flit), s, b.getStringAttr(body->op));
  } else {
    s = BeginOp::create(b, loc, stream(kb.one(), element), dms[0], ContextKind::Main);
    s = FetchOp::create(b, loc, stream(outer, packet), s);
    s = CollectOp::create(b, loc, stream(outer, flit), s);
    s = VectorInitOp::create(b, loc, stream(outer, flit), s);
    s = VectorIntraSliceTagOp::create(b, loc, stream(outer, flit), s);
    if (body->kind == Kind::ClipImm) {
      s = VectorClipOp::create(b, loc, stream(outer, flit), s, b.getStringAttr(body->op),
                               b.getF32FloatAttr(body->imm), Value(), UnitAttr());
    } else {
      // The float region is 4 lanes wide: an 8-element packet splits its top digit into time and back; a
      // 4-element one is trimmed from its padded flit and padded back.
      if (p != 8 && p != 4)
        return func.emitError("elementwise: float ops need 4- or 8-element packets; ") << n << " elements give " << p;
      MapAttr narrowTime = p == 8 ? MapAttr::pair(ctx, {outer, packet.stride(4)}) : outer;
      MapAttr narrowPacket = p == 8 ? a.modulo(4) : packet;
      if (p == 8)
        s = VectorNarrowSplitOp::create(b, loc, stream(narrowTime, narrowPacket), s);
      else
        s = VectorNarrowTrimOp::create(b, loc, stream(narrowTime, narrowPacket), s);
      if (body->kind == Kind::FpUnary)
        s = VectorFpUnaryOp::create(b, loc, stream(narrowTime, narrowPacket), s, b.getStringAttr(body->op));
      else
        s = VectorFpBinaryOp::create(b, loc, stream(narrowTime, narrowPacket), s, b.getStringAttr(body->op),
                                     b.getF32FloatAttr(body->imm), Value(), UnitAttr());
      if (p == 8)
        s = VectorWidenConcatOp::create(b, loc, stream(outer, flit), s);
      else
        s = VectorWidenPadOp::create(b, loc, stream(outer, flit), s);
    }
  }
  s = VectorFinalOp::create(b, loc, stream(outer, flit), s);
  s = CommitTrimOp::create(b, loc, stream(outer, packet), s);
  Value out = CommitOp::create(b, loc, dmType, s);
  Value hbm = ToHbmOp::create(b, loc, kb.hbmType(f32, a), out);
  kb.finish({hbm});
  return success();
}

} // namespace

OpClass mlir::furiosa::mapping::elementwiseClass() { return {"elementwise", match, map}; }

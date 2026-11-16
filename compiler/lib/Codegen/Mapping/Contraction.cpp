// Op class: a contraction y[m, n] = sum_k x[m, k] * w[n, k] over bf16 operands with fp32 accumulation, from a
// zero fill, optionally followed by an elementwise generic that casts the accumulator to the output dtype.
//
// Layout (derived from M, N, K): each slice holds r rows of w in its TRF (r a multiple of the 8 lanes, the
// fewest rows that fit N on at most 256 slices) and the whole x, broadcast. The stream carries x; K splits
// into the 32-element contraction packet and time, accumulated in the tree and the time reducer; the
// accumulator holds M x r/8 values per slice; the lane folder emits n % 8. Limits checked here: TRF capacity
// (r x K bf16 per slice), accumulator slots, K a multiple of 32.

#include "Mapping.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;
using namespace mlir::furiosa::mapping;

namespace {

constexpr int64_t kSlices = 256, kLanes = 8, kTrfBytes = 64 * 1024, kAccumulatorSlots = 128;

struct Contraction {
  linalg::GenericOp contract;
  linalg::GenericOp epilogue; // casts the accumulator; may be null when the output is the f32 accumulator
  Value x, w;                 // function arguments: x [M, K], w [N, K]
  int64_t m, n, k;
};

bool isDims(AffineMap map, ArrayRef<unsigned> dims) {
  if (map.getNumResults() != dims.size())
    return false;
  for (auto [e, d] : llvm::zip(map.getResults(), dims)) {
    auto de = dyn_cast<AffineDimExpr>(e);
    if (!de || de.getPosition() != d)
      return false;
  }
  return true;
}

/// Strips an extf from a body value; returns the block argument it reads, or null.
BlockArgument operandArg(Value v) {
  if (auto ext = v.getDefiningOp<arith::ExtFOp>())
    v = ext.getIn();
  return dyn_cast<BlockArgument>(v);
}

FailureOr<Contraction> find(func::FuncOp func, std::string &why) {
  SmallVector<linalg::GenericOp> generics(func.getOps<linalg::GenericOp>());
  Contraction c;
  for (linalg::GenericOp g : generics) {
    auto iters = g.getIteratorTypesArray();
    if (iters.size() == 3 && iters[2] == utils::IteratorType::reduction)
      c.contract = g;
  }
  if (!c.contract) {
    why = "no (parallel, parallel, reduction) generic";
    return failure();
  }
  linalg::GenericOp g = c.contract;
  auto iters = g.getIteratorTypesArray();
  if (iters[0] != utils::IteratorType::parallel || iters[1] != utils::IteratorType::parallel ||
      g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1) {
    why = "the contraction generic must be (parallel, parallel, reduction) with two inputs";
    return failure();
  }
  SmallVector<AffineMap> maps = g.getIndexingMapsArray();
  if (!isDims(maps[0], {0, 2}) || !isDims(maps[1], {1, 2}) || !isDims(maps[2], {0, 1})) {
    why = "maps must be x(m, k), w(n, k) -> y(m, n)";
    return failure();
  }
  // Body: acc + ext(a) * ext(b).
  Block &body = g.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  auto add = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add) {
    why = "the contraction body must accumulate with addf";
    return failure();
  }
  Value acc = body.getArgument(2);
  Value prod = add.getLhs() == acc ? add.getRhs() : add.getRhs() == acc ? add.getLhs() : Value();
  auto mul = prod ? prod.getDefiningOp<arith::MulFOp>() : arith::MulFOp();
  BlockArgument a = mul ? operandArg(mul.getLhs()) : BlockArgument(), b = mul ? operandArg(mul.getRhs()) : BlockArgument();
  if (!mul || !a || !b || a.getOwner() != &body || b.getOwner() != &body || a.getArgNumber() > 1 ||
      b.getArgNumber() > 1 || a == b) {
    why = "the contraction body must be acc + x * w";
    return failure();
  }
  for (unsigned i = 0; i < 2; ++i) {
    auto t = cast<RankedTensorType>(g.getDpsInputs()[i].getType());
    if (!t.getElementType().isBF16()) {
      why = "contraction operands must be bf16 (the NPU has no fp32/fp16 contraction; fp32 operands need the "
            "bf16 dtype policy, not implemented yet)";
      return failure();
    }
  }
  if (!cast<RankedTensorType>(g.getResultTypes()[0]).getElementType().isF32()) {
    why = "the accumulator must be f32";
    return failure();
  }
  auto fill = g.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat())) {
    why = "the accumulator must start from a zero fill";
    return failure();
  }
  auto isArg = [&](Value v) {
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && arg.getOwner() == &func.getBody().front();
  };
  c.x = g.getDpsInputs()[0];
  c.w = g.getDpsInputs()[1];
  if (!isArg(c.x) || !isArg(c.w)) {
    why = "both operands must be buffers (kernel arguments)";
    return failure();
  }
  auto xt = cast<RankedTensorType>(c.x.getType()), wt = cast<RankedTensorType>(c.w.getType());
  c.m = xt.getDimSize(0);
  c.k = xt.getDimSize(1);
  c.n = wt.getDimSize(0);

  // The epilogue: one elementwise generic reading the accumulator and casting it.
  for (linalg::GenericOp e : generics) {
    if (e == g)
      continue;
    if (c.epilogue || e.getNumDpsInputs() != 1 || e.getDpsInputs()[0] != g.getResult(0)) {
      why = "only a cast epilogue after the contraction is mapped yet";
      return failure();
    }
    Block &eb = e.getRegion().front();
    auto ey = cast<linalg::YieldOp>(eb.getTerminator());
    auto trunc = ey.getOperand(0).getDefiningOp<arith::TruncFOp>();
    if (!trunc || trunc.getIn() != eb.getArgument(0) || !trunc.getType().isBF16() ||
        !llvm::all_of(e.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); })) {
      why = "only a truncf-to-bf16 epilogue is mapped yet";
      return failure();
    }
    c.epilogue = e;
  }
  return c;
}

bool match(RaisedKernel &kernel) {
  std::string why;
  return succeeded(find(kernel.func, why));
}

LogicalResult map(RaisedKernel &kernel) {
  func::FuncOp func = kernel.func;
  std::string why;
  FailureOr<Contraction> found = find(func, why);
  if (failed(found))
    return func.emitError("contraction: ") << why;
  Contraction c = *found;
  if (kernel.written.size() != 1)
    return func.emitError("contraction: expected one written buffer");
  Value result = c.epilogue ? c.epilogue.getResult(0) : c.contract.getResult(0);
  auto returned = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  if (returned.getNumOperands() != 1 || returned.getOperand(0) != result)
    return func.emitError("contraction: the function must return the contraction's (cast) result");

  // Rows of w per slice: a multiple of the lanes, the fewest that put N on at most 256 slices.
  int64_t r = kLanes;
  while (c.n / r > kSlices || c.n % r)
    r += kLanes;
  if (r > c.n || c.n % r)
    return func.emitError("contraction: N = ") << c.n << " does not split into lanes of " << kLanes;
  if (c.k % 32)
    return func.emitError("contraction: K = ") << c.k << " is not a multiple of the 32-element contraction packet";
  if (r * c.k * 2 > kTrfBytes)
    return func.emitError("contraction: ") << r << " rows of K = " << c.k << " bf16 exceed the 64 KB TRF per slice "
                                            << "(K tiling is not implemented yet)";
  int64_t q = r / kLanes;
  if (c.m * q > kAccumulatorSlots)
    return func.emitError("contraction: M = ") << c.m << " needs " << c.m * q << " accumulator slots, more than "
                                                << kAccumulatorSlots << " (M tiling is not implemented yet)";

  KernelBuilder kb(kernel);
  MLIRContext *ctx = kb.getContext();
  Type bf16 = BFloat16Type::get(ctx), f32 = Float32Type::get(ctx);
  MapAttr M = kb.axis("M", c.m), N = kb.axis("N", c.n), K = kb.axis("K", c.k);
  auto pair = [&](ArrayRef<MapAttr> parts) {
    SmallVector<MapAttr> kept;
    for (MapAttr p : parts)
      if (p.size() != 1)
        kept.push_back(p);
    if (kept.empty())
      return kb.one();
    return kept.size() == 1 ? kept[0] : MapAttr::pair(ctx, kept);
  };
  MapAttr nRows = N.modulo(r), nLane = N.modulo(kLanes), nGroup = q > 1 ? N.stride(kLanes).modulo(q) : kb.one();
  MapAttr slice = N.stride(r);
  if (c.n / r < kSlices)
    slice = slice.padTo(kSlices);
  Type outType = c.epilogue ? bf16 : f32;

  int64_t xArg = cast<BlockArgument>(c.x).getArgNumber(), wArg = cast<BlockArgument>(c.w).getArgNumber();
  unsigned xp = kb.input(xArg, bf16, pair({M, K}));
  unsigned wp = kb.input(wArg, bf16, pair({N, K}));
  kb.output(kernel.written[0], outType, pair({M, N}));
  kb.beginBody();
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();

  Value xDm = ToDmOp::create(b, loc, kb.dmType(bf16, slice, pair({M, K})), kb.param(xp));
  Value wDm = ToDmOp::create(b, loc, kb.dmType(bf16, slice, pair({nRows, K})), kb.param(wp));
  auto stream = [&](Type et, MapAttr time, MapAttr packet) { return kb.streamType(et, slice, time, packet); };

  // The TRF: lane n % 8, element [n / 8 % q, k].
  MapAttr wTime = pair({nLane, nGroup, K.stride(16)});
  Value s = BeginOp::create(b, loc, stream(bf16, kb.one(), pair({nRows, K})), wDm, ContextKind::Sub);
  s = FetchOp::create(b, loc, stream(bf16, wTime, K.modulo(16)), s);
  s = CollectOp::create(b, loc, stream(bf16, wTime, K.modulo(16)), s);
  Value trf = ToTrfOp::create(
      b, loc, TrfTensorType::get(ctx, bf16, kb.chip(), kb.cluster(), slice, nLane, pair({nGroup, K})), s);

  MapAttr xTime = pair({M, K.stride(16)});
  s = BeginOp::create(b, loc, stream(bf16, kb.one(), pair({M, K})), xDm, ContextKind::Main);
  s = FetchOp::create(b, loc, stream(bf16, xTime, K.modulo(16)), s);
  s = CollectOp::create(b, loc, stream(bf16, xTime, K.modulo(16)), s);
  s = ContractOuterOp::create(b, loc, stream(bf16, pair({M, K.stride(32), nGroup}), K.modulo(32)), s, trf);
  s = ContractPacketOp::create(b, loc, stream(f32, pair({M, K.stride(32), nGroup}), kb.one()), s);
  s = ContractTimeOp::create(b, loc, stream(f32, pair({M, nGroup}), kb.one()), s);
  s = ContractLaneOp::create(b, loc, stream(f32, pair({M, nGroup}), nLane), s, LaneMode::Interleaved);
  MapAttr outPacket = nLane;
  if (c.epilogue)
    s = CastOp::create(b, loc, stream(bf16, pair({M, nGroup}), nLane.padTo(16)), s);
  s = CommitTrimOp::create(b, loc, stream(outType, pair({M, nGroup}), outPacket), s);
  Value y = CommitOp::create(b, loc, kb.dmType(outType, slice, pair({M, nRows})), s);
  Value hbm = ToHbmOp::create(b, loc, kb.hbmType(outType, pair({M, N})), y);
  kb.finish({hbm});
  return success();
}

} // namespace

OpClass mlir::furiosa::mapping::contractionClass() { return {"contraction", match, map}; }

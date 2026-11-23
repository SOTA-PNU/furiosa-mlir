// Op class: a row gather, out[i.., p..] = table[index[i..] * S + offset(p..)]. One linalg.generic reads the
// table with tensor.extract at an address that is a data-dependent row (an integer input times the row
// stride S) plus the position within the row, which its payload dims run over row-major, whole rows. The
// clamp the raising puts on every address only guards lanes Triton masks: a gather reads whole rows of
// in-range indices (an out-of-range index is undefined behavior in the source program too), so it is
// dropped.
//
// Lowering (furiosa-opt's indirect DMA): the index buffer goes to DM, the Vector Engine scales it to byte
// offsets (MulInt by the row's bytes), the offsets go back to HBM, and one gather DMA reads a row per
// offset into DM, rows across slices, which a DMA writes to the output.

#include "Mapping.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <map>

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;
using namespace mlir::furiosa::mapping;

namespace {

constexpr int64_t kSlices = 256, kDmBudget = 384 * 1024;

/// An index-valued body expression as constant + sum(coefficient * linalg.index) + sum(coefficient * input).
struct Linear {
  int64_t constant = 0;
  std::map<unsigned, int64_t> dims;
  std::map<unsigned, int64_t> inputs; // body block argument number -> coefficient

  Linear scaled(int64_t k) const {
    Linear r;
    r.constant = constant * k;
    for (auto [d, c] : dims)
      r.dims[d] = c * k;
    for (auto [a, c] : inputs)
      r.inputs[a] = c * k;
    return r;
  }
  Linear plus(const Linear &o, int64_t sign = 1) const {
    Linear r = *this;
    r.constant += sign * o.constant;
    for (auto [d, c] : o.dims)
      r.dims[d] += sign * c;
    for (auto [a, c] : o.inputs)
      r.inputs[a] += sign * c;
    return r;
  }
};

std::optional<int64_t> constantInt(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
      return i.getInt();
  return std::nullopt;
}

/// The linear form of an address computed in `body`; clamps (max/min against a constant) pass through.
FailureOr<Linear> linear(Value v, Block &body) {
  if (std::optional<int64_t> c = constantInt(v)) {
    Linear r;
    r.constant = *c;
    return r;
  }
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getOwner() != &body || !arg.getType().isIntOrIndex())
      return failure();
    Linear r;
    r.inputs[arg.getArgNumber()] = 1;
    return r;
  }
  Operation *op = v.getDefiningOp();
  if (!op || op->getBlock() != &body)
    return failure();
  if (auto index = dyn_cast<linalg::IndexOp>(op)) {
    Linear r;
    r.dims[index.getDim()] = 1;
    return r;
  }
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return linear(op->getOperand(0), body);
  if (isa<arith::MaxSIOp, arith::MinSIOp>(op)) {
    if (constantInt(op->getOperand(1)))
      return linear(op->getOperand(0), body);
    if (constantInt(op->getOperand(0)))
      return linear(op->getOperand(1), body);
    return failure();
  }
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    FailureOr<Linear> a = linear(op->getOperand(0), body), b = linear(op->getOperand(1), body);
    if (failed(a) || failed(b))
      return failure();
    return a->plus(*b, isa<arith::AddIOp>(op) ? 1 : -1);
  }
  if (isa<arith::MulIOp>(op)) {
    for (int side = 0; side < 2; ++side)
      if (std::optional<int64_t> k = constantInt(op->getOperand(side))) {
        FailureOr<Linear> x = linear(op->getOperand(1 - side), body);
        if (failed(x))
          return failure();
        return x->scaled(*k);
      }
  }
  return failure();
}

Value stripReshapes(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op))
      break;
    v = op->getOperand(0);
  }
  return v;
}

BlockArgument funcArg(Value v, func::FuncOp func) {
  auto arg = dyn_cast<BlockArgument>(stripReshapes(v));
  return arg && arg.getOwner() == &func.getBody().front() ? arg : BlockArgument();
}

struct Gather {
  linalg::GenericOp generic;
  BlockArgument table, index; // function arguments
  int64_t rows = 0, rowSize = 0, count = 0; // V, S, T
  Type elementType;
};

FailureOr<Gather> find(RaisedKernel &kernel, std::string &why) {
  func::FuncOp func = kernel.func;
  SmallVector<linalg::GenericOp> generics(func.getOps<linalg::GenericOp>());
  if (generics.size() != 1) {
    why = "a gather is one linalg.generic";
    return failure();
  }
  Gather g;
  g.generic = generics.front();
  Block &body = g.generic.getRegion().front();
  auto extract = cast<linalg::YieldOp>(body.getTerminator()).getOperand(0).getDefiningOp<tensor::ExtractOp>();
  if (!extract || g.generic.getNumDpsInits() != 1) {
    why = "the generic does not yield a tensor.extract";
    return failure();
  }
  g.table = funcArg(extract.getTensor(), func);
  if (!g.table) {
    why = "the gathered tensor is not a buffer";
    return failure();
  }
  // The address over the extracted tensor's row-major shape.
  auto sourceType = cast<RankedTensorType>(extract.getTensor().getType());
  Linear address;
  int64_t stride = 1;
  for (int k = sourceType.getRank() - 1; k >= 0; --k) {
    FailureOr<Linear> part = linear(extract.getIndices()[k], body);
    if (failed(part)) {
      why = "the address is not a linear function of the indices and the inputs";
      return failure();
    }
    address = address.plus(part->scaled(stride));
    stride *= sourceType.getDimSize(k);
  }
  std::map<unsigned, int64_t> dims, inputs;
  for (auto [d, c] : address.dims)
    if (c)
      dims[d] = c;
  for (auto [a, c] : address.inputs)
    if (c)
      inputs[a] = c;
  if (inputs.size() != 1 || inputs.begin()->second <= 0) {
    why = "the address needs exactly one data-dependent row term";
    return failure();
  }
  auto [argNo, rowStride] = *inputs.begin();
  if (argNo >= g.generic.getNumDpsInputs()) {
    why = "the row term reads the accumulator";
    return failure();
  }
  OpOperand *indexOperand = g.generic.getDpsInputOperand(argNo);
  g.index = funcArg(indexOperand->get(), func);
  auto indexType = cast<RankedTensorType>(indexOperand->get().getType());
  if (!g.index || !indexType.getElementType().isInteger(32)) {
    why = "the row index is not an i32 buffer";
    return failure();
  }
  // Index dims: the ones the index operand's map projects onto, in order; payload dims: the rest.
  SmallVector<int64_t> ranges = g.generic.getStaticLoopRanges();
  AffineMap indexMap = g.generic.getMatchingIndexingMap(indexOperand);
  SmallVector<unsigned> indexDims, payloadDims;
  for (auto [i, e] : llvm::enumerate(indexMap.getResults())) {
    // A constant 0 indexes a dimension of one entry.
    if (auto c = dyn_cast<AffineConstantExpr>(e); c && c.getValue() == 0 && indexType.getDimSize(i) == 1)
      continue;
    auto d = dyn_cast<AffineDimExpr>(e);
    if (!d || (!indexDims.empty() && d.getPosition() <= indexDims.back())) {
      why = "the index operand's map is not an ordered projection";
      return failure();
    }
    indexDims.push_back(d.getPosition());
  }
  g.count = 1;
  for (unsigned d : indexDims)
    g.count *= ranges[d];
  if (indexType.getNumElements() != g.count) {
    why = "the index buffer is not exactly the gathered index list";
    return failure();
  }
  // Dims of one iteration select nothing.
  for (unsigned d = 0; d < ranges.size(); ++d)
    if (!llvm::is_contained(indexDims, d) && ranges[d] != 1)
      payloadDims.push_back(d);
  // The payload dims run row-major over a whole row, after the index dims.
  int64_t expect = 1;
  for (unsigned d : llvm::reverse(payloadDims)) {
    if (!dims.count(d) || dims[d] != expect) {
      why = "the payload dims do not run row-major over the row";
      return failure();
    }
    expect *= ranges[d];
  }
  if (dims.size() != payloadDims.size() || address.constant != 0 || expect != rowStride ||
      (!payloadDims.empty() && !indexDims.empty() && payloadDims.front() < indexDims.back())) {
    why = "the gather does not read whole rows in index-major order";
    return failure();
  }
  g.rowSize = rowStride;
  auto tableType = cast<RankedTensorType>(g.table.getType());
  if (tableType.getNumElements() % g.rowSize) {
    why = "the table is not a whole number of rows";
    return failure();
  }
  g.rows = tableType.getNumElements() / g.rowSize;
  g.elementType = tableType.getElementType();
  // The kernel writes the whole output with the gathered rows.
  auto ret = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  if (kernel.written.size() != 1 || ret.getNumOperands() != 1 ||
      stripReshapes(ret.getOperand(0)) != g.generic.getResult(0) ||
      !g.generic.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>() ||
      cast<RankedTensorType>(ret.getOperand(0).getType()).getElementType() != g.elementType) {
    why = "the gathered rows are not the whole written buffer";
    return failure();
  }
  return g;
}

bool match(RaisedKernel &kernel) {
  std::string why;
  return succeeded(find(kernel, why));
}

LogicalResult map(RaisedKernel &kernel) {
  func::FuncOp func = kernel.func;
  std::string why;
  FailureOr<Gather> found = find(kernel, why);
  if (failed(found))
    return func.emitError("gather: ") << why;
  Gather g = *found;
  int64_t bytes = elementBytes(g.elementType);
  if (!bytes || (g.rowSize * bytes) % 8)
    return func.emitError("gather: a row of ") << g.rowSize << " elements is not a whole number of 8-byte DMA words";
  if (g.count > kSlices && g.count % kSlices)
    return func.emitError("gather: ") << g.count << " rows do not spread evenly over " << kSlices << " slices";
  // furiosa-opt's gather DMA partitions the slices by a power of two of rows (it rejects 12 rows on npu3).
  if (g.count < kSlices && !llvm::isPowerOf2_64(g.count))
    return func.emitError("gather: ") << g.count << " rows: the gather DMA takes a power of two below " << kSlices;

  KernelBuilder kb(kernel);
  MLIRContext *ctx = kb.getContext();
  Type i32 = IntegerType::get(ctx, 32);
  MapAttr T = kb.axis("T", g.count), V = kb.axis("V", g.rows), S = kb.axis("S", g.rowSize);
  unsigned indexParam = kb.input(g.index.getArgNumber(), i32, T);
  unsigned tableParam = kb.input(g.table.getArgNumber(), g.elementType, MapAttr::pair(ctx, {V, S}));
  kb.output(kernel.written[0], g.elementType, MapAttr::pair(ctx, {T, S}));
  kb.beginBody();
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();

  // The index list in one slice, an even count of i32 (DM writes are whole 8-byte words); a list that would not fit
  // a slice's DM with its offsets spreads over the slices as the rows do, whole packets of 8 in each.
  int64_t per = g.count / kSlices;
  bool spread = 8 * g.count > kDmBudget && g.count > kSlices && per % 8 == 0;
  MapAttr oneSlice = spread ? T.stride(per) : kb.one().padTo(kSlices);
  MapAttr list = spread ? T.modulo(per) : g.count % 2 ? T.padTo(g.count + 1) : T;
  Value indexDm;
  if (g.count % 2 && !spread) {
    Value v = ViewOp::create(b, loc, HbmViewType::get(ctx, i32, kb.chip(), T, false), kb.param(indexParam));
    v = PadOp::create(b, loc, HbmViewType::get(ctx, i32, kb.chip(), list, false), v);
    indexDm = ViewToDmOp::create(b, loc, kb.dmType(i32, oneSlice, list), v);
  } else {
    indexDm = ToDmOp::create(b, loc, kb.dmType(i32, oneSlice, list), kb.param(indexParam));
  }
  // Byte offsets: index * row bytes, through the Vector Engine's fixed-point ALU.
  auto stream = [&](MapAttr time, MapAttr packet) { return kb.streamType(i32, oneSlice, time, packet); };
  MapAttr time = spread ? T.stride(8).modulo(per / 8) : g.count > 8 ? T.stride(8) : kb.one();
  MapAttr packet = g.count > 8 ? T.modulo(8) : list;
  MapAttr flit = g.count > 8 ? packet : T.padTo(8);
  Value s = BeginOp::create(b, loc, stream(kb.one(), list), indexDm, ContextKind::Main);
  s = FetchOp::create(b, loc, stream(time, packet), s);
  s = CollectOp::create(b, loc, stream(time, flit), s);
  s = VectorInitOp::create(b, loc, stream(time, flit), s);
  s = VectorIntraSliceTagOp::create(b, loc, stream(time, flit), s);
  s = VectorFxpOp::create(b, loc, stream(time, flit), s, b.getStringAttr("MulInt"),
                          b.getI32IntegerAttr(int32_t(g.rowSize * bytes)));
  s = VectorFinalOp::create(b, loc, stream(time, flit), s);
  s = CommitTrimOp::create(b, loc, stream(time, packet), s);
  Value offsetsDm = CommitOp::create(b, loc, kb.dmType(i32, oneSlice, list), s);
  Value offsets = ToHbmOp::create(b, loc, kb.hbmType(i32, T), offsetsDm);

  // One row per slice (or an even share of rows per slice past 256 rows).
  MapAttr slice = g.count < kSlices ? T.padTo(kSlices) : g.count == kSlices ? T : T.stride(g.count / kSlices);
  MapAttr element = g.count <= kSlices ? S : MapAttr::pair(ctx, {T.modulo(g.count / kSlices), S});
  Value rows = GatherOp::create(b, loc, kb.dmType(g.elementType, slice, element), kb.param(tableParam), offsets, V, S);
  Value out = ToHbmOp::create(b, loc, kb.hbmType(g.elementType, MapAttr::pair(ctx, {T, S})), rows);
  kb.finish({out});
  return success();
}

} // namespace

OpClass mlir::furiosa::mapping::gatherClass() { return {"gather", match, map}; }

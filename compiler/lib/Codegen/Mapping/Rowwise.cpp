// Op class: row-wise kernels — elementwise generics and row reductions over a common [R, H] space (rows R,
// columns H), as in normalizations and softmax. The kernel's generics fold into one scalar expression DAG whose
// nodes are classified by what they vary along: constants, rows (Row), columns (Col) or both (Full). The
// Vector Engine then evaluates it in passes:
//
//  - Layout: one row per slice (R <= 256, padded to 256), a row's H columns in time x packet.
//  - A pass streams one Full (or Row) tensor through a path of ops whose other operands are immediates, VRF
//    tensors (Col inputs, materialized Row values) or the stream itself (stash, for x * x).
//  - Each op takes the Vector Engine stage and ALU its kind needs (Fp, the intra-slice reduce, FpDiv, Clip;
//    4-wide float ops between narrow and widen). A pass ends where the next op would go back a stage or reuse an
//    ALU; its value is committed to DM and the next pass streams it.
//  - Row-only subexpressions are computed once per row in Row passes, never per column.
//  - Two rewrites match the hardware: a * rsqrt(b) = a / sqrt(b) (no rsqrt), and a run of multiplications and
//    divisions by off-path operands is reordered multiplications first (Fp precedes FpDiv).

#include "Mapping.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/TypeSwitch.h"

#include <map>
#include <set>

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;
using namespace mlir::furiosa::mapping;

namespace {

constexpr int64_t kSlices = 256, kVrfBytes = 8 * 1024;

enum class Shape { Scalar = 0, Row = 1, Col = 2, Full = 3 };

Shape join(Shape a, Shape b) {
  if (a == b)
    return a;
  if (a == Shape::Scalar)
    return b;
  if (b == Shape::Scalar)
    return a;
  return Shape::Full; // Row x Col, or either with Full
}

/// A node of the folded expression DAG.
struct Node {
  enum Kind { Input, Const, Op, Reduce } kind;
  Shape shape = Shape::Scalar;
  int arg = -1;            // Input: pointer-argument index
  Type type;               // Input: the buffer's element type
  float value = 0;         // Const
  std::string op;          // Op: add, sub, mul, div, max, min, exp, sqrt, neg, ...; Reduce: Add, Max, Min
  SmallVector<int> args;   // Op / Reduce operands
};

struct Dag {
  std::vector<Node> nodes;
  int root = -1;
  Type outType;      // the stored element type
  int64_t r = 0, h = 0;

  int add(Node n) {
    nodes.push_back(std::move(n));
    return nodes.size() - 1;
  }
};

/// Folds the kernel's generics into one DAG; `why` explains a kernel outside the class.
FailureOr<Dag> fold(func::FuncOp func, ArrayRef<int64_t> written, std::string &why) {
  Dag dag;
  auto ret = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  if (ret.getNumOperands() != 1 || written.size() != 1) {
    why = "expected one written buffer";
    return failure();
  }
  auto outType = cast<RankedTensorType>(ret.getOperand(0).getType());
  if (outType.getRank() != 2) {
    why = "expected a rank-2 [rows, columns] output";
    return failure();
  }
  dag.r = outType.getDimSize(0);
  dag.h = outType.getDimSize(1);
  dag.outType = outType.getElementType();

  std::map<Operation *, int> genericNode; // a generic's yielded value at a point of its domain
  std::function<FailureOr<int>(Value, AffineMap)> tensorNode;
  std::function<FailureOr<int>(linalg::GenericOp)> genericValue;

  auto shapeOfMap = [&](AffineMap m, unsigned numDims) -> std::optional<Shape> {
    // Over the 2-D space (d0 = row, d1 = column) or a 1-D row space (d0 = row).
    SmallVector<unsigned> dims;
    for (AffineExpr e : m.getResults()) {
      auto d = dyn_cast<AffineDimExpr>(e);
      if (!d)
        return std::nullopt;
      dims.push_back(d.getPosition());
    }
    if (numDims == 1)
      return dims == SmallVector<unsigned>{0} ? std::optional<Shape>(Shape::Row) : std::nullopt;
    if (dims == SmallVector<unsigned>{0, 1})
      return Shape::Full;
    if (dims == SmallVector<unsigned>{0})
      return Shape::Row;
    if (dims == SmallVector<unsigned>{1})
      return Shape::Col;
    return std::nullopt;
  };

  // The node for a tensor read through `map` inside a generic.
  tensorNode = [&](Value t, AffineMap map) -> FailureOr<int> {
    unsigned nd = map.getNumDims();
    std::optional<Shape> shape = shapeOfMap(map, nd);
    if (!shape) {
      why = "an operand map is not a projection of (row, column)";
      return failure();
    }
    if (auto arg = dyn_cast<BlockArgument>(t); arg && arg.getOwner() == &func.getBody().front()) {
      Node n{Node::Input};
      n.shape = *shape;
      n.arg = arg.getArgNumber();
      n.type = cast<RankedTensorType>(t.getType()).getElementType();
      return dag.add(n);
    }
    auto producer = t.getDefiningOp<linalg::GenericOp>();
    if (!producer) {
      why = "an operand is neither a buffer nor a generic's result";
      return failure();
    }
    // A producer over the row space read through (d0), or over the full space read through (d0, d1).
    auto pIters = producer.getIteratorTypesArray();
    bool rowProducer = cast<RankedTensorType>(t.getType()).getRank() == 1;
    if (rowProducer != (*shape == Shape::Row) && !(rowProducer && *shape == Shape::Row)) {
      why = "a generic's result is read through a map that does not match its domain";
      return failure();
    }
    (void)pIters;
    return genericValue(producer);
  };

  genericValue = [&](linalg::GenericOp g) -> FailureOr<int> {
    if (auto it = genericNode.find(g); it != genericNode.end())
      return it->second;
    auto iters = g.getIteratorTypesArray();
    bool reduction = llvm::is_contained(iters, utils::IteratorType::reduction);
    if (g.getNumDpsInits() != 1) {
      why = "a generic with several results";
      return failure();
    }
    if (reduction && (iters.size() != 2 || iters[0] != utils::IteratorType::parallel ||
                      iters[1] != utils::IteratorType::reduction)) {
      why = "only reductions over the columns of each row are mapped";
      return failure();
    }
    Block &body = g.getRegion().front();
    llvm::DenseMap<Value, int> scalar; // body values -> nodes
    for (OpOperand *in : g.getDpsInputOperands()) {
      auto n = tensorNode(in->get(), g.getMatchingIndexingMap(in));
      if (failed(n))
        return failure();
      scalar[body.getArgument(in->getOperandNumber())] = *n;
    }
    Value acc = body.getArgument(g.getNumDpsInputs());
    std::function<FailureOr<int>(Value)> node = [&](Value v) -> FailureOr<int> {
      if (auto it = scalar.find(v); it != scalar.end())
        return it->second;
      llvm::APFloat f(0.0f);
      if (matchPattern(v, m_ConstantFloat(&f))) {
        bool lost;
        f.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven, &lost);
        Node n{Node::Const};
        n.value = f.convertToFloat();
        return scalar[v] = dag.add(n);
      }
      if (v == acc) {
        why = "the accumulator is read outside the reduction's combine";
        return failure();
      }
      Operation *op = v.getDefiningOp();
      if (!op || op->getBlock() != &body) {
        why = "a body value defined outside the body";
        return failure();
      }
      std::string name = llvm::TypeSwitch<Operation *, std::string>(op)
                             .Case<arith::AddFOp>([](auto) { return "add"; })
                             .Case<arith::SubFOp>([](auto) { return "sub"; })
                             .Case<arith::MulFOp>([](auto) { return "mul"; })
                             .Case<arith::DivFOp>([](auto) { return "div"; })
                             .Case<arith::MaximumFOp, arith::MaxNumFOp>([](auto) { return "max"; })
                             .Case<arith::MinimumFOp, arith::MinNumFOp>([](auto) { return "min"; })
                             .Case<arith::NegFOp>([](auto) { return "neg"; })
                             .Case<math::ExpOp>([](auto) { return "exp"; })
                             .Case<math::SqrtOp>([](auto) { return "sqrt"; })
                             .Case<math::RsqrtOp>([](auto) { return "rsqrt"; })
                             .Case<math::TanhOp>([](auto) { return "tanh"; })
                             .Case<math::LogOp>([](auto) { return "log"; })
                             .Case<math::ErfOp>([](auto) { return "erf"; })
                             .Case<math::SinOp>([](auto) { return "sin"; })
                             .Case<math::CosOp>([](auto) { return "cos"; })
                             .Case<arith::ExtFOp>([](auto) { return "ext"; })
                             .Case<arith::TruncFOp>([](auto) { return "trunc"; })
                             .Default([](auto) { return ""; });
      if (name.empty()) {
        why = "no Vector Engine op for '" + op->getName().getStringRef().str() + "'";
        return failure();
      }
      SmallVector<int> args;
      for (Value o : op->getOperands()) {
        auto a = node(o);
        if (failed(a))
          return failure();
        args.push_back(*a);
      }
      // Casts: an extension of an input is the f32 fetch; the final truncation is the output cast.
      if (name == "ext" || name == "trunc")
        return scalar[v] = args[0];
      Node n{Node::Op};
      n.op = name;
      n.args = args;
      for (int a : args)
        n.shape = join(n.shape, dag.nodes[a].shape);
      return scalar[v] = dag.add(n);
    };
    Value yielded = cast<linalg::YieldOp>(body.getTerminator()).getOperand(0);
    int result;
    if (reduction) {
      // yield = combine(acc, term), from an identity fill.
      Operation *comb = yielded.getDefiningOp();
      if (!comb || comb->getNumOperands() != 2 ||
          (comb->getOperand(0) != acc && comb->getOperand(1) != acc)) {
        why = "a reduction body must combine the accumulator with one term";
        return failure();
      }
      Value term = comb->getOperand(0) == acc ? comb->getOperand(1) : comb->getOperand(0);
      std::string op = isa<arith::AddFOp>(comb) ? "Add"
                       : isa<arith::MaximumFOp, arith::MaxNumFOp>(comb) ? "Max"
                       : isa<arith::MinimumFOp, arith::MinNumFOp>(comb) ? "Min"
                                                                        : "";
      auto fill = g.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
      llvm::APFloat init(0.0f);
      bool identity = fill && matchPattern(fill.getInputs()[0], m_ConstantFloat(&init)) &&
                      ((op == "Add" && init.isZero()) || (op == "Max" && init.isInfinity() && init.isNegative()) ||
                       (op == "Min" && init.isInfinity() && !init.isNegative()));
      if (op.empty() || !identity) {
        why = "a reduction must be add/max/min from its identity";
        return failure();
      }
      auto t = node(term);
      if (failed(t))
        return failure();
      Node n{Node::Reduce};
      n.op = op;
      n.args = {*t};
      n.shape = Shape::Row;
      result = dag.add(n);
    } else {
      auto v = node(yielded);
      if (failed(v))
        return failure();
      result = *v;
    }
    return genericNode[g] = result;
  };

  auto producer = ret.getOperand(0).getDefiningOp<linalg::GenericOp>();
  if (!producer) {
    why = "the result is not a generic's";
    return failure();
  }
  auto root = genericValue(producer);
  if (failed(root))
    return failure();
  dag.root = *root;
  if (dag.nodes[dag.root].shape != Shape::Full) {
    why = "the stored value does not vary along both rows and columns";
    return failure();
  }
  return dag;
}

/// a * rsqrt(b) -> a / sqrt(b): the Vector Engine has no rsqrt.
void rewriteRsqrt(Dag &dag) {
  for (size_t i = 0; i < dag.nodes.size(); ++i) {
    if (dag.nodes[i].kind != Node::Op || dag.nodes[i].op != "mul")
      continue;
    for (int side = 0; side < 2; ++side) {
      int c = dag.nodes[i].args[side];
      if (dag.nodes[c].kind != Node::Op || dag.nodes[c].op != "rsqrt")
        continue;
      Node sq{Node::Op};
      sq.op = "sqrt";
      sq.args = dag.nodes[c].args;
      sq.shape = dag.nodes[c].shape;
      int sqrtNode = dag.add(sq);
      Node &m = dag.nodes[i];
      m.op = "div";
      m.args = {m.args[1 - side], sqrtNode};
      break;
    }
  }
}

/// A planned VE op: its stage, the ALU it takes, and how it reads its other operand.
enum Stage { Tag = 0, Narrow = 4, Fp = 5, Reduce = 6, FpDiv = 7, Widen = 8, Clip = 10 };

class Planner {
public:
  Planner(KernelBuilder &kb, Dag &dag) : kb(kb), dag(dag), ctx(kb.getContext()) {}

  /// Declares the axes and the kernel's inputs and output (before the body).
  LogicalResult declare(func::FuncOp func, int64_t writtenArg);
  /// Emits the passes and returns the stored HbmTensor.
  FailureOr<Value> emitBody(func::FuncOp func);

private:
  KernelBuilder &kb;
  Dag &dag;
  MLIRContext *ctx;
  MapAttr R, H, slice;
  std::map<int, Value> materialized; // node -> DmTensor (Full: [slice, H], Row: [slice, 1 # 8])
  std::map<int, Value> vrfs;         // node -> VrfTensor
  std::map<int, unsigned> params;    // pointer argument -> kernel parameter
  std::map<int, Type> inputTypes;    // pointer argument -> element type
  std::map<int, Value> inputDm;      // pointer argument -> DmTensor

  Type f32() { return Float32Type::get(ctx); }
  MapAttr rowPacket(int64_t n) { return kb.one().padTo(n); }
  StreamType stream(Type et, MapAttr time, MapAttr packet) { return kb.streamType(et, slice, time, packet); }

  FailureOr<Value> vrfOf(int node, func::FuncOp func);
  FailureOr<Value> materialize(int node, func::FuncOp func, bool output);
  InFlightDiagnostic fail(func::FuncOp func, const Twine &msg) { return func.emitError("row-wise: ") << msg; }
};

/// The node as a VRF tensor: a Col input loaded through a sub pass, or a materialized Row value.
FailureOr<Value> Planner::vrfOf(int node, func::FuncOp func) {
  if (auto it = vrfs.find(node); it != vrfs.end())
    return it->second;
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  Node &n = dag.nodes[node];
  Value v;
  if (n.shape == Shape::Col) {
    if (n.kind != Node::Input)
      return fail(func, "a column value other than an input (computing over columns alone is not mapped)"),
             failure();
    if (dag.h * 4 > kVrfBytes)
      return fail(func, "a column VRF operand of ") << dag.h << " f32 exceeds the 8 KB VRF", failure();
    Value dm = inputDm.at(n.arg);
    bool bf16 = n.type.isBF16();
    int64_t p = bf16 ? 16 : 8;
    Value s = BeginOp::create(b, loc, stream(n.type, kb.one(), H), dm, ContextKind::Sub);
    s = FetchOp::create(b, loc, stream(n.type, H.stride(p), H.modulo(p)), s);
    if (bf16)
      s = FetchCastOp::create(b, loc, stream(f32(), H.stride(p), H.modulo(p)), s);
    s = CollectOp::create(b, loc, stream(f32(), H.stride(8), H.modulo(8)), s);
    v = ToVrfOp::create(b, loc, VrfTensorType::get(ctx, f32(), kb.chip(), kb.cluster(), slice, H), s);
  } else if (n.shape == Shape::Row) {
    auto dm = materialize(node, func, false);
    if (failed(dm))
      return failure();
    Value s = BeginOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), *dm, ContextKind::Sub);
    s = FetchOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), s);
    s = CollectOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), s);
    v = ToVrfOp::create(b, loc, VrfTensorType::get(ctx, f32(), kb.chip(), kb.cluster(), slice, rowPacket(8)), s);
  } else {
    return fail(func, "a full-shape operand off the streamed path (two streamed tensors in one op are not mapped)"),
           failure();
  }
  return vrfs[node] = v;
}

/// Builds the pass (or passes) that compute `node` and returns its DmTensor. `output` marks the stored root:
/// it is cast to the output dtype.
FailureOr<Value> Planner::materialize(int node, func::FuncOp func, bool output) {
  if (auto it = materialized.find(node); it != materialized.end())
    return it->second;
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  if (dag.nodes[node].kind == Node::Input && !output)
    return inputDm.at(dag.nodes[node].arg);
  Node &top = dag.nodes[node];
  if (top.shape != Shape::Full && top.shape != Shape::Row)
    return fail(func, "only row or full values are materialized"), failure();

  // The streamed path, root first: each op has one operand of the path's shape (a Row path passes through a
  // reduce into its Full operand); the others are immediates, VRFs or the same node (x * x, stashed).
  struct Step {
    int node;
    int operand = -1;   // the off-path operand node (-1: none)
    bool stash = false; // the off-path operand is the streamed value itself
    bool reversed = false; // the streamed value is the second operand (c - x, c / x)
  };
  SmallVector<Step> path;
  int leaf = node;
  while (true) {
    Node &n = dag.nodes[leaf];
    if (n.kind == Node::Input || (leaf != node && materialized.count(leaf)))
      break;
    if (n.kind == Node::Const)
      return fail(func, "a streamed path ends in a constant"), failure();
    if (n.kind == Node::Reduce) {
      path.push_back({leaf});
      leaf = n.args[0];
      continue;
    }
    Shape want = n.shape;
    SmallVector<int> streamed, other;
    for (int a : n.args)
      (dag.nodes[a].shape == want ? streamed : other).push_back(a);
    Step step{leaf};
    if (streamed.size() == 2 && streamed[0] == streamed[1]) {
      step.stash = true;
      path.push_back(step);
      leaf = streamed[0];
      continue;
    }
    if (streamed.size() != 1)
      return fail(func, "an op combines two different values of the streamed shape"), failure();
    if (!other.empty()) {
      step.operand = other[0];
      step.reversed = n.args[0] != streamed[0];
    }
    path.push_back(step);
    leaf = streamed[0];
  }
  std::reverse(path.begin(), path.end()); // leaf first

  // Reorder runs of mul/div by off-path operands: multiplications first (Fp precedes FpDiv).
  for (size_t i = 0; i < path.size();) {
    size_t j = i;
    auto isMulDiv = [&](const Step &s) {
      Node &n = dag.nodes[s.node];
      return n.kind == Node::Op && (n.op == "mul" || (n.op == "div" && !s.reversed)) && !s.stash && s.operand >= 0;
    };
    while (j < path.size() && isMulDiv(path[j]))
      ++j;
    std::stable_partition(path.begin() + i, path.begin() + j,
                          [&](const Step &s) { return dag.nodes[s.node].op == "mul"; });
    i = j == i ? i + 1 : j;
  }

  // The source: the leaf's DmTensor.
  Node &leafNode = dag.nodes[leaf];
  Value source;
  Type sourceType = f32();
  bool rowStream = false;
  if (leafNode.kind == Node::Input) {
    if (leafNode.shape != Shape::Full)
      return fail(func, "a streamed input must vary along rows and columns"), failure();
    source = inputDm.at(leafNode.arg);
    sourceType = leafNode.type;
  } else {
    source = materialized.at(leaf);
    rowStream = leafNode.shape == Shape::Row;
  }

  // VRF operands are loaded before the pass (their own passes come first).
  for (const Step &s : path)
    if (s.operand >= 0 && dag.nodes[s.operand].kind != Node::Const)
      if (failed(vrfOf(s.operand, func)))
        return failure();

  // Stream state.
  bool row = rowStream;
  int stage = Tag;
  bool way4 = false;
  std::set<std::string> alus;
  MapAttr fullTime8 = H.stride(8), fullPacket8 = H.modulo(8), fullTime4 = H.stride(4), fullPacket4 = H.modulo(4);
  auto time = [&]() { return row ? kb.one() : (way4 ? fullTime4 : fullTime8); };
  auto packet = [&]() { return row ? rowPacket(way4 ? 4 : 8) : (way4 ? fullPacket4 : fullPacket8); };
  Value s;
  auto begin = [&](Value dm, Type et) {
    if (row) {
      s = BeginOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), dm, ContextKind::Main);
      s = FetchOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), s);
      s = CollectOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), s);
    } else {
      bool bf16 = et.isBF16();
      int64_t p = bf16 ? 16 : 8;
      s = BeginOp::create(b, loc, stream(et, kb.one(), H), dm, ContextKind::Main);
      s = FetchOp::create(b, loc, stream(et, H.stride(p), H.modulo(p)), s);
      if (bf16)
        s = FetchCastOp::create(b, loc, stream(f32(), H.stride(p), H.modulo(p)), s);
      s = CollectOp::create(b, loc, stream(f32(), fullTime8, fullPacket8), s);
    }
    s = VectorInitOp::create(b, loc, stream(f32(), time(), packet()), s);
    s = VectorIntraSliceTagOp::create(b, loc, stream(f32(), time(), packet()), s);
    stage = Tag;
    way4 = false;
    alus.clear();
  };
  auto toWay4 = [&]() {
    if (way4)
      return;
    way4 = true;
    if (row)
      s = VectorNarrowTrimOp::create(b, loc, stream(f32(), time(), packet()), s);
    else
      s = VectorNarrowSplitOp::create(b, loc, stream(f32(), time(), packet()), s);
    stage = std::max(stage, int(Narrow));
  };
  auto toWay8 = [&]() {
    if (!way4)
      return;
    way4 = false;
    if (row)
      s = VectorWidenPadOp::create(b, loc, stream(f32(), time(), packet()), s);
    else
      s = VectorWidenConcatOp::create(b, loc, stream(f32(), time(), packet()), s);
    stage = std::max(stage, int(Widen));
  };
  // Ends the pass: commits the current value (node `at`) to DM.
  auto commit = [&](int at, bool cast) -> Value {
    toWay8();
    s = VectorFinalOp::create(b, loc, stream(f32(), time(), packet()), s);
    Type et = f32();
    MapAttr trim = row ? rowPacket(8) : fullPacket8;
    if (cast && !dag.outType.isF32()) {
      et = dag.outType;
      s = CastOp::create(b, loc, stream(et, time(), row ? rowPacket(16) : fullPacket8.padTo(16)), s);
    }
    s = CommitTrimOp::create(b, loc, stream(et, time(), trim), s);
    Value dm = CommitOp::create(b, loc, kb.dmType(et, slice, row ? rowPacket(8) : H), s);
    materialized[at] = dm;
    return dm;
  };
  // The off-path operand of a step as (imm, vrf, stash).
  auto operandOf = [&](const Step &st, FloatAttr &imm, Value &vrf, UnitAttr &stash) {
    imm = FloatAttr();
    vrf = Value();
    stash = UnitAttr();
    if (st.stash)
      stash = b.getUnitAttr();
    else if (dag.nodes[st.operand].kind == Node::Const)
      imm = b.getF32FloatAttr(dag.nodes[st.operand].value);
    else
      vrf = vrfs.at(st.operand);
  };

  begin(source, sourceType);
  int current = leaf;
  for (size_t i = 0; i < path.size(); ++i) {
    const Step &st = path[i];
    Node &n = dag.nodes[st.node];
    // What the op needs.
    std::string op = n.op;
    int need;            // stage
    bool needWay4 = true;
    std::string alu, veOp;
    enum { FpBinary, FpUnary, Div, ClipBin, ReduceOp } form;
    float negImm = 0;
    if (n.kind == Node::Reduce) {
      need = Reduce, alu = "ReduceAccTree", form = ReduceOp, veOp = n.op;
    } else if (op == "exp" || op == "sqrt" || op == "tanh" || op == "log" || op == "erf" || op == "sin" ||
               op == "cos") {
      need = Fp, form = FpUnary, alu = op == "exp" ? "FpExp" : "FpFpu";
      veOp = op == "exp" ? "Exp" : op == "sqrt" ? "Sqrt" : op == "tanh" ? "Tanh" : op == "log" ? "Log"
             : op == "erf" ? "Erf" : op == "sin" ? "Sin" : "Cos";
    } else if (op == "neg") {
      need = Fp, form = FpBinary, veOp = "MulF(FpMulAlu::Mul0)", alu = "FpMul0", negImm = -1;
    } else if (op == "mul") {
      need = Fp, form = FpBinary;
      alu = !alus.count("FpMul0") ? "FpMul0" : !alus.count("FpMul1") ? "FpMul1" : "FpFma";
      veOp = alu == "FpMul0" ? "MulF(FpMulAlu::Mul0)" : alu == "FpMul1" ? "MulF(FpMulAlu::Mul1)"
                                                                        : "MulF(FpMulAlu::Fma)";
    } else if ((op == "add" || op == "sub") && !st.reversed) {
      if (stage <= Fp && !alus.count("FpFma"))
        need = Fp, form = FpBinary, alu = "FpFma", veOp = op == "add" ? "AddF" : "SubF";
      else
        need = Clip, form = ClipBin, needWay4 = false, alu = "ClipAdd", veOp = "Add";
    } else if (op == "div" && !st.reversed) {
      if (stage <= FpDiv && !alus.count("ReduceFpDiv"))
        need = FpDiv, form = Div, alu = "ReduceFpDiv";
      else
        need = Fp, form = FpBinary, alu = "FpFpu", veOp = "DivF";
    } else if (op == "max" || op == "min") {
      need = Clip, form = ClipBin, needWay4 = false, alu = op == "max" ? "ClipMax" : "ClipMin";
      veOp = op == "max" ? "Max" : "Min";
    } else {
      return fail(func, "no Vector Engine form for '" + op + "' with the streamed value as its " +
                            (st.reversed ? "second" : "first") + " operand"),
             failure();
    }
    // A clip-stage subtraction adds the negated immediate; a VRF cannot be negated.
    bool clipSub = form == ClipBin && op == "sub";
    if (clipSub && (st.operand < 0 || dag.nodes[st.operand].kind != Node::Const))
      return fail(func, "a subtraction of a VRF operand after the float stage"), failure();
    // Pass break: a stage behind us, a used ALU, or a stash where the stage cannot stash.
    bool sameStageAllowed = need == Fp || need == Clip;
    bool breakHere = need < stage || (need == stage && !sameStageAllowed) || alus.count(alu) ||
                     (needWay4 && stage > Narrow && !way4) || (form == ReduceOp && row);
    if (breakHere) {
      if (current == leaf && i == 0)
        return fail(func, "an op cannot start a pass"), failure();
      Value dm = commit(current, false);
      row = dag.nodes[current].shape == Shape::Row;
      begin(dm, f32());
      // Retry this step in the new pass.
      --i;
      continue;
    }
    if (needWay4)
      toWay4();
    else
      toWay8();
    if (st.stash) {
      s = VectorStashOp::create(b, loc, stream(f32(), time(), packet()), s);
      stage = std::max(stage, int(Fp) - 1);
    }
    FloatAttr imm;
    Value vrf;
    UnitAttr stash;
    if (form != ReduceOp && form != FpUnary && !(op == "neg")) {
      operandOf(st, imm, vrf, stash);
      if (clipSub)
        imm = b.getF32FloatAttr(-dag.nodes[st.operand].value);
    }
    switch (form) {
    case FpUnary:
      s = VectorFpUnaryOp::create(b, loc, stream(f32(), time(), packet()), s, b.getStringAttr(veOp));
      break;
    case FpBinary:
      if (op == "neg")
        imm = b.getF32FloatAttr(negImm);
      s = VectorFpBinaryOp::create(b, loc, stream(f32(), time(), packet()), s, b.getStringAttr(veOp), imm, vrf, stash);
      break;
    case Div:
      s = VectorFpDivOp::create(b, loc, stream(f32(), time(), packet()), s, imm, vrf, stash);
      break;
    case ClipBin:
      s = VectorClipOp::create(b, loc, stream(f32(), time(), packet()), s, b.getStringAttr(veOp), imm, vrf, stash);
      break;
    case ReduceOp:
      row = true;
      s = VectorIntraSliceReduceOp::create(b, loc, stream(f32(), time(), packet()), s, b.getStringAttr(veOp),
                                           b.getStringAttr("H"));
      break;
    }
    stage = need;
    alus.insert(alu);
    current = st.node;
  }
  return commit(node, output);
}

LogicalResult Planner::declare(func::FuncOp func, int64_t writtenArg) {
  if (dag.r > kSlices)
    return fail(func, "more than 256 rows (row tiling is not implemented yet)");
  if (dag.h % 16)
    return fail(func, "a row length that is not a multiple of 16");
  R = kb.axis("R", dag.r);
  H = kb.axis("H", dag.h);
  slice = dag.r < kSlices ? R.padTo(kSlices) : R;
  for (const Node &n : dag.nodes)
    if (n.kind == Node::Input) {
      if (!n.type.isBF16() && !n.type.isF32())
        return fail(func, "inputs must be bf16 or f32");
      inputTypes[n.arg] = n.type;
    }
  for (auto [a, et] : inputTypes) {
    auto t = cast<RankedTensorType>(func.getArgumentTypes()[a]);
    bool full = t.getRank() == 2 && t.getDimSize(0) == dag.r && t.getDimSize(1) == dag.h;
    bool col = t.getRank() == 1 && t.getDimSize(0) == dag.h;
    if (!full && !col)
      return fail(func, "an input whose shape is neither [rows, columns] nor [columns]");
    params[a] = kb.input(a, et, full ? MapAttr::pair(ctx, {R, H}) : H);
  }
  if (!dag.outType.isBF16() && !dag.outType.isF32())
    return fail(func, "the output must be bf16 or f32");
  kb.output(writtenArg, dag.outType, MapAttr::pair(ctx, {R, H}));
  return success();
}

FailureOr<Value> Planner::emitBody(func::FuncOp func) {
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  for (auto [a, p] : params)
    inputDm[a] = ToDmOp::create(b, loc, kb.dmType(inputTypes[a], slice, H), kb.param(p));
  auto out = materialize(dag.root, func, true);
  if (failed(out))
    return failure();
  return ToHbmOp::create(b, loc, kb.hbmType(dag.outType, MapAttr::pair(ctx, {R, H})), *out).getResult();
}

bool match(RaisedKernel &kernel) {
  std::string why;
  return succeeded(fold(kernel.func, kernel.written, why));
}

LogicalResult map(RaisedKernel &kernel) {
  func::FuncOp func = kernel.func;
  std::string why;
  FailureOr<Dag> folded = fold(func, kernel.written, why);
  if (failed(folded))
    return func.emitError("row-wise: ") << why;
  Dag dag = *folded;
  rewriteRsqrt(dag);
  KernelBuilder kb(kernel);
  Planner planner(kb, dag);
  if (failed(planner.declare(func, kernel.written[0])))
    return failure();
  kb.beginBody();
  FailureOr<Value> hbm = planner.emitBody(func);
  if (failed(hbm))
    return failure();
  kb.finish({*hbm});
  return success();
}

} // namespace

OpClass mlir::furiosa::mapping::rowwiseClass() { return {"row-wise", match, map}; }

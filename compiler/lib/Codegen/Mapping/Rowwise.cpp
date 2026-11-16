// Op class: row-wise kernels — elementwise generics and row reductions over a common [R, H] space (rows R,
// columns H), as in normalizations and softmax. The kernel's generics fold into one scalar expression DAG whose
// nodes are classified by what they vary along: constants, rows (Row), columns (Col) or both (Full). The
// Vector Engine then evaluates it in passes:
//
//  - Layout: one row per slice (padded to 256), a row's H columns in time x packet. More than 256 rows run as
//    row tiles of up to 256, each read from and written to an HBM tile view of the buffers.
//  - A pass streams one Full (or Row) tensor through a path of ops whose other operands are immediates, VRF
//    tensors (Col inputs, materialized Row values, or a second Full tensor, a row per slice) or the stream
//    itself (stash, for x * x).
//  - Each op takes the Vector Engine stage and ALU its kind needs (Fp, the intra-slice reduce, FpDiv, Clip;
//    4-wide float ops between narrow and widen). A pass ends where the next op would go back a stage or reuse an
//    ALU; its value is committed to DM and the next pass streams it.
//  - Row-only subexpressions are computed once per row in Row passes, never per column.
//  - Two rewrites match the hardware: a * rsqrt(b) = a / sqrt(b) (no rsqrt), and a run of multiplications and
//    divisions by off-path operands is reordered multiplications first (Fp precedes FpDiv).
//  - Each written buffer is a root of the DAG (e.g. an in-place residual sum and the row normalized from it); a
//    root another root reads is committed once, in its output dtype, and streamed from there.
//  - A single row (R = 1) may come rank-reduced: rank-0 row values and constant-0 row indices.

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
  SmallVector<int> roots;     // one per written buffer, in result order
  SmallVector<Type> outTypes; // their stored element types
  int64_t r = 0, h = 0;

  /// Adds a node, or returns the equal one already there: generics recompute shared subexpressions (each
  /// reads the buffers again), and the DAG keeps one of each.
  int add(Node n) {
    std::string key = std::to_string(int(n.kind)) + "|" + std::to_string(n.arg) + "|" +
                      std::to_string(llvm::bit_cast<uint32_t>(n.value)) + "|" + n.op + "|" + std::to_string(int(n.shape));
    for (int a : n.args)
      key += "," + std::to_string(a);
    if (auto it = interned.find(key); it != interned.end())
      return it->second;
    nodes.push_back(std::move(n));
    return interned[key] = nodes.size() - 1;
  }
  std::map<std::string, int> interned;
};

/// Folds the kernel's generics into one DAG; `why` explains a kernel outside the class.
FailureOr<Dag> fold(func::FuncOp func, ArrayRef<int64_t> written, std::string &why) {
  Dag dag;
  auto ret = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  if (ret.getNumOperands() < 1 || ret.getNumOperands() != written.size()) {
    why = "expected a result per written buffer";
    return failure();
  }
  for (Value result : ret.getOperands()) {
    auto outType = cast<RankedTensorType>(result.getType());
    if (outType.getRank() != 2 || (dag.r && (outType.getDimSize(0) != dag.r || outType.getDimSize(1) != dag.h))) {
      why = "expected rank-2 [rows, columns] outputs of one shape";
      return failure();
    }
    dag.r = outType.getDimSize(0);
    dag.h = outType.getDimSize(1);
    dag.outTypes.push_back(outType.getElementType());
  }

  std::map<Operation *, int> genericNode; // a generic's yielded value at a point of its domain
  std::function<FailureOr<int>(linalg::GenericOp)> genericValue;

  // What a tensor read through `m` in a generic with `iters` varies along. The space is (row, column) for two
  // loops, rows for one parallel loop, a row's columns for one reduction loop (a single row, rank-reduced). A
  // constant 0 indexes a dimension of size 1: the single row (or a row's only column).
  auto shapeOfMap = [&](AffineMap m, ArrayRef<utils::IteratorType> iters, RankedTensorType t) -> std::optional<Shape> {
    auto roleOf = [&](unsigned d) {
      if (iters.size() == 2)
        return d == 0 ? Shape::Row : Shape::Col;
      return iters[0] == utils::IteratorType::reduction ? Shape::Col : Shape::Row;
    };
    if (iters.empty() || iters.size() > 2)
      return std::nullopt;
    if (t.getRank() == 0)
      return Shape::Row; // the value of the single row
    Shape shape = Shape::Scalar;
    SmallVector<Shape> seen;
    for (auto [i, e] : llvm::enumerate(m.getResults())) {
      Shape role;
      if (auto d = dyn_cast<AffineDimExpr>(e)) {
        role = roleOf(d.getPosition());
      } else if (auto c = dyn_cast<AffineConstantExpr>(e); c && c.getValue() == 0 && t.getDimSize(i) == 1) {
        role = t.getRank() == 2 && i == 1 ? Shape::Col : Shape::Row;
      } else {
        return std::nullopt;
      }
      if (llvm::is_contained(seen, role))
        return std::nullopt;
      seen.push_back(role);
      shape = join(shape, role);
    }
    // A [rows, columns] tensor reads row-major: the row first.
    if (seen.size() == 2 && seen[0] != Shape::Row)
      return std::nullopt;
    return shape;
  };

  // The node for a tensor read through `map` inside a generic.
  std::function<FailureOr<int>(Value, AffineMap, ArrayRef<utils::IteratorType>)> readNode;
  readNode = [&](Value t, AffineMap map, ArrayRef<utils::IteratorType> iters) -> FailureOr<int> {
    std::optional<Shape> shape = shapeOfMap(map, iters, cast<RankedTensorType>(t.getType()));
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
    bool rowProducer = cast<RankedTensorType>(t.getType()).getRank() <= 1;
    if (rowProducer != (*shape == Shape::Row)) {
      why = "a generic's result is read through a map that does not match its domain";
      return failure();
    }
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
    bool rowReduction = iters.size() == 2 && iters[0] == utils::IteratorType::parallel &&
                        iters[1] == utils::IteratorType::reduction;
    bool singleRowReduction = iters.size() == 1 && dag.r == 1;
    if (reduction && !rowReduction && !singleRowReduction) {
      why = "only reductions over the columns of each row are mapped";
      return failure();
    }
    Block &body = g.getRegion().front();
    llvm::DenseMap<Value, int> scalar; // body values -> nodes
    for (OpOperand *in : g.getDpsInputOperands()) {
      auto n = readNode(in->get(), g.getMatchingIndexingMap(in), iters);
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

  for (Value result : ret.getOperands()) {
    auto producer = result.getDefiningOp<linalg::GenericOp>();
    if (!producer) {
      why = "a result is not a generic's";
      return failure();
    }
    auto root = genericValue(producer);
    if (failed(root))
      return failure();
    if (dag.nodes[*root].shape != Shape::Full) {
      why = "a stored value does not vary along both rows and columns";
      return failure();
    }
    dag.roots.push_back(*root);
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

/// 1 / (1 + exp(-x)) -> sigmoid(x), and exp(-x) -> negexp(x): the Vector Engine computes both directly, and
/// has no reciprocal for the first.
void rewriteSigmoid(Dag &dag) {
  auto isConst = [&](int n, float v) { return dag.nodes[n].kind == Node::Const && dag.nodes[n].value == v; };
  auto isOp = [&](int n, StringRef op) { return dag.nodes[n].kind == Node::Op && dag.nodes[n].op == op; };
  for (size_t i = 0; i < dag.nodes.size(); ++i) {
    Node &n = dag.nodes[i];
    if (isOp(i, "exp") && isOp(n.args[0], "neg")) {
      n.op = "negexp";
      n.args = dag.nodes[n.args[0]].args;
    }
  }
  for (size_t i = 0; i < dag.nodes.size(); ++i) {
    Node &n = dag.nodes[i];
    if (!isOp(i, "div") || !isConst(n.args[0], 1.0f) || !isOp(n.args[1], "add"))
      continue;
    Node &sum = dag.nodes[n.args[1]];
    int e = isConst(sum.args[0], 1.0f) ? sum.args[1] : isConst(sum.args[1], 1.0f) ? sum.args[0] : -1;
    if (e < 0 || !isOp(e, "negexp"))
      continue;
    n.op = "sigmoid";
    n.args = dag.nodes[e].args;
  }
}

/// A planned VE op: its stage, the ALU it takes, and how it reads its other operand.
enum Stage { Tag = 0, Narrow = 4, Fp = 5, Reduce = 6, FpDiv = 7, Widen = 8, Clip = 10 };

class Planner {
public:
  Planner(KernelBuilder &kb, Dag &dag) : kb(kb), dag(dag), ctx(kb.getContext()) {}

  /// Declares the axes and the kernel's inputs and outputs (before the body).
  LogicalResult declare(func::FuncOp func, ArrayRef<int64_t> written);
  /// Emits the passes and returns the stored HbmTensors, one per root (none when the rows are tiled: the
  /// outputs are then written in place).
  FailureOr<SmallVector<Value>> emitBody(func::FuncOp func);
  bool tiled() const { return dag.r > kSlices; }

private:
  KernelBuilder &kb;
  Dag &dag;
  MLIRContext *ctx;
  MapAttr R, H, slice;
  MapAttr rows; // the rows of the current tile: R, or R resized to the tile
  std::map<int, Value> materialized; // node -> DmTensor (Full: [slice, H], Row: [slice, 1 # 8])
  std::map<int, Value> vrfs;         // node -> VrfTensor
  std::map<int, unsigned> params;    // pointer argument -> kernel parameter
  std::map<int, Type> inputTypes;    // pointer argument -> element type
  std::map<int, Value> inputDm;      // pointer argument -> DmTensor
  std::map<int, bool> fullInput;     // pointer argument -> [rows, columns] (else [columns])
  SmallVector<unsigned> outParams;   // tiled: the in-place output parameters
  int64_t vrfBytes = 0;              // VRF taken per slice

  Type f32() { return Float32Type::get(ctx); }
  MapAttr rowPacket(int64_t n) { return kb.one().padTo(n); }
  StreamType stream(Type et, MapAttr time, MapAttr packet) { return kb.streamType(et, slice, time, packet); }

  /// The passes of one row tile (or all rows): the roots' DmTensors when tiled, else their HbmTensors.
  FailureOr<SmallVector<Value>> emitTile(func::FuncOp func);
  FailureOr<Value> vrfOf(int node, func::FuncOp func);
  /// `outType` set: the node is stored, cast to that dtype.
  FailureOr<Value> materialize(int node, func::FuncOp func, Type outType = {});
  /// The ops a node's value still costs to compute (nothing for inputs and committed values).
  int weight(int node);
  InFlightDiagnostic fail(func::FuncOp func, const Twine &msg) { return func.emitError("row-wise: ") << msg; }
};

int Planner::weight(int node) {
  Node &n = dag.nodes[node];
  if (n.kind == Node::Input || n.kind == Node::Const || materialized.count(node))
    return 0;
  int w = 1;
  for (int a : n.args)
    w += weight(a);
  return w;
}

/// The node as a VRF tensor: a Col input or a Full value (a row per slice) loaded through a sub pass, or a
/// materialized Row value.
FailureOr<Value> Planner::vrfOf(int node, func::FuncOp func) {
  if (auto it = vrfs.find(node); it != vrfs.end())
    return it->second;
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  Node &n = dag.nodes[node];
  Value v;
  if (n.shape == Shape::Col || n.shape == Shape::Full) {
    if (n.shape == Shape::Col && n.kind != Node::Input)
      return fail(func, "a column value other than an input (computing over columns alone is not mapped)"),
             failure();
    if (vrfBytes + dag.h * 4 > kVrfBytes)
      return fail(func, "VRF operands of ") << dag.h << " f32 per slice exceed the 8 KB VRF", failure();
    vrfBytes += dag.h * 4;
    FailureOr<Value> dm = n.kind == Node::Input ? FailureOr<Value>(inputDm.at(n.arg)) : materialize(node, func);
    if (failed(dm))
      return failure();
    Type et = cast<DmTensorType>(dm->getType()).getElementType();
    bool bf16 = et.isBF16();
    int64_t p = bf16 ? 16 : 8;
    Value s = BeginOp::create(b, loc, stream(et, kb.one(), H), *dm, ContextKind::Sub);
    s = FetchOp::create(b, loc, stream(et, H.stride(p), H.modulo(p)), s);
    if (bf16)
      s = FetchCastOp::create(b, loc, stream(f32(), H.stride(p), H.modulo(p)), s);
    s = CollectOp::create(b, loc, stream(f32(), H.stride(8), H.modulo(8)), s);
    v = ToVrfOp::create(b, loc, VrfTensorType::get(ctx, f32(), kb.chip(), kb.cluster(), slice, H), s);
  } else if (n.shape == Shape::Row) {
    if (vrfBytes + 32 > kVrfBytes)
      return fail(func, "row VRF operands exceed the 8 KB VRF"), failure();
    vrfBytes += 32;
    auto dm = materialize(node, func);
    if (failed(dm))
      return failure();
    Value s = BeginOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), *dm, ContextKind::Sub);
    s = FetchOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), s);
    s = CollectOp::create(b, loc, stream(f32(), kb.one(), rowPacket(8)), s);
    v = ToVrfOp::create(b, loc, VrfTensorType::get(ctx, f32(), kb.chip(), kb.cluster(), slice, rowPacket(8)), s);
  } else {
    return fail(func, "a constant operand off the streamed path is an immediate, not a VRF"), failure();
  }
  return vrfs[node] = v;
}

/// Builds the pass (or passes) that compute `node` and returns its DmTensor, cast to `outType` when the node is
/// stored.
FailureOr<Value> Planner::materialize(int node, func::FuncOp func, Type outType) {
  bool output = bool(outType);
  if (auto it = materialized.find(node); it != materialized.end())
    if (!output || cast<DmTensorType>(it->second.getType()).getElementType() == outType)
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
    bool zip = false;      // the off-path operand is a second input of the leaf's type, interleaved with it
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
    if (streamed.size() == 2) {
      // Two different values of the streamed shape: stream the costlier (the first operand of a sub or div),
      // and read the other from the VRF, a row per slice.
      bool ordered = n.op == "sub" || n.op == "div";
      int first = n.args[0], second = n.args[1];
      if (!ordered && weight(second) > weight(first))
        std::swap(first, second);
      streamed = {first};
      other = {second};
      // Two inputs of one dtype stream interleaved (pair mode) instead of filling the VRF.
      Node &a = dag.nodes[first], &o = dag.nodes[second];
      if (a.kind == Node::Input && o.kind == Node::Input && a.type == o.type)
        step.zip = true;
    }
    if (streamed.size() != 1)
      return fail(func, "an op combines more than two values of the streamed shape"), failure();
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
    sourceType = cast<DmTensorType>(source.getType()).getElementType();
    rowStream = leafNode.shape == Shape::Row;
  }

  // VRF operands are loaded before the pass (their own passes come first).
  for (const Step &s : path)
    if (s.operand >= 0 && dag.nodes[s.operand].kind != Node::Const && !s.zip)
      if (failed(vrfOf(s.operand, func)))
        return failure();
  bool zipFirst = !path.empty() && path[0].zip && leafNode.kind == Node::Input;

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
  auto commit = [&](int at, Type cast) -> Value {
    toWay8();
    s = VectorFinalOp::create(b, loc, stream(f32(), time(), packet()), s);
    Type et = f32();
    MapAttr trim = row ? rowPacket(8) : fullPacket8;
    if (cast && !cast.isF32()) {
      et = cast;
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

  bool pair = false; // the pass started in pair mode: no reduction in it
  if (zipFirst) {
    // begin_interleaved over the two inputs in the op's operand order, then unzip into the pair phase.
    Node &zipNode = dag.nodes[path[0].node];
    Value lhs = inputDm.at(dag.nodes[zipNode.args[0]].arg), rhs = inputDm.at(dag.nodes[zipNode.args[1]].arg);
    MapAttr I = kb.axis("I", 2);
    int64_t p = 8; // 8-element packets (16 bytes of bf16) keep the collected time (H / 8, I) after a cast
    s = BeginInterleavedOp::create(b, loc, stream(sourceType, I, H), lhs, rhs, ContextKind::Main, I);
    s = FetchOp::create(b, loc, stream(sourceType, MapAttr::pair(ctx, {H.stride(p), I}), H.modulo(p)), s);
    if (sourceType.isBF16())
      s = FetchCastOp::create(b, loc, stream(f32(), MapAttr::pair(ctx, {H.stride(p), I}), H.modulo(p)), s);
    s = CollectOp::create(b, loc, stream(f32(), MapAttr::pair(ctx, {fullTime8, I}), fullPacket8), s);
    s = VectorInitOp::create(b, loc, stream(f32(), MapAttr::pair(ctx, {fullTime8, I}), fullPacket8), s);
    s = VectorIntraSliceUnzipOp::create(b, loc, stream(f32(), fullTime8, fullPacket8), s, I);
    stage = Tag;
    way4 = false;
    alus.clear();
    pair = true;
  } else {
    begin(source, sourceType);
  }
  int current = leaf;
  for (size_t i = 0; i < path.size(); ++i) {
    const Step &st = path[i];
    Node &n = dag.nodes[st.node];
    // What the op needs.
    std::string op = n.op;
    int need;            // stage
    bool needWay4 = true;
    std::string alu, veOp;
    enum { FpBinary, FpUnary, Div, ClipBin, ReduceOp, FpZip, ClipZip } form;
    float negImm = 0;
    if (st.zip && i == 0 && zipFirst) {
      bool clip = op == "max" || op == "min";
      need = clip ? Clip : Fp, form = clip ? ClipZip : FpZip, needWay4 = !clip;
      alu = clip ? "ClipZip" : op == "mul" ? "FpMul0" : op == "div" ? "FpFpu" : "FpFma";
      veOp = op == "add" ? "AddF" : op == "sub" ? "SubF" : op == "mul" ? "MulF(FpMulAlu::Mul0)" : op == "div" ? "DivF"
             : op == "max" ? "Max" : "Min";
      if (!clip && op != "add" && op != "sub" && op != "mul" && op != "div")
        return fail(func, "no Vector Engine zip for '" + op + "'"), failure();
    } else if (n.kind == Node::Reduce) {
      need = Reduce, alu = "ReduceAccTree", form = ReduceOp, veOp = n.op;
    } else if (op == "exp" || op == "negexp" || op == "sigmoid" || op == "sqrt" || op == "tanh" || op == "log" ||
               op == "erf" || op == "sin" || op == "cos") {
      need = Fp, form = FpUnary, alu = op == "exp" || op == "negexp" ? "FpExp" : "FpFpu";
      veOp = op == "exp" ? "Exp" : op == "negexp" ? "NegExp" : op == "sigmoid" ? "Sigmoid" : op == "sqrt" ? "Sqrt"
             : op == "tanh" ? "Tanh" : op == "log" ? "Log" : op == "erf" ? "Erf" : op == "sin" ? "Sin" : "Cos";
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
                     (needWay4 && stage > Narrow && !way4) || (form == ReduceOp && (row || pair));
    if (breakHere) {
      if (current == leaf && i == 0)
        return fail(func, "an op cannot start a pass"), failure();
      Value dm = commit(current, Type());
      row = dag.nodes[current].shape == Shape::Row;
      begin(dm, f32());
      pair = false;
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
    if (form != ReduceOp && form != FpUnary && form != FpZip && form != ClipZip && !(op == "neg")) {
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
    case FpZip:
      s = VectorFpZipOp::create(b, loc, stream(f32(), time(), packet()), s, b.getStringAttr(veOp));
      break;
    case ClipZip:
      s = VectorClipZipOp::create(b, loc, stream(f32(), time(), packet()), s, b.getStringAttr(veOp));
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
  return commit(node, outType);
}

LogicalResult Planner::declare(func::FuncOp func, ArrayRef<int64_t> written) {
  if (dag.h % 16)
    return fail(func, "a row length that is not a multiple of 16");
  R = kb.axis("R", dag.r);
  H = kb.axis("H", dag.h);
  rows = R;
  slice = dag.r < kSlices ? R.padTo(kSlices) : R;
  for (const Node &n : dag.nodes)
    if (n.kind == Node::Input) {
      if (!n.type.isBF16() && !n.type.isF32())
        return fail(func, "inputs must be bf16 or f32");
      inputTypes[n.arg] = n.type;
    }
  for (auto [a, et] : inputTypes) {
    auto t = cast<RankedTensorType>(func.getArgumentTypes()[a]);
    bool full = t.getRank() == 2 && t.getNumElements() == dag.r * dag.h; // row-major: rows may be regrouped
    bool col = t.getRank() == 1 && t.getDimSize(0) == dag.h;
    if (!full && !col)
      return fail(func, "an input whose shape is neither [rows, columns] nor [columns]");
    params[a] = kb.input(a, et, full ? MapAttr::pair(ctx, {R, H}) : H);
    fullInput[a] = full;
  }
  for (auto [k, outType] : llvm::enumerate(dag.outTypes)) {
    if (!outType.isBF16() && !outType.isF32())
      return fail(func, "the outputs must be bf16 or f32");
    if (tiled())
      outParams.push_back(kb.inout(written[k], outType, MapAttr::pair(ctx, {R, H})));
    else
      kb.output(written[k], outType, MapAttr::pair(ctx, {R, H}));
  }
  return success();
}

FailureOr<SmallVector<Value>> Planner::emitBody(func::FuncOp func) {
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  if (!tiled()) {
    for (auto [a, p] : params)
      inputDm[a] = ToDmOp::create(b, loc, kb.dmType(inputTypes[a], slice, H), kb.param(p));
    return emitTile(func);
  }
  // Row tiles: each reads its rows of the [rows, columns] inputs through a tile view, and writes its rows
  // of the outputs through a mutable one.
  for (int64_t start = 0; start < dag.r; start += kSlices) {
    int64_t len = std::min(kSlices, dag.r - start);
    rows = R.resize(len);
    slice = len < kSlices ? rows.padTo(kSlices) : rows;
    materialized.clear();
    vrfs.clear();
    inputDm.clear();
    vrfBytes = 0;
    auto tileOf = [&](Value hbm, Type et, bool mut) -> Value {
      MapAttr whole = MapAttr::pair(ctx, {R, H});
      Value v = ViewOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), whole, mut), hbm);
      MapAttr part = MapAttr::pair(ctx, {rows.padTo(dag.r, mut ? PadKind::Bottom : PadKind::Top), H});
      return TileOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), part, mut), v, R, b.getI64IntegerAttr(len),
                            b.getI64IntegerAttr(start));
    };
    for (auto [a, p] : params) {
      DmTensorType dm = kb.dmType(inputTypes[a], slice, H);
      inputDm[a] = fullInput[a] ? ViewToDmOp::create(b, loc, dm, tileOf(kb.param(p), inputTypes[a], false)).getResult()
                                : ToDmOp::create(b, loc, dm, kb.param(p)).getResult();
    }
    FailureOr<SmallVector<Value>> dms = emitTile(func);
    if (failed(dms))
      return failure();
    for (auto [k, dm] : llvm::enumerate(*dms))
      DmToHbmViewOp::create(b, loc, dm, tileOf(kb.param(outParams[k]), dag.outTypes[k], true));
  }
  return SmallVector<Value>();
}

FailureOr<SmallVector<Value>> Planner::emitTile(func::FuncOp func) {
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  // A root another root reads goes first, so the other streams its stored value.
  SmallVector<size_t> order;
  for (size_t k = 0; k < dag.roots.size(); ++k)
    order.push_back(k);
  auto reaches = [&](int from, int to) {
    SmallVector<int> stack{from};
    llvm::DenseSet<int> seen;
    while (!stack.empty()) {
      int n = stack.pop_back_val();
      if (n == to)
        return true;
      if (seen.insert(n).second)
        llvm::append_range(stack, dag.nodes[n].args);
    }
    return false;
  };
  llvm::stable_sort(order, [&](size_t x, size_t y) {
    return dag.roots[x] != dag.roots[y] && reaches(dag.roots[y], dag.roots[x]);
  });
  SmallVector<Value> outs(dag.roots.size());
  for (size_t k : order) {
    auto out = materialize(dag.roots[k], func, dag.outTypes[k]);
    if (failed(out))
      return failure();
    outs[k] = tiled() ? *out
                      : ToHbmOp::create(b, loc, kb.hbmType(dag.outTypes[k], MapAttr::pair(ctx, {R, H})), *out).getResult();
  }
  return outs;
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
  rewriteSigmoid(dag);
  // Without reductions, row or column values, the DAG is elementwise and its rows are free to regroup:
  // shorter rows keep a second streamed tensor within the VRF.
  if (llvm::all_of(dag.nodes, [](const Node &n) { return n.shape == Shape::Full || n.kind == Node::Const; })) {
    constexpr int64_t kRow = 1024;
    for (int64_t q = 1; q <= dag.h && dag.h > kRow; ++q)
      if (dag.h % q == 0 && dag.h / q <= kRow && (dag.h / q) % 16 == 0) {
        dag.r *= q;
        dag.h /= q;
        break;
      }
  }
  KernelBuilder kb(kernel);
  Planner planner(kb, dag);
  if (failed(planner.declare(func, kernel.written)))
    return failure();
  kb.beginBody();
  FailureOr<SmallVector<Value>> hbm = planner.emitBody(func);
  if (failed(hbm))
    return failure();
  kb.finish(*hbm);
  return success();
}

} // namespace

OpClass mlir::furiosa::mapping::rowwiseClass() { return {"row-wise", match, map}; }

// Op class: row-wise kernels — elementwise generics and row reductions over a common [R, H] space (rows R,
// columns H), as in normalizations and softmax. The kernel's generics fold into one scalar expression DAG whose
// nodes are classified by what they vary along: constants, rows (Row), columns (Col) or both (Full). The
// Vector Engine then evaluates it in passes:
//
//  - Layout: one row per slice (padded to 256), a row's H columns in time x packet. More than 256 rows run as
//    row tiles of up to 256, each read from and written to an HBM tile view of the buffers. A row length that is
//    not a multiple of 16 pads to one (whole packets): rows cross between HBM and DM through a padded scratch, a
//    reduction's float input padded with its identity.
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
#include "../Tile/TileLoops.h"
#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/TypeSwitch.h"

#include <cmath>
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
  unsigned rows = 0;       // Input: the row dims it varies along (bit i: row dim i), broadcast along the others
  bool col = false;        // Input: whether it varies along the columns
  float value = 0;         // Const
  std::string op;          // Op: add, sub, mul, div, max, min, exp, sqrt, neg, select, eqkey, ...; Reduce: Add, Max,
                           // Min, MinI, MaxI (i32)
  std::string cmp;         // Op select: the tag comparison its key (args[0]) passes where it takes args[1]
  SmallVector<int> args;   // Op / Reduce operands
  bool integer = false;    // an i32 value: streams carry its bits as f32 between the i32 ops that reread them, and
                           // DM holds it as i32
};

struct Dag {
  std::vector<Node> nodes;
  SmallVector<int> roots;     // one per written buffer, in result order
  SmallVector<Type> outTypes; // their stored element types
  SmallVector<int64_t> rowExtents; // the row dims (all but the last output dim); r is their product
  int64_t r = 0, h = 0;
  /// The row tiles: windows of `tileStep` along row dim `tileDim` (furiosa-tile's loop, or rowTile's choice for
  /// an untiled kernel); none (step 0) when every row has its slice.
  int tileDim = -1;
  int64_t tileStep = 0;

  /// Adds a node, or returns the equal one already there: generics recompute shared subexpressions (each
  /// reads the buffers again), and the DAG keeps one of each.
  int add(Node n) {
    std::string key = std::to_string(int(n.kind)) + "|" + std::to_string(n.arg) + "|" +
                      std::to_string(llvm::bit_cast<uint32_t>(n.value)) + "|" + n.op + "|" + n.cmp + "|" + std::to_string(int(n.shape)) +
                      "|" + std::to_string(n.rows) + "|" + std::to_string(n.col) + "|" + std::to_string(n.integer);
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
  // A kernel furiosa-tile tiled: each stored value is a row tile loop's result. The DAG is its body's, one tile
  // of rows along one row dim; the buffers keep their whole shapes (the planner makes the tile views).
  // A reshape of a result moves no data: the buffer holds the generic's elements in the same order.
  SmallVector<Value> results(ret.getOperands());
  for (Value &v : results)
    while (v.getDefiningOp<tensor::CollapseShapeOp>() || v.getDefiningOp<tensor::ExpandShapeOp>())
      v = v.getDefiningOp()->getOperand(0);
  SmallVector<Value> stored(results);
  Block *body = &func.getBody().front();
  Value iv;
  for (Value &v : stored) {
    if (!v.getDefiningOp<scf::ForOp>())
      continue;
    FailureOr<std::optional<tile::TileNest>> nest = tile::tileNest(v, why);
    if (failed(nest))
      return failure();
    auto tiledDims = llvm::count_if((*nest)->loopOf, [](std::optional<unsigned> l) { return l.has_value(); });
    if ((*nest)->loops.size() != 1 || tiledDims != 1 || (iv && iv != (*nest)->loops[0].iv)) {
      why = "row tiles other than one loop over one row dim";
      return failure();
    }
    iv = (*nest)->loops[0].iv;
    dag.tileDim = llvm::find_if((*nest)->loopOf, [](std::optional<unsigned> l) { return l.has_value(); }) -
                  (*nest)->loopOf.begin();
    dag.tileStep = (*nest)->loops[0].step;
    body = (*nest)->insert->getBlock();
    v = (*nest)->tile;
  }
  if (iv && llvm::any_of(results, [](Value v) { return !v.getDefiningOp<scf::ForOp>(); })) {
    why = "stored values of which only some are row-tiled";
    return failure();
  }
  // The outputs: [rows..., columns] (the highest rank), or [rows...] for values of a row.
  RankedTensorType full;
  for (Value result : results) {
    auto t = cast<RankedTensorType>(result.getType());
    if (!full || t.getRank() > full.getRank())
      full = t;
    dag.outTypes.push_back(t.getElementType());
  }
  for (Value result : results) {
    auto t = cast<RankedTensorType>(result.getType());
    if (t.getShape() != full.getShape() && t.getShape() != full.getShape().drop_back() &&
        !llvm::all_of(results, [&](Value v) { return v.getType() == t; })) {
      why = "expected [rows..., columns] outputs of one shape, and [rows...] ones";
      return failure();
    }
  }
  // Only row values stored: the rows are their shape, and the columns the extent a reduction over the rows
  // reduces.
  SmallVector<int64_t> tileShape(full.getShape());
  if (dag.tileStep)
    tileShape[dag.tileDim] = dag.tileStep;
  linalg::GenericOp reducer;
  for (auto g : body->getOps<linalg::GenericOp>())
    if (!reducer && g.getNumReductionLoops() == 1 &&
        cast<RankedTensorType>(g.getResultTypes()[0]).getShape() == ArrayRef<int64_t>(tileShape))
      reducer = g;
  bool sameShapes = llvm::all_of(results, [&](Value v) { return v.getType() == results[0].getType(); });
  bool onlyRows = sameShapes && reducer;
  if (!onlyRows && full.getRank() < 1) {
    why = "expected a [rows..., columns] output (a single row when 1-D)";
    return failure();
  }
  if (onlyRows) {
    int64_t reduced = 1;
    if (reducer)
      for (auto [d, it] : llvm::enumerate(reducer.getIteratorTypesArray()))
        if (it == utils::IteratorType::reduction)
          reduced = reducer.getStaticLoopRanges()[d];
    dag.rowExtents.assign(full.getShape().begin(), full.getShape().end());
    dag.r = full.getNumElements();
    dag.h = reduced;
  } else {
    dag.rowExtents.assign(full.getShape().begin(), full.getShape().end() - 1);
    dag.r = full.getNumElements() / full.getShape().back();
    dag.h = full.getShape().back();
  }
  const unsigned k = dag.rowExtents.size();

  std::map<Operation *, int> genericNode; // a generic's yielded value at a point of its domain
  std::function<FailureOr<int>(linalg::GenericOp)> genericValue;

  // What a tensor read through `m` in a generic with `iters` varies along. The space is (rows..., column) for
  // k + 1 loops, the rows for k parallel loops, and a row's columns for one reduction loop (a single row,
  // rank-reduced). A constant 0 indexes a dimension of size 1. A read that misses some row dims is broadcast
  // along them (`rows` records the ones it has); its shape is that of the rows it fills, Row or Full.
  struct Read {
    Shape shape;
    unsigned rows = 0;
    bool col = false;
  };
  unsigned nonUnit = 0;
  for (unsigned i = 0; i < k; ++i)
    if (dag.rowExtents[i] != 1)
      nonUnit |= 1u << i;
  // The role of each loop of a generic: a row dim, or the column (-1). A reduction loop is the column; the
  // parallel loops are the row dims (all, or the ones of more than one row, when unit dims were dropped),
  // then the column unless there is a reduction or they are only rows, whichever the loop extents fit.
  SmallVector<unsigned> nonUnitDims;
  for (unsigned i = 0; i < k; ++i)
    if (dag.rowExtents[i] != 1)
      nonUnitDims.push_back(i);
  auto rolesOf = [&](ArrayRef<utils::IteratorType> iters, ArrayRef<int64_t> extents) -> std::optional<SmallVector<int>> {
    SmallVector<unsigned> parallel;
    int reduction = -1;
    for (auto [d, it] : llvm::enumerate(iters)) {
      if (it == utils::IteratorType::reduction) {
        if (reduction >= 0)
          return std::nullopt;
        reduction = d;
      } else {
        parallel.push_back(d);
      }
    }
    SmallVector<unsigned> allDims;
    for (unsigned i = 0; i < k; ++i)
      allDims.push_back(i);
    for (ArrayRef<unsigned> rowDims : {ArrayRef<unsigned>(allDims), ArrayRef<unsigned>(nonUnitDims)})
      for (bool withCol : {true, false}) {
        if (withCol && reduction >= 0)
          continue;
        if (parallel.size() != rowDims.size() + (withCol ? 1 : 0))
          continue;
        SmallVector<int> roles(iters.size(), -1);
        bool fits = reduction < 0 || extents[reduction] == dag.h;
        for (auto [i, d] : llvm::enumerate(rowDims)) {
          roles[parallel[i]] = d;
          fits &= extents[parallel[i]] == (int(d) == dag.tileDim ? dag.tileStep : dag.rowExtents[d]);
        }
        if (withCol)
          fits &= extents[parallel.back()] == dag.h;
        if (fits)
          return roles;
      }
    return std::nullopt;
  };
  auto shapeOfMap = [&](AffineMap m, ArrayRef<int> roles, RankedTensorType t) -> std::optional<Read> {
    Read read;
    int last = -1;
    for (auto [i, e] : llvm::enumerate(m.getResults())) {
      if (auto c = dyn_cast<AffineConstantExpr>(e); c && c.getValue() == 0 && t.getDimSize(i) == 1)
        continue; // a unit dim
      auto d = dyn_cast<AffineDimExpr>(e);
      if (!d || int(d.getPosition()) <= last)
        return std::nullopt; // not a row-major projection
      last = d.getPosition();
      if (roles[d.getPosition()] < 0)
        read.col = true;
      else
        read.rows |= 1u << roles[d.getPosition()];
    }
    read.rows &= nonUnit;
    read.shape = !read.col ? Shape::Row : read.rows || !nonUnit ? Shape::Full : Shape::Col;
    return read;
  };

  // The node for a tensor read through `map` inside a generic.
  std::function<FailureOr<int>(Value, AffineMap, ArrayRef<int>)> readNode;
  readNode = [&](Value t, AffineMap map, ArrayRef<int> roles) -> FailureOr<int> {
    std::optional<Read> read = shapeOfMap(map, roles, cast<RankedTensorType>(t.getType()));
    if (!read) {
      why = "an operand map is not a projection of (rows..., column)";
      return failure();
    }
    std::optional<Shape> shape = read->shape;
    // In a row tile: a buffer's tile (its tiled dim at the loop's offset, the rest whole) reads the buffer.
    if (auto slice = t.getDefiningOp<tensor::ExtractSliceOp>(); slice && iv && slice->getBlock() == body) {
      auto src = dyn_cast<BlockArgument>(slice.getSource());
      bool tileOfIt = src && src.getOwner() == &func.getBody().front() && slice.hasUnitStride() &&
                      slice.getSourceType().getRank() == slice.getType().getRank();
      for (int64_t d = 0; tileOfIt && d < slice.getSourceType().getRank(); ++d) {
        FailureOr<AffineExpr> at = tile::offsetExpr(slice.getMixedOffsets()[d], {iv}, func.getContext());
        int64_t size = slice.getStaticSizes()[d], whole = slice.getSourceType().getDimSize(d);
        tileOfIt = succeeded(at) && ((size == whole && *at == getAffineConstantExpr(0, func.getContext())) ||
                                     (size == dag.tileStep && *at == getAffineDimExpr(0, func.getContext())));
      }
      if (!tileOfIt) {
        why = "a row tile reads a slice that is not a buffer's tile";
        return failure();
      }
      t = slice.getSource();
    }
    if (auto arg = dyn_cast<BlockArgument>(t); arg && arg.getOwner() == &func.getBody().front()) {
      Node n{Node::Input};
      n.shape = *shape;
      n.arg = arg.getArgNumber();
      n.type = cast<RankedTensorType>(t.getType()).getElementType();
      n.integer = n.type.isInteger(32);
      n.rows = read->rows;
      n.col = read->col;
      return dag.add(n);
    }
    // A fill is a constant (an i32 one, its bits).
    if (auto fill = t.getDefiningOp<linalg::FillOp>()) {
      llvm::APFloat f(0.0f);
      if (matchPattern(fill.getInputs()[0], m_ConstantFloat(&f))) {
        bool lost;
        f.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven, &lost);
        Node n{Node::Const};
        n.value = f.convertToFloat();
        return dag.add(n);
      }
      if (APInt i; matchPattern(fill.getInputs()[0], m_ConstantInt(&i)) && i.getBitWidth() == 32) {
        Node n{Node::Const};
        n.value = llvm::bit_cast<float>(uint32_t(i.getZExtValue()));
        return dag.add(n);
      }
    }
    auto producer = t.getDefiningOp<linalg::GenericOp>();
    if (!producer) {
      why = "an operand is neither a buffer nor a generic's result";
      return failure();
    }
    // A producer over the rows read without the column, or over (rows, column) read with it, all rows.
    bool rowProducer = cast<RankedTensorType>(t.getType()).getRank() <= int64_t(k) && !read->col;
    if (rowProducer != (*shape == Shape::Row) || read->rows != nonUnit) {
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
    std::optional<SmallVector<int>> roles = rolesOf(iters, g.getStaticLoopRanges());
    if (!roles) {
      why = reduction ? "only reductions over the columns of each row are mapped"
                      : "a generic whose loops are not the rows and columns of the outputs";
      return failure();
    }
    Block &body = g.getRegion().front();
    llvm::DenseMap<Value, int> scalar; // body values -> nodes
    for (OpOperand *in : g.getDpsInputOperands()) {
      auto n = readNode(in->get(), g.getMatchingIndexingMap(in), *roles);
      if (failed(n))
        return failure();
      scalar[body.getArgument(in->getOperandNumber())] = *n;
    }
    Value acc = body.getArgument(g.getNumDpsInputs());
    std::function<FailureOr<int>(arith::SelectOp)> selectNode;
    std::function<FailureOr<int>(arith::SelectOp, arith::CmpIOp)> intSelectNode;
    auto addSelect = [&](Node n) -> int {
      for (int arg : n.args)
        n.shape = join(n.shape, dag.nodes[arg].shape);
      return dag.add(n);
    };
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
      if (APInt i; matchPattern(v, m_ConstantInt(&i)) &&
                   (i.getBitWidth() == 32 || (v.getType().isIndex() && i.isSignedIntN(32)))) {
        Node n{Node::Const};
        n.value = llvm::bit_cast<float>(uint32_t(i.getSExtValue()));
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
      if (auto sel = dyn_cast<arith::SelectOp>(op)) {
        auto n = selectNode(sel);
        if (failed(n))
          return failure();
        return scalar[v] = *n;
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
                             .Case<math::Exp2Op>([](auto) { return "exp2"; })
                             .Case<math::SqrtOp>([](auto) { return "sqrt"; })
                             .Case<math::RsqrtOp>([](auto) { return "rsqrt"; })
                             .Case<math::TanhOp>([](auto) { return "tanh"; })
                             .Case<math::LogOp>([](auto) { return "log"; })
                             .Case<math::ErfOp>([](auto) { return "erf"; })
                             .Case<math::SinOp>([](auto) { return "sin"; })
                             .Case<math::CosOp>([](auto) { return "cos"; })
                             .Case<arith::ExtFOp>([](auto) { return "ext"; })
                             .Case<arith::TruncFOp>([](auto) { return "trunc"; })
                             .Case<arith::AddIOp>([](auto) { return "addi"; })
                             .Case<arith::SubIOp>([](auto) { return "subi"; })
                             .Case<arith::MulIOp>([](auto) { return "muli"; })
                             .Case<arith::MaxSIOp>([](auto) { return "maxi"; })
                             .Case<arith::MinSIOp>([](auto) { return "mini"; })
                             .Case<arith::XOrIOp>([](auto) { return "xori"; })
                             .Case<arith::AndIOp>([](auto) { return "andi"; })
                             .Case<arith::OrIOp>([](auto) { return "ori"; })
                             .Case<arith::ShLIOp>([](auto) { return "shli"; })
                             .Case<arith::ShRSIOp>([](auto) { return "shrsi"; })
                             .Case<math::AbsIOp>([](auto) { return "absi"; })
                             .Case<arith::IndexCastOp>([](auto) { return "ext"; })
                             .Case<math::RoundOp>([](math::RoundOp r) { return r.getType().isF32() ? "round" : ""; })
                             .Case<arith::SIToFPOp>([](arith::SIToFPOp c) {
                               return c.getIn().getType().isInteger(32) && c.getType().isF32() ? "itof" : "";
                             })
                             .Default([](auto) { return ""; });
      // Integer ops compute on i32s (index values are i32s cast to index and back).
      Type rt = op->getResult(0).getType();
      if (StringRef(name).ends_with("i") && !rt.isInteger(32) && !rt.isIndex())
        name = "";
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
      // Casts: an extension of an input is the f32 fetch; the final truncation is the output cast; an index cast
      // keeps an i32's value.
      if (name == "ext" || name == "trunc")
        return scalar[v] = args[0];
      // round(x) = the integer nearest x (ties away from zero) back in f32: the Vector Engine's conversions.
      if (name == "round") {
        Node toInt{Node::Op};
        toInt.op = "ftoi";
        toInt.integer = true;
        toInt.args = args;
        toInt.shape = dag.nodes[args[0]].shape;
        name = "itof";
        args = {dag.add(toInt)};
      }
      // 2^x = e^(x ln 2): the Vector Engine has e^x (within its ulp budget, not bit-exact to exp2).
      if (name == "exp2") {
        Node ln2{Node::Const};
        ln2.value = float(M_LN2);
        Node scaled{Node::Op};
        scaled.op = "mul";
        scaled.args = {args[0], dag.add(ln2)};
        scaled.shape = dag.nodes[args[0]].shape;
        name = "exp";
        args = {dag.add(scaled)};
      }
      Node n{Node::Op};
      n.op = name;
      n.args = args;
      n.integer = v.getType().isInteger(32) || v.getType().isIndex();
      for (int a : args)
        n.shape = join(n.shape, dag.nodes[a].shape);
      return scalar[v] = dag.add(n);
    };
    // select(x <pred> y, a, b) takes a where its key passes the tag comparison: x itself against a constant y
    // (the tag unit compares with immediates); x - y > 0 (< 0) for a strict order, exact since IEEE subtraction
    // keeps the sign and equal infinities give NaN, false either way; for equality the key (x + 0) ^ (y + 0),
    // zero exactly when the bits match once -0 is +0, a NaN x kept nonzero (eqkey's own tag).
    selectNode = [&](arith::SelectOp sel) -> FailureOr<int> {
      if (auto ci = sel.getCondition().getDefiningOp<arith::CmpIOp>(); ci && ci->getBlock() == &body)
        return intSelectNode(sel, ci);
      auto c = sel.getCondition().getDefiningOp<arith::CmpFOp>();
      if (!c || c->getBlock() != &body) {
        why = "a select whose condition is not a float comparison in the body";
        return failure();
      }
      auto x = node(c.getLhs()), y = node(c.getRhs()), a = node(sel.getTrueValue()), f = node(sel.getFalseValue());
      if (failed(x) || failed(y) || failed(a) || failed(f))
        return failure();
      using P = arith::CmpFPredicate;
      P pred = c.getPredicate();
      if (dag.nodes[*x].kind == Node::Const) {
        std::swap(x, y);
        pred = pred == P::OGT ? P::OLT : pred == P::OLT ? P::OGT : pred;
      }
      auto opNode = [&](StringRef name, SmallVector<int> args) {
        Node n{Node::Op};
        n.op = name.str();
        n.args = args;
        for (int a : args)
          n.shape = join(n.shape, dag.nodes[a].shape);
        return dag.add(n);
      };
      Node n{Node::Op};
      n.op = "select";
      n.integer = sel.getType().isInteger(32);
      int key;
      if (dag.nodes[*y].kind == Node::Const && (pred == P::OGT || pred == P::OLT || pred == P::OEQ)) {
        std::string k = rustF32(dag.nodes[*y].value);
        key = *x;
        n.cmp = (pred == P::OGT ? "Greater(" : pred == P::OLT ? "Less(" : "Equal(") + k + ")";
      } else if (pred == P::OGT || pred == P::OLT) {
        key = opNode("sub", {*x, *y});
        n.cmp = pred == P::OGT ? "Greater(0.0f32)" : "Less(0.0f32)";
      } else if (pred == P::OEQ) {
        int zero = dag.add(Node{Node::Const});
        key = opNode("eqkey", {opNode("add", {*x, zero}), opNode("add", {*y, zero})});
        n.cmp = "Equal(0i32)"; // the key is committed as i32 (see commit)
      } else {
        why = "no exact Vector Engine comparison for predicate " + arith::stringifyCmpFPredicate(pred).str();
        return failure();
      }
      n.args = {key, *a, *f};
      return addSelect(n);
    };
    // select(x <pred> y, a, b) on i32s: the tag unit compares i32 keys exactly, x against a constant y, else
    // x - y against 0; >=, <= and != take the other branch of <, > and ==.
    intSelectNode = [&](arith::SelectOp sel, arith::CmpIOp c) -> FailureOr<int> {
      auto x = node(c.getLhs()), y = node(c.getRhs()), a = node(sel.getTrueValue()), f = node(sel.getFalseValue());
      if (failed(x) || failed(y) || failed(a) || failed(f))
        return failure();
      using P = arith::CmpIPredicate;
      P pred = c.getPredicate();
      if (dag.nodes[*x].kind == Node::Const) {
        std::swap(x, y);
        pred = pred == P::sgt ? P::slt : pred == P::slt ? P::sgt : pred == P::sge ? P::sle : pred == P::sle ? P::sge : pred;
      }
      if (pred == P::ne || pred == P::sge || pred == P::sle) {
        pred = arith::invertPredicate(pred);
        std::swap(a, f);
      }
      if (pred != P::eq && pred != P::sgt && pred != P::slt) {
        why = "no i32 tag comparison for predicate " + arith::stringifyCmpIPredicate(pred).str();
        return failure();
      }
      Node n{Node::Op};
      n.op = "select";
      n.integer = sel.getType().isInteger(32) || sel.getType().isIndex();
      int key = *x;
      int32_t bound = 0;
      if (dag.nodes[*y].kind == Node::Const) {
        bound = llvm::bit_cast<int32_t>(dag.nodes[*y].value);
      } else {
        // shortcut: x - y wraps for operands 2^31 or more apart; compare in two steps when such ranges matter.
        Node sub{Node::Op};
        sub.op = "subi";
        sub.integer = true;
        sub.args = {*x, *y};
        sub.shape = join(dag.nodes[*x].shape, dag.nodes[*y].shape);
        key = dag.add(sub);
      }
      n.cmp = (pred == P::sgt ? "Greater(" : pred == P::slt ? "Less(" : "Equal(") + std::to_string(bound) + "i32)";
      n.args = {key, *a, *f};
      return addSelect(n);
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
                       (op == "Min" && init.isInfinity() && !init.isNegative()) ||
                       (init.isNaN() && isa<arith::MaxNumFOp, arith::MinNumFOp>(comb))); // maxnum's identity
      if (APInt i; isa<arith::MinSIOp, arith::MaxSIOp, arith::AddIOp>(comb)) {
        op = isa<arith::MinSIOp>(comb) ? "MinI" : isa<arith::MaxSIOp>(comb) ? "MaxI" : "AddI";
        identity = fill && matchPattern(fill.getInputs()[0], m_ConstantInt(&i)) && i.getBitWidth() == 32 &&
                   (op == "MinI" ? i.isMaxSignedValue() : op == "MaxI" ? i.isMinSignedValue() : i.isZero());
      }
      if (op.empty() || !identity) {
        why = "a reduction must be add/max/min (or i32 add/min/max) from its identity";
        return failure();
      }
      auto t = node(term);
      if (failed(t))
        return failure();
      Node n{Node::Reduce};
      n.op = op;
      n.integer = StringRef(op).ends_with("I");
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

  for (Value result : stored) {
    auto producer = result.getDefiningOp<linalg::GenericOp>();
    if (!producer) {
      why = "a result is not a generic's";
      return failure();
    }
    auto root = genericValue(producer);
    if (failed(root))
      return failure();
    bool rowResult = cast<RankedTensorType>(result.getType()).getRank() == int64_t(k);
    if (dag.nodes[*root].shape != (rowResult ? Shape::Row : Shape::Full)) {
      why = "a stored value does not vary along what its buffer spans";
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

/// c - x -> -x + c when x is the streamed operand (c of a smaller shape: a constant, a row value): the Vector
/// Engine subtracts only from the streamed value. Negation is exact, so the sum is the difference.
void rewriteReversedSub(Dag &dag) {
  for (size_t i = 0; i < dag.nodes.size(); ++i) {
    bool integer = dag.nodes[i].op == "subi";
    if (dag.nodes[i].kind != Node::Op || (dag.nodes[i].op != "sub" && !integer))
      continue;
    int c = dag.nodes[i].args[0], x = dag.nodes[i].args[1];
    if (dag.nodes[c].shape == dag.nodes[i].shape)
      continue;
    Node neg{Node::Op};
    neg.op = integer ? "muli" : "neg"; // an i32 negates as x * -1 (wrapping, like 0 - x)
    neg.integer = integer;
    neg.args = {x};
    if (integer) {
      Node minusOne{Node::Const};
      minusOne.value = llvm::bit_cast<float>(int32_t(-1));
      neg.args.push_back(dag.add(minusOne));
    }
    neg.shape = dag.nodes[x].shape;
    int negNode = dag.add(neg);
    Node &n = dag.nodes[i];
    n.op = integer ? "addi" : "add";
    n.args = {negNode, c};
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
enum Stage {
  Tag = 0, Logic = 1, Fxp = 2, FxpToFp = 3, Narrow = 4, Fp = 5, Reduce = 6, FpDiv = 7, Widen = 8, FpToFxp = 9, Clip = 10
};

class Planner {
public:
  Planner(KernelBuilder &kb, Dag &dag, RaisedKernel &raised) : kb(kb), dag(dag), raised(raised), ctx(kb.getContext()) {}

  /// Declares the axes and the kernel's inputs and outputs (before the body).
  LogicalResult declare(func::FuncOp func, ArrayRef<int64_t> written);
  /// Emits the passes and returns the stored HbmTensors, one per root (none when the rows are tiled: the
  /// outputs are then written in place).
  FailureOr<SmallVector<Value>> emitBody(func::FuncOp func);
  bool tiled() const { return dag.tileStep > 0; }

private:
  KernelBuilder &kb;
  Dag &dag;
  RaisedKernel &raised;
  MLIRContext *ctx;
  SmallVector<std::pair<int, MapAttr>> rowAxes; // (row dim, axis), outermost first; dim -1: the single row
  int64_t inner = 1;                             // rows inside the outermost row axis
  MapAttr H, slice; // H: the column axis, padded to whole packets
  MapAttr columns;   // the columns HBM holds (H resized to them)
  std::map<int, float> padFill; // padded rows: input node -> the identity of the reduction that reads it
  std::map<int, Value> materialized; // node -> DmTensor (Full: [slice, H], Row: [slice, 1 # 8])
  std::map<int, Value> vrfs;         // node -> VrfTensor
  std::map<int, unsigned> params;    // input node -> kernel parameter
  std::map<int, Value> inputDm;      // input node -> DmTensor
  SmallVector<unsigned> outParams;   // tiled: the in-place output parameters

  /// The slice mapping of `len` rows of the outermost row axis (all of the inner ones), padded to 256.
  MapAttr rowsMap(int64_t len);
  /// The HBM mapping of an input: the row axes it varies along, then the column axis.
  MapAttr inputMap(const Node &n);
  /// The HBM mapping of an output: every row axis, then the column axis (none for row values, one f32 a row).
  MapAttr outMap(bool row = false);
  /// The HBM scratch a value crosses between HBM and DM through when the DMA, which moves 8-byte units, cannot
  /// move it as the view holds it: a row value (its f32 padded to 8 bytes) or rows padded to whole packets. The
  /// view's rows (a tile's padding dropped), then the DM element; byte-granular HBM-to-HBM copies fill and drain
  /// it. Null when the view moves directly.
  MapAttr scratchMap(MapAttr viewMap, bool col);
  /// The columns of a padded scratch's view (of mapping `staged`), the part a copy to or from HBM moves.
  Value interior(Value scratchView, MapAttr staged, bool mut);
  /// The DM element of an input: the columns, or a row value's padded f32.
  MapAttr inputElement(const Node &n) { return n.col ? H : rowPacket(2); }
  /// An input's DmTensor from (a view of) its HBM tensor, through a padded scratch when it needs one.
  Value loadInput(const Node &n, Value hbmView, MapAttr viewMap);
  /// Writes a stored root's DmTensor to (a mutable view of) its HBM tensor; a row value goes through a padded
  /// scratch.
  void storeOutput(size_t k, Value dm, Value hbmView, MapAttr viewMap);
  bool rootRow(size_t k) { return dag.nodes[dag.roots[k]].shape == Shape::Row; }
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
    if (vrfBytes + H.size() * 4 > kVrfBytes)
      return fail(func, "VRF operands of ") << H.size() << " f32 per slice exceed the 8 KB VRF", failure();
    vrfBytes += H.size() * 4;
    FailureOr<Value> dm = n.kind == Node::Input ? FailureOr<Value>(inputDm.at(node)) : materialize(node, func);
    if (failed(dm))
      return failure();
    Type et = cast<DmTensorType>(dm->getType()).getElementType();
    Type vt = et.isInteger(32) ? et : f32(); // an i32 value's VRF is i32
    bool bf16 = et.isBF16();
    int64_t p = bf16 ? 16 : 8;
    Value s = BeginOp::create(b, loc, stream(et, kb.one(), H), *dm, ContextKind::Sub);
    s = FetchOp::create(b, loc, stream(et, H.stride(p), H.modulo(p)), s);
    if (bf16)
      s = FetchCastOp::create(b, loc, stream(f32(), H.stride(p), H.modulo(p)), s);
    s = CollectOp::create(b, loc, stream(vt, H.stride(8), H.modulo(8)), s);
    v = ToVrfOp::create(b, loc, VrfTensorType::get(ctx, vt, kb.chip(), kb.cluster(), slice, H), s);
  } else if (n.shape == Shape::Row) {
    if (vrfBytes + 32 > kVrfBytes)
      return fail(func, "row VRF operands exceed the 8 KB VRF"), failure();
    vrfBytes += 32;
    auto dm = materialize(node, func);
    if (failed(dm))
      return failure();
    MapAttr held = cast<DmTensorType>(dm->getType()).getElement(); // 1 # 8 when computed, 1 # 2 when read
    Type vt = cast<DmTensorType>(dm->getType()).getElementType().isInteger(32) ? Type(IntegerType::get(ctx, 32)) : f32();
    Value s = BeginOp::create(b, loc, stream(vt, kb.one(), held), *dm, ContextKind::Sub);
    s = FetchOp::create(b, loc, stream(vt, kb.one(), held), s);
    s = CollectOp::create(b, loc, stream(vt, kb.one(), rowPacket(8)), s);
    v = ToVrfOp::create(b, loc, VrfTensorType::get(ctx, vt, kb.chip(), kb.cluster(), slice, rowPacket(8)), s);
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
    return inputDm.at(node);
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
    int operand2 = -1;     // a select's false value
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
    // A select streams its key, an equality key its first operand of the key's shape (it is symmetric).
    if (n.op == "select" || n.op == "eqkey") {
      Step step{leaf};
      int key = n.args[0], other = n.args[1];
      if (n.op == "eqkey" && dag.nodes[key].shape != n.shape)
        std::swap(key, other);
      if (dag.nodes[key].shape != n.shape)
        return fail(func, "a select whose key does not vary along its result"), failure();
      step.operand = other;
      if (n.op == "select")
        step.operand2 = n.args[2];
      path.push_back(step);
      leaf = key;
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
      bool ordered = n.op == "sub" || n.op == "div" || n.op == "subi";
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
      step.reversed = n.args[0] != streamed[0] && (n.op == "sub" || n.op == "div" || n.op == "subi"); // the others commute
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
    if (leafNode.shape != Shape::Full && leafNode.shape != Shape::Row)
      return fail(func, "a streamed input must vary along the rows"), failure();
    source = inputDm.at(leaf);
    sourceType = leafNode.type;
    rowStream = leafNode.shape == Shape::Row;
  } else {
    source = materialized.at(leaf);
    sourceType = cast<DmTensorType>(source.getType()).getElementType();
    rowStream = leafNode.shape == Shape::Row;
  }

  // VRF operands are loaded before the pass (their own passes come first).
  for (const Step &s : path)
    for (int o : {s.operand, s.operand2})
      if (o >= 0 && dag.nodes[o].kind != Node::Const && !s.zip)
        if (failed(vrfOf(o, func)))
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
  // The tag comparisons a pass starting at `st` needs: a select's, or an equality key's ordered test.
  auto tagsOf = [&](const Step &st) -> ArrayAttr {
    Node &n = dag.nodes[st.node];
    if (n.op == "select")
      return b.getStrArrayAttr({n.cmp, "False", "False", "False"});
    if (n.op == "eqkey")
      return b.getStrArrayAttr({"Less(f32::INFINITY)", "Equal(f32::INFINITY)", "False", "False"});
    if (n.op == "absi")
      return b.getStrArrayAttr({"Less(0i32)", "False", "False", "False"});
    return {};
  };
  bool tagged = false; // the pass began with the tags of the op it is at
  // An i32 stream (an equality key) is tagged as i32, then reread as f32 bits.
  auto begin = [&](Value dm, Type et, ArrayAttr tags) {
    Type vt = et.isInteger(32) ? et : f32();
    if (row) {
      MapAttr held = cast<DmTensorType>(dm.getType()).getElement();
      s = BeginOp::create(b, loc, stream(vt, kb.one(), held), dm, ContextKind::Main);
      s = FetchOp::create(b, loc, stream(vt, kb.one(), held), s);
      s = CollectOp::create(b, loc, stream(vt, kb.one(), rowPacket(8)), s);
    } else {
      bool bf16 = et.isBF16();
      int64_t p = bf16 ? 16 : 8;
      s = BeginOp::create(b, loc, stream(et, kb.one(), H), dm, ContextKind::Main);
      s = FetchOp::create(b, loc, stream(et, H.stride(p), H.modulo(p)), s);
      if (bf16)
        s = FetchCastOp::create(b, loc, stream(f32(), H.stride(p), H.modulo(p)), s);
      s = CollectOp::create(b, loc, stream(vt, fullTime8, fullPacket8), s);
    }
    s = VectorInitOp::create(b, loc, stream(vt, time(), packet()), s);
    // An i32 comparison (an i32 key's) tags the stream read as i32.
    if (tags && StringRef(cast<StringAttr>(tags[0]).getValue()).ends_with("i32)") && vt == f32()) {
      vt = IntegerType::get(ctx, 32);
      s = VectorReinterpretOp::create(b, loc, stream(vt, time(), packet()), s);
    }
    if (tags)
      s = VectorTagCmpOp::create(b, loc, stream(vt, time(), packet()), s, tags);
    else
      s = VectorIntraSliceTagOp::create(b, loc, stream(vt, time(), packet()), s);
    if (vt != f32())
      s = VectorReinterpretOp::create(b, loc, stream(f32(), time(), packet()), s);
    tagged = bool(tags);
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
    Type et = f32();
    // An equality key leaves as i32: the tag unit flushes subnormal f32s (verified on npu3), not i32s. So does an
    // i32 value.
    if (dag.nodes[at].op == "eqkey" || dag.nodes[at].integer) {
      et = IntegerType::get(ctx, 32);
      s = VectorReinterpretOp::create(b, loc, stream(et, time(), packet()), s);
    }
    s = VectorFinalOp::create(b, loc, stream(et, time(), packet()), s);
    MapAttr trim = row ? rowPacket(8) : fullPacket8;
    if (cast && cast != et) {
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
    Value lhs = inputDm.at(zipNode.args[0]), rhs = inputDm.at(zipNode.args[1]);
    MapAttr I = kb.axis("I", 2);
    int64_t p = 8; // 8-element packets (16 bytes of bf16) keep the collected time (H / 8, I) after a cast
    s = BeginInterleavedOp::create(b, loc, stream(sourceType, I, H), lhs, rhs, ContextKind::Main, I);
    s = FetchOp::create(b, loc, stream(sourceType, MapAttr::pair(ctx, {H.stride(p), I}), H.modulo(p)), s);
    if (sourceType.isBF16())
      s = FetchCastOp::create(b, loc, stream(f32(), MapAttr::pair(ctx, {H.stride(p), I}), H.modulo(p)), s);
    Type vt = sourceType.isInteger(32) ? sourceType : f32();
    s = CollectOp::create(b, loc, stream(vt, MapAttr::pair(ctx, {fullTime8, I}), fullPacket8), s);
    s = VectorInitOp::create(b, loc, stream(vt, MapAttr::pair(ctx, {fullTime8, I}), fullPacket8), s);
    s = VectorIntraSliceUnzipOp::create(b, loc, stream(vt, fullTime8, fullPacket8), s, I);
    if (vt != f32())
      s = VectorReinterpretOp::create(b, loc, stream(f32(), fullTime8, fullPacket8), s);
    stage = Tag;
    way4 = false;
    alus.clear();
    pair = true;
  } else {
    begin(source, sourceType, path.empty() ? ArrayAttr() : tagsOf(path[0]));
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
    enum { FpBinary, FpUnary, Div, ClipBin, ReduceOp, FpZip, ClipZip, Select, EqKey, IntOp, IntZip, AbsI, IToF, FToI } form;
    std::string method; // IntOp, IntZip: the furiosa-opt method
    float negImm = 0;
    // i32 ops: add, sub and mul in the fixed-point stage (an add in the clip stage once past it), max and min in
    // the clip stage, bitwise ops in the logic stage; each reads the stream as i32.
    static const std::map<std::string, std::tuple<Stage, std::string, std::string, std::string>> intOps = {
        {"addi", {Fxp, "FxpAdd", "vector_fxp", "FxpBinaryOp::AddFxp"}},
        {"subi", {Fxp, "FxpAdd", "vector_fxp", "FxpBinaryOp::SubFxp"}},
        {"muli", {Fxp, "FxpMul", "vector_fxp", "FxpBinaryOp::MulInt"}},
        {"maxi", {Clip, "ClipMax", "vector_clip", "ClipBinaryOpI32::Max"}},
        {"mini", {Clip, "ClipMin", "vector_clip", "ClipBinaryOpI32::Min"}},
        {"xori", {Logic, "LogicXor", "vector_logic", "LogicBinaryOpI32::BitXor"}},
        {"andi", {Logic, "LogicAnd", "vector_logic", "LogicBinaryOpI32::BitAnd"}},
        {"ori", {Logic, "LogicOr", "vector_logic", "LogicBinaryOpI32::BitOr"}},
        {"shli", {Fxp, "FxpLshift", "vector_fxp", "FxpBinaryOp::LeftShift"}},
        {"shrsi", {Fxp, "FxpRshift", "vector_fxp", "FxpBinaryOp::ArithRightShift"}}};
    if (auto it = intOps.find(op); it != intOps.end()) {
      auto [stg, a, m, v] = it->second;
      need = stg, alu = a, method = m, veOp = v, needWay4 = false;
      form = st.zip && i == 0 && zipFirst ? IntZip : IntOp;
      if (op == "addi" && form == IntOp && (stage > Fxp || alus.count("FxpAdd")))
        need = Clip, alu = "ClipAdd", method = "vector_clip", veOp = "ClipBinaryOpI32::AddFxp";
      if (form == IntZip)
        method += "_zip";
      else if (st.stash || st.reversed || st.operand < 0 ||
               (dag.nodes[st.operand].kind != Node::Const && !dag.nodes[st.operand].integer))
        return fail(func, "an i32 '" + op + "' whose other operand is neither an immediate nor an i32 value"),
               failure();
    } else if (op == "ftoi") {
      need = FpToFxp, needWay4 = false, alu = "FpToFxp", form = FToI;
    } else if (op == "itof") {
      need = FxpToFp, needWay4 = false, alu = "FxpToFp", form = IToF;
    } else if (op == "absi") {
      // |x| = (x ^ m) + (m & 1) with m = -1 where x < 0, else 0: tag branches in the logic and fixed-point stages.
      need = Logic, needWay4 = false, alu = "LogicXor", form = AbsI;
    } else if (st.zip && i == 0 && zipFirst) {
      bool clip = op == "max" || op == "min";
      need = clip ? Clip : Fp, form = clip ? ClipZip : FpZip, needWay4 = !clip;
      alu = clip ? "ClipZip" : op == "mul" ? "FpMul0" : op == "div" ? "FpFpu" : "FpFma";
      veOp = op == "add" ? "AddF" : op == "sub" ? "SubF" : op == "mul" ? "MulF(FpMulAlu::Mul0)" : op == "div" ? "DivF"
             : op == "max" ? "Max" : "Min";
      if (!clip && op != "add" && op != "sub" && op != "mul" && op != "div")
        return fail(func, "no Vector Engine zip for '" + op + "'"), failure();
    } else if (n.op == "select" || n.op == "eqkey") {
      // Logic ops right after the tags the pass began with (Way8).
      need = Logic, needWay4 = false, alu = n.op == "select" ? "LogicSelect" : "LogicXor";
      form = n.op == "select" ? Select : EqKey;
    } else if (n.kind == Node::Reduce) {
      need = Reduce, alu = "ReduceAccTree", form = ReduceOp, veOp = n.op;
    } else if (op == "sin" || op == "cos") {
      // The Vector Engine's sin and cos are exact only within [-2 pi, 2 pi] (harness/characterize) and nothing
      // bounds a computed angle here; tables of index-only angles are hoisted and never reach this point.
      return fail(func, op + " of a value not known to lie within [-2 pi, 2 pi] (no range reduction yet)"), failure();
    } else if (op == "exp" || op == "negexp" || op == "sigmoid" || op == "sqrt" || op == "tanh" || op == "log" ||
               op == "erf") {
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
      // The clip stage adds; it subtracts only an immediate (negated). A VRF subtraction needs FpFma, in a
      // new pass if this one has used it.
      bool immediate = st.operand >= 0 && dag.nodes[st.operand].kind == Node::Const;
      if ((stage <= Fp && !alus.count("FpFma")) || (op == "sub" && !immediate && !st.stash))
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
                     (needWay4 && stage > Narrow && !way4) || (form == ReduceOp && (row || pair)) ||
                     (pair && st.stash) || // pair mode has neither the stash nor the intra-slice reducer
                     ((form == Select || form == EqKey || form == AbsI) && (!tagged || stage != Tag || !alus.empty())) ||
                     (form == AbsI && alus.count("FxpAdd"));
    if (breakHere) {
      if (current == leaf && i == 0)
        return fail(func, "an op cannot start a pass"), failure();
      Value dm = commit(current, Type());
      row = dag.nodes[current].shape == Shape::Row;
      begin(dm, cast<DmTensorType>(dm.getType()).getElementType(), tagsOf(st));
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
    if (form != ReduceOp && form != FpUnary && form != FpZip && form != ClipZip && form != Select && form != EqKey &&
        form != IntZip && form != AbsI && form != IToF && form != FToI && !(op == "neg")) {
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
    case IToF:
      // An i32 is a fixed-point value with no fraction bits (integer width 31).
      s = VectorReinterpretOp::create(b, loc, stream(IntegerType::get(ctx, 32), time(), packet()), s);
      s = VectorFxpToFpOp::create(b, loc, stream(f32(), time(), packet()), s, b.getI32IntegerAttr(31));
      break;
    case FToI:
      // Rounds to the nearest integer (ties away from zero), saturating; the bits ride on as f32.
      s = VectorFpToFxpOp::create(b, loc, stream(IntegerType::get(ctx, 32), time(), packet()), s, b.getI32IntegerAttr(31));
      s = VectorReinterpretOp::create(b, loc, stream(f32(), time(), packet()), s);
      break;
    case IntOp:
    case IntZip:
    case AbsI: {
      Type it = IntegerType::get(ctx, 32);
      s = VectorReinterpretOp::create(b, loc, stream(it, time(), packet()), s);
      auto alu = [&](StringRef method, StringRef veOp, FloatAttr imm, ArrayAttr branches, Value vrf = {}) {
        s = VectorAluOp::create(b, loc, stream(it, time(), packet()), s, b.getStringAttr(method),
                                b.getStringAttr(veOp), StringAttr(), imm, vrf, UnitAttr(), branches);
      };
      if (form == AbsI) {
        auto i32 = [&](int32_t v) { return b.getF32FloatAttr(llvm::bit_cast<float>(v)); };
        auto slots = [&](int32_t taken, int32_t other) {
          return b.getArrayAttr({b.getArrayAttr({b.getStringAttr("1---"), i32(taken)}),
                                 b.getArrayAttr({b.getStringAttr("all"), i32(other)})});
        };
        alu("vector_logic", "LogicBinaryOpI32::BitXor", {}, slots(-1, 0));
        alu("vector_fxp", "FxpBinaryOp::AddFxp", {}, slots(1, 0));
        alus.insert("FxpAdd");
        need = Fxp;
      } else {
        alu(method, veOp, imm, {}, vrf);
      }
      s = VectorReinterpretOp::create(b, loc, stream(f32(), time(), packet()), s);
      break;
    }
    case ClipZip:
      s = VectorClipZipOp::create(b, loc, stream(f32(), time(), packet()), s, b.getStringAttr(veOp));
      break;
    case ReduceOp: {
      // An i32 reduce rereads the bits as i32 and leaves them in the f32 stream again.
      bool i32 = veOp == "MinI" || veOp == "MaxI" || veOp == "AddI";
      Type it = IntegerType::get(ctx, 32);
      if (i32)
        s = VectorReinterpretOp::create(b, loc, stream(it, time(), packet()), s);
      row = true;
      s = VectorIntraSliceReduceOp::create(b, loc, stream(i32 ? it : f32(), time(), packet()), s,
                                           b.getStringAttr(veOp == "AddI" ? "AddSat" : i32 ? veOp.substr(0, 3) : veOp),
                                           b.getStringAttr("H"));
      if (i32)
        s = VectorReinterpretOp::create(b, loc, stream(f32(), time(), packet()), s);
      break;
    }
    case Select: {
      // cond ? a : b, bit-exact: clear the key, OR in a where the tag holds, XOR in b where it does not (an rf
      // slot comes last, so "rf where it holds" is written "0 where it does not, else rf").
      // An i32 select runs on the stream read as i32, with i32 VRFs and immediates.
      Type vt = n.integer ? Type(IntegerType::get(ctx, 32)) : f32();
      if (n.integer)
        s = VectorReinterpretOp::create(b, loc, stream(vt, time(), packet()), s);
      auto logic = [&](StringRef op, FloatAttr imm, ArrayAttr branches, Value vrf) {
        s = VectorAluOp::create(b, loc, stream(vt, time(), packet()), s, b.getStringAttr("vector_logic"),
                                b.getStringAttr(((n.integer ? "LogicBinaryOpI32::" : "LogicBinaryOpF32::") + op).str()),
                                StringAttr(), imm, vrf, UnitAttr(), branches);
      };
      auto slot = [&](StringRef bits, Attribute v) { return b.getArrayAttr({b.getStringAttr(bits), v}); };
      FloatAttr zero = b.getF32FloatAttr(0);
      logic("BitAnd", zero, {}, {});
      Node &a = dag.nodes[st.operand], &f = dag.nodes[st.operand2];
      if (a.kind == Node::Const)
        logic("BitOr", {}, b.getArrayAttr({slot("1---", b.getF32FloatAttr(a.value)), slot("all", zero)}), {});
      else
        logic("BitOr", {}, b.getArrayAttr({slot("0---", zero), slot("all", b.getStringAttr("rf"))}), vrfs.at(st.operand));
      if (f.kind == Node::Const)
        logic("BitXor", {}, b.getArrayAttr({slot("1---", zero), slot("all", b.getF32FloatAttr(f.value))}), {});
      else
        logic("BitXor", {}, b.getArrayAttr({slot("1---", zero), slot("all", b.getStringAttr("rf"))}), vrfs.at(st.operand2));
      if (n.integer)
        s = VectorReinterpretOp::create(b, loc, stream(f32(), time(), packet()), s);
      break;
    }
    case EqKey:
      // x ^ y where x is ordered (tag bit 0 or 1), x itself (nonzero) where it is NaN.
      s = VectorAluOp::create(b, loc, stream(f32(), time(), packet()), s, b.getStringAttr("vector_logic"),
                              b.getStringAttr("LogicBinaryOpF32::BitXor"), StringAttr(), FloatAttr(), vrfs.at(st.operand),
                              UnitAttr(),
                              b.getArrayAttr({b.getArrayAttr({b.getStringAttr("00--"), b.getF32FloatAttr(0)}),
                                              b.getArrayAttr({b.getStringAttr("all"), b.getStringAttr("rf")})}));
      break;
    }
    stage = need;
    alus.insert(alu);
    current = st.node;
  }
  return commit(node, outType);
}

LogicalResult Planner::declare(func::FuncOp func, ArrayRef<int64_t> written) {
  // Rows move in packets of 8 f32 (16 bf16): a row length that is not a multiple of 16 pads to one. Copies move and
  // stores write the columns alone, so only a reduction reads the padding: a float input it reduces directly pads
  // with the reduction's identity.
  bool pad = dag.h % 16 && dag.h != 1;
  for (const Node &n : dag.nodes) {
    if (!pad || n.kind != Node::Reduce)
      continue;
    int in = n.args[0];
    float id = n.op == "Add" ? 0.0f : n.op == "Max" ? -INFINITY : INFINITY;
    // shortcut: i32 reductions and reductions of computed values over padded rows are refused (an i32 fill needs an
    // i32 memset; a computed value's padding is not its input's); pad those with a masked stream when a case needs it.
    if (n.integer || dag.nodes[in].kind != Node::Input || (padFill.count(in) && padFill[in] != id))
      return fail(func, "a reduction over padded rows of a value other than a float input");
    padFill[in] = id;
  }
  // A row axis per row dim of more than one row (a single row keeps one axis of one), and the column axis.
  for (auto [i, extent] : llvm::enumerate(dag.rowExtents))
    if (extent != 1)
      rowAxes.push_back({int(i), kb.axis(rowAxes.empty() ? "R" : "R" + std::to_string(rowAxes.size()), extent)});
  if (rowAxes.empty())
    rowAxes.push_back({-1, kb.axis("R", 1)});
  H = kb.axis("H", pad ? (dag.h + 15) / 16 * 16 : dag.h);
  columns = pad ? H.resize(dag.h) : H;
  inner = dag.r / rowAxes.front().second.size();
  if (kSlices % inner && inner * rowAxes.front().second.size() != kSlices)
    return fail(func, "rows inside the outermost row dim (") << inner << ") do not divide the 256 slices";
  // The NPU tiles a window of L rows over an extent E only when L grows into a divisor of E within [L, 2L).
  if (tiled()) {
    int64_t extent = rowAxes.front().second.size(), step = dag.tileStep;
    if (rowAxes.front().first != dag.tileDim || step * inner > kSlices)
      return fail(func, "row tiles that are not windows of the outermost row dim within the slices");
    auto windows = [&](int64_t len) {
      for (int64_t d = len; d < 2 * len && d <= extent; ++d)
        if (extent % d == 0)
          return true;
      return false;
    };
    if (!windows(step) || (extent % step && !windows(extent % step)))
      return fail(func, "row tiles of ") << step << " over " << extent << " rows, which the NPU cannot window";
  }
  slice = rowsMap(rowAxes.front().second.size());
  for (auto [i, n] : llvm::enumerate(dag.nodes)) {
    if (n.kind != Node::Input)
      continue;
    if (!n.type.isBF16() && !n.type.isF32() && !n.type.isInteger(32))
      return fail(func, "inputs must be bf16, f32 or i32");
    MapAttr m = inputMap(n);
    if (cast<RankedTensorType>(func.getArgumentTypes()[n.arg]).getNumElements() != m.size())
      return fail(func, "an input whose shape is not the rows and columns its map reads");
    params[i] = kb.input(n.arg, n.type, m);
  }
  for (auto [k, outType] : llvm::enumerate(dag.outTypes)) {
    if (!outType.isBF16() && !outType.isF32() && !outType.isInteger(32))
      return fail(func, "the outputs must be bf16, f32 or i32");
    if (rootRow(k) && outType.isBF16())
      return fail(func, "a row value is stored as f32 or i32");
    if (tiled())
      outParams.push_back(kb.inout(written[k], outType, outMap(rootRow(k))));
    else
      kb.output(written[k], outType, outMap(rootRow(k)));
  }
  return success();
}

MapAttr Planner::rowsMap(int64_t len) {
  MapAttr outer = rowAxes.front().second;
  if (len != outer.size())
    outer = outer.resize(len);
  if (len * inner < kSlices)
    outer = outer.padTo(kSlices / inner);
  SmallVector<MapAttr> parts{outer};
  for (auto &[d, axis] : llvm::drop_begin(rowAxes))
    parts.push_back(axis);
  return MapAttr::pair(ctx, parts);
}

MapAttr Planner::inputMap(const Node &n) {
  SmallVector<MapAttr> parts;
  for (auto &[d, axis] : rowAxes)
    if (d >= 0 && (n.rows >> d) & 1)
      parts.push_back(axis);
  if (n.col)
    parts.push_back(columns);
  return MapAttr::pair(ctx, parts);
}

Value Planner::loadInput(const Node &n, Value hbmView, MapAttr viewMap) {
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  DmTensorType dm = kb.dmType(n.type, slice, inputElement(n));
  MapAttr staged = scratchMap(viewMap, n.col);
  if (!staged)
    return ViewToDmOp::create(b, loc, dm, hbmView);
  Value scratch = AllocHbmOp::create(b, loc, kb.hbmType(n.type, staged));
  Value into = ViewOp::create(b, loc, HbmViewType::get(ctx, n.type, kb.chip(), staged, true), scratch);
  if (auto it = padFill.find(&n - dag.nodes.data()); it != padFill.end()) {
    // Eight bytes of the fill, broadcast by the DMA over the scratch; the copy then writes the columns.
    Value fill =
        AllocDmOp::create(b, loc, kb.dmType(n.type, kb.one().padTo(kSlices), rowPacket(n.type.isBF16() ? 4 : 2)));
    MemsetOp::create(b, loc, fill, b.getF32FloatAttr(it->second));
    DmToHbmViewOp::create(b, loc, fill, ViewOp::create(b, loc, into.getType(), scratch));
  }
  HbmToHbmViewOp::create(b, loc, hbmView, n.col ? interior(into, staged, true) : into);
  return ToDmOp::create(b, loc, dm, scratch);
}

Value Planner::interior(Value scratchView, MapAttr staged, bool mut) {
  SmallVector<MapAttr> parts;
  for (Attribute a : staged.getKind() == MapKind::Pair ? staged.getOperands() : ArrayRef<Attribute>(staged))
    parts.push_back(cast<MapAttr>(a));
  parts.back() = columns.padTo(H.size(), mut ? PadKind::Bottom : PadKind::Top);
  auto t = cast<HbmViewType>(scratchView.getType());
  return TileOp::create(kb.builder(), kb.loc(),
                        HbmViewType::get(ctx, t.getElementType(), t.getChip(), MapAttr::pair(ctx, parts), mut),
                        scratchView, H, kb.builder().getI64IntegerAttr(columns.size()),
                        kb.builder().getI64IntegerAttr(0));
}

MapAttr Planner::scratchMap(MapAttr viewMap, bool col) {
  if (col && H == columns)
    return {};
  SmallVector<MapAttr> live;
  for (Attribute a : viewMap.getKind() == MapKind::Pair ? viewMap.getOperands() : ArrayRef<Attribute>(viewMap)) {
    MapAttr m = cast<MapAttr>(a);
    live.push_back(m.getKind() == MapKind::Pad ? m.operand(0) : m);
  }
  if (col)
    live.pop_back(); // the columns, whole packets in the scratch
  live.push_back(col ? H : rowPacket(2));
  return MapAttr::pair(ctx, live);
}

void Planner::storeOutput(size_t k, Value dm, Value hbmView, MapAttr viewMap) {
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  Type et = dag.outTypes[k];
  MapAttr staged = scratchMap(viewMap, !rootRow(k));
  if (!staged) {
    DmToHbmViewOp::create(b, loc, dm, hbmView);
    return;
  }
  Value scratch = ToHbmOp::create(b, loc, kb.hbmType(et, staged), dm);
  Value from = ViewOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), staged, false), scratch);
  HbmToHbmViewOp::create(b, loc, rootRow(k) ? from : interior(from, staged, false), hbmView);
}

MapAttr Planner::outMap(bool row) {
  SmallVector<MapAttr> parts;
  for (auto &[d, axis] : rowAxes)
    parts.push_back(axis);
  if (!row)
    parts.push_back(columns);
  return MapAttr::pair(ctx, parts);
}

FailureOr<SmallVector<Value>> Planner::emitBody(func::FuncOp func) {
  OpBuilder &b = kb.builder();
  Location loc = kb.loc();
  if (!tiled()) {
    for (auto [i, p] : params)
      inputDm[i] = loadInput(dag.nodes[i],
                             ViewOp::create(b, loc,
                                            HbmViewType::get(ctx, dag.nodes[i].type, kb.chip(), inputMap(dag.nodes[i]),
                                                             false),
                                            kb.param(p)),
                             inputMap(dag.nodes[i]));
    return emitTile(func);
  }
  // Row tiles along the outermost row axis: each reads its rows of the inputs that have that axis through a
  // tile view (the others broadcast as they are), and writes its rows of the outputs through a mutable one.
  MapAttr outer = rowAxes.front().second;
  int64_t extent = outer.size(), step = dag.tileStep;
  for (int64_t start = 0; start < extent; start += step) {
    int64_t len = std::min(step, extent - start);
    slice = rowsMap(len);
    materialized.clear();
    vrfs.clear();
    inputDm.clear();
    vrfBytes = 0;
    // The tile of a view whose mapping has the outer axis, and its mapping.
    auto tileOf = [&](Value hbm, Type et, MapAttr whole, bool mut) -> std::pair<Value, MapAttr> {
      Value v = ViewOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), whole, mut), hbm);
      SmallVector<MapAttr> parts;
      for (Attribute a : whole.getKind() == MapKind::Pair ? whole.getOperands() : ArrayRef<Attribute>(whole))
        parts.push_back(cast<MapAttr>(a) == outer
                            ? outer.resize(len).padTo(extent, mut ? PadKind::Bottom : PadKind::Top)
                            : cast<MapAttr>(a));
      MapAttr part = MapAttr::pair(ctx, parts);
      return {TileOp::create(b, loc, HbmViewType::get(ctx, et, kb.chip(), part, mut), v, outer,
                             b.getI64IntegerAttr(len), b.getI64IntegerAttr(start)),
              part};
    };
    for (auto [i, p] : params) {
      Node &n = dag.nodes[i];
      bool hasOuter = rowAxes.front().first >= 0 && (n.rows >> rowAxes.front().first) & 1;
      if (hasOuter) {
        auto [view, map] = tileOf(kb.param(p), n.type, inputMap(n), false);
        inputDm[i] = loadInput(n, view, map);
      } else {
        Value view = ViewOp::create(b, loc, HbmViewType::get(ctx, n.type, kb.chip(), inputMap(n), false), kb.param(p));
        inputDm[i] = loadInput(n, view, inputMap(n));
      }
    }
    FailureOr<SmallVector<Value>> dms = emitTile(func);
    if (failed(dms))
      return failure();
    for (auto [k, dm] : llvm::enumerate(*dms)) {
      auto [view, map] = tileOf(kb.param(outParams[k]), dag.outTypes[k], outMap(rootRow(k)), true);
      storeOutput(k, dm, view, map);
    }
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
    if (tiled()) {
      outs[k] = *out;
    } else if (rootRow(k) || H != columns) {
      // A fresh dense buffer of row values or padded rows, through the padded scratch.
      MapAttr rows = outMap(rootRow(k));
      outs[k] = AllocHbmOp::create(b, loc, kb.hbmType(dag.outTypes[k], rows));
      storeOutput(k, *out, ViewOp::create(b, loc, HbmViewType::get(ctx, dag.outTypes[k], kb.chip(), rows, true), outs[k]),
                  rows);
    } else {
      outs[k] = ToHbmOp::create(b, loc, kb.hbmType(dag.outTypes[k], outMap()), *out);
    }
  }
  return outs;
}

/// Without reductions, row or column values or broadcast reads, the DAG is elementwise and its elements are
/// free to regroup into rows: of 16-element multiples, and at most kRow long to keep a second streamed tensor
/// within the VRF. Returns whether it regrouped.
bool regroup(Dag &dag) {
  unsigned allRows = 0;
  for (auto [i, e] : llvm::enumerate(dag.rowExtents))
    if (e != 1)
      allRows |= 1u << i;
  if (!llvm::all_of(dag.nodes, [&](const Node &n) {
        return n.kind == Node::Const || (n.shape == Shape::Full && (n.kind != Node::Input || n.rows == allRows));
      }))
    return false;
  // Each input, and a select's computed operand, may be a VRF operand of some pass: rows short enough for all of
  // them at once.
  int64_t operands = llvm::count_if(dag.nodes, [](const Node &n) { return n.kind == Node::Input || n.op == "select"; });
  int64_t kRow = std::min<int64_t>(1024, kVrfBytes / 4 / std::max<int64_t>(operands, 2) / 16 * 16);
  int64_t total = dag.r * dag.h;
  for (int64_t h = std::min(total, kRow); h >= 16 && (dag.h > kRow || dag.h % 16); --h)
    if (total % h == 0 && h % 16 == 0) {
      dag.r = total / h;
      dag.h = h;
      dag.rowExtents = {dag.r};
      for (Node &n : dag.nodes)
        if (n.kind == Node::Input)
          n.rows = dag.r > 1 ? 1 : 0;
      return true;
    }
  return false;
}

/// The row tiles, the one decision both furiosa-tile (through `tiling`) and the mapping of an untiled kernel
/// use: rows beyond the 256 slices take windows of the outermost row dim, as many of its rows as fill the slices
/// with all the rows inside it.
void rowTile(Dag &dag) {
  if (dag.r <= kSlices)
    return;
  for (auto [d, extent] : llvm::enumerate(dag.rowExtents))
    if (extent != 1) {
      dag.tileDim = d;
      dag.tileStep = std::max<int64_t>(1, kSlices / (dag.r / extent));
      return;
    }
}

bool match(RaisedKernel &kernel) {
  std::string why;
  bool ok = succeeded(fold(kernel.func, kernel.written, why));
  // FURIOSA_DEBUG=1 says why the class passes on a kernel.
  if (!ok && getenv("FURIOSA_DEBUG"))
    llvm::errs() << "row-wise: " << kernel.func.getName() << ": " << why << "\n";
  return ok;
}

LogicalResult map(RaisedKernel &kernel) {
  func::FuncOp func = kernel.func;
  std::string why;
  FailureOr<Dag> folded = fold(func, kernel.written, why);
  if (failed(folded))
    return func.emitError("row-wise: ") << why;
  Dag dag = *folded;
  rewriteRsqrt(dag);
  rewriteReversedSub(dag);
  rewriteSigmoid(dag);
  // A tiled kernel brings its row tiles; an untiled one takes rowTile's.
  if (!dag.tileStep) {
    regroup(dag);
    rowTile(dag);
  }
  KernelBuilder kb(kernel);
  Planner planner(kb, dag, kernel);
  if (failed(planner.declare(func, kernel.written)))
    return failure();
  kb.beginBody();
  FailureOr<SmallVector<Value>> hbm = planner.emitBody(func);
  if (failed(hbm))
    return failure();
  kb.finish(*hbm);
  return success();
}

/// furiosa-tile's loops for a kernel: rowTile's windows as a tile of the root generic's loop over that row dim.
/// Nothing for a kernel within the slices, a regrouped one (its rows are no loop of a generic), stored values
/// from several generics, or windows that do not divide the dim.
std::optional<Tiling> tiling(RaisedKernel &kernel) {
  std::string why;
  FailureOr<Dag> folded = fold(kernel.func, kernel.written, why);
  if (failed(folded) || folded->tileStep)
    return std::nullopt;
  Dag dag = *folded;
  if (regroup(dag))
    return std::nullopt;
  rowTile(dag);
  if (!dag.tileStep || dag.rowExtents[dag.tileDim] % dag.tileStep)
    return std::nullopt;
  auto ret = cast<func::ReturnOp>(kernel.func.getBody().front().getTerminator());
  auto root = ret.getOperand(0).getDefiningOp<linalg::GenericOp>();
  if (!root || llvm::any_of(ret.getOperands(), [&](Value v) { return v.getDefiningOp() != root; }))
    return std::nullopt;
  auto result = cast<OpResult>(ret.getOperand(0));
  AffineMap out = root.getMatchingIndexingMap(root.getDpsInitOperand(result.getResultNumber()));
  auto loop = dyn_cast<AffineDimExpr>(out.getResult(dag.tileDim));
  if (!loop)
    return std::nullopt;
  Tiling t;
  t.root = root;
  t.rowSizes.assign(root.getNumLoops(), 0);
  t.rowSizes[loop.getPosition()] = dag.tileStep;
  return t;
}

} // namespace

OpClass mlir::furiosa::mapping::rowwiseClass() { return {"row-wise", match, map, tiling}; }

// Structured linalg -> Program (Program.h): folds a kernel's generics, fills, pads, slices and reshapes into one
// expression over its output space, then splits divided dims into digits so every index is linear.

#include "Program.h"
#include "TileLoops.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <functional>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::tile;

//===----------------------------------------------------------------------===//
// Program helpers
//===----------------------------------------------------------------------===//

int64_t Lin::coeff(unsigned axis) const {
  for (auto [a, c] : terms)
    if (a == axis)
      return c;
  return 0;
}

void Lin::add(unsigned axis, int64_t c) {
  for (auto it = terms.begin(); it != terms.end(); ++it)
    if (it->first == axis) {
      it->second += c;
      if (!it->second)
        terms.erase(it);
      return;
    }
  if (c)
    terms.push_back({axis, c});
}

bool Lin::operator==(const Lin &o) const {
  if (offset != o.offset || terms.size() != o.terms.size())
    return false;
  for (auto [a, c] : terms)
    if (o.coeff(a) != c)
      return false;
  return true;
}

SmallVector<unsigned> Read::axes() const {
  std::set<unsigned> s;
  for (const Lin &l : index)
    for (auto [a, c] : l.terms)
      s.insert(a);
  return SmallVector<unsigned>(s.begin(), s.end());
}

SmallVector<unsigned> Program::axesOf(unsigned node) const {
  std::set<unsigned> s;
  std::function<void(unsigned)> walk = [&](unsigned n) {
    const Node &x = nodes[n];
    if (x.kind == Node::Leaf)
      for (unsigned a : reads[x.read].axes())
        s.insert(a);
    for (unsigned a : x.args)
      walk(a);
  };
  walk(node);
  // A reduction's own axes are not free in its result.
  std::function<void(unsigned)> strip = [&](unsigned n) {
    const Node &x = nodes[n];
    if (x.kind == Node::Reduce)
      for (unsigned a : x.axes)
        s.erase(a);
    for (unsigned a : x.args)
      strip(a);
  };
  strip(node);
  return SmallVector<unsigned>(s.begin(), s.end());
}

static void printLin(llvm::raw_ostream &os, const Lin &l) {
  bool first = true;
  for (auto [a, c] : l.terms) {
    os << (first ? "" : " + ") << (c == 1 ? "" : std::to_string(c) + "*") << "a" << a;
    first = false;
  }
  if (l.offset || first)
    os << (first ? "" : " + ") << l.offset;
}

void Program::print(llvm::raw_ostream &os) const {
  os << "axes:";
  for (auto [i, a] : llvm::enumerate(axes))
    os << " a" << i << "=" << a.size << (a.reduction ? "r" : "");
  os << "\nout:";
  for (unsigned a : out)
    os << " a" << a;
  os << "\nstore arg" << outArg << "[";
  llvm::interleave(outIndex, os, [&](const Lin &l) { printLin(os, l); }, ", ");
  os << "]\n";
  for (auto [i, n] : llvm::enumerate(nodes)) {
    os << "  %" << i << " = ";
    switch (n.kind) {
    case Node::Leaf: {
      const Read &r = reads[n.read];
      os << "arg" << r.arg << "[";
      llvm::interleave(r.index, os, [&](const Lin &l) { printLin(os, l); }, ", ");
      os << "]";
      if (r.fill)
        os << " fill " << *r.fill;
      break;
    }
    case Node::Const:
      os << n.value;
      break;
    case Node::Op:
      os << n.op;
      for (unsigned a : n.args)
        os << " %" << a;
      break;
    case Node::Reduce:
      os << "reduce." << n.op << " %" << n.args[0] << " over";
      for (unsigned a : n.axes)
        os << " a" << a;
      break;
    }
    os << "\n";
  }
  os << "root %" << root << "\n";
}

//===----------------------------------------------------------------------===//
// Digits
//===----------------------------------------------------------------------===//

namespace {

/// An affine expression as coefficients of atoms (digits or opaque sub-expressions) plus a constant.
struct Form {
  llvm::MapVector<AffineExpr, int64_t> terms;
  int64_t constant = 0;
  void add(const Form &o, int64_t scale) {
    for (auto [atom, c] : o.terms)
      terms[atom] += c * scale;
    constant += o.constant * scale;
  }
};

std::set<unsigned> dimsOf(AffineExpr e) {
  std::set<unsigned> out;
  e.walk([&](AffineExpr x) {
    if (auto d = dyn_cast<AffineDimExpr>(x))
      out.insert(d.getPosition());
  });
  return out;
}

/// `e` as sum(coefficient * dim) + constant, if it is one.
std::optional<std::pair<std::map<unsigned, int64_t>, int64_t>> linearInDims(AffineExpr e) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return std::make_pair(std::map<unsigned, int64_t>{}, c.getValue());
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return std::make_pair(std::map<unsigned, int64_t>{{d.getPosition(), 1}}, int64_t(0));
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return std::nullopt;
  if (e.getKind() == AffineExprKind::Add) {
    auto l = linearInDims(bin.getLHS()), r = linearInDims(bin.getRHS());
    if (!l || !r)
      return std::nullopt;
    for (auto [d, c] : r->first)
      l->first[d] += c;
    l->second += r->second;
    return l;
  }
  if (e.getKind() == AffineExprKind::Mul) {
    auto c = dyn_cast<AffineConstantExpr>(bin.getRHS());
    auto x = linearInDims(bin.getLHS());
    if (!c || !x)
      return std::nullopt;
    for (auto &[d, k] : x->first)
      k *= c.getValue();
    x->second *= c.getValue();
    return x;
  }
  return std::nullopt;
}

/// The places each dim must split at: the constants a map divides or reduces it by (x floordiv c, x mod c with
/// x mentioning that dim alone), and, for a divided sum of several dims (a flattened index), the place within
/// each dim where the divisor falls (c / coefficient, when it is a proper divisor of the dim's extent).
std::map<unsigned, std::set<int64_t>> divisors(ArrayRef<AffineExpr> exprs, ArrayRef<int64_t> extents) {
  std::map<unsigned, std::set<int64_t>> found;
  for (AffineExpr e : exprs)
    e.walk([&](AffineExpr x) {
      if (x.getKind() != AffineExprKind::FloorDiv && x.getKind() != AffineExprKind::Mod)
        return;
      auto bin = cast<AffineBinaryOpExpr>(x);
      auto c = dyn_cast<AffineConstantExpr>(bin.getRHS());
      if (!c || c.getValue() <= 1)
        return;
      std::set<unsigned> dims = dimsOf(bin.getLHS());
      if (dims.size() == 1) {
        found[*dims.begin()].insert(c.getValue());
        return;
      }
      auto lin = linearInDims(bin.getLHS());
      if (!lin)
        return;
      for (auto [d, k] : lin->first) {
        if (k <= 0 || c.getValue() % k)
          continue;
        int64_t place = c.getValue() / k;
        if (place > 1 && place < extents[d] && extents[d] % place == 0)
          found[d].insert(place);
      }
    });
  return found;
}

/// Splits dims into mixed-radix digits and linearizes expressions over them.
class Digits {
public:
  /// `chains[d]`: the divisors of dim d, increasing, each dividing the next and the extent.
  Digits(MLIRContext *ctx, ArrayRef<int64_t> extents, const std::map<unsigned, SmallVector<int64_t>> &chains)
      : ctx(ctx) {
    for (auto [d, extent] : llvm::enumerate(extents)) {
      AffineExpr dim = getAffineDimExpr(d, ctx);
      auto it = chains.find(d);
      SmallVector<int64_t> cs = it == chains.end() ? SmallVector<int64_t>{} : it->second;
      // Digits most significant first: d floordiv cK, ..., (d floordiv c1) mod (c2 / c1), d mod c1.
      SmallVector<Digit> ds;
      for (int k = cs.size(); k >= 0; --k) {
        int64_t place = k == 0 ? 1 : cs[k - 1];
        int64_t size = k == (int)cs.size() ? extent / place : cs[k] / place;
        AffineExpr e = k == 0 ? (cs.empty() ? dim : dim % cs[0])
                       : k == (int)cs.size() ? dim.floorDiv(cs[k - 1])
                                             : dim.floorDiv(cs[k - 1]) % (cs[k] / cs[k - 1]);
        ds.push_back({e, place, size});
        place_[e] = {unsigned(d), place};
        size_[e] = size;
      }
      table[d] = ds;
    }
  }

  struct Digit {
    AffineExpr expr;
    int64_t place, size;
  };
  std::map<unsigned, SmallVector<Digit>> table;

  /// `e` over digits; atoms that are not digits stay opaque.
  Form linear(AffineExpr e) {
    Form f;
    if (auto c = dyn_cast<AffineConstantExpr>(e)) {
      f.constant = c.getValue();
      return f;
    }
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      for (const Digit &g : table[d.getPosition()])
        f.terms[g.expr] += g.place;
      return f;
    }
    auto bin = cast<AffineBinaryOpExpr>(e);
    if (e.getKind() == AffineExprKind::Add) {
      f = linear(bin.getLHS());
      f.add(linear(bin.getRHS()), 1);
      return f;
    }
    if (e.getKind() == AffineExprKind::Mul) {
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
        f.add(linear(bin.getLHS()), c.getValue());
        return f;
      }
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
        f.add(linear(bin.getRHS()), c.getValue());
        return f;
      }
    }
    if (e.getKind() == AffineExprKind::FloorDiv || e.getKind() == AffineExprKind::Mod) {
      auto c = dyn_cast<AffineConstantExpr>(bin.getRHS());
      Form inner = linear(bin.getLHS());
      // A sum of digits (of any dims) that splits at c without a carry: every term at or above c is a multiple of
      // c, and the terms below c (at their largest) plus the constant stay below c.
      if (c && c.getValue() > 0 && inner.constant >= 0) {
        int64_t cv = c.getValue(), below = inner.constant;
        bool ok = true;
        for (auto [atom, coeff] : inner.terms) {
          if (!coeff)
            continue;
          auto it = size_.find(atom);
          if (it == size_.end() || coeff < 0) {
            ok = false;
            break;
          }
          if (coeff >= cv)
            ok &= coeff % cv == 0;
          else
            below += coeff * (it->second - 1);
        }
        if (ok && below < cv) {
          for (auto [atom, coeff] : inner.terms) {
            if (!coeff)
              continue;
            if (e.getKind() == AffineExprKind::FloorDiv && coeff >= cv)
              f.terms[atom] += coeff / cv;
            if (e.getKind() == AffineExprKind::Mod && coeff < cv)
              f.terms[atom] += coeff;
          }
          if (e.getKind() == AffineExprKind::Mod)
            f.constant += inner.constant;
          return f;
        }
      }
      // Digits of one dim at exactly their place values (plus a constant below the divisor for floordiv), and
      // a divisor of that dim's chain: the digits above (floordiv) or below (mod) the divisor survive.
      std::optional<unsigned> dim;
      bool ok = c && inner.constant >= 0;
      for (auto [atom, coeff] : inner.terms) {
        if (!coeff)
          continue;
        auto it = place_.find(atom);
        if (it == place_.end() || it->second.second != coeff || (dim && *dim != it->second.first)) {
          ok = false;
          break;
        }
        dim = it->second.first;
      }
      if (ok && dim) {
        int64_t cv = c.getValue();
        bool isDivisor = llvm::any_of(table[*dim], [&](const Digit &g) { return g.place == cv; });
        // The constant must not carry across the divisor: below every digit at or above it.
        if (isDivisor && inner.constant < cv) {
          bool carries = false;
          int64_t below = inner.constant;
          for (auto [atom, coeff] : inner.terms)
            if (coeff && place_[atom].second < cv)
              below += (llvm::find_if(table[*dim], [&](const Digit &g) { return g.expr == atom; })->size - 1) *
                       coeff;
          carries = below >= cv;
          if (!carries) {
            for (auto [atom, coeff] : inner.terms) {
              if (!coeff)
                continue;
              int64_t value = place_[atom].second;
              bool above = value >= cv;
              if (e.getKind() == AffineExprKind::FloorDiv && above)
                f.terms[atom] += value / cv;
              if (e.getKind() == AffineExprKind::Mod && !above)
                f.terms[atom] += value;
            }
            if (e.getKind() == AffineExprKind::Mod)
              f.constant += inner.constant;
            return f;
          }
        }
      }
    }
    f.terms[e] = 1;
    return f;
  }

  MLIRContext *ctx;
  llvm::DenseMap<AffineExpr, std::pair<unsigned, int64_t>> place_; // digit -> (dim, place value)
  llvm::DenseMap<AffineExpr, int64_t> size_;                      // digit -> extent
};

//===----------------------------------------------------------------------===//
// Folding
//===----------------------------------------------------------------------===//

/// A read before linearization: indices are AffineExprs over the global dims.
struct RawRead {
  unsigned arg;
  Type type;
  SmallVector<int64_t> shape;
  SmallVector<AffineExpr> index;
  std::optional<double> fill;
};

struct RawNode {
  Node::Kind kind;
  unsigned read = 0;
  double value = 0;
  std::string op;
  SmallVector<unsigned> args;
  SmallVector<unsigned> axes; // Reduce: global dims
};

class Folder {
public:
  Folder(func::FuncOp func, std::string &why) : func(func), why(why), ctx(func.getContext()) {}

  func::FuncOp func;
  std::string &why;
  MLIRContext *ctx;
  SmallVector<int64_t> extents;
  SmallVector<bool> isReduction;
  std::vector<RawRead> reads;
  std::vector<RawNode> nodes;
  std::map<std::string, unsigned> memo;
  /// Index values with a known expression over the dims (tile loops' induction variables).
  llvm::DenseMap<Value, AffineExpr> bound;
  /// The generic loop each reduction dim iterates (for the tiling pass).
  std::map<unsigned, std::pair<Operation *, unsigned>> reductionLoop;
  /// K loops (furiosa-tile) being folded: the loop of each iteration argument; the dims tiles split, as (tile
  /// dim, dim in the tile, tile size), merged once the Program is built.
  llvm::DenseMap<Value, scf::ForOp> kLoopOf;
  SmallVector<std::tuple<unsigned, unsigned, int64_t>> tiled;

  /// An index operand (a constant, a bound value, or an affine.apply of those) as an expression over the dims.
  FailureOr<AffineExpr> indexExpr(OpFoldResult ofr) {
    if (std::optional<int64_t> c = getConstantIntValue(ofr))
      return getAffineConstantExpr(*c, ctx);
    Value v = cast<Value>(ofr);
    if (auto it = bound.find(v); it != bound.end())
      return it->second;
    if (auto apply = v.getDefiningOp<affine::AffineApplyOp>()) {
      SmallVector<AffineExpr> operands;
      for (Value o : apply.getMapOperands()) {
        auto e = indexExpr(o);
        if (failed(e))
          return failure();
        operands.push_back(*e);
      }
      AffineMap m = apply.getAffineMap();
      return m.getResult(0).replaceDimsAndSymbols(ArrayRef(operands).take_front(m.getNumDims()),
                                                  ArrayRef(operands).drop_front(m.getNumDims()));
    }
    why = "an index that is neither constant nor a tile loop's";
    return failure();
  }

  unsigned dim(int64_t extent, bool reduction) {
    extents.push_back(extent);
    isReduction.push_back(reduction);
    return extents.size() - 1;
  }

  std::string key(const RawNode &n) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << int(n.kind) << "|" << n.op << "|" << llvm::bit_cast<uint64_t>(n.value) << "|";
    if (n.kind == Node::Leaf) {
      const RawRead &r = reads[n.read];
      os << r.arg << "|";
      for (AffineExpr e : r.index)
        os << e << ",";
      os << "|" << (r.fill ? std::to_string(llvm::bit_cast<uint64_t>(*r.fill)) : "none");
    }
    for (unsigned a : n.args)
      os << "%" << a;
    for (unsigned a : n.axes)
      os << "d" << a;
    return s;
  }

  unsigned add(RawNode n) {
    // Reductions are never shared: their axes are fresh per occurrence.
    if (n.kind != Node::Reduce) {
      std::string k = key(n);
      if (auto it = memo.find(k); it != memo.end())
        return it->second;
      nodes.push_back(n);
      return memo[k] = nodes.size() - 1;
    }
    nodes.push_back(n);
    return nodes.size() - 1;
  }

  unsigned constant(double v) {
    RawNode n{Node::Const};
    n.value = v;
    return add(n);
  }

  /// The node for `tensor` at `idx` (one AffineExpr per tensor dim). `fill`: the value of positions an
  /// enclosing tensor.pad added.
  FailureOr<unsigned> valueAt(Value tensor, SmallVector<AffineExpr> idx, std::optional<double> fill) {
    if (auto arg = dyn_cast<BlockArgument>(tensor); arg && arg.getOwner() == &func.getBody().front()) {
      auto type = cast<RankedTensorType>(tensor.getType());
      reads.push_back({arg.getArgNumber(), type.getElementType(), SmallVector<int64_t>(type.getShape()), idx, fill});
      RawNode n{Node::Leaf};
      n.read = reads.size() - 1;
      return add(n);
    }
    if (auto it = kLoopOf.find(tensor); it != kLoopOf.end())
      return valueAt(it->second.getInitArgs()[0], idx, fill);
    Operation *op = tensor.getDefiningOp();
    if (!op) {
      why = "a tensor that is neither an argument nor an op result";
      return failure();
    }
    if (isa<scf::ForOp>(op)) {
      // A K loop (furiosa-tile): one tile of a generic's reduction dim per iteration, accumulated in its
      // iteration argument. generic() gives that dim a tile dim alongside it.
      std::optional<TileLoop> k = kLoop(tensor);
      if (!k) {
        why = "an scf.for that is not a K tile loop";
        return failure();
      }
      kLoopOf[k->iterArg] = k->op;
      return valueAt(k->yield, idx, fill);
    }
    if (auto pad = dyn_cast<tensor::PadOp>(op)) {
      llvm::APFloat value(0.0f);
      Value pv = pad.getConstantPaddingValue();
      if (!pv || !matchPattern(pv, m_ConstantFloat(&value))) {
        why = "a tensor.pad with a non-constant padding value";
        return failure();
      }
      double v = value.convertToDouble();
      if (fill && *fill != v) {
        why = "nested pads with different values";
        return failure();
      }
      SmallVector<int64_t> low(pad.getStaticLow());
      if (llvm::is_contained(low, ShapedType::kDynamic)) {
        why = "a dynamic tensor.pad";
        return failure();
      }
      for (unsigned j = 0; j < idx.size(); ++j)
        idx[j] = idx[j] - low[j];
      return valueAt(pad.getSource(), idx, v);
    }
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op)) {
      SmallVector<int64_t> strides(slice.getStaticStrides());
      if (llvm::is_contained(strides, ShapedType::kDynamic) || llvm::is_contained(slice.getStaticSizes(), ShapedType::kDynamic)) {
        why = "a dynamic tensor.extract_slice";
        return failure();
      }
      auto srcType = slice.getSourceType();
      llvm::SmallBitVector dropped = slice.getDroppedDims();
      SmallVector<AffineExpr> src;
      unsigned k = 0;
      for (auto [j, offset] : llvm::enumerate(slice.getMixedOffsets())) {
        auto off = indexExpr(offset);
        if (failed(off))
          return failure();
        AffineExpr inner = dropped.test(j) ? getAffineConstantExpr(0, ctx) : idx[k++];
        src.push_back(inner * strides[j] + *off);
      }
      return valueAt(slice.getSource(), src, fill);
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      // Source dim = the expanded group's indices at their row-major strides.
      auto outShape = expand.getResultType().getShape();
      SmallVector<AffineExpr> src;
      for (ReassociationIndices group : expand.getReassociationIndices()) {
        AffineExpr e = getAffineConstantExpr(0, ctx);
        int64_t stride = 1;
        for (int i = group.size() - 1; i >= 0; --i) {
          e = e + idx[group[i]] * stride;
          stride *= outShape[group[i]];
        }
        src.push_back(e);
      }
      return valueAt(expand.getSrc(), src, fill);
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      // A collapsed buffer is read as the flat buffer it is: its index stays linear however it was formed.
      if (auto arg = dyn_cast<BlockArgument>(collapse.getSrc()); arg && arg.getOwner() == &func.getBody().front()) {
        auto type = collapse.getResultType();
        reads.push_back({arg.getArgNumber(), type.getElementType(), SmallVector<int64_t>(type.getShape()), idx, fill});
        RawNode n{Node::Leaf};
        n.read = reads.size() - 1;
        return add(n);
      }
      // Source dims = the digits of the collapsed index.
      auto srcShape = collapse.getSrcType().getShape();
      // Unit dims no group covers (collapsed into a rank-0 result) index 0.
      SmallVector<AffineExpr> src(srcShape.size(), getAffineConstantExpr(0, ctx));
      for (auto [g, group] : llvm::enumerate(collapse.getReassociationIndices())) {
        int64_t stride = 1;
        for (int i = group.size() - 1; i >= 0; --i) {
          AffineExpr e = idx[g];
          if (stride > 1)
            e = e.floorDiv(stride);
          if (i > 0)
            e = e % srcShape[group[i]];
          src[group[i]] = e;
          stride *= srcShape[group[i]];
        }
      }
      return valueAt(collapse.getSrc(), src, fill);
    }
    if (auto f = dyn_cast<linalg::FillOp>(op)) {
      llvm::APFloat value(0.0f);
      if (!matchPattern(f.getInputs()[0], m_ConstantFloat(&value))) {
        why = "a linalg.fill with a non-constant value";
        return failure();
      }
      return constant(value.convertToDouble());
    }
    if (auto g = dyn_cast<linalg::GenericOp>(op))
      return generic(g, cast<OpResult>(tensor).getResultNumber(), idx, fill);
    why = "no folding for '" + op->getName().getStringRef().str() + "'";
    return failure();
  }

  /// The generic's result `r` at `idx`: its parallel dims bind to the indices, its reduction dims are fresh.
  FailureOr<unsigned> generic(linalg::GenericOp g, unsigned r, ArrayRef<AffineExpr> idx, std::optional<double> fill) {
    if (fill) {
      why = "a padded generic result";
      return failure();
    }
    auto iters = g.getIteratorTypesArray();
    SmallVector<int64_t> ranges = g.getStaticLoopRanges();
    AffineMap outMap = g.getMatchingIndexingMap(g.getDpsInitOperand(r));
    SmallVector<AffineExpr> dims(iters.size());
    SmallVector<unsigned> fresh;
    for (auto [k, e] : llvm::enumerate(outMap.getResults())) {
      if (auto d = dyn_cast<AffineDimExpr>(e))
        dims[d.getPosition()] = idx[k];
      else if (!isa<AffineConstantExpr>(e)) {
        why = "a generic output map that is not a projection";
        return failure();
      }
    }
    for (unsigned d = 0; d < iters.size(); ++d) {
      if (dims[d])
        continue;
      if (iters[d] != utils::IteratorType::reduction) {
        if (ranges[d] != 1) {
          why = "a parallel dim the output does not index";
          return failure();
        }
        dims[d] = getAffineConstantExpr(0, ctx);
        continue;
      }
      // A dim a K loop around this generic tiles: the loop's tile dim (the iteration) comes first.
      Value init = g.getDpsInits()[r];
      for (auto it = kLoopOf.find(init); it != kLoopOf.end(); it = kLoopOf.find(it->second.getInitArgs()[0])) {
        scf::ForOp loop = it->second;
        if (loop->getAttrOfType<IntegerAttr>("furiosa.tile_dim").getInt() != int64_t(d))
          continue;
        int64_t step = *getConstantIntValue(loop.getStep()), ub = *getConstantIntValue(loop.getUpperBound());
        unsigned kt = dim(ub / step, true);
        fresh.push_back(kt);
        reductionLoop[kt] = {g.getOperation(), d};
        bound[loop.getInductionVar()] = getAffineDimExpr(kt, ctx) * step;
        tiled.push_back({kt, unsigned(extents.size()), step});
      }
      unsigned fd = dim(ranges[d], true);
      fresh.push_back(fd);
      reductionLoop[fd] = {g.getOperation(), d};
      dims[d] = getAffineDimExpr(fd, ctx);
    }
    Block &body = g.getRegion().front();
    llvm::DenseMap<Value, unsigned> values;
    for (OpOperand *in : g.getDpsInputOperands()) {
      BlockArgument a = body.getArgument(in->getOperandNumber());
      if (a.use_empty())
        continue;
      AffineMap m = g.getMatchingIndexingMap(in);
      SmallVector<AffineExpr> sub;
      for (AffineExpr e : m.getResults())
        sub.push_back(e.replaceDims(dims));
      auto n = valueAt(in->get(), sub, std::nullopt);
      if (failed(n))
        return failure();
      values[a] = *n;
    }
    Value acc = body.getArgument(g.getNumDpsInputs() + r);
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    Value yielded = yield.getOperand(r);
    if (!fresh.empty()) {
      Operation *comb = yielded.getDefiningOp();
      if (!comb || comb->getNumOperands() != 2 || (comb->getOperand(0) != acc && comb->getOperand(1) != acc)) {
        why = "a reduction body that does not combine the accumulator with one term";
        return failure();
      }
      std::string op = isa<arith::AddFOp>(comb)                            ? "add"
                       : isa<arith::MaximumFOp, arith::MaxNumFOp>(comb)    ? "max"
                       : isa<arith::MinimumFOp, arith::MinNumFOp>(comb)    ? "min"
                                                                           : "";
      if (op.empty()) {
        why = "a reduction combiner other than add/max/min";
        return failure();
      }
      Value term = comb->getOperand(0) == acc ? comb->getOperand(1) : comb->getOperand(0);
      auto t = scalar(term, body, values, acc);
      if (failed(t))
        return failure();
      RawNode red{Node::Reduce};
      red.op = op;
      red.args = {*t};
      red.axes = fresh;
      unsigned rn = add(red);
      // The initial value: the combiner's identity drops out.
      auto init = valueAt(g.getDpsInits()[r], SmallVector<AffineExpr>(idx), std::nullopt);
      if (failed(init))
        return failure();
      const RawNode &in = nodes[*init];
      bool identity = in.kind == Node::Const &&
                      ((op == "add" && in.value == 0) || (op == "max" && std::isinf(in.value) && in.value < 0) ||
                       (op == "min" && std::isinf(in.value) && in.value > 0));
      if (identity)
        return rn;
      RawNode c{Node::Op};
      c.op = op;
      c.args = {*init, rn};
      return add(c);
    }
    if (!acc.use_empty()) {
      auto init = valueAt(g.getDpsInits()[r], SmallVector<AffineExpr>(idx), std::nullopt);
      if (failed(init))
        return failure();
      values[acc] = *init;
    }
    return scalar(yielded, body, values, Value());
  }

  /// A body value as a node.
  FailureOr<unsigned> scalar(Value v, Block &body, llvm::DenseMap<Value, unsigned> &values, Value acc) {
    if (auto it = values.find(v); it != values.end())
      return it->second;
    if (v == acc && acc) {
      why = "the accumulator read outside the combiner";
      return failure();
    }
    llvm::APFloat f(0.0f);
    if (matchPattern(v, m_ConstantFloat(&f)))
      return values[v] = constant(f.convertToDouble());
    Operation *op = v.getDefiningOp();
    if (!op || op->getBlock() != &body) {
      // A value from outside the body (a function-level constant is matched above).
      why = "a body value defined outside the body";
      return failure();
    }
    if (isa<arith::ExtFOp, arith::TruncFOp>(op)) {
      auto in = scalar(op->getOperand(0), body, values, acc);
      if (failed(in))
        return failure();
      return values[v] = *in;
    }
    // 2^x = e^(x ln 2): the Vector Engine has e^x (within its ulp budget, not bit-exact to exp2).
    if (isa<math::Exp2Op>(op)) {
      auto in = scalar(op->getOperand(0), body, values, acc);
      if (failed(in))
        return failure();
      RawNode scaled{Node::Op}, e{Node::Op};
      scaled.op = "mul";
      scaled.args = {*in, constant(0.6931471805599453)};
      e.op = "exp";
      e.args = {add(scaled)};
      return values[v] = add(e);
    }
    std::string name = llvm::TypeSwitch<Operation *, std::string>(op)
                           .Case<arith::AddFOp>([](auto) { return "add"; })
                           .Case<arith::SubFOp>([](auto) { return "sub"; })
                           .Case<arith::MulFOp>([](auto) { return "mul"; })
                           .Case<arith::DivFOp>([](auto) { return "div"; })
                           .Case<arith::MaximumFOp, arith::MaxNumFOp>([](auto) { return "max"; })
                           .Case<arith::MinimumFOp, arith::MinNumFOp>([](auto) { return "min"; })
                           .Case<arith::NegFOp>([](auto) { return "neg"; })
                           .Case<math::AbsFOp>([](auto) { return "abs"; })
                           .Case<math::ExpOp>([](auto) { return "exp"; })
                           .Case<math::SqrtOp>([](auto) { return "sqrt"; })
                           .Case<math::RsqrtOp>([](auto) { return "rsqrt"; })
                           .Case<math::TanhOp>([](auto) { return "tanh"; })
                           .Case<math::LogOp>([](auto) { return "log"; })
                           .Case<math::ErfOp>([](auto) { return "erf"; })
                           .Case<math::SinOp>([](auto) { return "sin"; })
                           .Case<math::CosOp>([](auto) { return "cos"; })
                           .Case<arith::SelectOp>([](auto) { return "select"; })
                           .Case<arith::CmpFOp>([](arith::CmpFOp c) {
                             return "cmp." + arith::stringifyCmpFPredicate(c.getPredicate()).str();
                           })
                           .Default([](auto) { return ""; });
    if (name.empty()) {
      why = "no folding for body op '" + op->getName().getStringRef().str() + "'";
      return failure();
    }
    RawNode n{Node::Op};
    n.op = name;
    for (Value o : op->getOperands()) {
      auto a = scalar(o, body, values, acc);
      if (failed(a))
        return failure();
      n.args.push_back(*a);
    }
    return values[v] = add(n);
  }
};

/// Ranges of a linear index over the axes.
std::pair<int64_t, int64_t> range(const Lin &l, ArrayRef<Axis> axes) {
  int64_t lo = l.offset, hi = l.offset;
  for (auto [a, c] : l.terms) {
    int64_t span = c * (axes[a].size - 1);
    (span < 0 ? lo : hi) += span;
  }
  return {lo, hi};
}

} // namespace

FailureOr<Program> mlir::furiosa::tile::analyze(func::FuncOp func, ArrayRef<int64_t> written, std::string &why) {
  MLIRContext *ctx = func.getContext();
  auto ret = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  if (ret.getNumOperands() != 1 || written.size() != 1) {
    why = "expected one written buffer";
    return failure();
  }
  Folder folder(func, why);

  // The stored value: reshapes down to the generic that computes it.
  Value v = ret.getOperand(0);
  SmallVector<Operation *> reshapes;
  while (isa_and_nonnull<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(v.getDefiningOp())) {
    reshapes.push_back(v.getDefiningOp());
    v = v.getDefiningOp()->getOperand(0);
  }
  // Tile loops (furiosa-tile, TileLoops.h): a nest of row tile loops inserting one tile of the stored value. Each
  // tiled dim gets two dims, the tile index (with the output dims) and the index in the tile (after them); the
  // axes merge back into one once the Program is built.
  SmallVector<AffineExpr> rootIdx, outIdx;
  FailureOr<std::optional<TileNest>> nest = tileNest(v, why);
  if (failed(nest))
    return failure();
  if (*nest) {
    TileNest n = **nest;
    auto shape = n.insert.getDestType().getShape();
    SmallVector<unsigned> outer(shape.size());
    for (auto [j, extent] : llvm::enumerate(shape))
      outer[j] = folder.dim(n.loopOf[j] ? n.loops[*n.loopOf[j]].trips : extent, false);
    for (auto [j, extent] : llvm::enumerate(shape)) {
      AffineExpr d = getAffineDimExpr(outer[j], ctx);
      if (!n.loopOf[j]) {
        rootIdx.push_back(d);
        outIdx.push_back(d);
        continue;
      }
      const TileLoop &l = n.loops[*n.loopOf[j]];
      unsigned in = folder.dim(l.step, false);
      folder.bound[l.iv] = d * l.step;
      rootIdx.push_back(getAffineDimExpr(in, ctx));
      outIdx.push_back(d * l.step + getAffineDimExpr(in, ctx));
      folder.tiled.push_back({outer[j], in, l.step});
    }
    v = n.tile;
  }
  // K loops (furiosa-tile) around the stored generic: it reduces across their iterations (generic() below).
  while (std::optional<TileLoop> k = kLoop(v)) {
    folder.kLoopOf[k->iterArg] = k->op;
    v = k->yield;
  }
  auto last = v.getDefiningOp<linalg::GenericOp>();
  if (!last) {
    why = "the stored value is not a generic's result";
    return failure();
  }
  // The output dims: the last generic's dims, or, when it reduces (a contraction stored as it is), the dims its
  // output map projects to (generic() makes the reduction dims fresh reduction axes).
  AffineMap lastOut = last.getMatchingIndexingMap(last.getDpsInitOperand(cast<OpResult>(v).getResultNumber()));
  SmallVector<int64_t> ranges = last.getStaticLoopRanges();
  SmallVector<unsigned> outLoop;
  if (!rootIdx.empty()) {
    // The tile's own dims are bound through its output map (generic() below).
  } else if (llvm::all_of(last.getIteratorTypesArray(), [](auto t) { return t == utils::IteratorType::parallel; })) {
    for (auto [i, e] : llvm::enumerate(ranges)) {
      folder.dim(e, false);
      outLoop.push_back(i);
    }
    rootIdx = SmallVector<AffineExpr>(lastOut.getResults());
  } else {
    for (AffineExpr e : lastOut.getResults()) {
      auto d = dyn_cast<AffineDimExpr>(e);
      if (!d) {
        why = "a reducing last generic whose output map is not a projection";
        return failure();
      }
      rootIdx.push_back(getAffineDimExpr(folder.dim(ranges[d.getPosition()], false), ctx));
      outLoop.push_back(d.getPosition());
    }
  }
  unsigned numOut = folder.extents.size();
  if (outIdx.empty())
    outIdx = rootIdx;
  // Through the reshapes, outermost last: the written buffer's indices over the output dims.
  for (Operation *op : llvm::reverse(reshapes)) {
    SmallVector<AffineExpr> next;
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      auto shape = collapse.getSrcType().getShape();
      for (ReassociationIndices group : collapse.getReassociationIndices()) {
        AffineExpr e = getAffineConstantExpr(0, ctx);
        int64_t stride = 1;
        for (int i = group.size() - 1; i >= 0; --i) {
          e = e + outIdx[group[i]] * stride;
          stride *= shape[group[i]];
        }
        next.push_back(e);
      }
    } else {
      auto expand = cast<tensor::ExpandShapeOp>(op);
      auto shape = expand.getResultType().getShape();
      // Unit dims no group covers (a rank-0 source expanded) index 0.
      next.assign(shape.size(), getAffineConstantExpr(0, ctx));
      for (auto [g, group] : llvm::enumerate(expand.getReassociationIndices())) {
        int64_t stride = 1;
        for (int i = group.size() - 1; i >= 0; --i) {
          AffineExpr e = outIdx[g];
          if (stride > 1)
            e = e.floorDiv(stride);
          if (i > 0)
            e = e % shape[group[i]];
          next[group[i]] = e;
          stride *= shape[group[i]];
        }
      }
    }
    outIdx = next;
  }
  SmallVector<AffineExpr> dims;
  for (unsigned d = 0; d < numOut; ++d)
    dims.push_back(getAffineDimExpr(d, ctx));
  auto root = folder.generic(last, cast<OpResult>(v).getResultNumber(), rootIdx, std::nullopt);
  if (failed(root))
    return failure();
  // generic() bound the last generic's dims to its own output map's results; re-express the root at the dims.
  // (The map is a projection, so the root is already over dims 0..numOut-1.)

  // Digits: every divisor a map applies to a dim, if the chain nests and divides the extent.
  SmallVector<AffineExpr> all(outIdx);
  for (const RawRead &r : folder.reads)
    llvm::append_range(all, r.index);
  std::map<unsigned, SmallVector<int64_t>> chains;
  for (auto &[d, cs] : divisors(all, folder.extents)) {
    SmallVector<int64_t> chain(cs.begin(), cs.end());
    bool ok = true;
    for (unsigned i = 0; i < chain.size(); ++i)
      ok &= (i == 0 || chain[i] % chain[i - 1] == 0) && folder.extents[d] % chain[i] == 0;
    if (!ok) {
      why = "dim " + std::to_string(d) + " is divided by constants that do not nest in its extent";
      return failure();
    }
    chains[d] = chain;
  }
  Digits digits(ctx, folder.extents, chains);

  Program p;
  llvm::DenseMap<AffineExpr, unsigned> axisOf; // digit -> axis
  for (unsigned d = 0; d < folder.extents.size(); ++d)
    for (const Digits::Digit &g : digits.table[d]) {
      if (g.size == 1)
        continue;
      axisOf[g.expr] = p.axes.size();
      p.axes.push_back({g.size, (bool)folder.isReduction[d]});
      p.axisDim.push_back({d, g.place});
      if (d < numOut)
        p.out.push_back(p.axes.size() - 1);
    }
  p.dimExtent = folder.extents;
  p.outLoop = outLoop;
  p.rootOp = last;
  auto toLin = [&](AffineExpr e) -> FailureOr<Lin> {
    Form f = digits.linear(e);
    Lin l;
    l.offset = f.constant;
    for (auto [atom, c] : f.terms) {
      if (!c)
        continue;
      auto it = axisOf.find(atom);
      if (it == axisOf.end()) {
        // A unit digit is zero; anything else is not linear in the digits.
        bool unit = false;
        for (auto &[d, ds] : digits.table)
          for (const Digits::Digit &g : ds)
            unit |= g.expr == atom && g.size == 1;
        if (unit)
          continue;
        std::string s;
        llvm::raw_string_ostream os(s);
        os << "an index that is not linear in the loop digits: " << e;
        why = s;
        return failure();
      }
      l.add(it->second, c);
    }
    return l;
  };
  for (const RawRead &r : folder.reads) {
    Read read{r.arg, r.type, r.shape, {}, r.fill};
    for (AffineExpr e : r.index) {
      auto l = toLin(e);
      if (failed(l))
        return failure();
      read.index.push_back(*l);
    }
    // Drop the fill when no index can leave the buffer.
    bool inside = true;
    for (auto [j, l] : llvm::enumerate(read.index)) {
      auto [lo, hi] = range(l, p.axes);
      inside &= lo >= 0 && hi < read.shape[j];
    }
    if (inside)
      read.fill.reset();
    else if (!read.fill) {
      why = "a read that can leave its buffer without padding";
      return failure();
    }
    p.reads.push_back(read);
  }
  // Global dims -> axes for reductions.
  for (const RawNode &n : folder.nodes) {
    Node m{n.kind};
    m.read = n.read;
    m.value = n.value;
    m.op = n.op;
    m.args = n.args;
    for (unsigned d : n.axes)
      for (const Digits::Digit &g : digits.table[d])
        if (auto it = axisOf.find(g.expr); it != axisOf.end())
          m.axes.push_back(it->second);
    p.nodes.push_back(m);
  }
  p.root = *root;
  p.outArg = written[0];
  auto outType = cast<RankedTensorType>(func.getArgumentTypes()[p.outArg]);
  p.outType = outType.getElementType();
  p.outShape = SmallVector<int64_t>(outType.getShape());
  if (outIdx.size() != p.outShape.size()) {
    why = "the stored value's rank differs from the written buffer's";
    return failure();
  }
  for (AffineExpr e : outIdx) {
    auto l = toLin(e);
    if (failed(l))
      return failure();
    p.outIndex.push_back(*l);
  }
  // Tile loops: the tile index and the index in the tile merge into the one axis they split (every index uses
  // them as t * size + i), which keeps the loop's tiling as a tile digit of that axis.
  auto axisOfDim = [&](unsigned d) -> std::optional<unsigned> {
    std::optional<unsigned> a;
    for (const Digits::Digit &g : digits.table[d])
      if (auto it = axisOf.find(g.expr); it != axisOf.end()) {
        if (a)
          return std::nullopt; // a tiled dim split into digits: not merged
        a = it->second;
      }
    return a;
  };
  p.reductionLoop = folder.reductionLoop;
  auto allLins = [&] {
    SmallVector<Lin *> lins;
    for (Read &r : p.reads)
      for (Lin &l : r.index)
        lins.push_back(&l);
    for (Lin &l : p.outIndex)
      lins.push_back(&l);
    return lins;
  };
  // Drops axis `ai`, which no index uses any more: every later axis moves down by one.
  auto dropAxis = [&](unsigned ai) {
    auto shift = [&](unsigned a) { return a > ai ? a - 1 : a; };
    for (Lin *l : allLins())
      for (auto &[a, k] : l->terms)
        a = shift(a);
    llvm::erase(p.out, ai);
    for (unsigned &a : p.out)
      a = shift(a);
    for (Node &n : p.nodes) {
      llvm::erase(n.axes, ai);
      for (unsigned &a : n.axes)
        a = shift(a);
    }
    for (auto &[a, k] : p.loopTiles)
      a = shift(a);
    p.axes.erase(p.axes.begin() + ai);
    p.axisDim.erase(p.axisDim.begin() + ai);
    for (auto &[da, ta] : axisOf)
      ta = shift(ta);
  };
  for (auto [dt, di, t] : folder.tiled) {
    std::optional<unsigned> at = axisOfDim(dt), ai = axisOfDim(di);
    if (!at)
      continue; // one tile
    if (!ai) {
      p.loopTiles.push_back({*at, 1});
      continue;
    }
    SmallVector<Lin *> lins = allLins();
    for (Lin *l : lins)
      if (l->coeff(*at) != t * l->coeff(*ai)) {
        why = "a tile loop's index used apart from its tile";
        return failure();
      }
    for (Lin *l : lins) {
      int64_t c = l->coeff(*ai);
      SmallVector<std::pair<unsigned, int64_t>> terms;
      for (auto [a, k] : l->terms)
        if (a != *ai)
          terms.push_back({a, a == *at ? c : k});
      l->terms = terms;
    }
    p.axes[*at].size *= t;
    p.axisDim[*at].second = 1;
    p.dimExtent[dt] *= t;
    p.loopTiles.push_back({*at, t});
    dropAxis(*ai);
  }
  // Adjacent digits of one dim that every index reads together merge back into one axis: in one buffer dim at
  // coefficients size(lo) * c and c, or in adjacent buffer dims, the lower one whole (those dims then flatten into
  // one). A loop the maps reshape apart only at the buffers' dims (a contraction's 256 rows read from a 2 x 128
  // buffer) is one axis again.
  for (bool merged = true; merged;) {
    merged = false;
    for (unsigned hi = 0; hi < p.axes.size() && !merged; ++hi)
      for (unsigned lo = 0; lo < p.axes.size() && !merged; ++lo) {
        if (hi == lo || p.axisDim[hi].first != p.axisDim[lo].first ||
            p.axisDim[hi].second != p.axisDim[lo].second * p.axes[lo].size ||
            llvm::any_of(p.loopTiles, [&](auto &t) { return t.first == hi || t.first == lo; }))
          continue;
        // Whether the index list reads hi and lo together; `apply` flattens their buffer dims.
        auto together = [&](SmallVector<Lin> &index, SmallVector<int64_t> &shape, bool fill, bool apply) {
          std::optional<unsigned> jh, jl;
          for (auto [j, l] : llvm::enumerate(index)) {
            if (l.uses(hi) && std::exchange(jh, j))
              return false;
            if (l.uses(lo) && std::exchange(jl, j))
              return false;
          }
          if (!jh && !jl)
            return true;
          if (!jh || !jl)
            return false;
          if (*jh == *jl)
            return index[*jh].coeff(hi) == index[*jh].coeff(lo) * p.axes[lo].size;
          Lin &h = index[*jh], &l = index[*jl];
          if (*jl != *jh + 1 || fill || h.coeff(hi) != 1 || l.terms.size() != 1 || l.offset || l.coeff(lo) != 1 ||
              shape[*jl] != p.axes[lo].size)
            return false;
          if (apply) {
            for (auto &[a, c] : h.terms)
              c *= shape[*jl];
            h.offset *= shape[*jl];
            h.add(lo, 1);
            shape[*jh] *= shape[*jl];
            index.erase(index.begin() + *jl);
            shape.erase(shape.begin() + *jl);
          }
          return true;
        };
        bool ok = together(p.outIndex, p.outShape, false, false);
        for (Read &r : p.reads)
          ok &= together(r.index, r.shape, r.fill.has_value(), false);
        if (!ok)
          continue;
        together(p.outIndex, p.outShape, false, true);
        for (Read &r : p.reads)
          together(r.index, r.shape, r.fill.has_value(), true);
        for (Lin *l : allLins())
          llvm::erase_if(l->terms, [&](auto &t) { return t.first == hi; });
        p.axes[lo].size *= p.axes[hi].size;
        dropAxis(hi);
        merged = true;
      }
  }
  return p;
}

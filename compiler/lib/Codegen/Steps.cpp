// Steps: a structured kernel that no op class maps whole runs as a sequence of sub-kernels (steps), each a
// function in the raised-kernel contract that one op class maps, with HBM temporaries between them.
//
// The split is greedy: every unit op (a linalg op other than a fill, or an insert_slice) starts as its own
// step, and a producer step merges into its consumer while the producer feeds nothing else and some op
// class maps the merged step. Cheap ops (tensor.empty, fills, constants, reshapes) are cloned into every
// step that reads them. The kernel function becomes the plan: a tensor.empty per temporary, one func.call
// per step on the buffers and temporaries it reads and writes, and reshapes, which move no data. So the
// split kernel still runs on the CPU (L1 judges it), and the crate launches the steps in call order.

#include "Mapping/Mapping.h"
#include "furiosa/Codegen/Codegen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"

#include <array>

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::mapping;

namespace {

constexpr llvm::StringLiteral kStepsAttr = "furiosa.steps";

bool isReshape(Operation *op) { return isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op); }

/// Ops cloned into every step that reads them: they move no data, or describe a read (a padded view) an op
/// class may absorb.
bool isCheap(Operation *op) {
  return isReshape(op) || isa<tensor::EmptyOp, linalg::FillOp, arith::ConstantOp, tensor::PadOp>(op);
}

/// A box of a buffer at a kept loop's iteration (an offset the body computes from the induction variable): the
/// view the plan takes of that buffer each iteration, not a step.
bool isIterationSlice(Operation *op) {
  auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
  return slice && !slice.getOffsets().empty() && isa<scf::ForOp>(op->getParentOp());
}

/// The bytes an iteration slice views of its source, {offset, per-iteration stride, size}, when they are one
/// contiguous range at the iteration index (the loop counting 0, 1, ...); the plan aliases nothing else.
std::optional<std::array<int64_t, 3>> iterationView(tensor::ExtractSliceOp slice) {
  auto loop = cast<scf::ForOp>(slice->getParentOp());
  if (!matchPattern(loop.getLowerBound(), m_Zero()) || !matchPattern(loop.getStep(), m_One()))
    return std::nullopt;
  RankedTensorType src = slice.getSourceType();
  int64_t width = src.getElementType().getIntOrFloatBitWidth() / 8, stride = width, offset = 0, perIter = -1;
  bool inner = true; // the dims inside the box's first partial dim must be whole
  for (int64_t d = src.getRank() - 1; d >= 0; --d) {
    int64_t size = slice.getStaticSizes()[d], at = slice.getStaticOffsets()[d];
    if (ShapedType::isDynamic(at)) {
      if (slice.getDynamicOffset(d) != loop.getInductionVar() || perIter >= 0)
        return std::nullopt;
      perIter = stride;
    } else {
      offset += at * stride;
    }
    if (!inner && size != 1)
      return std::nullopt;
    inner &= size == src.getDimSize(d);
    stride *= src.getDimSize(d);
  }
  return std::array<int64_t, 3>{offset, perIter, slice.getType().getNumElements() * width};
}

/// A kept loop whose body is split into steps: the plan runs it.
bool isSplitLoop(Operation *op) { return isa<scf::ForOp>(op) && op->hasAttr(kStepsAttr); }

/// The values `op` reads from its own block: its operands, and for a split loop what its body reads.
SmallVector<Value> blockUses(Operation *op) {
  SmallVector<Value> vs(op->getOperands());
  if (isSplitLoop(op))
    op->walk([&](Operation *nested) {
      for (Value v : nested->getOperands())
        if (Operation *def = v.getDefiningOp(); def && def->getBlock() == op->getBlock())
          vs.push_back(v);
    });
  return vs;
}

/// Where values leave a block's steps: its terminator, or a split loop that reads them.
bool isAnchor(Operation *op) { return isa<func::ReturnOp, scf::YieldOp>(op) || isSplitLoop(op); }

/// Everything else in a block is part of some step: a step an op class may not map (a kept loop, say)
/// still gives a valid plan, and L2 names it.
bool isUnit(Operation *op) {
  return !isCheap(op) && !isAnchor(op) && !isa<func::CallOp>(op) && !isIterationSlice(op) && op->getNumResults() > 0;
}

/// The value a reshape chain starts from.
Value throughReshapes(Value v) {
  while (v.getDefiningOp() && isReshape(v.getDefiningOp()))
    v = v.getDefiningOp()->getOperand(0);
  return v;
}

/// An identity generic copying `v` into a fresh tensor: a step of its own, the copy class's relayout.
Value copyOf(OpBuilder &b, Location loc, Value v) {
  auto t = cast<RankedTensorType>(v.getType());
  Value empty = tensor::EmptyOp::create(b, loc, t.getShape(), t.getElementType());
  AffineMap id = b.getMultiDimIdentityMap(t.getRank());
  return linalg::GenericOp::create(
             b, loc, TypeRange{t}, ValueRange{v}, ValueRange{empty}, ArrayRef<AffineMap>{id, id},
             SmallVector<utils::IteratorType>(t.getRank(), utils::IteratorType::parallel),
             [](OpBuilder &nb, Location l, ValueRange args) { linalg::YieldOp::create(nb, l, args[0]); })
      .getResult(0);
}

/// Whether some op class maps `func` (mapped on a copy, its diagnostics dropped): an op class's match is a
/// quick filter, its map the authority.
/// The tensor constants of a function's body (tables hoistIndexOnly made): data the host uploads.
SmallVector<arith::ConstantOp> tensorConstants(func::FuncOp func) {
  SmallVector<arith::ConstantOp> constants;
  for (auto c : func.getBody().front().getOps<arith::ConstantOp>())
    if (isa<RankedTensorType>(c.getType()))
      constants.push_back(c);
  return constants;
}

/// Turns each tensor constant into a trailing argument (so op classes see a buffer) and returns their data.
SmallVector<DenseElementsAttr> externalize(func::FuncOp func) {
  SmallVector<DenseElementsAttr> data;
  for (arith::ConstantOp c : tensorConstants(func)) {
    auto attr = cast<DenseElementsAttr>(c.getValue());
    std::string name = "c" + std::to_string(data.size());
    // Marked as a table: op classes may tell index-only data (a mask) from the kernel's.
    auto table = DictionaryAttr::get(func.getContext(),
                                     NamedAttribute(StringAttr::get(func.getContext(), "furiosa.table"),
                                                    UnitAttr::get(func.getContext())));
    if (failed(func.insertArgument(func.getNumArguments(), c.getType(), table,
                                   NameLoc::get(StringAttr::get(func.getContext(), name)))))
      break;
    c.replaceAllUsesWith(func.getArgument(func.getNumArguments() - 1));
    c.erase();
    data.push_back(attr);
  }
  return data;
}

/// furiosa-opt compiles a kernel in time superlinear in its unrolled DMA statements (a 733-DMA conv took hours;
/// the corpus's largest kernels have about 110): a step candidate beyond this many is split instead.
constexpr unsigned kStepDmas = 256;

bool mapsWhole(func::FuncOp func) {
  OpBuilder b(func.getContext());
  OwningOpRef<ModuleOp> scratch = ModuleOp::create(b, func.getLoc());
  if (ModuleOp parent = func->getParentOfType<ModuleOp>())
    for (NamedAttribute attr : parent->getAttrs())
      (*scratch)->setAttr(attr.getName(), attr.getValue());
  auto copy = cast<func::FuncOp>(func->clone());
  scratch->push_back(copy);
  externalize(copy);
  for (const OpClass &c : opClasses())
    if (mapsOnCopy(copy, c, kStepDmas))
      return true;
  // FURIOSA_DEBUG=1 prints why a candidate step does not map.
  if (getenv("FURIOSA_DEBUG")) {
    ScopedDiagnosticHandler print(func.getContext(), [&](Diagnostic &d) {
      llvm::errs() << "candidate " << func.getName() << ": " << d << "\n";
      return success();
    });
    (void)mapKernel(copy);
  }
  return false;
}

/// Whether some op class maps `g` as a step of its own: its inputs as buffers, a fill or empty init kept (as the steps
/// clone them), index-only values as tables (as hoistIndexOnly leaves the steps), and the tensors its body reads
/// from outside (a gather's table) as buffers too.
bool mapsAlone(linalg::GenericOp g) {
  OpBuilder b(g.getContext());
  Location loc = g.getLoc();
  SmallVector<Value> ins(g.getDpsInputs());
  SetVector<Value> above;
  getUsedValuesDefinedAbove(g.getRegion(), above);
  SmallVector<Value> read;
  for (Value v : above)
    if (isa<RankedTensorType>(v.getType()) && !v.getDefiningOp<arith::ConstantOp>())
      read.push_back(v);
  SmallVector<Type> types;
  for (Value v : ins)
    types.push_back(v.getType());
  types.push_back(g.getResultTypes()[0]);
  for (Value v : read)
    types.push_back(v.getType());
  auto func = func::FuncOp::create(b, loc, "alone", b.getFunctionType(types, g.getResultTypes()));
  func->setAttr("furiosa.written", b.getI64ArrayAttr({int64_t(ins.size())}));
  Block *body = func.addEntryBlock();
  b.setInsertionPointToStart(body);
  IRMapping map;
  for (auto [v, arg] : llvm::zip(ins, body->getArguments()))
    map.map(v, arg);
  for (auto [v, arg] : llvm::zip(read, body->getArguments().take_back(read.size())))
    map.map(v, arg);
  Value init = g.getDpsInits()[0];
  if (auto fill = init.getDefiningOp<linalg::FillOp>(); fill && fill.getInputs()[0].getDefiningOp<arith::ConstantOp>()) {
    b.clone(*fill.getInputs()[0].getDefiningOp(), map);
    b.clone(*fill.getOutputs()[0].getDefiningOp(), map);
    b.clone(*fill, map);
  } else if (init.getDefiningOp<tensor::EmptyOp>()) {
    b.clone(*init.getDefiningOp(), map);
  } else {
    map.map(init, body->getArgument(ins.size()));
  }
  Operation *copy = b.clone(*g, map);
  func::ReturnOp::create(b, loc, copy->getResults());
  OwningOpRef<ModuleOp> holder = ModuleOp::create(loc);
  holder->push_back(func);
  hoistIndexOnly(*holder);
  return mapsWhole(func);
}

/// A step: the unit ops it owns (it produces their values for the other steps), and those of other steps it
/// recomputes rather than reading their results.
struct Group {
  SmallVector<Operation *> ops, dups;
  bool contains(Operation *op) const { return llvm::is_contained(ops, op) || llvm::is_contained(dups, op); }
};

class Splitter {
public:
  /// Splits `block`: the kernel's body, or the body of one of its kept loops.
  Splitter(func::FuncOp kernel, Block &block) : kernel(kernel), block(block) {
    for (auto [i, op] : llvm::enumerate(block.getOperations()))
      position[&op] = i;
  }
  Splitter(func::FuncOp kernel) : Splitter(kernel, kernel.getBody().front()) {}

  /// `single`: one step of the whole kernel (it maps whole, but its tables must become buffers).
  void run(bool single = false);

private:
  func::FuncOp kernel;
  Block &block;
  llvm::DenseMap<Operation *, size_t> position;
  llvm::DenseMap<Operation *, int> groupOf; // unit op -> the group that owns it
  std::vector<Group> groups;
  std::vector<llvm::SmallSetVector<unsigned, 2>> reads; // per group: the kernel buffers it reads
  llvm::DenseMap<Value, Value> resultOf;               // a step output -> its value after the call
  llvm::StringMap<bool> mapped; // candidate text -> whether it maps: the scan retries candidates after each merge

  /// A group's ops and dups, in program order.
  SmallVector<Operation *> members(const Group &g) {
    SmallVector<Operation *> all(g.ops);
    llvm::append_range(all, g.dups);
    llvm::sort(all, [&](Operation *x, Operation *y) { return position.lookup(x) < position.lookup(y); });
    return all;
  }
  /// The unit ops whose values `g`'s members read (through cheap ops), outside `g`.
  llvm::SetVector<Operation *> producedFor(const Group &g, bool throughLoops = false);
  /// The groups (or -1 for the kernel's return) whose owned ops read `v`, looking through cheap ops.
  void consumers(Value v, llvm::SmallSetVector<int, 4> &out);
  /// The values of `g`'s own ops that something outside it reads: the return, or a group that reads it
  /// without recomputing it (`all`: any op outside `g`, for a candidate).
  SmallVector<Value> outputsOf(int g, bool all);
  /// A step function for `g` that writes `outputs`, not inserted anywhere; fills `inputs` with the values
  /// it reads (kernel buffers, or outputs of earlier steps, by their original value).
  func::FuncOp outline(const Group &g, StringRef name, ArrayRef<Value> outputs, SmallVectorImpl<Value> &inputs);
  /// Whether an op class maps `g` in place of group `at`.
  bool mappable(int at, const Group &g);
  /// Drops groups whose values no live group needs: every reader recomputes them.
  void prune();
  /// Orders the steps: after the steps whose results they read, and a step that leaves its result in a
  /// buffer after every step that reads the buffer's original value, so the buffer can take it in place.
  void order();
};

llvm::SetVector<Operation *> Splitter::producedFor(const Group &g, bool throughLoops) {
  llvm::SetVector<Operation *> producers;
  for (Operation *op : members(g))
    op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands()) {
        std::function<void(Value)> trace = [&](Value v) {
          Operation *def = v.getDefiningOp();
          if (!def || def->getBlock() != &block || g.contains(def))
            return;
          if (groupOf.count(def))
            producers.insert(def);
          else if (isCheap(def) || (throughLoops && isSplitLoop(def))) // for order: after the loop's inputs too
            for (Value o : blockUses(def))
              trace(o);
        };
        trace(operand);
      }
    });
  return producers;
}

void Splitter::consumers(Value v, llvm::SmallSetVector<int, 4> &out) {
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    while (user->getBlock() != &block)
      user = user->getParentOp();
    if (isAnchor(user))
      out.insert(-1);
    else if (auto it = groupOf.find(user); it != groupOf.end())
      out.insert(it->second);
    else if (isCheap(user))
      for (Value r : user->getResults())
        consumers(r, out);
  }
}

SmallVector<Value> Splitter::outputsOf(int g, bool all) {
  SmallVector<Value> outputs;
  const Group &group = groups[g];
  for (Operation *op : members(group)) {
    if (!llvm::is_contained(group.ops, op))
      continue;
    for (Value r : op->getResults()) {
      bool read = false;
      std::function<void(Value)> visit = [&](Value v) {
        for (OpOperand &use : v.getUses()) {
          Operation *user = use.getOwner();
          while (user->getBlock() != &block)
            user = user->getParentOp();
          // A user this group recomputes may be another group's own too: that group still reads the value.
          bool mine = group.contains(user);
          if (!mine && (isAnchor(user) || all)) {
            read = true;
          } else if (!mine && isCheap(user)) {
            for (Value x : user->getResults())
              visit(x);
          } else {
            // A live group that reads the value without recomputing its producer.
            for (const Group &other : groups)
              if (&other != &group && other.contains(user) && !other.contains(op))
                read = true;
          }
        }
      };
      visit(r);
      if (read)
        outputs.push_back(r);
    }
  }
  return outputs;
}

void Splitter::prune() {
  int n = groups.size();
  std::vector<bool> live(n);
  SmallVector<int> work;
  for (int g = 0; g < n; ++g) {
    bool needed = llvm::any_of(groups[g].ops, [](Operation *op) { return isa<tensor::InsertSliceOp>(op); });
    for (Operation *op : groups[g].ops)
      for (Value r : op->getResults()) {
        llvm::SmallSetVector<int, 4> readers;
        consumers(r, readers);
        needed |= readers.contains(-1);
      }
    if (needed && !live[g]) {
      live[g] = true;
      work.push_back(g);
    }
  }
  while (!work.empty()) {
    int g = work.pop_back_val();
    for (Operation *def : producedFor(groups[g])) {
      int p = groupOf.lookup(def);
      if (!live[p]) {
        live[p] = true;
        work.push_back(p);
      }
    }
  }
  for (int g = 0; g < n; ++g)
    if (!live[g])
      groups[g].ops.clear(), groups[g].dups.clear();
}

void Splitter::order() {
  llvm::erase_if(groups, [](const Group &g) { return g.ops.empty(); });
  auto last = [&](const Group &g) {
    size_t at = 0;
    for (Operation *op : g.ops)
      at = std::max(at, position.lookup(op));
    return at;
  };
  llvm::sort(groups, [&](const Group &x, const Group &y) { return last(x) < last(y); });
  int n = groups.size();
  groupOf.clear();
  for (int g = 0; g < n; ++g)
    for (Operation *op : groups[g].ops)
      groupOf[op] = g;
  std::vector<llvm::SmallSetVector<int, 4>> after(n); // after[g]: steps that must run before g
  std::vector<llvm::SmallSetVector<unsigned, 2>> writes(n);
  reads.assign(n, {});
  for (int g = 0; g < n; ++g) {
    for (Operation *def : producedFor(groups[g], /*throughLoops=*/true))
      after[g].insert(groupOf.lookup(def));
    for (Operation *op : members(groups[g]))
      op->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          Value v = throughReshapes(operand);
          if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() == &block)
            reads[g].insert(arg.getArgNumber());
        }
      });
  }
  // The buffers a step's results are returned in.
  auto written = isa<func::ReturnOp>(block.getTerminator()) ? kernel->getAttrOfType<ArrayAttr>("furiosa.written")
                                                             : ArrayAttr();
  for (OpOperand &use : block.getTerminator()->getOpOperands()) {
    Value v = use.get();
    while (v.getDefiningOp() && isCheap(v.getDefiningOp()) && v.getDefiningOp()->getNumOperands())
      v = v.getDefiningOp()->getOperand(0);
    if (Operation *def = v.getDefiningOp(); def && groupOf.count(def) && written)
      writes[groupOf[def]].insert(cast<IntegerAttr>(written[use.getOperandNumber()]).getInt());
  }
  // An insert_slice writes in place the buffer its destination is (through reshapes and earlier inserts).
  for (int g = 0; g < n; ++g)
    for (Operation *op : groups[g].ops)
      if (auto insert = dyn_cast<tensor::InsertSliceOp>(op)) {
        Value d = insert.getDest();
        while (Operation *def = d.getDefiningOp()) {
          if (isReshape(def))
            d = def->getOperand(0);
          else if (auto earlier = dyn_cast<tensor::InsertSliceOp>(def))
            d = earlier.getDest();
          else
            break;
        }
        if (auto arg = dyn_cast<BlockArgument>(d); arg && arg.getOwner() == &block)
          writes[g].insert(arg.getArgNumber());
      }
  for (int g = 0; g < n; ++g)
    for (unsigned buffer : writes[g])
      for (int h = 0; h < n; ++h)
        if (h != g && reads[h].contains(buffer))
          after[g].insert(h);
  // Kahn's algorithm, earliest program position first; on a cycle keep program order.
  std::vector<int> sorted, pending(n);
  std::vector<bool> done(n);
  for (int g = 0; g < n; ++g)
    pending[g] = after[g].size();
  for (int step = 0; step < n; ++step) {
    int next = -1;
    for (int g = 0; g < n && next < 0; ++g)
      if (!done[g] && !pending[g])
        next = g;
    if (next < 0)
      return;
    done[next] = true;
    sorted.push_back(next);
    for (int g = 0; g < n; ++g)
      if (after[g].contains(next))
        --pending[g];
  }
  std::vector<Group> reordered;
  std::vector<llvm::SmallSetVector<unsigned, 2>> reorderedReads;
  for (int g : sorted) {
    reordered.push_back(groups[g]);
    reorderedReads.push_back(reads[g]);
  }
  groups = std::move(reordered);
  reads = std::move(reorderedReads);
  groupOf.clear();
  for (int g = 0; g < n; ++g)
    for (Operation *op : groups[g].ops)
      groupOf[op] = g;
}

func::FuncOp Splitter::outline(const Group &g, StringRef name, ArrayRef<Value> outputs,
                               SmallVectorImpl<Value> &inputs) {
  // Cheap producers are cloned in; everything else the members read is an input.
  llvm::SetVector<Operation *> cloned;
  llvm::SetVector<Value> external;
  // Whether an op lies inside one of the members' own regions (as opposed to around the block).
  auto nested = [&](Operation *op) { return op && op != block.getParentOp() && block.findAncestorOpInBlock(*op); };
  std::function<void(Value)> need = [&](Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() != &block) {
      if (!nested(arg.getOwner()->getParentOp()))
        external.insert(v); // an argument around the block (a loop body reads the kernel's buffers)
      return;
    }
    Operation *def = v.getDefiningOp();
    if (def && (g.contains(def) || cloned.contains(def)))
      return;
    if (def && def->getBlock() != &block && nested(def))
      return;
    if (def && isIterationSlice(def)) {
      external.insert(v); // the plan's view of a buffer at the iteration
      return;
    }
    // A reshape of something the step reads stays in the plan (it moves no data): the step reads the
    // reshaped buffer, so op classes see a plain argument.
    Operation *root = throughReshapes(v).getDefiningOp();
    bool readsReshaped = def && isReshape(def) &&
                         (!root || resultOf.count(throughReshapes(v)) || !(isCheap(root) || g.contains(root)));
    if (def && isCheap(def) && !resultOf.count(v) && !readsReshaped) {
      // Its own operands, and what its regions capture (a pad's padding value).
      def->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands())
          if (!(operand.getDefiningOp() && def->isAncestor(operand.getDefiningOp())))
            need(operand);
      });
      cloned.insert(def);
      return;
    }
    external.insert(v);
  };
  SmallVector<Operation *> all = members(g);
  for (Operation *op : all)
    op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        need(operand);
    });
  inputs.assign(external.begin(), external.end());

  // Each output is written to an argument: the buffer an insert_slice updates, or a fresh destination.
  OpBuilder b(kernel.getContext());
  SmallVector<Type> argTypes;
  for (Value v : inputs)
    argTypes.push_back(v.getType());
  SmallVector<int64_t> written;
  for (Value out : outputs) {
    // An insert_slice updates the buffer it writes (through reshapes) in place; order() put the steps that
    // read the buffer's earlier value first, and stepsPlan rejects a plan that would read it stale.
    auto insert = out.getDefiningOp<tensor::InsertSliceOp>();
    auto dest = inputs.end();
    Value d = insert ? insert.getDest() : Value();
    while (d && (dest = llvm::find(inputs, d)) == inputs.end())
      d = d.getDefiningOp() && isReshape(d.getDefiningOp()) ? d.getDefiningOp()->getOperand(0) : Value();
    if (dest != inputs.end()) {
      written.push_back(dest - inputs.begin());
    } else {
      written.push_back(argTypes.size());
      argTypes.push_back(out.getType());
    }
  }
  SmallVector<Type> resultTypes;
  for (int64_t w : written)
    resultTypes.push_back(argTypes[w]);
  auto func = func::FuncOp::create(b, kernel.getLoc(), name, b.getFunctionType(argTypes, resultTypes));
  func->setAttr("furiosa.written", b.getI64ArrayAttr(written));
  Block *body = func.addEntryBlock();
  IRMapping map;
  for (auto [i, v] : llvm::enumerate(inputs)) {
    map.map(v, body->getArgument(i));
    // Arguments keep the kernel's names (NameLocs) so the mapped step names its parameters.
    if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() == &block)
      body->getArgument(i).setLoc(arg.getLoc());
  }
  for (auto [k, w] : llvm::enumerate(written))
    if (w >= int64_t(inputs.size()))
      body->getArgument(w).setLoc(NameLoc::get(b.getStringAttr("t" + std::to_string(k))));
  b.setInsertionPointToStart(body);
  SmallVector<Operation *> clones(cloned.begin(), cloned.end());
  llvm::append_range(clones, all);
  llvm::sort(clones, [&](Operation *x, Operation *y) { return position.lookup(x) < position.lookup(y); });
  for (Operation *op : clones)
    b.clone(*op, map);
  SmallVector<Value> results;
  for (auto [out, w] : llvm::zip(outputs, written))
    results.push_back(reshapeTo(b, kernel.getLoc(), map.lookup(out), cast<RankedTensorType>(argTypes[w])));
  func::ReturnOp::create(b, kernel.getLoc(), results);
  // Reshapes of fills and of other reshapes inside the step fold away (upstream canonicalizations).
  MLIRContext *ctx = kernel.getContext();
  RewritePatternSet patterns(ctx);
  linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsGreedily(func.getBody(), std::move(patterns));
  return func;
}

bool Splitter::mappable(int at, const Group &g) {
  // The candidate stands in for the group it would replace, which must not count as another reader.
  Group kept = std::exchange(groups[at], g);
  SmallVector<Value> inputs, outputs = outputsOf(at, /*all=*/true);
  func::FuncOp func = outline(g, "candidate", outputs, inputs);
  groups[at] = std::move(kept);
  std::string text;
  llvm::raw_string_ostream os(text);
  func.print(os);
  auto [it, fresh] = mapped.try_emplace(text, false);
  if (fresh)
    it->second = mapsWhole(func);
  func.erase();
  return it->second;
}

void Splitter::run(bool single) {
  for (Operation &op : block)
    if (isUnit(&op)) {
      if (single && !groups.empty()) {
        groupOf[&op] = 0;
        groups[0].ops.push_back(&op);
        continue;
      }
      groupOf[&op] = groups.size();
      groups.push_back({{&op}, {}});
    }
  // A loop body outlines even one step; a block around split loops is a plan even with none of its own.
  if (groups.size() < (single || isa<scf::YieldOp>(block.getTerminator()) ? 1u : 2u) &&
      !llvm::any_of(block, [](Operation &op) { return isSplitLoop(&op); }))
    return;

  // A producer merges into its only consumer, or is recomputed by one of several, while an op class maps
  // the merged step.
  for (bool changed = !single; changed;) {
    changed = false;
    for (int g = groups.size() - 1; g >= 0 && !changed; --g) {
      if (groups[g].ops.empty())
        continue;
      llvm::SmallSetVector<int, 4> producers;
      for (Operation *def : producedFor(groups[g]))
        producers.insert(groupOf.lookup(def));
      for (int p : producers) {
        llvm::SmallSetVector<int, 4> readers;
        for (Operation *op : groups[p].ops)
          for (Value r : op->getResults())
            consumers(r, readers);
        Group merged = groups[g];
        bool move = readers.size() == 1 && readers.front() == g;
        if (move) {
          // An op the consumer recomputed (a dup) moves in as its own: a group holds each op once.
          llvm::erase_if(merged.dups, [&](Operation *op) { return llvm::is_contained(groups[p].ops, op); });
          llvm::append_range(merged.ops, groups[p].ops);
        } else {
          if (llvm::all_of(groups[p].ops, [&](Operation *op) { return groups[g].contains(op); }))
            continue;
          for (Operation *op : groups[p].ops)
            if (!merged.contains(op))
              merged.dups.push_back(op);
        }
        for (Operation *op : groups[p].dups)
          if (!merged.contains(op))
            merged.dups.push_back(op);
        if (!mappable(g, merged))
          continue;
        if (move) {
          for (Operation *op : groups[p].ops)
            groupOf[op] = g;
          groups[p].ops.clear();
          groups[p].dups.clear();
        }
        groups[g] = merged;
        changed = true;
        break;
      }
      // No single producer merges: try every computing (non-copy) producer at once, since a step may map
      // only once all of its row values, say, are computed inside it.
      if (changed || producers.size() < 2)
        continue;
      Group merged = groups[g];
      SmallVector<int> moved;
      for (int p : producers) {
        if (llvm::any_of(groups[p].ops, [](Operation *op) { return isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op); }))
          continue;
        llvm::SmallSetVector<int, 4> readers;
        for (Operation *op : groups[p].ops)
          for (Value r : op->getResults())
            consumers(r, readers);
        if (readers.size() == 1 && readers.front() == g) {
          llvm::erase_if(merged.dups, [&](Operation *op) { return llvm::is_contained(groups[p].ops, op); });
          llvm::append_range(merged.ops, groups[p].ops);
          moved.push_back(p);
        } else {
          for (Operation *op : groups[p].ops)
            if (!merged.contains(op))
              merged.dups.push_back(op);
        }
        for (Operation *op : groups[p].dups)
          if (!merged.contains(op))
            merged.dups.push_back(op);
      }
      if (merged.ops.size() == groups[g].ops.size() && merged.dups.size() == groups[g].dups.size())
        continue;
      if (!mappable(g, merged))
        continue;
      for (int p : moved) {
        for (Operation *op : groups[p].ops)
          groupOf[op] = g;
        groups[p].ops.clear();
        groups[p].dups.clear();
      }
      groups[g] = merged;
      changed = true;
    }
  }
  prune();
  order();
  // A loop body outlines even one step; a block around split loops is a plan even with none of its own.
  if (groups.size() < (single || isa<scf::YieldOp>(block.getTerminator()) ? 1u : 2u) &&
      !llvm::any_of(block, [](Operation &op) { return isSplitLoop(&op); }))
    return;

  // Outline each step and call it from the kernel, in order.
  ModuleOp module = kernel->getParentOfType<ModuleOp>();
  // Step names kernel__sN: a hashed table and a running N (probing from 0 with module.lookupSymbol, a scan of the
  // module per probe, was cubic in the steps: minutes for tl.sort's thousands).
  SymbolTable symbols(module);
  int next = 0;
  auto freshName = [&] {
    std::string name;
    do
      name = (kernel.getName() + "__s" + Twine(next++)).str();
    while (symbols.lookup(name));
    return name;
  };
  Operation *terminator = block.getTerminator();
  auto returnOp = dyn_cast<func::ReturnOp>(terminator);
  auto written = returnOp ? kernel->getAttrOfType<ArrayAttr>("furiosa.written") : ArrayAttr();
  OpBuilder b(kernel.getContext());
  SmallVector<Operation *> outlinedOps;
  SmallVector<std::pair<Value, unsigned>> misplaced; // results a step left in a temporary: (value, buffer)
  for (auto [s, group] : llvm::enumerate(groups)) {
    SmallVector<Value> inputs, outputs = outputsOf(s, /*all=*/false);
    func::FuncOp step = outline(group, freshName(), outputs, inputs);
    step.setPrivate();
    symbols.insert(step);
    auto stepWritten = step->getAttrOfType<ArrayAttr>("furiosa.written");
    // Call operands: the inputs (as earlier steps left them), then a destination per fresh output: the buffer
    // the kernel returns it into (through reshapes, which move no data), or a new temporary.
    b.setInsertionPoint(terminator); // calls run in step order
    SmallVector<Value> operands;
    for (Value v : inputs)
      operands.push_back(resultOf.count(v) ? resultOf[v] : v);
    for (auto [k, out] : llvm::enumerate(outputs)) {
      int64_t w = cast<IntegerAttr>(stepWritten[k]).getInt();
      if (w < int64_t(inputs.size()))
        continue;
      Value dest;
      for (OpOperand &use : terminator->getOpOperands()) {
        // Through reshapes, and inserts that update `out` in place on the way to the return.
        Value v = use.get();
        while (v != out && v.getDefiningOp() && (isReshape(v.getDefiningOp()) || isa<tensor::InsertSliceOp>(v.getDefiningOp())))
          v = isa<tensor::InsertSliceOp>(v.getDefiningOp()) ? v.getDefiningOp<tensor::InsertSliceOp>().getDest()
                                                           : v.getDefiningOp()->getOperand(0);
        if (v != out || !written)
          continue;
        unsigned index = cast<IntegerAttr>(written[use.getOperandNumber()]).getInt();
        Value buffer = kernel.getArgument(index);
        auto bt = cast<RankedTensorType>(buffer.getType()), ot = cast<RankedTensorType>(out.getType());
        // The buffer takes the result once every other step that reads it has run (order() put them first).
        bool free = bt.getNumElements() == ot.getNumElements() && bt.getElementType() == ot.getElementType();
        for (size_t h = s + 1; h < groups.size() && free; ++h)
          free = !reads[h].contains(index);
        // A step that reads the buffer itself cannot take it as its destination too (the runtime refuses the
        // aliasing): its result goes to a temporary, copied into the buffer last.
        if (free && reads[s].contains(index))
          misplaced.push_back({out, index});
        else if (free)
          dest = reshapeTo(b, kernel.getLoc(), buffer, ot);
        break;
      }
      if (!dest)
        dest = tensor::EmptyOp::create(b, kernel.getLoc(), cast<RankedTensorType>(out.getType()).getShape(),
                                       cast<RankedTensorType>(out.getType()).getElementType());
      operands.push_back(dest);
    }
    auto call = func::CallOp::create(b, kernel.getLoc(), step, operands);
    for (auto [out, result] : llvm::zip(outputs, call.getResults()))
      resultOf[out] = reshapeTo(b, kernel.getLoc(), result, cast<RankedTensorType>(out.getType()));
    llvm::append_range(outlinedOps, group.ops);
  }
  for (auto [out, index] : misplaced) {
    Value value = resultOf[out];
    auto t = cast<RankedTensorType>(value.getType());
    OpBuilder detached(kernel.getContext());
    auto copy = func::FuncOp::create(detached, kernel.getLoc(), freshName(), b.getFunctionType({t, t}, {t}));
    copy->setAttr("furiosa.written", b.getI64ArrayAttr({1}));
    OpBuilder cb = OpBuilder::atBlockBegin(copy.addEntryBlock());
    func::ReturnOp::create(cb, kernel.getLoc(), copyOf(cb, kernel.getLoc(), copy.getArgument(0)));
    copy.setPrivate();
    symbols.insert(copy);
    b.setInsertionPoint(terminator);
    resultOf[out] = func::CallOp::create(b, kernel.getLoc(), copy,
                                         ValueRange{value, reshapeTo(b, kernel.getLoc(), kernel.getArgument(index), t)})
                        .getResult(0);
  }
  // The kernel now only calls steps: what remains reads the step results; the unit ops (outlined, or
  // recomputed by every step that read them) go.
  for (auto &[out, value] : resultOf)
    out.replaceUsesWithIf(value, [&](OpOperand &use) {
      Operation *user = use.getOwner();
      while (user->getBlock() != &block)
        user = user->getParentOp();
      return !isUnit(user);
    });
  SmallVector<Operation *> units;
  for (Operation &op : block)
    if (isUnit(&op))
      units.push_back(&op);
  for (Operation *op : llvm::reverse(units)) {
    op->dropAllUses();
    op->erase();
  }
  for (bool erased = true; erased;) {
    erased = false;
    SmallVector<Operation *> ops;
    for (Operation &op : block)
      ops.push_back(&op);
    for (Operation *op : llvm::reverse(ops))
      if (isCheap(op) && op->use_empty()) {
        op->erase();
        erased = true;
      }
  }
  // The kernel's own cheap ops on step results move right after their operands, before the later steps that
  // read them (those made with the calls are in place already).
  // A split loop moves likewise, after what it reads.
  llvm::SetVector<Operation *> late;
  for (bool grew = true; grew;) {
    grew = false;
    for (Operation &op : block)
      if (((isCheap(&op) && position.count(&op)) || isSplitLoop(&op)) && !late.contains(&op) &&
          llvm::any_of(blockUses(&op), [&](Value v) {
            Operation *def = v.getDefiningOp();
            return def && (isa<func::CallOp>(def) || late.contains(def));
          }))
        grew = late.insert(&op);
  }
  llvm::SmallPtrSet<Operation *, 8> moved;
  while (moved.size() < late.size())
    for (Operation *op : late)
      if (!moved.contains(op) && llvm::all_of(blockUses(op), [&](Value v) {
            Operation *def = v.getDefiningOp();
            return !def || !late.contains(def) || moved.contains(def);
          })) {
        Operation *last = nullptr;
        for (Value v : blockUses(op))
          if (Operation *def = v.getDefiningOp(); def && def->getBlock() == &block && (!last || last->isBeforeInBlock(def)))
            last = def;
        op->moveAfter(last);
        moved.insert(op);
      }
  block.getParentOp()->setAttr(kStepsAttr, b.getUnitAttr());
}

} // namespace

namespace {
/// A buffer argument the kernel returns into another written buffer (an atomic's old value) is a copy, which
/// order() runs before a step overwrites its source.
void copyReturnedArguments(func::FuncOp kernel) {
  Block &block = kernel.getBody().front();
  auto ret = dyn_cast<func::ReturnOp>(block.getTerminator());
  auto written = kernel->getAttrOfType<ArrayAttr>("furiosa.written");
  if (!ret || !written)
    return;
  OpBuilder b(ret);
  for (OpOperand &use : ret->getOpOperands()) {
    auto arg = dyn_cast<BlockArgument>(throughReshapes(use.get()));
    if (!arg || arg.getOwner() != &block ||
        int64_t(arg.getArgNumber()) == cast<IntegerAttr>(written[use.getOperandNumber()]).getInt())
      continue;
    use.set(copyOf(b, ret.getLoc(), use.get()));
  }
}

} // namespace

LogicalResult mlir::furiosa::splitSteps(ModuleOp module) {
  // The mapping oracle builds fopt kernels; steps add tensor and linalg ops to a kernel that may have had none.
  module.getContext()->loadDialect<fopt::FoptDialect, tensor::TensorDialect, linalg::LinalgDialect>();
  guardNaNs(module);
  roundToF8(module);
  SmallVector<func::FuncOp> kernels;
  for (auto func : module.getOps<func::FuncOp>())
    if (func.isPublic() && !func->hasAttr(kStepsAttr))
      kernels.push_back(func);
  // Kernels an op class maps as raised stay as they are: normalizing could only lose that (a batched
  // contraction mapped whole would split into a step per batch).
  if (llvm::all_of(kernels, mapsWhole))
    return success();
  if (failed(unrollKeptLoops(module)))
    return failure();
  normalizeKernels(module, mapsAlone);
  normalizeContractions(module, mapsAlone);
  relayoutFactorReads(module, mapsAlone);
  // Reads the normalization rewrote alike (each operand gets its own reshape and box) become one again.
  PassManager pm(module.getContext());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module)))
    return failure();
  hoistIndexOnly(module);
  sliceKeptLoops(module);
  for (func::FuncOp kernel : kernels)
    copyReturnedArguments(kernel);
  for (func::FuncOp kernel : kernels) {
    // A kept loop's body splits into steps of its own, which the plan runs once per iteration; the fills that
    // start its carried values become tables.
    for (auto loop : llvm::to_vector(kernel.getBody().front().getOps<scf::ForOp>())) {
      // A slice the plan cannot view as one range (a box of several rows per iteration) unrolls the loop: its
      // iterations' boxes are then copy steps.
      bool viewable = true;
      loop.getBody()->walk([&](tensor::ExtractSliceOp s) { viewable &= !isIterationSlice(s) || iterationView(s); });
      if (!viewable) {
        if (failed(loopUnrollFull(loop)))
          return loop.emitError("cannot unroll a kept loop");
        // The iterations' index arithmetic folds into static box offsets (upstream canonicalizations).
        RewritePatternSet patterns(kernel.getContext());
        tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, kernel.getContext());
        (void)applyPatternsGreedily(kernel, std::move(patterns));
        continue;
      }
      for (OpOperand &init : loop.getInitArgsMutable()) {
        Attribute c;
        auto fill = init.get().getDefiningOp<linalg::FillOp>();
        if (fill && matchPattern(fill.getInputs()[0], m_Constant(&c))) {
          OpBuilder b(loop);
          init.set(arith::ConstantOp::create(b, loop.getLoc(),
                                             DenseElementsAttr::get(cast<ShapedType>(init.get().getType()), c)));
        }
      }
      Splitter(kernel, *loop.getBody()).run();
    }
    if (!mapsWhole(kernel))
      Splitter(kernel).run();
    else if (!tensorConstants(kernel).empty())
      Splitter(kernel).run(/*single=*/true);
  }
  // A bad split must fail here, not print in generic form for a later stage to choke on.
  return verify(module);
}

FailureOr<std::string> mlir::furiosa::stepsPlan(ModuleOp module) {
  func::FuncOp kernel;
  for (auto func : module.getOps<func::FuncOp>())
    if (func.isPublic())
      kernel = func;
  if (!kernel)
    return module.emitError("no kernel function");
  if (!kernel->hasAttr(kStepsAttr))
    return std::string("null");
  // Slots: the kernel's buffers, then the temporaries; reshapes keep their operand's slot.
  llvm::DenseMap<Value, llvm::json::Object> slot;
  llvm::json::Array temps, steps, consts, returns;
  auto slotOf = [&](Value v) -> llvm::json::Value {
    auto it = slot.find(v);
    return it == slot.end() ? llvm::json::Value(nullptr) : llvm::json::Value(llvm::json::Object(it->second));
  };
  llvm::StringMap<Value> latest; // slot -> the value it holds now
  auto key = [](const llvm::json::Value &s) {
    std::string text;
    llvm::raw_string_ostream(text) << s;
    return text;
  };
  auto root = [](Value v) {
    while (v.getDefiningOp() && isReshape(v.getDefiningOp()))
      v = v.getDefiningOp()->getOperand(0);
    return v;
  };
  for (BlockArgument arg : kernel.getArguments()) {
    slot[arg] = llvm::json::Object{{"buffer", int64_t(arg.getArgNumber())}};
    latest[key(slotOf(arg))] = arg;
  }
  // A table the host uploads: its slot.
  auto addConst = [&](DenseElementsAttr data, Operation *op) -> FailureOr<llvm::json::Object> {
    auto t = cast<RankedTensorType>(data.getType());
    if (!t.getElementType().isIntOrFloat() || t.getElementType().getIntOrFloatBitWidth() % 8)
      return op->emitError("a table of ") << t.getElementType() << " has no byte layout";
    std::vector<char> bytes;
    int64_t width = t.getElementType().getIntOrFloatBitWidth() / 8;
    ArrayRef<char> raw = data.getRawData();
    for (int64_t e = 0; e < t.getNumElements(); ++e) {
      int64_t from = data.isSplat() ? 0 : e * width;
      bytes.insert(bytes.end(), raw.begin() + from, raw.begin() + from + width);
    }
    llvm::json::Array shape;
    for (int64_t d : t.getShape())
      shape.push_back(d);
    std::string dtype;
    llvm::raw_string_ostream(dtype) << t.getElementType();
    llvm::json::Object at{{"const", int64_t(consts.size())}};
    consts.push_back(llvm::json::Object{{"dtype", dtype}, {"shape", std::move(shape)}, {"data", llvm::encodeBase64(bytes)}});
    return at;
  };
  int64_t carriedCount = 0; // kept loops' carried values, numbered across the plan
  std::function<LogicalResult(Block &, llvm::json::Array &)> emit = [&](Block &block,
                                                                       llvm::json::Array &steps) -> LogicalResult {
    for (Operation &op : block) {
      if (auto empty = dyn_cast<tensor::EmptyOp>(op)) {
        auto t = empty.getType();
        llvm::json::Array shape;
        for (int64_t d : t.getShape())
          shape.push_back(d);
        std::string dtype;
        llvm::raw_string_ostream(dtype) << t.getElementType();
        slot[empty] = llvm::json::Object{{"temp", int64_t(temps.size())}};
        latest[key(slotOf(empty))] = empty;
        temps.push_back(llvm::json::Object{{"dtype", dtype}, {"shape", std::move(shape)}});
      } else if (auto c = dyn_cast<arith::ConstantOp>(op); c && isa<RankedTensorType>(c.getType())) {
        FailureOr<llvm::json::Object> at = addConst(cast<DenseElementsAttr>(c.getValue()), c);
        if (failed(at))
          return failure();
        slot[c] = std::move(*at);
      } else if (isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp>(op)) {
        // Index arithmetic of a loop iteration: the slices below say what it addresses.
      } else if (isReshape(&op)) {
        auto it = slot.find(op.getOperand(0));
        if (it == slot.end())
          return op.emitError("a reshape of an unknown slot");
        slot[op.getResult(0)] = llvm::json::Object(it->second);
      } else if (isIterationSlice(&op)) {
        // A contiguous box at the iteration: the source's bytes from offset + iteration * iter_stride.
        auto slice = cast<tensor::ExtractSliceOp>(op);
        auto it = slot.find(slice.getSource());
        if (it == slot.end() || it->second.get("offset"))
          return op.emitError("an iteration slice of an unknown slot");
        std::optional<std::array<int64_t, 3>> bytes = iterationView(slice);
        if (!bytes)
          return op.emitError("an iteration slice that is not one contiguous range at the iteration index");
        llvm::json::Object view(it->second);
        view["offset"] = (*bytes)[0];
        view["iter_stride"] = (*bytes)[1];
        view["bytes"] = (*bytes)[2];
        slot[slice.getResult()] = std::move(view);
      } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (!loop->hasAttr(kStepsAttr) || !loop.getStaticTripCount())
          return op.emitError("a kept loop the plan cannot run");
        int64_t first = carriedCount;
        carriedCount += loop.getNumRegionIterArgs();
        llvm::json::Array carried, body, yields;
        for (auto [k, init] : llvm::enumerate(loop.getInitArgs())) {
          llvm::json::Value s = slotOf(init);
          if (s.kind() == llvm::json::Value::Null)
            return op.emitError("a carried value that starts in no buffer");
          carried.push_back(std::move(s));
          slot[loop.getRegionIterArgs()[k]] = llvm::json::Object{{"carried", int64_t(first + k)}};
        }
        if (failed(emit(*loop.getBody(), body)))
          return failure();
        for (Value y : loop.getBody()->getTerminator()->getOperands()) {
          llvm::json::Value s = slotOf(y);
          if (s.kind() == llvm::json::Value::Null)
            return op.emitError("a carried value that leaves an iteration in no buffer");
          yields.push_back(std::move(s));
        }
        for (auto [k, result] : llvm::enumerate(loop.getResults()))
          slot[result] = llvm::json::Object{{"carried", int64_t(first + k)}};
        steps.push_back(llvm::json::Object{
            {"loop", llvm::json::Object{{"count", loop.getStaticTripCount()->getSExtValue()},
                                        {"carried", std::move(carried)},
                                        {"steps", std::move(body)},
                                        {"yields", std::move(yields)}}}});
      } else if (isa<scf::YieldOp>(op)) {
        // The loop reads its yields.
      } else if (auto call = dyn_cast<func::CallOp>(op)) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        auto stepWritten = callee->getAttrOfType<ArrayAttr>("furiosa.written");
        llvm::json::Array args;
        for (Value operand : call.getOperands()) {
          llvm::json::Value s = slotOf(operand);
          if (s.kind() == llvm::json::Value::Null)
            return call.emitError("a step operand that is no buffer or temporary");
          // A slot holds one version at a time: a step reads the latest (loop views name theirs).
          const llvm::json::Object *o = s.getAsObject();
          if (!o->get("carried") && !o->get("const") && !o->get("offset") && latest.lookup(key(s)) != root(operand))
            return call.emitError("a step reads a buffer or temporary that an earlier step overwrote");
          args.push_back(std::move(s));
        }
        for (auto [k, result] : llvm::enumerate(call.getResults())) {
          auto it = slot.find(call.getOperand(cast<IntegerAttr>(stepWritten[k]).getInt()));
          slot[result] = llvm::json::Object(it->second);
          latest[key(slotOf(result))] = result;
        }
        // The step on its own, as a module codegen maps.
        OpBuilder b(module.getContext());
        auto standalone = ModuleOp::create(b, module.getLoc());
        for (NamedAttribute attr : module->getAttrs())
          standalone->setAttr(attr.getName(), attr.getValue());
        auto copy = cast<func::FuncOp>(callee->clone());
        copy.setPublic();
        standalone.push_back(copy);
        // Its tables become trailing arguments, read from constant buffers the host uploads.
        for (DenseElementsAttr data : externalize(copy)) {
          FailureOr<llvm::json::Object> at = addConst(data, call);
          if (failed(at))
            return failure();
          args.push_back(std::move(*at));
        }
        std::string text;
        llvm::raw_string_ostream os(text);
        standalone.print(os, OpPrintingFlags().enableDebugInfo(true, false));
        standalone->erase();
        steps.push_back(llvm::json::Object{{"func", callee.getName().str()}, {"args", std::move(args)},
                                           {"module", text}});
      } else if (auto ret = dyn_cast<func::ReturnOp>(op)) {
        auto written = kernel->getAttrOfType<ArrayAttr>("furiosa.written");
        for (auto [r, v] : llvm::enumerate(ret.getOperands())) {
          auto it = slot.find(v);
          int64_t buffer = cast<IntegerAttr>(written[r]).getInt();
          // A kept loop's final carried value is the buffer itself: the host hands it back as the result.
          if (it != slot.end() && it->second.get("carried") && !it->second.get("offset")) {
            returns.push_back(llvm::json::Object{{"buffer", buffer}, {"carried", *it->second.getInteger("carried")}});
            continue;
          }
          if (it == slot.end() || it->second.getInteger("buffer") != buffer)
            return ret.emitError("result ") << r << " is not left in buffer " << buffer;
        }
      } else {
        return op.emitError("a split kernel holds only temporaries, tables, reshapes, kept loops and step calls");
      }
    }
    return success();
  };
  if (failed(emit(kernel.getBody().front(), steps)))
    return failure();
  llvm::json::Object plan{{"kernel", kernel.getName().str()},
                          {"temps", std::move(temps)},
                          {"consts", std::move(consts)},
                          {"returns", std::move(returns)},
                          {"steps", std::move(steps)}};
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::json::Value(std::move(plan));
  return text;
}

namespace {

struct SplitStepsPass : PassWrapper<SplitStepsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitStepsPass)

  StringRef getArgument() const override { return "furiosa-split-steps"; }
  StringRef getDescription() const override {
    return "Split structured kernels no op class maps whole into steps one class each maps";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registerCodegenDialects(registry); }
  void runOnOperation() override {
    if (failed(splitSteps(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createSplitStepsPass() { return std::make_unique<SplitStepsPass>(); }

namespace {

/// One stage of splitSteps as a pass of its own, for the per-pass tests.
struct StagePass : PassWrapper<StagePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StagePass)

  StagePass(StringRef argument, StringRef description, std::function<LogicalResult(ModuleOp)> stage)
      : argument(argument), description(description), stage(std::move(stage)) {}
  StagePass(const StagePass &) = default;

  StringRef getArgument() const override { return argument; }
  StringRef getDescription() const override { return description; }
  void getDependentDialects(DialectRegistry &registry) const override { registerCodegenDialects(registry); }
  void runOnOperation() override {
    getContext().loadDialect<fopt::FoptDialect>(); // mapsAlone builds fopt kernels
    if (failed(stage(getOperation())))
      signalPassFailure();
  }

  StringRef argument, description;
  std::function<LogicalResult(ModuleOp)> stage;
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createNormalizeKernelsPass() {
  return std::make_unique<StagePass>("furiosa-normalize-kernels",
                                     "Promote affine extracts, delinearize mixed-radix reads, masks to padding",
                                     [](ModuleOp m) {
                                       normalizeKernels(m);
                                       return success();
                                     });
}

std::unique_ptr<Pass> mlir::furiosa::createNormalizeContractionsPass() {
  return std::make_unique<StagePass>("furiosa-normalize-contractions",
                                     "Take contractions apart into the 2-D form op classes map", [](ModuleOp m) {
                                       normalizeContractions(m, mapsAlone);
                                       return success();
                                     });
}

std::unique_ptr<Pass> mlir::furiosa::createHoistIndexOnlyPass() {
  return std::make_unique<StagePass>("furiosa-hoist-index-only",
                                     "Replace index-only generic body values with constant tables", [](ModuleOp m) {
                                       hoistIndexOnly(m);
                                       return success();
                                     });
}

std::unique_ptr<Pass> mlir::furiosa::createKeptLoopsPass() {
  return std::make_unique<StagePass>(
      "furiosa-kept-loops", "Lift kept loops' generics over their iterations (or unroll), then structure them",
      [](ModuleOp m) { return unrollKeptLoops(m); });
}

std::unique_ptr<Pass> mlir::furiosa::createSliceKeptLoopsPass() {
  return std::make_unique<StagePass>("furiosa-slice-kept-loops",
                                     "Tile lifted generics back to their iteration's slice", [](ModuleOp m) {
                                       sliceKeptLoops(m);
                                       return success();
                                     });
}

std::unique_ptr<Pass> mlir::furiosa::createStepsPlanPass() {
  return std::make_unique<StagePass>("furiosa-steps-plan", "Print a split kernel's launch plan (JSON) to stdout",
                                     [](ModuleOp m) -> LogicalResult {
                                       FailureOr<std::string> plan = stepsPlan(m);
                                       if (failed(plan))
                                         return failure();
                                       llvm::outs() << *plan << "\n";
                                       return success();
                                     });
}

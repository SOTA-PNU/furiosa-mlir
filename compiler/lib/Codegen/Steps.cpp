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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"

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

/// Everything else at the top level is part of some step: a step an op class may not map (a kept loop, say)
/// still gives a valid plan, and L2 names it.
bool isUnit(Operation *op) {
  return !isCheap(op) && !isa<func::ReturnOp, func::CallOp>(op) && op->getNumResults() > 0;
}

/// The value a reshape chain starts from.
Value throughReshapes(Value v) {
  while (v.getDefiningOp() && isReshape(v.getDefiningOp()))
    v = v.getDefiningOp()->getOperand(0);
  return v;
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
    if (failed(func.insertArgument(func.getNumArguments(), c.getType(), {},
                                   NameLoc::get(StringAttr::get(func.getContext(), name)))))
      break;
    c.replaceAllUsesWith(func.getArgument(func.getNumArguments() - 1));
    c.erase();
    data.push_back(attr);
  }
  return data;
}

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
    if (mapsOnCopy(copy, c))
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

/// Whether some op class maps `g` as a step of its own: its inputs as buffers, a fill init kept.
bool mapsAlone(linalg::GenericOp g) {
  OpBuilder b(g.getContext());
  Location loc = g.getLoc();
  SmallVector<Value> ins(g.getDpsInputs());
  SmallVector<Type> types;
  for (Value v : ins)
    types.push_back(v.getType());
  types.push_back(g.getResultTypes()[0]);
  auto func = func::FuncOp::create(b, loc, "alone", b.getFunctionType(types, g.getResultTypes()));
  func->setAttr("furiosa.written", b.getI64ArrayAttr({int64_t(ins.size())}));
  Block *body = func.addEntryBlock();
  b.setInsertionPointToStart(body);
  IRMapping map;
  for (auto [v, arg] : llvm::zip(ins, body->getArguments()))
    map.map(v, arg);
  Value init = g.getDpsInits()[0];
  if (auto fill = init.getDefiningOp<linalg::FillOp>(); fill && fill.getInputs()[0].getDefiningOp<arith::ConstantOp>()) {
    b.clone(*fill.getInputs()[0].getDefiningOp(), map);
    b.clone(*fill.getOutputs()[0].getDefiningOp(), map);
    b.clone(*fill, map);
  } else {
    map.map(init, body->getArgument(ins.size()));
  }
  Operation *copy = b.clone(*g, map);
  func::ReturnOp::create(b, loc, copy->getResults());
  bool ok = mapsWhole(func);
  func.erase();
  return ok;
}

/// A step: the unit ops it owns (it produces their values for the other steps), and those of other steps it
/// recomputes rather than reading their results.
struct Group {
  SmallVector<Operation *> ops, dups;
  bool contains(Operation *op) const { return llvm::is_contained(ops, op) || llvm::is_contained(dups, op); }
};

class Splitter {
public:
  Splitter(func::FuncOp kernel) : kernel(kernel), block(kernel.getBody().front()) {
    for (auto [i, op] : llvm::enumerate(block.getOperations()))
      position[&op] = i;
  }

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

  /// A group's ops and dups, in program order.
  SmallVector<Operation *> members(const Group &g) {
    SmallVector<Operation *> all(g.ops);
    llvm::append_range(all, g.dups);
    llvm::sort(all, [&](Operation *x, Operation *y) { return position.lookup(x) < position.lookup(y); });
    return all;
  }
  /// The unit ops whose values `g`'s members read (through cheap ops), outside `g`.
  llvm::SetVector<Operation *> producedFor(const Group &g);
  /// The groups (or -1 for the kernel's return) whose owned ops read `v`, looking through cheap ops.
  void consumers(Value v, llvm::SmallSetVector<int, 4> &out);
  /// The values of `g`'s own ops that something outside it reads: the return, or a group that reads it
  /// without recomputing it (`all`: any op outside `g`, for a candidate).
  SmallVector<Value> outputsOf(int g, bool all);
  /// A step function for `g` that writes `outputs`, not inserted anywhere; fills `inputs` with the values
  /// it reads (kernel buffers, or outputs of earlier steps, by their original value).
  func::FuncOp outline(const Group &g, StringRef name, ArrayRef<Value> outputs, SmallVectorImpl<Value> &inputs);
  bool mappable(const Group &g);
  /// Drops groups whose values no live group needs: every reader recomputes them.
  void prune();
  /// Orders the steps: after the steps whose results they read, and a step that leaves its result in a
  /// buffer after every step that reads the buffer's original value, so the buffer can take it in place.
  void order();
};

llvm::SetVector<Operation *> Splitter::producedFor(const Group &g) {
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
          else if (isCheap(def))
            for (Value o : def->getOperands())
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
    if (isa<func::ReturnOp>(user))
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
          if (group.contains(user))
            continue;
          if (isa<func::ReturnOp>(user) || all) {
            read = true;
          } else if (isCheap(user)) {
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
    for (Operation *def : producedFor(groups[g]))
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
  auto written = kernel->getAttrOfType<ArrayAttr>("furiosa.written");
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
  std::function<void(Value)> need = [&](Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() != &block)
      return; // an argument of one of the ops' own regions
    Operation *def = v.getDefiningOp();
    if (def && (g.contains(def) || cloned.contains(def)))
      return;
    if (def && def->getBlock() != &block)
      return; // defined inside one of the ops' own regions
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

bool Splitter::mappable(const Group &g) {
  groups.push_back(g);
  SmallVector<Value> inputs, outputs = outputsOf(groups.size() - 1, /*all=*/true);
  func::FuncOp func = outline(g, "candidate", outputs, inputs);
  groups.pop_back();
  bool ok = mapsWhole(func);
  func.erase();
  return ok;
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
  if (groups.size() < (single ? 1u : 2u))
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
          llvm::append_range(merged.ops, groups[p].ops);
        } else {
          if (llvm::all_of(groups[p].ops, [&](Operation *op) { return groups[g].contains(op); }))
            continue;
          llvm::append_range(merged.dups, groups[p].ops);
        }
        for (Operation *op : groups[p].dups)
          if (!merged.contains(op))
            merged.dups.push_back(op);
        if (!mappable(merged))
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
      if (!mappable(merged))
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
  if (groups.size() < (single ? 1u : 2u))
    return;

  // Outline each step and call it from the kernel, in order.
  ModuleOp module = kernel->getParentOfType<ModuleOp>();
  auto returnOp = cast<func::ReturnOp>(block.getTerminator());
  auto written = kernel->getAttrOfType<ArrayAttr>("furiosa.written");
  OpBuilder b(kernel.getContext());
  SmallVector<Operation *> outlinedOps;
  for (auto [s, group] : llvm::enumerate(groups)) {
    SmallVector<Value> inputs, outputs = outputsOf(s, /*all=*/false);
    std::string name = (kernel.getName() + "__s" + Twine(s)).str();
    func::FuncOp step = outline(group, name, outputs, inputs);
    step.setPrivate();
    module.push_back(step);
    auto stepWritten = step->getAttrOfType<ArrayAttr>("furiosa.written");
    // Call operands: the inputs (as earlier steps left them), then a destination per fresh output: the buffer
    // the kernel returns it into (through reshapes, which move no data), or a new temporary.
    b.setInsertionPoint(returnOp); // calls run in step order
    SmallVector<Value> operands;
    for (Value v : inputs)
      operands.push_back(resultOf.count(v) ? resultOf[v] : v);
    for (auto [k, out] : llvm::enumerate(outputs)) {
      int64_t w = cast<IntegerAttr>(stepWritten[k]).getInt();
      if (w < int64_t(inputs.size()))
        continue;
      Value dest;
      for (OpOperand &use : returnOp->getOpOperands()) {
        Value v = use.get();
        while (v.getDefiningOp() && isReshape(v.getDefiningOp()) && v != out)
          v = v.getDefiningOp()->getOperand(0);
        if (v != out || !written)
          continue;
        unsigned index = cast<IntegerAttr>(written[use.getOperandNumber()]).getInt();
        Value buffer = kernel.getArgument(index);
        auto bt = cast<RankedTensorType>(buffer.getType()), ot = cast<RankedTensorType>(out.getType());
        // The buffer takes the result once every other step that reads it has run (order() put them first).
        bool free = bt.getNumElements() == ot.getNumElements() && bt.getElementType() == ot.getElementType();
        for (size_t h = s + 1; h < groups.size() && free; ++h)
          free = !reads[h].contains(index);
        if (free)
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
  llvm::SetVector<Operation *> late;
  for (bool grew = true; grew;) {
    grew = false;
    for (Operation &op : block)
      if (isCheap(&op) && position.count(&op) && !late.contains(&op) && llvm::any_of(op.getOperands(), [&](Value v) {
            Operation *def = v.getDefiningOp();
            return def && (isa<func::CallOp>(def) || late.contains(def));
          }))
        grew = late.insert(&op);
  }
  llvm::SmallPtrSet<Operation *, 8> moved;
  while (moved.size() < late.size())
    for (Operation *op : late)
      if (!moved.contains(op) && llvm::all_of(op->getOperands(), [&](Value v) {
            Operation *def = v.getDefiningOp();
            return !def || !late.contains(def) || moved.contains(def);
          })) {
        Operation *last = nullptr;
        for (Value v : op->getOperands())
          if (Operation *def = v.getDefiningOp(); def && def->getBlock() == &block && (!last || last->isBeforeInBlock(def)))
            last = def;
        op->moveAfter(last);
        moved.insert(op);
      }
  kernel->setAttr(kStepsAttr, b.getUnitAttr());
}

} // namespace

LogicalResult mlir::furiosa::splitSteps(ModuleOp module) {
  module.getContext()->loadDialect<fopt::FoptDialect>(); // the mapping oracle builds fopt kernels
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
  normalizeKernels(module);
  normalizeContractions(module, mapsAlone);
  // Reads the normalization rewrote alike (each operand gets its own reshape and box) become one again.
  PassManager pm(module.getContext());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module)))
    return failure();
  hoistIndexOnly(module);
  for (func::FuncOp kernel : kernels) {
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
  llvm::json::Array temps, steps, consts;
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
  for (Operation &op : kernel.getBody().front()) {
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
    } else if (isReshape(&op)) {
      auto it = slot.find(op.getOperand(0));
      if (it == slot.end())
        return op.emitError("a reshape of an unknown slot");
      slot[op.getResult(0)] = llvm::json::Object(it->second);
    } else if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      auto stepWritten = callee->getAttrOfType<ArrayAttr>("furiosa.written");
      llvm::json::Array args;
      for (Value operand : call.getOperands()) {
        llvm::json::Value s = slotOf(operand);
        if (s.kind() == llvm::json::Value::Null)
          return call.emitError("a step operand that is no buffer or temporary");
        // A slot holds one version at a time: a step reads the latest.
        if (latest.lookup(key(s)) != root(operand))
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
        auto t = cast<RankedTensorType>(data.getType());
        if (!t.getElementType().isIntOrFloat() || t.getElementType().getIntOrFloatBitWidth() % 8)
          return call.emitError("a table of ") << t.getElementType() << " has no byte layout";
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
        args.push_back(llvm::json::Object{{"const", int64_t(consts.size())}});
        consts.push_back(llvm::json::Object{
            {"dtype", dtype}, {"shape", std::move(shape)}, {"data", llvm::encodeBase64(bytes)}});
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
        if (it == slot.end() || it->second.getInteger("buffer") != buffer)
          return ret.emitError("result ") << r << " is not left in buffer " << buffer;
      }
    } else {
      return op.emitError("a split kernel holds only temporaries, reshapes and step calls");
    }
  }
  llvm::json::Object plan{{"kernel", kernel.getName().str()},
                          {"temps", std::move(temps)},
                          {"consts", std::move(consts)},
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

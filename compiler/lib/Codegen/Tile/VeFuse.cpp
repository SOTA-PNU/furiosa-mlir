// furiosa-ve-fuse: groups the hardware-abstract Vector Engine ops (fopt.ve) into passes and lowers them. A pass
// fetches one DM tensor (or two, interleaved), runs its ops in the engine's stage order, each on an ALU of its
// own, and commits. An op joins the pass of its input when that input has no other use and the op fits after
// the pass's last op: its stage is not behind (only the float and clip stages take several ops), its ALU is free,
// and a 4-wide float op does not follow the widening. Multiplications take Mul0, Mul1, then the FMA unit; an add
// of an immediate or VRF moves to the clip stage once the FMA unit is taken or passed (the stage rules the
// row-wise class plans by). An op that zips two tensors, tags the stream or branches on the tags starts a pass.

#include "Plan.h"
#include "furiosa/Codegen/Codegen.h"
#include "furiosa/Codegen/Fopt.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <optional>
#include <set>

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;

namespace {

// The Vector Engine's stages, in pass order (as row-wise numbers them).
enum Stage { Tag = 0, Narrow = 4, Fp = 5, FpDiv = 7, Widen = 8, Clip = 10, Last = 11 };

/// An op placed in a pass: its stage, ALU, whether it runs 4-wide, and its call (the ALU's op).
struct Placed {
  VeOp ve;
  int stage;
  std::string alu;
  bool way4;
  std::string method, op;
};

struct PassState {
  int stage = Tag;
  bool way4 = false;
  std::set<std::string> alus;

  void add(const Placed &p) {
    if (p.way4 != way4)
      stage = std::max(stage, int(p.way4 ? Narrow : Widen));
    way4 = p.way4;
    stage = std::max(stage, p.stage);
    alus.insert(p.alu);
  }
};

/// Where `ve` goes in a pass at `st` (a new pass when `st` is null), or nullopt when it cannot join.
std::optional<Placed> place(VeOp ve, const PassState *st) {
  static const PassState fresh;
  bool head = !st;
  if (head)
    st = &fresh;
  else if (ve.getZip() || ve.getTags() || ve.getBranches()) // branches read the tags the pass began with
    return std::nullopt;
  StringRef m = ve.getMethod(), op = ve.getOp();
  Placed p{ve, Last, m.str(), m.starts_with("vector_fp"), m.str(), op.str()};
  if (m == "vector_fp_unary") {
    p.stage = Fp, p.alu = op.ends_with("Exp") ? "FpExp" : "FpFpu";
  } else if (m == "vector_fp_binary" || m == "vector_fp_zip") {
    p.stage = Fp;
    if (op.contains("MulF") && m == "vector_fp_binary") {
      p.alu.clear();
      for (auto [alu, name] : {std::pair{"FpMul0", "Mul0"}, {"FpMul1", "Mul1"}, {"FpFma", "Fma"}})
        if (!st->alus.count(alu)) {
          p.alu = alu, p.op = std::string("FpBinaryOp::MulF(FpMulAlu::") + name + ")";
          break;
        }
      if (p.alu.empty())
        return std::nullopt;
    } else {
      p.alu = op.contains("MulF") ? "FpMul0" : op.contains("DivF") ? "FpFpu" : "FpFma";
    }
    if (m == "vector_fp_binary" && op.contains("AddF") && !ve.getMode() && !ve.getBranches() &&
        (st->stage > Fp || st->alus.count("FpFma")))
      p = {ve, Clip, "ClipAdd", false, "vector_clip", "ClipBinaryOpF32::Add"};
  } else if (m == "vector_fp_div") {
    p.stage = FpDiv, p.alu = "ReduceFpDiv";
  } else if (m == "vector_clip" || m == "vector_clip_zip") {
    p.stage = Clip;
    p.alu = op.ends_with("Max") ? "ClipMax" : op.ends_with("Min") ? "ClipMin" : "ClipAdd"; // zipped or not
  }
  if (head)
    return p;
  bool sameStageAllowed = p.stage == Fp || p.stage == Clip;
  if (p.stage == Last || p.stage < st->stage || (p.stage == st->stage && !sameStageAllowed) ||
      st->alus.count(p.alu) || (p.way4 && st->stage > Narrow && !st->way4))
    return std::nullopt;
  return p;
}

MapAttr pairOf(MLIRContext *ctx, ArrayRef<MapAttr> parts) { return MapAttr::pair(ctx, parts); }

/// The time digits and packet of a DM element: its last digit split into lanes.
std::pair<SmallVector<MapAttr>, MapAttr> timePacket(MapAttr element) {
  SmallVector<MapAttr> parts;
  if (element.getKind() == MapKind::Pair)
    for (Attribute a : element.getOperands())
      parts.push_back(cast<MapAttr>(a));
  else
    parts.push_back(element);
  MapAttr c = parts.pop_back_val();
  if (c.size() > tile::kLanes) {
    parts.push_back(c.stride(tile::kLanes));
    c = c.modulo(tile::kLanes);
  }
  return {parts, c};
}

/// Emits one pass computing `ops` (a chain, each op reading the previous one's result) before the last op, and
/// replaces the chain with its commit.
void lowerPass(ArrayRef<Placed> ops) {
  VeOp head = ops.front().ve, last = ops.back().ve;
  MLIRContext *ctx = head.getContext();
  OpBuilder b(last);
  Location loc = head.getLoc();
  auto t = cast<DmTensorType>(head.getInput().getType());
  Type f32 = Float32Type::get(ctx);
  auto [timeParts, packet] = timePacket(t.getElement());
  MapAttr time = pairOf(ctx, timeParts);
  SmallVector<MapAttr> t4(timeParts);
  t4.push_back(packet.stride(4));
  MapAttr time4 = pairOf(ctx, t4), packet4 = packet.modulo(4);
  auto st = [&](MapAttr tm, MapAttr pk) {
    return StreamType::get(ctx, f32, t.getChip(), t.getCluster(), t.getSlice(), tm, pk);
  };
  Value s;
  if (Value rhs = head.getZip()) {
    MapAttr axis = head.getAxisAttr(), ti = pairOf(ctx, {time, axis});
    s = BeginInterleavedOp::create(b, loc, st(axis, t.getElement()), head.getInput(), rhs, ContextKind::Main, axis);
    s = FetchOp::create(b, loc, st(ti, packet), s);
    s = CollectOp::create(b, loc, st(ti, packet), s);
    s = VectorInitOp::create(b, loc, st(ti, packet), s);
    s = VectorIntraSliceUnzipOp::create(b, loc, st(time, packet), s, axis);
  } else {
    s = BeginOp::create(b, loc, st(MapAttr::broadcast(ctx, 1), t.getElement()), head.getInput(), ContextKind::Main);
    s = FetchOp::create(b, loc, st(time, packet), s);
    s = CollectOp::create(b, loc, st(time, packet), s);
    s = VectorInitOp::create(b, loc, st(time, packet), s);
    if (ArrayAttr tags = head.getTagsAttr())
      s = VectorTagCmpOp::create(b, loc, st(time, packet), s, tags);
    else
      s = VectorIntraSliceTagOp::create(b, loc, st(time, packet), s);
  }
  // Float ops run 4-wide: the packet's halves split into time, and join back before a clip op and the commit.
  bool way4 = false;
  auto widen = [&]() {
    if (way4)
      s = VectorWidenConcatOp::create(b, loc, st(time, packet), s);
    way4 = false;
  };
  for (const Placed &p : ops) {
    if (p.way4 && !way4) {
      s = VectorNarrowSplitOp::create(b, loc, st(time4, packet4), s);
      way4 = true;
    } else if (!p.way4) {
      widen();
    }
    VeOp ve = p.ve;
    s = VectorAluOp::create(b, loc, way4 ? st(time4, packet4) : st(time, packet), s, b.getStringAttr(p.method),
                            b.getStringAttr(p.op), ve.getModeAttr(), ve.getImmAttr(), ve.getVrf(), UnitAttr(),
                            ve.getBranchesAttr());
  }
  widen();
  s = VectorFinalOp::create(b, loc, st(time, packet), s);
  s = CommitTrimOp::create(b, loc, st(time, packet), s);
  last.getResult().replaceAllUsesWith(CommitOp::create(b, loc, t, s).getResult());
  for (const Placed &p : llvm::reverse(ops))
    p.ve->erase();
}

/// Groups the fopt.ve ops of `block` (and of the blocks nested in it) into passes and lowers each.
void fuseBlock(Block &block) {
  SmallVector<SmallVector<Placed>> passes;
  DenseMap<Operation *, std::pair<size_t, PassState>> open; // a pass's last op -> (the pass, its state)
  for (Operation &o : block) {
    for (Region &r : o.getRegions())
      for (Block &inner : r)
        fuseBlock(inner);
    auto ve = dyn_cast<VeOp>(&o);
    if (!ve)
      continue;
    Operation *prev = ve.getInput().getDefiningOp();
    if (auto it = open.find(prev); it != open.end() && prev->hasOneUse()) {
      auto [index, state] = it->second;
      open.erase(it);
      if (auto p = place(ve, &state)) {
        state.add(*p);
        passes[index].push_back(*p);
        open[ve] = {index, state};
        continue;
      }
    }
    Placed p = *place(ve, nullptr);
    PassState state;
    state.add(p);
    passes.push_back({p});
    open[ve] = {passes.size() - 1, state};
  }
  for (auto &ops : passes)
    lowerPass(ops);
}

struct VeFusePass : PassWrapper<VeFusePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VeFusePass)

  StringRef getArgument() const override { return "furiosa-ve-fuse"; }
  StringRef getDescription() const override {
    return "Group the abstract Vector Engine ops (fopt.ve) into passes in the engine's stage order and lower them";
  }
  void getDependentDialects(DialectRegistry &registry) const override { registerCodegenDialects(registry); }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>())
      for (Block &block : func.getBody())
        fuseBlock(block);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::furiosa::createVeFusePass() { return std::make_unique<VeFusePass>(); }

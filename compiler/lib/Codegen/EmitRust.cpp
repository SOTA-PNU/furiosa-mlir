// fopt -> Rust: an EmitC-style transcription. Each kernel becomes one Rust module body: axes!, a type alias
// per parameter and result, and the #[device] function whose statements are the fopt ops in order. A
// Tensor Unit pass (begin .. commit) prints as one builder chain under a typed let.

#include "furiosa/Codegen/Codegen.h"
#include "furiosa/Codegen/Fopt.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::furiosa;
using namespace mlir::furiosa::fopt;

namespace {

std::string rustScalar(Type type) {
  if (type.isF32())
    return "f32";
  if (type.isBF16())
    return "bf16";
  if (type.isInteger(32))
    return "i32";
  if (type.isInteger(16))
    return "i16";
  if (type.isInteger(8))
    return "i8";
  if (isa<Float8E4M3FNType>(type))
    return "f8e4m3";
  if (isa<Float8E5M2Type>(type))
    return "f8e5m2";
  return "";
}

std::string m(MapAttr map) { return "m![" + map.rust() + "]"; }

std::string rustType(Type type) {
  return llvm::TypeSwitch<Type, std::string>(type)
      .Case([](HbmTensorType t) {
        return "HbmTensor<" + rustScalar(t.getElementType()) + ", " + m(t.getChip()) + ", " + m(t.getElement()) + ">";
      })
      .Case([](DmTensorType t) {
        return "DmTensor<" + rustScalar(t.getElementType()) + ", " + m(t.getChip()) + ", " + m(t.getCluster()) + ", " +
               m(t.getSlice()) + ", " + m(t.getElement()) + ">";
      })
      .Case([](TrfTensorType t) {
        return "TrfTensor<" + rustScalar(t.getElementType()) + ", " + m(t.getChip()) + ", " + m(t.getCluster()) +
               ", " + m(t.getSlice()) + ", " + m(t.getLane()) + ", " + m(t.getElement()) + ">";
      })
      .Case([](VrfTensorType t) {
        return "VrfTensor<" + rustScalar(t.getElementType()) + ", " + m(t.getChip()) + ", " + m(t.getCluster()) +
               ", " + m(t.getSlice()) + ", " + m(t.getElement()) + ">";
      })
      .Default([](Type) { return std::string(); });
}

std::string rustF32(float value) {
  if (std::isnan(value))
    return "f32::NAN";
  if (std::isinf(value))
    return value > 0 ? "f32::INFINITY" : "f32::NEG_INFINITY";
  llvm::SmallString<32> text;
  llvm::APFloat(value).toString(text, /*FormatPrecision=*/0, /*FormatMaxPadding=*/0);
  std::string s(text);
  if (s.find_first_of(".eE") == std::string::npos)
    s += ".0";
  return s + "f32";
}

bool isKeyword(StringRef name) {
  static const char *keywords[] = {"as",   "break", "const", "continue", "crate", "else",   "enum",  "extern",
                                   "false", "fn",   "for",   "if",       "impl",  "in",     "let",   "loop",
                                   "match", "mod",  "move",  "mut",      "pub",   "ref",    "return", "self",
                                   "Self",  "static", "struct", "super", "trait", "true",   "type",  "unsafe",
                                   "use",   "where", "while", "async",   "await", "dyn",    "device"};
  return llvm::is_contained(keywords, name);
}

class Emitter {
public:
  Emitter(llvm::raw_ostream &os) : os(os) {}

  LogicalResult emit(func::FuncOp kernel);

private:
  llvm::raw_ostream &os;
  llvm::DenseMap<Value, std::string> names;
  llvm::StringSet<> taken;

  std::string fresh(StringRef base) {
    std::string name;
    for (char c : base)
      name += llvm::isAlnum(c) || c == '_' ? c : '_';
    if (name.empty() || llvm::isDigit(name.front()))
      name = "v_" + name;
    if (isKeyword(name))
      name += "_";
    std::string candidate = name;
    for (int i = 1; taken.contains(candidate); ++i)
      candidate = name + "_" + std::to_string(i);
    taken.insert(candidate);
    return candidate;
  }

  std::string nameOf(Value v) { return names.lookup(v); }

  /// An ALU op's second operand: an immediate, `&vrf`, or `Stash`.
  template <typename Op> std::string operand(Op o) {
    if (o.getStash())
      return "Stash";
    if (Value v = o.getVrf())
      return "&" + nameOf(v);
    return rustF32(o.getImm()->convertToFloat());
  }

  /// The builder chain that ends in `stream`: device.<ctx>.begin(..).<stage>()...
  FailureOr<std::string> chain(Value stream);
};

FailureOr<std::string> Emitter::chain(Value stream) {
  SmallVector<Operation *> stages;
  Value v = stream;
  while (Operation *op = v.getDefiningOp()) {
    stages.push_back(op);
    if (isa<BeginOp, BeginInterleavedOp>(op))
      break;
    // A chain link reads its stream as the first operand; any other operand (a TRF, a VRF) is a named value.
    if (op->getNumOperands() < 1 || !isa<StreamType>(op->getOperand(0).getType()) ||
        llvm::any_of(op->getOperands().drop_front(), [](Value v) { return isa<StreamType>(v.getType()); }))
      return op->emitOpError("cannot be printed as a builder chain link");
    v = op->getOperand(0);
  }
  if (stages.empty() || !isa<BeginOp, BeginInterleavedOp>(stages.back()))
    return failure();

  std::string s;
  llvm::raw_string_ostream out(s);
  for (Operation *op : llvm::reverse(stages)) {
    auto st = op->getNumResults() ? dyn_cast<StreamType>(op->getResult(0).getType()) : StreamType();
    auto tp = [&]() { return m(st.getTime()) + ", " + m(st.getPacket()); };
    LogicalResult ok =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](BeginOp b) {
              out << "device." << stringifyContextKind(b.getCtx()) << ".begin(" << nameOf(b.getSource()) << ".view())";
              return success();
            })
            .Case([&](BeginInterleavedOp b) {
              out << "device." << stringifyContextKind(b.getCtx()) << ".begin_interleaved::<" << b.getAxis().rust()
                  << ", _, _, _, _, _>(" << nameOf(b.getLhs()) << ".view(), " << nameOf(b.getRhs()) << ".view())";
              return success();
            })
            .Case([&](FetchOp) {
              out << "\n.fetch::<" << tp() << ">()";
              return success();
            })
            .Case([&](FetchCastOp) {
              out << "\n.fetch_cast::<" << rustScalar(st.getElementType()) << ">()";
              return success();
            })
            .Case([&](CollectOp) {
              out << "\n.collect::<" << tp() << ">()";
              return success();
            })
            .Case([&](VectorInitOp) {
              out << "\n.vector_init()";
              return success();
            })
            .Case([&](VectorIntraSliceTagOp) {
              out << "\n.vector_intra_slice_tag(TagMode::Zero)";
              return success();
            })
            .Case([&](VectorIntraSliceUnzipOp u) {
              out << "\n.vector_intra_slice_unzip::<" << u.getAxis().rust() << ", " << m(st.getTime()) << ">()";
              return success();
            })
            .Case([&](VectorNarrowSplitOp) {
              out << "\n.vector_narrow_split::<" << tp() << ">()";
              return success();
            })
            .Case([&](VectorWidenConcatOp) {
              out << "\n.vector_widen_concat::<" << tp() << ">()";
              return success();
            })
            .Case([&](VectorNarrowTrimOp) {
              out << "\n.vector_narrow_trim::<" << m(st.getPacket()) << ">()";
              return success();
            })
            .Case([&](VectorWidenPadOp) {
              out << "\n.vector_widen_pad::<" << m(st.getPacket()) << ">()";
              return success();
            })
            .Case([&](VectorClipZipOp o) {
              out << "\n.vector_clip_zip(ClipBinaryOpF32::" << o.getOp() << ")";
              return success();
            })
            .Case([&](VectorFpZipOp o) {
              out << "\n.vector_fp_zip(FpBinaryOp::" << o.getOp() << ")";
              return success();
            })
            .Case([&](VectorFpUnaryOp o) {
              out << "\n.vector_fp_unary(FpUnaryOp::" << o.getOp() << ")";
              return success();
            })
            .Case([&](VectorClipOp o) {
              out << "\n.vector_clip(ClipBinaryOpF32::" << o.getOp() << ", " << operand(o) << ")";
              return success();
            })
            .Case([&](VectorFpBinaryOp o) {
              out << "\n.vector_fp_binary(FpBinaryOp::" << o.getOp() << ", " << operand(o) << ")";
              return success();
            })
            .Case([&](VectorFpDivOp o) {
              out << "\n.vector_fp_div(" << operand(o) << ")";
              return success();
            })
            .Case([&](VectorStashOp) {
              out << "\n.vector_stash()";
              return success();
            })
            .Case([&](VectorIntraSliceReduceOp o) {
              out << "\n.vector_intra_slice_reduce::<" << o.getAxis() << ", " << tp() << ">(IntraSliceReduceOpF32::"
                  << o.getOp() << ")";
              return success();
            })
            .Case([&](VectorFinalOp) {
              out << "\n.vector_final()";
              return success();
            })
            .Case([&](CastOp) {
              out << "\n.cast::<" << rustScalar(st.getElementType()) << ", " << m(st.getPacket()) << ">()";
              return success();
            })
            .Case([&](CommitTrimOp) {
              out << "\n.commit_trim::<" << m(st.getPacket()) << ">()";
              return success();
            })
            .Case([&](ContractOuterOp o) {
              out << "\n.contract_outer::<" << tp() << ", _, _, _>(&" << nameOf(o.getTrf()) << ")";
              return success();
            })
            .Case([&](ContractPacketOp) {
              out << "\n.contract_packet::<" << m(st.getPacket()) << ">()";
              return success();
            })
            .Case([&](ContractTimeOp) {
              out << "\n.contract_time::<" << m(st.getTime()) << ">()";
              return success();
            })
            .Case([&](ContractLaneOp o) {
              out << "\n.contract_lane::<" << tp() << ">(LaneMode::"
                  << (o.getMode() == LaneMode::Interleaved ? "Interleaved" : "Sequential") << ")";
              return success();
            })
            .Default([&](Operation *op) -> LogicalResult {
              return op->emitOpError("has no Rust form in a builder chain");
            });
    if (failed(ok))
      return failure();
  }
  return s;
}

LogicalResult Emitter::emit(func::FuncOp kernel) {
  auto chip = kernel->getAttrOfType<IntegerAttr>(kChipAttr);
  auto pe = kernel->getAttrOfType<IntegerAttr>(kPeAttr);
  auto axes = kernel->getAttrOfType<ArrayAttr>(kAxesAttr);
  if (!chip || !axes)
    return kernel.emitError("a kernel needs fopt.chip and fopt.axes");

  os << "//! Generated by furiosa-compile from the raised kernel `" << kernel.getName() << "`; do not edit.\n\n";
  os << "use furiosa_opt_std::prelude::*;\n\n";
  os << "axes![";
  llvm::interleave(
      axes, os,
      [&](Attribute decl) {
        auto pair = cast<ArrayAttr>(decl);
        os << cast<StringAttr>(pair[0]).getValue() << " = " << cast<IntegerAttr>(pair[1]).getInt();
      },
      ", ");
  os << "];\n\n";

  FunctionType type = kernel.getFunctionType();
  for (auto [i, t] : llvm::enumerate(type.getInputs())) {
    auto arg = kernel.getArgAttrOfType<IntegerAttr>(i, "fopt.arg");
    os << "/// Parameter " << i << (arg ? ": pointer argument " + std::to_string(arg.getInt()) : std::string()) << ".\n";
    os << "pub type Arg" << i << " = " << rustType(t) << ";\n";
  }
  for (auto [i, t] : llvm::enumerate(type.getResults())) {
    auto arg = kernel.getResultAttrOfType<IntegerAttr>(i, "fopt.arg");
    os << "/// Result " << i << (arg ? ": pointer argument " + std::to_string(arg.getInt()) : std::string()) << ".\n";
    os << "pub type Out" << i << " = " << rustType(t) << ";\n";
  }
  os << "\n";

  taken.insert("device");
  os << "#[device(chip = " << chip.getInt();
  if (pe && pe.getInt() != 8)
    os << ", pe = " << pe.getInt();
  os << ")]\n";
  os << "pub fn " << kernel.getName() << "(device: &mut Device";
  for (auto [i, arg] : llvm::enumerate(kernel.getArguments())) {
    auto name = kernel.getArgAttrOfType<StringAttr>(i, kNameAttr);
    names[arg] = fresh(name ? name.getValue() : "arg" + std::to_string(i));
    os << ", " << names[arg] << ": &Arg" << i;
  }
  os << ")";
  if (type.getNumResults() == 1)
    os << " -> Out0";
  else if (type.getNumResults() > 1) {
    os << " -> (";
    for (unsigned i = 0; i < type.getNumResults(); ++i)
      os << (i ? ", " : "") << "Out" << i;
    os << ")";
  }
  os << " {\n";

  auto returnOp = cast<func::ReturnOp>(kernel.getBody().front().getTerminator());
  // A to_hbm whose only use is the return prints in place, as the function's tail expression.
  auto isTail = [&](Operation *op) {
    return isa<ToHbmOp>(op) && op->getResult(0).hasOneUse() && *op->getResult(0).getUsers().begin() == returnOp;
  };
  llvm::DenseMap<Value, std::string> tails;
  for (Operation &op : kernel.getBody().front()) {
    if (isa<func::ReturnOp>(op))
      break;
    LogicalResult ok =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case([&](ToDmOp o) {
              names[o] = fresh(nameOf(o.getSource()) + "_dm");
              os << "    let " << names[o] << ": " << rustType(o.getType()) << " = " << nameOf(o.getSource())
                 << ".to_dm(&mut device.tdma);\n";
              return success();
            })
            .Case([&](ToHbmOp o) {
              std::string expr = nameOf(o.getSource()) + ".to_hbm(&mut device.tdma)";
              if (isTail(o)) {
                tails[o] = expr;
              } else {
                names[o] = fresh(nameOf(o.getSource()) + "_hbm");
                os << "    let " << names[o] << ": " << rustType(o.getType()) << " = " << expr << ";\n";
              }
              return success();
            })
            .Case([&](ToVrfOp o) -> LogicalResult {
              FailureOr<std::string> c = chain(o.getInput());
              if (failed(c))
                return o.emitOpError("does not end a builder chain from a begin");
              names[o] = fresh("vrf" + std::to_string(names.size()));
              os << "    let " << names[o] << ": " << rustType(o.getType()) << " = " << *c << "\n.to_vrf();\n";
              return success();
            })
            .Case([&](ToTrfOp o) -> LogicalResult {
              FailureOr<std::string> c = chain(o.getInput());
              if (failed(c))
                return o.emitOpError("does not end a builder chain from a begin");
              names[o] = fresh("trf" + std::to_string(names.size()));
              os << "    let " << names[o] << ": " << rustType(o.getType()) << " = " << *c << "\n.to_trf();\n";
              return success();
            })
            .Case([&](CommitOp o) -> LogicalResult {
              FailureOr<std::string> c = chain(o.getInput());
              if (failed(c))
                return o.emitOpError("does not end a builder chain from a begin");
              names[o] = fresh("t" + std::to_string(names.size()));
              os << "    let " << names[o] << ": " << rustType(o.getType()) << " = " << *c << "\n.commit();\n";
              return success();
            })
            .Default([&](Operation *op) -> LogicalResult {
              if (op->getNumResults() == 1 && isa<StreamType>(op->getResult(0).getType()))
                return success(); // printed as part of its chain
              return op->emitOpError("has no Rust statement form");
            });
    if (failed(ok))
      return failure();
  }
  auto value = [&](Value v) { return tails.count(v) ? tails[v] : nameOf(v); };
  if (returnOp.getNumOperands() == 1) {
    os << "    " << value(returnOp.getOperand(0)) << "\n";
  } else if (returnOp.getNumOperands() > 1) {
    os << "    (";
    llvm::interleave(returnOp.getOperands(), os, [&](Value v) { os << value(v); }, ", ");
    os << ")\n";
  }
  os << "}\n";
  return success();
}

} // namespace

LogicalResult mlir::furiosa::translateFoptToRust(ModuleOp module, llvm::raw_ostream &os) {
  SmallVector<func::FuncOp> kernels;
  for (auto func : module.getOps<func::FuncOp>())
    if (func->hasAttr(kChipAttr))
      kernels.push_back(func);
  if (kernels.size() != 1)
    return module.emitError("expected one fopt kernel per module (its axes! is module-wide), found ")
           << kernels.size();
  Emitter emitter(os);
  return emitter.emit(kernels.front());
}

FailureOr<std::string> mlir::furiosa::foptSignature(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  for (auto func : module.getOps<func::FuncOp>())
    if (func->hasAttr(kChipAttr))
      kernels.push_back(func);
  if (kernels.size() != 1)
    return module.emitError("expected one fopt kernel per module, found ") << kernels.size();
  func::FuncOp kernel = kernels.front();
  llvm::json::Array inputs, outputs;
  for (unsigned i = 0; i < kernel.getNumArguments(); ++i) {
    auto arg = kernel.getArgAttrOfType<IntegerAttr>(i, "fopt.arg");
    if (!arg)
      return kernel.emitError("parameter ") << i << " has no fopt.arg";
    inputs.push_back(arg.getInt());
  }
  for (unsigned i = 0; i < kernel.getNumResults(); ++i) {
    auto arg = kernel.getResultAttrOfType<IntegerAttr>(i, "fopt.arg");
    if (!arg)
      return kernel.emitError("result ") << i << " has no fopt.arg";
    outputs.push_back(arg.getInt());
  }
  auto pe = kernel->getAttrOfType<IntegerAttr>(kPeAttr);
  llvm::json::Object sig{{"kernel", kernel.getName().str()},
                         {"chip", kernel->getAttrOfType<IntegerAttr>(kChipAttr).getInt()},
                         {"pe", pe ? pe.getInt() : 8},
                         {"inputs", std::move(inputs)},
                         {"outputs", std::move(outputs)}};
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::json::Value(std::move(sig));
  return text;
}

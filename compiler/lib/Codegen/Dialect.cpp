// The fopt dialect: mapping attributes, types and op verifiers.

#include "furiosa/Codegen/Fopt.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::furiosa::fopt;

#include "furiosa/Codegen/FoptDialect.cpp.inc"
#include "furiosa/Codegen/FoptEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "furiosa/Codegen/FoptAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "furiosa/Codegen/FoptTypes.cpp.inc"

#define GET_OP_CLASSES
#include "furiosa/Codegen/FoptOps.cpp.inc"

void FoptDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "furiosa/Codegen/FoptAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "furiosa/Codegen/FoptTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "furiosa/Codegen/FoptOps.cpp.inc"
      >();
}

int64_t mlir::furiosa::fopt::elementBytes(Type type) {
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isBF16() || type.isF16() || type.isInteger(16))
    return 2;
  if (type.isInteger(8) || isa<Float8E4M3FNType, Float8E5M2Type>(type))
    return 1;
  return 0;
}

//===----------------------------------------------------------------------===//
// MapAttr
//===----------------------------------------------------------------------===//

MapAttr MapAttr::axis(MLIRContext *ctx, StringRef name, int64_t size) {
  return get(ctx, MapKind::Axis, name, size, PadKind::Top, {});
}

MapAttr MapAttr::broadcast(MLIRContext *ctx, int64_t n) { return get(ctx, MapKind::Broadcast, "", n, PadKind::Top, {}); }

MapAttr MapAttr::pair(MLIRContext *ctx, ArrayRef<MapAttr> parts) {
  // Pairs flatten; a pair of one is that one; an empty pair is the identity m![1].
  SmallVector<Attribute> flat;
  for (MapAttr part : parts) {
    if (part.getKind() == MapKind::Pair)
      llvm::append_range(flat, part.getOperands());
    else if (!(part.getKind() == MapKind::Broadcast && part.getValue() == 1))
      flat.push_back(part);
  }
  if (flat.empty())
    return broadcast(ctx, 1);
  if (flat.size() == 1)
    return cast<MapAttr>(flat.front());
  return get(ctx, MapKind::Pair, "", 0, PadKind::Top, flat);
}

MapAttr MapAttr::stride(int64_t n) const { return get(getContext(), MapKind::Stride, "", n, PadKind::Top, {*this}); }
MapAttr MapAttr::modulo(int64_t n) const { return get(getContext(), MapKind::Modulo, "", n, PadKind::Top, {*this}); }
MapAttr MapAttr::resize(int64_t n) const { return get(getContext(), MapKind::Resize, "", n, PadKind::Top, {*this}); }
MapAttr MapAttr::padTo(int64_t n, PadKind kind) const {
  return get(getContext(), MapKind::Pad, "", n, kind, {*this});
}
MapAttr MapAttr::operand(unsigned i) const { return cast<MapAttr>(getOperands()[i]); }

int64_t MapAttr::size() const {
  switch (getKind()) {
  case MapKind::Axis:
  case MapKind::Broadcast:
  case MapKind::Modulo:
  case MapKind::Resize:
  case MapKind::Pad:
    return getValue();
  case MapKind::Stride:
    return operand(0).size() / getValue();
  case MapKind::Pair: {
    int64_t n = 1;
    for (Attribute part : getOperands())
      n *= cast<MapAttr>(part).size();
    return n;
  }
  }
  llvm_unreachable("mapping kind");
}

static void printMap(MapAttr map, llvm::raw_ostream &os, bool withSizes, bool nested) {
  auto inner = [&](MapAttr x) {
    // Postfix operators bind tighter than `,`: a pair under one needs parentheses.
    bool paren = x.getKind() == MapKind::Pair;
    if (paren)
      os << "(";
    printMap(x, os, withSizes, true);
    if (paren)
      os << ")";
  };
  switch (map.getKind()) {
  case MapKind::Axis:
    os << map.getName();
    if (withSizes)
      os << "{" << map.getValue() << "}";
    return;
  case MapKind::Broadcast:
    os << map.getValue();
    return;
  case MapKind::Stride:
    inner(map.operand(0));
    os << " / " << map.getValue();
    return;
  case MapKind::Modulo:
    inner(map.operand(0));
    os << " % " << map.getValue();
    return;
  case MapKind::Resize:
    inner(map.operand(0));
    os << " = " << map.getValue();
    return;
  case MapKind::Pad:
    inner(map.operand(0));
    os << (map.getPad() == PadKind::Top ? " # " : map.getPad() == PadKind::Zero ? " #{0} " : " #{!} ")
       << map.getValue();
    return;
  case MapKind::Pair:
    llvm::interleave(
        map.getOperands(), os, [&](Attribute part) { printMap(cast<MapAttr>(part), os, withSizes, false); }, ", ");
    return;
  }
}

std::string MapAttr::rust() const {
  std::string s;
  llvm::raw_string_ostream os(s);
  printMap(*this, os, /*withSizes=*/false, false);
  return s;
}

std::string MapAttr::text() const {
  std::string s;
  llvm::raw_string_ostream os(s);
  printMap(*this, os, /*withSizes=*/true, false);
  return s;
}

namespace {
/// Recursive descent over the textual form:
///   expr    := term (',' term)*
///   term    := primary postfix*
///   primary := IDENT '{' INT '}' | INT | '(' expr ')'
///   postfix := '/' INT | '%' INT | '=' INT | '#' ('{' ('0' | '!' | '*') '}')? INT
struct MapParser {
  MLIRContext *ctx;
  StringRef rest;
  std::string &error;

  void skip() { rest = rest.ltrim(); }
  bool eat(char c) {
    skip();
    if (rest.starts_with(StringRef(&c, 1))) {
      rest = rest.drop_front();
      return true;
    }
    return false;
  }
  FailureOr<int64_t> integer() {
    skip();
    size_t n = 0;
    while (n < rest.size() && llvm::isDigit(rest[n]))
      ++n;
    int64_t value;
    if (n == 0 || rest.take_front(n).getAsInteger(10, value)) {
      error = "expected an integer at '" + rest.str() + "'";
      return failure();
    }
    rest = rest.drop_front(n);
    return value;
  }
  FailureOr<MapAttr> primary() {
    skip();
    if (eat('(')) {
      FailureOr<MapAttr> e = expr();
      if (failed(e) || !eat(')')) {
        error = error.empty() ? "expected ')'" : error;
        return failure();
      }
      return e;
    }
    if (!rest.empty() && llvm::isDigit(rest.front())) {
      FailureOr<int64_t> n = integer();
      if (failed(n))
        return failure();
      return MapAttr::broadcast(ctx, *n);
    }
    size_t n = 0;
    while (n < rest.size() && (llvm::isAlnum(rest[n]) || rest[n] == '_'))
      ++n;
    if (n == 0) {
      error = "expected an axis, a literal or '(' at '" + rest.str() + "'";
      return failure();
    }
    StringRef name = rest.take_front(n);
    rest = rest.drop_front(n);
    if (!eat('{')) {
      error = "axis " + name.str() + " needs its size: " + name.str() + "{N}";
      return failure();
    }
    FailureOr<int64_t> size = integer();
    if (failed(size) || !eat('}')) {
      error = error.empty() ? "expected '}'" : error;
      return failure();
    }
    return MapAttr::axis(ctx, name, *size);
  }
  FailureOr<MapAttr> term() {
    FailureOr<MapAttr> x = primary();
    while (succeeded(x)) {
      if (eat('/')) {
        FailureOr<int64_t> n = integer();
        if (failed(n))
          return failure();
        x = x->stride(*n);
      } else if (eat('%')) {
        FailureOr<int64_t> n = integer();
        if (failed(n))
          return failure();
        x = x->modulo(*n);
      } else if (eat('=')) {
        FailureOr<int64_t> n = integer();
        if (failed(n))
          return failure();
        x = x->resize(*n);
      } else if (eat('#')) {
        PadKind kind = PadKind::Top;
        if (eat('{')) {
          if (eat('0'))
            kind = PadKind::Zero;
          else if (eat('!'))
            kind = PadKind::Bottom;
          else if (!eat('*')) {
            error = "expected 0, ! or * in a padding kind";
            return failure();
          }
          if (!eat('}')) {
            error = "expected '}' after a padding kind";
            return failure();
          }
        }
        FailureOr<int64_t> n = integer();
        if (failed(n))
          return failure();
        x = x->padTo(*n, kind);
      } else {
        break;
      }
    }
    return x;
  }
  FailureOr<MapAttr> expr() {
    SmallVector<MapAttr> parts;
    do {
      FailureOr<MapAttr> t = term();
      if (failed(t))
        return failure();
      parts.push_back(*t);
    } while (eat(','));
    return MapAttr::pair(ctx, parts);
  }
};
} // namespace

FailureOr<MapAttr> MapAttr::parseText(MLIRContext *ctx, StringRef text, std::string &error) {
  MapParser p{ctx, text, error};
  FailureOr<MapAttr> map = p.expr();
  p.skip();
  if (succeeded(map) && !p.rest.empty()) {
    error = "unexpected '" + p.rest.str() + "'";
    return failure();
  }
  return map;
}

Attribute MapAttr::parse(AsmParser &parser, Type) {
  std::string text, error;
  if (parser.parseLess() || parser.parseString(&text) || parser.parseGreater())
    return {};
  FailureOr<MapAttr> map = parseText(parser.getContext(), text, error);
  if (failed(map)) {
    parser.emitError(parser.getCurrentLocation(), "bad mapping \"") << text << "\": " << error;
    return {};
  }
  return *map;
}

void MapAttr::print(AsmPrinter &printer) const { printer << "<\"" << text() << "\">"; }

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

static bool validSlice(int64_t n) { return n == 64 || n == 128 || n == 256; }
static bool validCluster(int64_t n) { return n == 1 || n == 2; }

/// The legality furiosa-opt's constraints put on where a DM-resident tensor lives.
static LogicalResult verifyPlacement(Operation *op, MapAttr cluster, MapAttr slice) {
  if (!validCluster(cluster.size()))
    return op->emitOpError("cluster mapping spans ") << cluster.size() << " clusters; must be 1 or 2";
  if (!validSlice(slice.size()))
    return op->emitOpError("slice mapping spans ") << slice.size() << " slices; must be 64, 128 or 256";
  return success();
}

LogicalResult ToDmOp::verify() {
  auto src = getSource().getType();
  auto dst = getResult().getType();
  if (src.getElementType() != dst.getElementType())
    return emitOpError("changes the element type");
  if (src.getChip() != dst.getChip())
    return emitOpError("to_dm keeps the chip mapping");
  return verifyPlacement(*this, dst.getCluster(), dst.getSlice());
}

LogicalResult ToHbmOp::verify() {
  auto src = getSource().getType();
  auto dst = getResult().getType();
  if (src.getElementType() != dst.getElementType())
    return emitOpError("changes the element type");
  if (src.getChip() != dst.getChip())
    return emitOpError("to_hbm keeps the chip mapping");
  return success();
}

/// Every stream value is one link of a builder chain: exactly one use.
static LogicalResult verifyStreamUse(Operation *op) {
  for (Value result : op->getResults())
    if (isa<StreamType>(result.getType()) && !result.hasOneUse())
      return op->emitOpError("a stream result must have exactly one use (it is one link of a builder chain)");
  return success();
}

/// A stage keeps the stream's chip, cluster and slice.
static LogicalResult verifySamePlacement(Operation *op, StreamType in, StreamType out) {
  if (in.getChip() != out.getChip() || in.getCluster() != out.getCluster() || in.getSlice() != out.getSlice())
    return op->emitOpError("a stage keeps the stream's chip, cluster and slice");
  return success();
}

static int64_t packetBytes(StreamType type) { return type.getPacket().size() * elementBytes(type.getElementType()); }

LogicalResult BeginOp::verify() {
  auto src = getSource().getType();
  auto out = getResult().getType();
  if (src.getChip() != out.getChip() || src.getCluster() != out.getCluster() || src.getSlice() != out.getSlice() ||
      src.getElementType() != out.getElementType())
    return emitOpError("begin keeps the tensor's element type, chip, cluster and slice");
  if (failed(verifyPlacement(*this, out.getCluster(), out.getSlice())))
    return failure();
  return verifyStreamUse(*this);
}

LogicalResult BeginInterleavedOp::verify() {
  auto lhs = getLhs().getType();
  auto out = getResult().getType();
  if (lhs != getRhs().getType())
    return emitOpError("interleaved tensors must have identical types");
  if (lhs.getChip() != out.getChip() || lhs.getCluster() != out.getCluster() || lhs.getSlice() != out.getSlice() ||
      lhs.getElementType() != out.getElementType())
    return emitOpError("begin_interleaved keeps the tensors' element type, chip, cluster and slice");
  if (getAxis().getKind() != MapKind::Axis || getAxis().getValue() != 2)
    return emitOpError("the interleave axis must be a size-2 axis");
  return verifyStreamUse(*this);
}

LogicalResult FetchOp::verify() {
  auto in = getInput().getType();
  auto out = getResult().getType();
  if (in.getElementType() != out.getElementType())
    return emitOpError("fetch keeps the element type (use fetch_cast)");
  if (packetBytes(out) % 8 != 0)
    return emitOpError("a fetched packet must be a multiple of 8 bytes, got ") << packetBytes(out);
  if (failed(verifySamePlacement(*this, in, out)))
    return failure();
  return verifyStreamUse(*this);
}

LogicalResult FetchCastOp::verify() {
  auto out = getResult().getType();
  if (!out.getElementType().isF32() && !out.getElementType().isInteger(32))
    return emitOpError("fetch_cast widens to f32 or i32");
  if (failed(verifySamePlacement(*this, getInput().getType(), out)))
    return failure();
  return verifyStreamUse(*this);
}

LogicalResult CollectOp::verify() {
  auto in = getInput().getType();
  auto out = getResult().getType();
  if (in.getElementType() != out.getElementType())
    return emitOpError("collect keeps the element type");
  if (packetBytes(out) != 32)
    return emitOpError("collect's packet is one 32-byte flit, got ") << packetBytes(out) << " bytes";
  if (failed(verifySamePlacement(*this, in, out)))
    return failure();
  return verifyStreamUse(*this);
}

/// The vector stages and the stages after them keep placement; their element type is the stream's.
static LogicalResult verifyVectorStage(Operation *op, StreamType in, StreamType out) {
  if (in.getElementType() != out.getElementType())
    return op->emitOpError("keeps the element type");
  if (failed(verifySamePlacement(op, in, out)))
    return failure();
  return verifyStreamUse(op);
}

static LogicalResult verifyVectorInput(Operation *op, StreamType in) {
  if (!in.getElementType().isF32() && !in.getElementType().isInteger(32))
    return op->emitOpError("the Vector Engine takes f32 or i32 streams");
  return success();
}

LogicalResult VectorInitOp::verify() {
  if (failed(verifyVectorInput(*this, getInput().getType())))
    return failure();
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorFinalOp::verify() {
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

/// Narrowing moves to the 4-wide float region and widening back: the packet halves or doubles.
LogicalResult VectorNarrowSplitOp::verify() {
  if (packetBytes(getResult().getType()) != 16)
    return emitOpError("narrows to a 4-element (16-byte) packet, got ") << packetBytes(getResult().getType());
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorWidenConcatOp::verify() {
  if (packetBytes(getResult().getType()) != 32)
    return emitOpError("widens to an 8-element (32-byte) packet, got ") << packetBytes(getResult().getType());
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorNarrowTrimOp::verify() {
  if (packetBytes(getResult().getType()) != 16)
    return emitOpError("trims to a 4-element (16-byte) packet, got ") << packetBytes(getResult().getType());
  if (getInput().getType().getTime() != getResult().getType().getTime())
    return emitOpError("keeps time");
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorWidenPadOp::verify() {
  if (packetBytes(getResult().getType()) != 32)
    return emitOpError("pads to an 8-element (32-byte) packet, got ") << packetBytes(getResult().getType());
  if (getInput().getType().getTime() != getResult().getType().getTime())
    return emitOpError("keeps time");
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorIntraSliceTagOp::verify() {
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorIntraSliceUnzipOp::verify() {
  if (getAxis().getKind() != MapKind::Axis || getAxis().getValue() != 2)
    return emitOpError("unzips a size-2 interleave axis");
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

static LogicalResult verifyOpName(Operation *op, StringRef name, ArrayRef<StringRef> allowed) {
  if (!llvm::is_contained(allowed, name))
    return op->emitOpError("unknown op '") << name << "'; expected one of " << llvm::join(allowed, ", ");
  return success();
}

static constexpr StringRef kClipOps[] = {"Min", "Max", "AbsMin", "AbsMax", "Add"};
static constexpr StringRef kFpBinaryOps[] = {"AddF", "SubF", "MulF(FpMulAlu::Mul0)", "DivF"};
static constexpr StringRef kFpUnaryOps[] = {"Exp", "NegExp", "Sqrt", "Tanh", "Sigmoid", "Erf", "Log", "Sin", "Cos"};

LogicalResult VectorClipZipOp::verify() {
  if (failed(verifyOpName(*this, getOp(), kClipOps)))
    return failure();
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorClipOp::verify() {
  if (failed(verifyOpName(*this, getOp(), kClipOps)))
    return failure();
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorFpZipOp::verify() {
  if (failed(verifyOpName(*this, getOp(), kFpBinaryOps)))
    return failure();
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorFpBinaryOp::verify() {
  if (failed(verifyOpName(*this, getOp(), kFpBinaryOps)))
    return failure();
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult VectorFpUnaryOp::verify() {
  if (failed(verifyOpName(*this, getOp(), kFpUnaryOps)))
    return failure();
  return verifyVectorStage(*this, getInput().getType(), getResult().getType());
}

LogicalResult CastOp::verify() {
  if (failed(verifySamePlacement(*this, getInput().getType(), getResult().getType())))
    return failure();
  return verifyStreamUse(*this);
}

LogicalResult CommitTrimOp::verify() {
  auto out = getResult().getType();
  int64_t bytes = packetBytes(out);
  if (bytes != 8 && bytes != 16 && bytes != 24 && bytes != 32)
    return emitOpError("a trimmed packet is 8, 16, 24 or 32 bytes, got ") << bytes;
  if (failed(verifySamePlacement(*this, getInput().getType(), out)))
    return failure();
  return verifyStreamUse(*this);
}

LogicalResult CommitOp::verify() {
  auto in = getInput().getType();
  auto out = getResult().getType();
  if (in.getElementType() != out.getElementType() || in.getChip() != out.getChip() ||
      in.getCluster() != out.getCluster() || in.getSlice() != out.getSlice())
    return emitOpError("commit keeps the stream's element type, chip, cluster and slice");
  return success();
}

// The fopt dialect: furiosa-opt's eDSL as MLIR, one op per Rust call, so emitting Rust is transcription.
//
// A kernel is a func.func carrying `fopt.chip` / `fopt.pe` (its #[device] attribute) and `fopt.axes` (its
// axes! declaration). Tensor types mirror furiosa-opt's: HbmTensor, DmTensor, TrfTensor, VrfTensor, and the
// `tu` stream that flows through one Tensor Unit pass. Stage ops consume and produce a stream; a pass is the
// chain from `begin` to `commit` (or `to_trf` / `to_vrf`), and each stream value has exactly one use, as the
// builder chain it becomes.

#ifndef FURIOSA_CODEGEN_FOPT_OPS
#define FURIOSA_CODEGEN_FOPT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Fopt_Dialect : Dialect {
  let name = "fopt";
  let cppNamespace = "::mlir::furiosa::fopt";
  let summary = "furiosa-opt kernels, one op per eDSL call";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Mapping expressions
//===----------------------------------------------------------------------===//

def Fopt_MapKind : I32EnumAttr<"MapKind", "mapping expression node", [
  I32EnumAttrCase<"Axis", 0, "axis">,         // a named axis of `value` positions
  I32EnumAttrCase<"Broadcast", 1, "broadcast">, // a literal `value`: that many copies
  I32EnumAttrCase<"Stride", 2, "stride">,     // x / value
  I32EnumAttrCase<"Modulo", 3, "modulo">,     // x % value
  I32EnumAttrCase<"Resize", 4, "resize">,     // x = value
  I32EnumAttrCase<"Pad", 5, "pad">,           // x # value
  I32EnumAttrCase<"Pair", 6, "pair">          // operands, major first
]> {
  let cppNamespace = "::mlir::furiosa::fopt";
  let genSpecializedAttr = 0;
}

def Fopt_PadKind : I32EnumAttr<"PadKind", "padding kind", [
  I32EnumAttrCase<"Top", 0, "top">,       // `#`: arbitrary values
  I32EnumAttrCase<"Zero", 1, "zero">,     // `#{0}`
  I32EnumAttrCase<"Bottom", 2, "bottom">  // `#{!}`: readable, not writable
]> {
  let cppNamespace = "::mlir::furiosa::fopt";
  let genSpecializedAttr = 0;
}

def Fopt_MapAttr : AttrDef<Fopt_Dialect, "Map"> {
  let mnemonic = "m";
  let summary = "a furiosa-opt mapping expression, m![...], as a tree";
  let description = [{
    Written `#fopt.m<"N{2048} / 8 # 256">`: furiosa-opt's mapping syntax with each axis's size in braces.
    Stored as a tree of nodes (kind, axis name, value, padding kind, operands), so passes compute sizes and
    rewrite mappings structurally; the string is only the textual form.
  }];
  let parameters = (ins
    EnumParameter<Fopt_MapKind>:$kind,
    StringRefParameter<"axis name of an Axis node", "\"\"">:$name,
    "int64_t":$value,
    EnumParameter<Fopt_PadKind>:$pad,
    ArrayRefParameter<"::mlir::Attribute", "operand nodes">:$operands
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    static MapAttr axis(::mlir::MLIRContext *ctx, ::llvm::StringRef name, int64_t size);
    static MapAttr broadcast(::mlir::MLIRContext *ctx, int64_t n);
    static MapAttr pair(::mlir::MLIRContext *ctx, ::llvm::ArrayRef<MapAttr> parts);
    MapAttr stride(int64_t n) const;
    MapAttr modulo(int64_t n) const;
    MapAttr resize(int64_t n) const;
    MapAttr padTo(int64_t n, PadKind kind = PadKind::Top) const;
    MapAttr operand(unsigned i) const;
    /// Positions the mapping spans, padding included.
    int64_t size() const;
    /// The expression in furiosa-opt's syntax (no axis sizes), as written inside m![...].
    std::string rust() const;
    /// The expression with axis sizes, as written inside #fopt.m<"...">.
    std::string text() const;
    /// Parses the textual form.
    static ::mlir::FailureOr<MapAttr> parseText(::mlir::MLIRContext *ctx, ::llvm::StringRef text, std::string &error);
  }];
}

def Fopt_ContextKind : I32EnumAttr<"ContextKind", "Tensor Unit context", [
  I32EnumAttrCase<"Main", 0, "main">,
  I32EnumAttrCase<"Sub", 1, "sub">
]> {
  let cppNamespace = "::mlir::furiosa::fopt";
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class Fopt_Type<string name, string typeMnemonic> : TypeDef<Fopt_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Fopt_HbmTensor : Fopt_Type<"HbmTensor", "hbm"> {
  let summary = "HbmTensor<D, Chip, Element>";
  let parameters = (ins "::mlir::Type":$elementType, "MapAttr":$chip, "MapAttr":$element);
  let assemblyFormat = "`<` $elementType `,` $chip `,` $element `>`";
}

def Fopt_DmTensor : Fopt_Type<"DmTensor", "dm"> {
  let summary = "DmTensor<D, Chip, Cluster, Slice, Element>";
  let parameters = (ins "::mlir::Type":$elementType, "MapAttr":$chip, "MapAttr":$cluster, "MapAttr":$slice,
                        "MapAttr":$element);
  let assemblyFormat = "`<` $elementType `,` $chip `,` $cluster `,` $slice `,` $element `>`";
}

def Fopt_TrfTensor : Fopt_Type<"TrfTensor", "trf"> {
  let summary = "TrfTensor<D, Chip, Cluster, Slice, Lane, Element>";
  let parameters = (ins "::mlir::Type":$elementType, "MapAttr":$chip, "MapAttr":$cluster, "MapAttr":$slice,
                        "MapAttr":$lane, "MapAttr":$element);
  let assemblyFormat = "`<` $elementType `,` $chip `,` $cluster `,` $slice `,` $lane `,` $element `>`";
}

def Fopt_VrfTensor : Fopt_Type<"VrfTensor", "vrf"> {
  let summary = "VrfTensor<D, Chip, Cluster, Slice, Element>";
  let parameters = (ins "::mlir::Type":$elementType, "MapAttr":$chip, "MapAttr":$cluster, "MapAttr":$slice,
                        "MapAttr":$element);
  let assemblyFormat = "`<` $elementType `,` $chip `,` $cluster `,` $slice `,` $element `>`";
}

def Fopt_Stream : Fopt_Type<"Stream", "tu"> {
  let summary = "a Tensor Unit stream: TuTensor<.., D, Chip, Cluster, Slice, Time, Packet>";
  let parameters = (ins "::mlir::Type":$elementType, "MapAttr":$chip, "MapAttr":$cluster, "MapAttr":$slice,
                        "MapAttr":$time, "MapAttr":$packet);
  let assemblyFormat = "`<` $elementType `,` $chip `,` $cluster `,` $slice `,` $time `,` $packet `>`";
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

class Fopt_Op<string mnemonic, list<Trait> traits = []> : Op<Fopt_Dialect, mnemonic, traits>;

// DMA

def Fopt_ToDmOp : Fopt_Op<"to_dm", [Pure]> {
  let summary = "hbm.to_dm::<Cluster, Slice, Element>(&mut device.tdma)";
  let arguments = (ins Fopt_HbmTensor:$source);
  let results = (outs Fopt_DmTensor:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Fopt_ToHbmOp : Fopt_Op<"to_hbm", [Pure]> {
  let summary = "dm.to_hbm::<Element>(&mut device.tdma)";
  let arguments = (ins Fopt_DmTensor:$source);
  let results = (outs Fopt_HbmTensor:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

// Tensor Unit passes. Each stage's result type carries the type arguments of its Rust call.

class Fopt_StageOp<string mnemonic, list<Trait> traits = []> : Fopt_Op<mnemonic, !listconcat([Pure], traits)> {
  let results = (outs Fopt_Stream:$result);
  let hasVerifier = 1;
}

def Fopt_BeginOp : Fopt_StageOp<"begin"> {
  let summary = "device.{main|sub}.begin(dm.view())";
  let arguments = (ins Fopt_DmTensor:$source, Fopt_ContextKind:$ctx);
  let assemblyFormat = "$ctx $source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
}

def Fopt_BeginInterleavedOp : Fopt_StageOp<"begin_interleaved"> {
  let summary = "device.{main|sub}.begin_interleaved::<I, ..>(a.view(), b.view())";
  let arguments = (ins Fopt_DmTensor:$lhs, Fopt_DmTensor:$rhs, Fopt_ContextKind:$ctx, Fopt_MapAttr:$axis);
  let assemblyFormat = [{
    $ctx $lhs `,` $rhs `axis` $axis attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs)) `->`
    qualified(type($result))
  }];
}

class Fopt_UnaryStageOp<string mnemonic, string summaryText> : Fopt_StageOp<mnemonic> {
  let summary = summaryText;
  let arguments = (ins Fopt_Stream:$input);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
}

def Fopt_FetchOp : Fopt_UnaryStageOp<"fetch", ".fetch::<Time, Packet>()">;
def Fopt_FetchCastOp : Fopt_UnaryStageOp<"fetch_cast", ".fetch_cast::<D>()">;
def Fopt_CollectOp : Fopt_UnaryStageOp<"collect", ".collect::<Time, Packet>()">;
def Fopt_VectorInitOp : Fopt_UnaryStageOp<"vector_init", ".vector_init()">;
def Fopt_VectorFinalOp : Fopt_UnaryStageOp<"vector_final", ".vector_final()">;
def Fopt_VectorNarrowSplitOp : Fopt_UnaryStageOp<"vector_narrow_split", ".vector_narrow_split::<Time, Packet>()">;
def Fopt_VectorWidenConcatOp : Fopt_UnaryStageOp<"vector_widen_concat", ".vector_widen_concat::<Time, Packet>()">;
def Fopt_VectorNarrowTrimOp : Fopt_UnaryStageOp<"vector_narrow_trim", ".vector_narrow_trim::<Packet>()">;
def Fopt_VectorWidenPadOp : Fopt_UnaryStageOp<"vector_widen_pad", ".vector_widen_pad::<Packet>()">;
def Fopt_CastOp : Fopt_UnaryStageOp<"cast", ".cast::<D, Packet>()">;
def Fopt_CommitTrimOp : Fopt_UnaryStageOp<"commit_trim", ".commit_trim::<Packet>()">;

def Fopt_VectorIntraSliceTagOp : Fopt_StageOp<"vector_intra_slice_tag"> {
  let summary = ".vector_intra_slice_tag(TagMode::Zero)";
  let arguments = (ins Fopt_Stream:$input);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
}

def Fopt_VectorIntraSliceUnzipOp : Fopt_StageOp<"vector_intra_slice_unzip"> {
  let summary = ".vector_intra_slice_unzip::<I, Time>()";
  let arguments = (ins Fopt_Stream:$input, Fopt_MapAttr:$axis);
  let assemblyFormat = "$input `axis` $axis attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
}

// Vector Engine ALU ops. `op` names the furiosa-opt enum case (e.g. "Add" in ClipBinaryOpF32::Add);
// a binary op with an immediate carries it in `imm`.

class Fopt_VectorOp<string mnemonic, string summaryText, dag extra> : Fopt_StageOp<mnemonic> {
  let summary = summaryText;
  let arguments = !con((ins Fopt_Stream:$input, StrAttr:$op), extra);
  let assemblyFormat = "$op $input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
}

def Fopt_VectorClipZipOp : Fopt_VectorOp<"vector_clip_zip", ".vector_clip_zip(ClipBinaryOpF32::<op>)", (ins)>;
def Fopt_VectorFpZipOp : Fopt_VectorOp<"vector_fp_zip", ".vector_fp_zip(FpBinaryOp::<op>)", (ins)>;
def Fopt_VectorFpUnaryOp : Fopt_VectorOp<"vector_fp_unary", ".vector_fp_unary(FpUnaryOp::<op>)", (ins)>;
// An ALU op whose second operand is exactly one of: an immediate (`imm`), a VRF tensor (`vrf`), or the stashed
// stream (`stash`).
class Fopt_VectorOperandOp<string mnemonic, string summaryText, dag opArgs> : Fopt_StageOp<mnemonic> {
  let summary = summaryText;
  let arguments = !con((ins Fopt_Stream:$input), opArgs,
                       (ins OptionalAttr<F32Attr>:$imm, Optional<Fopt_VrfTensor>:$vrf, UnitAttr:$stash));
}

def Fopt_VectorClipOp : Fopt_VectorOperandOp<"vector_clip", ".vector_clip(ClipBinaryOpF32::<op>, operand)",
                                             (ins StrAttr:$op)> {
  let assemblyFormat = [{
    $op $input (`vrf` $vrf^ `:` qualified(type($vrf)))? attr-dict `:` qualified(type($input)) `->`
    qualified(type($result))
  }];
}
def Fopt_VectorFpBinaryOp : Fopt_VectorOperandOp<"vector_fp_binary", ".vector_fp_binary(FpBinaryOp::<op>, operand)",
                                                 (ins StrAttr:$op)> {
  let assemblyFormat = [{
    $op $input (`vrf` $vrf^ `:` qualified(type($vrf)))? attr-dict `:` qualified(type($input)) `->`
    qualified(type($result))
  }];
}
def Fopt_VectorFpDivOp : Fopt_VectorOperandOp<"vector_fp_div", ".vector_fp_div(operand)", (ins)> {
  let assemblyFormat = [{
    $input (`vrf` $vrf^ `:` qualified(type($vrf)))? attr-dict `:` qualified(type($input)) `->` qualified(type($result))
  }];
}

def Fopt_VectorStashOp : Fopt_UnaryStageOp<"vector_stash", ".vector_stash()">;

def Fopt_VectorIntraSliceReduceOp : Fopt_StageOp<"vector_intra_slice_reduce"> {
  let summary = ".vector_intra_slice_reduce::<Axis, Time, Packet>(IntraSliceReduceOpF32::<op>)";
  let arguments = (ins Fopt_Stream:$input, StrAttr:$op, StrAttr:$axis);
  let assemblyFormat = [{
    $op $input `axis` $axis attr-dict `:` qualified(type($input)) `->` qualified(type($result))
  }];
}

def Fopt_ToVrfOp : Fopt_Op<"to_vrf", [Pure]> {
  let summary = ".to_vrf() (a sub-context pass)";
  let arguments = (ins Fopt_Stream:$input);
  let results = (outs Fopt_VrfTensor:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

// Contraction (main context): the TRF operand is loaded by a sub-context pass ending in to_trf.

def Fopt_LaneMode : I32EnumAttr<"LaneMode", "contract_lane mode", [
  I32EnumAttrCase<"Interleaved", 0, "interleaved">,
  I32EnumAttrCase<"Sequential", 1, "sequential">
]> {
  let cppNamespace = "::mlir::furiosa::fopt";
}

def Fopt_ToTrfOp : Fopt_Op<"to_trf", [Pure]> {
  let summary = ".to_trf()";
  let arguments = (ins Fopt_Stream:$input);
  let results = (outs Fopt_TrfTensor:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Fopt_ContractOuterOp : Fopt_StageOp<"contract_outer"> {
  let summary = ".contract_outer::<Time, Packet, _, _, _>(&trf)";
  let arguments = (ins Fopt_Stream:$input, Fopt_TrfTensor:$trf);
  let assemblyFormat = [{
    $input `,` $trf attr-dict `:` qualified(type($input)) `,` qualified(type($trf)) `->` qualified(type($result))
  }];
}

def Fopt_ContractPacketOp : Fopt_UnaryStageOp<"contract_packet", ".contract_packet::<Packet>()">;
def Fopt_ContractTimeOp : Fopt_UnaryStageOp<"contract_time", ".contract_time::<Time>()">;

def Fopt_ContractLaneOp : Fopt_StageOp<"contract_lane"> {
  let summary = ".contract_lane::<Time, Packet>(LaneMode::<mode>)";
  let arguments = (ins Fopt_Stream:$input, Fopt_LaneMode:$mode);
  let assemblyFormat = "$mode $input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
}

def Fopt_CommitOp : Fopt_Op<"commit", [Pure]> {
  let summary = ".commit::<Element>()";
  let arguments = (ins Fopt_Stream:$input);
  let results = (outs Fopt_DmTensor:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// HBM views and indirect DMA
//===----------------------------------------------------------------------===//

// A view is an expression, not a statement: each view value has one use and prints inline there, as
// `t.view()` / `t.view_mut()` followed by its tile and pad calls.

def Fopt_HbmView : Fopt_Type<"HbmView", "hbmv"> {
  let summary = "HbmTensorView<D, Chip, Element> (or HbmTensorViewMut when mutable)";
  let parameters = (ins "::mlir::Type":$elementType, "MapAttr":$chip, "MapAttr":$element, "bool":$isMut);
  let assemblyFormat = "`<` $elementType `,` $chip `,` $element `,` $isMut `>`";
}

def Fopt_ViewOp : Fopt_Op<"view", [Pure]> {
  let summary = "t.view() or, into a mutable view, t.view_mut()";
  let arguments = (ins Fopt_HbmTensor:$source);
  let results = (outs Fopt_HbmView:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Fopt_TileOp : Fopt_Op<"tile", [Pure]> {
  let summary = "v.tile::<Index, LEN, Element2>(start)";
  let arguments = (ins Fopt_HbmView:$source, Fopt_MapAttr:$index, I64Attr:$len, I64Attr:$start);
  let results = (outs Fopt_HbmView:$result);
  let assemblyFormat = [{
    $source `index` $index `len` $len `start` $start attr-dict `:` qualified(type($source)) `->`
    qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Fopt_PadOp : Fopt_Op<"pad", [Pure]> {
  let summary = "v.pad::<Element2>() (one outermost padding factor)";
  let arguments = (ins Fopt_HbmView:$source);
  let results = (outs Fopt_HbmView:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Fopt_ViewToDmOp : Fopt_Op<"view_to_dm", [Pure]> {
  let summary = "v.to_dm::<Cluster, Slice, Element>(&mut device.tdma)";
  let arguments = (ins Fopt_HbmView:$source);
  let results = (outs Fopt_DmTensor:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Fopt_DmToHbmViewOp : Fopt_Op<"dm_to_hbm_view"> {
  let summary = "dm.view().to_hbm_view(&mut device.tdma, dst)";
  let arguments = (ins Fopt_DmTensor:$source, Fopt_HbmView:$dest);
  let assemblyFormat = "$source `,` $dest attr-dict `:` qualified(type($source)) `,` qualified(type($dest))";
  let hasVerifier = 1;
}

def Fopt_HbmToHbmViewOp : Fopt_Op<"hbm_to_hbm_view"> {
  let summary = "src.to_hbm_view(&mut device.tdma, dst) (an HBM-to-HBM copy between views)";
  let arguments = (ins Fopt_HbmView:$source, Fopt_HbmView:$dest);
  let assemblyFormat = "$source `,` $dest attr-dict `:` qualified(type($source)) `,` qualified(type($dest))";
  let hasVerifier = 1;
}

def Fopt_GatherOp : Fopt_Op<"gather", [Pure]> {
  let summary = "table.gather::<Indexed, Payload>().by_byte_offsets::<Domain>(&offsets).to_dm(&mut device.tdma)";
  let description = [{
    `result[d, p] = table[offsets[d] / stride(Indexed), p]`: the DMA reads, per entry of the HBM index (whose
    element mapping is the domain), one payload of the table at that byte offset.
  }];
  let arguments = (ins Fopt_HbmTensor:$table, Fopt_HbmTensor:$offsets, Fopt_MapAttr:$indexed, Fopt_MapAttr:$payload);
  let results = (outs Fopt_DmTensor:$result);
  let assemblyFormat = [{
    $table `[` $offsets `]` `indexed` $indexed `payload` $payload attr-dict `:` qualified(type($table)) `,`
    qualified(type($offsets)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Fopt_VectorFxpOp : Fopt_StageOp<"vector_fxp"> {
  let summary = ".vector_fxp(FxpBinaryOp::<op>, imm) (an i32 ALU op with an immediate)";
  let arguments = (ins Fopt_Stream:$input, StrAttr:$op, I32Attr:$imm);
  let assemblyFormat = "$op $input `,` $imm attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
}

#endif // FURIOSA_CODEGEN_FOPT_OPS

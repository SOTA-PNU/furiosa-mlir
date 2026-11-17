// fopt ops for tiled kernels (lib/Codegen/Tile/): device-side allocation, fills, whole-tensor relabeling, and
// the general Vector Engine ALU call. Included by FoptOps.td.

#ifndef FURIOSA_CODEGEN_FOPT_TILE_OPS
#define FURIOSA_CODEGEN_FOPT_TILE_OPS

def Fopt_AllocHbmOp : Fopt_Op<"alloc_hbm"> {
  let summary = "HbmTensor::<D, Chip, Element>::new() (an uninitialized device allocation)";
  let results = (outs Fopt_HbmTensor:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def Fopt_AllocDmOp : Fopt_Op<"alloc_dm"> {
  let summary = "DmTensor::new() (an uninitialized DM allocation)";
  let results = (outs Fopt_DmTensor:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def Fopt_MemsetOp : Fopt_Op<"memset"> {
  let summary = "t.view_mut().memset(value, &mut device.sub)";
  let arguments = (ins Fopt_DmTensor:$target, F32Attr:$value);
  let assemblyFormat = "$target `,` $value attr-dict `:` qualified(type($target))";
}

def Fopt_ReshapeHbmOp : Fopt_Op<"reshape_hbm"> {
  let summary = "unsafe { t.reshape::<Chip, Element2>() } (relabels a whole tensor; consumes it)";
  let description = [{
    The same cells under another mapping of equal size and the same physical order. Reshaping a whole tensor is
    the form the NPU lowers correctly (reshaping a mutable view and then tiling it miscompiles), so tiled kernels
    relabel whole tensors only. The source must have no other use.
  }];
  let arguments = (ins Fopt_HbmTensor:$source);
  let results = (outs Fopt_HbmTensor:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def Fopt_ViewReshapeOp : Fopt_Op<"view_reshape", [Pure]> {
  let summary = "unsafe { v.reshape::<Chip, Element2>() } (relabels a whole read view)";
  let arguments = (ins Fopt_HbmView:$source);
  let results = (outs Fopt_HbmView:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

// One Vector Engine ALU call. `method` is the furiosa-opt method (vector_fp_unary, vector_fp_binary, vector_clip,
// vector_fp_div, vector_fp_zip, vector_clip_zip), `op` the Rust enum path of its operation (empty for
// vector_fp_div), `mode` an optional BinaryArgMode case (the call becomes <method>_with_mode). The second operand
// is at most one of: an immediate, a VRF tensor, the stash, or tag branches. A branch is [guard, value]; guard is
// the four tag-bit requirements LSB first, each '1', '0' or '-' ("all" for the unconditional slot), and value an
// f32 immediate or the string "rf" for the trailing VRF/stash slot.
def Fopt_VectorAluOp : Fopt_StageOp<"vector_alu"> {
  let summary = ".<method>[_with_mode](<op>, [<mode>,] <operand>)";
  let arguments = (ins Fopt_Stream:$input, StrAttr:$method, StrAttr:$op, OptionalAttr<StrAttr>:$mode,
                       OptionalAttr<F32Attr>:$imm, Optional<Fopt_VrfTensor>:$vrf, UnitAttr:$stash,
                       OptionalAttr<ArrayAttr>:$branches);
  let assemblyFormat = [{
    $method $op $input (`vrf` $vrf^ `:` qualified(type($vrf)))? attr-dict `:` qualified(type($input)) `->`
    qualified(type($result))
  }];
}

def Fopt_VectorTagCmpOp : Fopt_StageOp<"vector_tag_cmp"> {
  let summary = ".vector_intra_slice_tag(TagMode::Comparison([cmp0, cmp1, cmp2, cmp3]))";
  let description = [{ `cmps` holds the four Cmp cases as Rust, e.g. "Less(0.0f32)", "False". }];
  let arguments = (ins Fopt_Stream:$input, StrArrayAttr:$cmps);
  let assemblyFormat = "$cmps $input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
}

// Kept device loops: `for i in 0..count { body }`. The body block's one index argument is the induction variable;
// tile_at views inside index with it, so one DMA statement covers every iteration (the NPU keeps HBM loops,
// where unrolled DMAs cost compile time superlinearly).
def Fopt_ForOp : Fopt_Op<"for", [SingleBlock, NoTerminator]> {
  let summary = "for i in 0..count { body }";
  let arguments = (ins I64Attr:$count);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def Fopt_TileAtOp : Fopt_Op<"tile_at", [Pure]> {
  let summary = "v.tile::<Index, LEN, Element2>(start + sum(coeffs[i] * ivs[i]))";
  let arguments = (ins Fopt_HbmView:$source, Fopt_MapAttr:$index, I64Attr:$len, I64Attr:$start,
                       Variadic<Index>:$ivs, DenseI64ArrayAttr:$coeffs);
  let results = (outs Fopt_HbmView:$result);
  let hasVerifier = 1;
}

// Hardware-abstract data movement (lowered by furiosa-dma, lib/Codegen/Tile/Dma.cpp). The tiled lowering says
// what each tile holds; the pass decides how the data gets there: the staging buffer's layout in HBM (padded
// copies and their fills, tap-major or row-major K, zero-padded inner digits, tap-loop classes) and each tile's
// view and DMA to DM.

def Fopt_StageOp : Fopt_Op<"stage"> {
  let summary = "a kernel buffer read at a linear index over the program's axes, staged in HBM (abstract)";
  let description = [{
    `spec` describes the read and the layout its tiles are cut from: `index` (per buffer dim, [offset, axis,
    coeff, ...]), `shape`, `fill`; the program's axis `sizes` and padded `extents`; the layout's `rows` axes and
    their `canonical` digits; for a contraction operand its K split (`taps`, `inner`, `tap_pad`, `inner_pad`,
    `k_tile`, `kf`); inside a Vector Engine reduction its `ext` axes and `rf` (and `rc`) axes; the `tag` naming
    the staging axes and `axes_at`, where they go in the kernel's axes. The result is only read by fopt.read.
  }];
  let arguments = (ins Fopt_HbmTensor:$source, DictionaryAttr:$spec);
  let results = (outs Fopt_HbmTensor:$result);
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
}

def Fopt_ReadOp : Fopt_Op<"read"> {
  let summary = "one tile of a staged read, moved to DM (abstract)";
  let description = [{
    `at` selects the tile: [digit, start, iv] per tiled digit of the staging layout, the tile at start (plus the
    induction variable `ivs[iv]` of a device loop over the tiles, when iv >= 0). `k_index` is the contraction's K
    tile (with `with_k`), `rc_index` the reduction chunk.
  }];
  let arguments = (ins Fopt_HbmTensor:$staged, Variadic<Index>:$ivs, ArrayAttr:$at, BoolAttr:$with_k,
                       I64Attr:$k_index, I64Attr:$rc_index);
  let results = (outs Fopt_DmTensor:$result);
  let assemblyFormat = [{
    $staged (`ivs` `(` $ivs^ `)`)? attr-dict `:` qualified(type($staged)) `->` qualified(type($result))
  }];
}

#endif // FURIOSA_CODEGEN_FOPT_TILE_OPS

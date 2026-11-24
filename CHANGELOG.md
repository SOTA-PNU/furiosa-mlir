# Changelog

main carries the next version as `X.Y.Z.dev0`. A release is a `release/X.Y.Z` branch cut from a verified commit,
with the commit "Release X.Y.Z" and the tag `vX.Y.Z`. "On npu3" means harness level L4 (or the operator matrix's
npu stage) against GPU goldens.

## 1.6.0 (unreleased, main)

- Raise, Swin-T's window attention (unshifted windows), three fallbacks taken only where the old path failed (each
  with a lit test): a masked store's digits resolve over the lanes its mask keeps (3eec6a9); a value is rewritten
  within each instance's tile when the whole coordinate ranges keep it apart (4881ada); a masked load's address is
  rewritten over the lanes its mask keeps (d678368). Swin-T's SHIFT=0 window attention matches a NumPy reference at
  all four L1 stages (1.4e-6); the shifted windows (a store that wraps around the image) still fail the raise.
- Tile: the plan keeps a kernel's unrolled K-tile reads within furiosa-opt's dynamic addresses (6352b6c): a K tile
  is admitted only if K tiles x operands <= 64, half of furiosa-opt's 128, chosen from the build-time curve (64
  tiles 69 s, 65 tiles 745-1395 s); above it the plan falls back to whole-K column tiles. Llama's and Qwen2.5's
  128256/152064-wide heads build in 31-33 s at 4.8 GB (they failed after 163 s / 1484 s) and match on npu3 within
  a bf16 ulp. The corpus and all 907 matrix cases are unchanged.
- Tolerances of `qwen3_4b` by the CPU-vs-GPU drift (49c9c8d): graph 2.6e-2, npu 5.2e-2.

## 1.5.0 (release/1.5.0, at 8aa63d7)

- A third held-out batch (`heldout3_*`, e1ca4c1: sliding-window attention, scatter-add, batch-norm statistics, avg pool
  with padding, bilinear upsampling, MoE top-2 gating, segment sum, an fp8 scaled matmul, residual add with layer
  norm, scans) and a held-out model, ViT-B/16 in pure Triton with torchvision's weights (9a6ee09). First try at 1.4.0:
  1 of 10 kernels passes L4; ViT stops at L1 (its patch-embedding store remaps rows by // and % across 64-row
  blocks). The fp8 matmul gave a wrong result without a diagnostic (the rounding to e4m3 was dropped): fixed below.
- Truncations to fp8 round as the GPU converts (931abfb): mapping widens fp8 to f32 and had dropped the casts, so a
  `tl.dot` on `x.to(tl.float8e4nv)` ran on the unquantized values. A truncation to e4m3fn or e5m2 now rounds in f32
  ops the Vector Engine has (nearest even, saturating; exact for every bf16 and 2M random f32 against ml_dtypes), and
  L1's JIT encodes it. `heldout3_fp8_matmul` passes L1 to L4 (rel 0).
- A contraction factor the body computes from a weight laid out [k, n] reads the weight relaid out (a transpose step),
  padded after when K is masked (21f5bd3).
- Raise: a store whose rows are remapped by `//` and `%` by a constant its tile crosses keeps the digits in every
  program instance (6f360ee; ViT's patch rows `n * 197 + 1 + p`). ViT's 88 launches pass L1; its patch embedding then
  stops in codegen (an assertion in the K-loop digit split).
- Raise: a loop over column tiles of a row the tile does not divide stores at `c0 + arange` (the tile counter takes
  the column's digits above the tile), and a loop-carried value read inside the loop is the accumulation of the
  iterations before (7f88f38). `heldout3_cumsum` (exclusive scan) passes L1; both its kernels stop in codegen.
- Normalize: a loop split into the digits a `//` and `%` read takes apart, which only the reduction's carrier spans,
  reshapes the carrier like an output; a split that would leave a digit unspanned is not made (66d40f0; it asserted
  in `getStaticLoopRanges`). ViT's patch embedding maps as one contraction; all of vit_b16 passes L2a.
- A read whose address comes from the loop indices alone but not affinely (a float floor, a flat index over tiles
  its divisor does not align with) gathers from a hoisted address table instead of staying in gather form; past 256
  rows the addresses are inserted into zeros (f3ed504). `heldout3_upsample_bilinear` passes L1 to L4 (rel 0).
- Tiled class: a contraction whose result an enclosing reduction's body reads is refused (e3d9bd3; furiosa-opt had
  rejected the kernel at the NPU build after the mapping was accepted), and a 0/1 mask table is no contraction
  operand (2a32c04): a masked f32 sum went through the bf16 contraction path and was off by 1e-3. Steps marks the
  tables it externalizes. `heldout3_batch_norm` passes L1 to L4 (rel 4.4e-7).
- Raise, four general mechanisms for batch 3 (each with a lit test; corpus and all 907 matrix cases' stages
  unchanged):
  - An index no single expression of the coordinates gives (a min/max whose operands' ranges overlap, an index the
    program instances disagree on) is computed from its operands where it is read (615c338). A check that instances
    add the same data terms now compares the terms, not only their count.
  - A loop whose lower bound differs between instances and is no expression of the program ids counts from the
    least bound, each iteration guarded (5f85e8f).
  - An atomic whose address adds data (`dw[ids[t] * D + d] += g`) folds, per element, the lanes whose data selects
    that row, starting from the old value (0ab25cb).
  - A loop whose bounds are loaded runs over every counter value its loads can reach, guarded by the bounds and the
    step; accepted only for loops that write nothing and only accumulate (15078aa).
  `heldout3_avgpool` passes L1 to L4 (rel 0); scatter-add, segment sum and sliding-window attention pass L1 and stop
  in codegen.
- N chips, first step (9a12d59): `FURIOSA_CHIPS=N` compiles every kernel for N chips (`#[device(chip = N)]`, the chip
  mapping in every type) and uploads every buffer to each chip; for now every chip computes the same values.
  `FURIOSA_VISIBLE_CHIPS` names the NPUs (default the last N). Every run locks each NPU it opens and checks right
  before that each is idle, waiting otherwise; L4 passes only if the run opened exactly its NPUs. N = 1 is
  byte-identical. At N = 2 (npu2 and npu3) eight cases pass L4, from vector_add to LeNet-5.
- Tiled class: a K window over a dim the K blocks do not divide (32 keys of 100) reads a copy padded to whole blocks
  (984cc71; furiosa-dma had failed with "an inner contraction digit wider than its padded extent"). Only kernels
  that failed so take the copy. `heldout3_sliding_window_attention` passes L1 to L4 (rel 0.0024, tolerance 6.4e-3).
- N chips, splitting (2a5fbb4): with `FURIOSA_CHIPS=N > 1`, `furiosa-shard` splits one parallel dim of every
  kernel over the chips. It traces each result's dims, outermost first, through the kernel's linalg ops; a window,
  a loop index used as data, a reduction loop, an op outside linalg, a block N does not divide, or a block that does
  not map to fopt refuses a dim (`FURIOSA_DEBUG=1` says why). Chip c gets block c of each split argument by one
  HBM-to-HBM DMA, and each split result is all-gathered into every chip's copy by one DMA across chips; a kernel
  with no dim to split, or one that reads a buffer it writes, runs whole on every chip. N = 1 is unchanged. At
  N = 2 on npu2 and npu3 the micro kernels, all tutorials, 19 held-out kernels, LeNet-5, AlexNet, VGG16,
  ResNet-101 and MobileNetV3 pass L4 against the same goldens. Measured after both commits: at N = 2 all 51 corpus
  cases that pass at N = 1 pass L4 (and 181 of 181 operator-matrix cases of add, exp, sum, argmax, dot, gather,
  atomics, sort, softmax, cumsum, trans); at N = 4 on npu0 to npu3, ten cases from vector_add to Qwen3-0.6B.
  Speedup is modest (ms, N = 1 / 2 / 4, warm): tutorial matmul 11.4 / 9.7 / 8.8, VGG16 2136 / 1897 / 1804,
  ResNet-101 1574 / 1280 / 1160, Qwen3-0.6B prefill + decode 14790 / 11980 / 10735. A cross-chip gather runs at
  about 8.6 GB/s per chip, a few ms per inference; the limits are kernels that stay whole (VGG16 6 of 18:
  maxpool windows; Qwen3 37 of 66: attention, RoPE, RMSNorm, argmax, embedding) and weight blocks copied every
  launch. ViT-B/16's N = 2 NPU build was killed (memory).
- Selections and row-wise terms for MoE gating and segment offsets:
  - A seeded selection reduction (`tl.argmax`'s form) may compare a value computed from the element, other inputs
    and the index (softmax probabilities, a second pick masked to -1), when the seed computes it op for op at
    element 0 (8b59f24).
  - A masked term over padded rows pads the one input whose identity padding folds the term (f84756a; segment
    offsets over 13 columns padded to 16).
  - A column input streams where it meets a row value (`cols == pick[:, None]`); `tl.histogram`'s values stream so
    too, with no relayout step (e3286a2).
  `heldout3_moe_topk_gating` passes L1 to L4 (rel 9.8e-8); segment sum's offsets kernel maps and is exact on npu3.
- ViT-B/16 passes L0 to L4 on npu3: top-1 equal to the GPU's, rel 0.0015 at the logits (tolerance 0.0058); one
  graph run 3.44 s including device open and 350 MB of weights; its NPU build 1m52s. Three general changes besides
  66d40f0:
  - An index-only value equal at every point (a mask true over its whole range) is a constant, and what it makes
    trivial folds (8bb6122).
  - The row-wise class divides a constant by the streamed value (`1 / sqrt(var / N + eps)`) with the divider's
    operands swapped (d448fc6); ViT's layer norm maps as one row-wise kernel.
  - A mapping whose Vector Engine reduction replicates its reads beyond 64M elements does not map, and the kernel
    splits into smaller steps (fcfa5bf): attention at 197 tokens replicated 155M and its furiosa-opt build passed
    50 GB of memory. The corpus's largest is 40M.
  Six corpus cases change fopt (group norm, layer-norm and rmsnorm backward, add + layer norm, the layer-norm
  tutorial, Qwen3-0.6B) and pass L0 to L4; the matrix is unchanged.
- Cumulative sums over rows that do not divide the slices:
  - Row-wise: a pass reads at most 5 VRF operands (d72932a). The NPU build refuses a sixth ("cannot load more than
    5"), but the CPU emulation accepts it, so L3 passed where L2 failed.
  - Row-wise: rows whose dims do not divide the 256 slices map padded or in windows of an inner dim, tried after the
    tiled class so nothing it maps moves, and never for a contraction of two buffers (fe5055f; cumsum's [19, 300]
    rows as windows of 300 with 19 padded to 32, one device loop of 50).
  - Normalize: a guarded integer reduction step `select(p, acc + t, acc)` adds `select(p, t, 0)` (d00ed6e); an
    integer padding read through a reshape is a copy step (4f9ed25); a loop splits into digits whose place does not
    divide it, padded to whole digits and sliced back (072e14e). Integers only: for floats, the skipped term may be
    inf, so a float form keeps its diagnostic.
  `heldout3_cumsum` passes L1 to L4 (exact). The matrix is unchanged (its scans and reductions pass 171 of 171 on
  npu3); upsample and sliding-window attention merge some steps and pass L4.
- Held-out models, wave 2 (written from the math by authors who did not read the compiler, checked on the GPU
  against torchvision's own model: top-1 identical on the golden input and a real photo):
  - EfficientNet-B0 (7fed84b): MBConv with strided 3x3/5x5 depthwise convs, squeeze-and-excitation, SiLU, explicit
    batch norm; 83 launches; max|logit diff| 2.7e-6, NPU tolerance rtol 7.4e-2.
  - ConvNeXt-Tiny (1473bd7): 7x7 depthwise conv into channels-last rows, LayerNorm over channels, erf GELU, layer
    scale with a transposed store, patch-GEMM stem and downsampling; 83 launches; max|logit diff| 2.9e-6, rtol
    1.5e-2.
  - Swin-T (fb109d7): window attention gathered from the cyclically shifted image with the relative position bias
    and the shifted-window mask, stored back by the reverse roll, patch merging; 95 launches; max|logit diff|
    4.3e-6.
  - DenseNet-121 (f50aea8): each dense block one NHWC buffer that layers append their channels to (a store into a
    channel slice), BN-ReLU as prologue and epilogue, transitions with a 2x2 average pool; 126 launches.
  - BERT-base (8ba8d7b): embeddings with LayerNorm, 12 post-LN layers with a key padding mask from per-sequence
    lengths, erf GELU, the tanh pooler; fp32, a batch of 2 padded to 128; max diff 1.1e-5 against transformers.
  - GPT-2 (c1c10cc): prefill and decode graphs, Conv1D weights read in place, a fused QKV GEMM feeding per-layer KV
    caches, causal flash attention, tanh GELU, the tied head; greedy tokens identical to transformers'.
  - Whisper-tiny (327ff98): the encoder (two conv1d as implicit GEMMs, sinusoidal positions, 4 layers over 1500
    frames) and a decoder step with causal self-attention and cross-attention; next token identical.
- Larger models: the Qwen3 graphs build every size from a Config, qwen3_0_6b's TTIR unchanged byte for byte
  (0ff6f16); `qwen3_1_7b` runs them at Qwen3-1.7B's config with the official weights, greedy tokens identical to
  transformers' (42b818b), as do `qwen3_4b` (6dd5987) and `qwen3_8b` (5115db6, untied head); `llama3_1_8b`
  (b99cd56) is a new pure-Triton module written from the math (GQA with Llama 3's rope scaling, SwiGLU, an untied
  head over 128256 tokens; weights unsloth/Llama-3.1-8B-Instruct, bf16), and so is `qwen2_5_7b` (b6f2864: QKV bias, no
  q/k norm). All five match transformers' greedy tokens in prefill and decode (logit drift 1.8-3.3%). All stop at L1:
  an LLVM assertion (`cast<AffineDimExpr>`) in RMSNorm with residual run as one program at hidden >= 2048 (and
  Llama's argmax), Llama's GQA prefill attention ("an address varies between program instances"), and silent
  segfaults after tiling in GEMMs with N >= 12288 (SwiGLU, Llama's 128256-token head). Fixes are in progress.
- Row-wise: a stored K loop (a contraction over K > 1024) is not a row tile nest, and the class declines it
  (9d69cb4). fold() had dereferenced the empty nest: undefined behavior that crashed or mapped from run to run, in
  every SwiGLU from Qwen3-1.7B up. Those SwiGLUs now map.
- Normalize: a long reduction whose input is read at 0 in a unit dim is reshaped before the split (18ae9a9). The
  upstream split asserted (`cast<AffineDimExpr>`) on RMSNorm with residual run as one program at hidden >= 2048, in
  every Qwen3 size from 1.7B and Qwen2.5-7B; those kernels now map.
- Raise: a program id is recovered from the tile origin along a view dim, digit by digit (5bd6d61): rows a program
  computes as `t * GROUP + j` (Llama's GQA attention, 4 query heads per KV head in one tile) or `seq * T + block`
  (BERT's per-sequence attention) raise. Llama's prefill attention matches on npu3 (max err 0.0073, bf16); its
  furiosa-opt build takes about 40 minutes.
- A raise failure's diagnostics travel in its exception (d705248): the harness line shows the first error instead of
  "PassManager::run failed".
- Golden data keeps checkpoint weights by reference (55b474a): a golden of a case with `weights=` stores only its
  inputs and states; every consumer (the harness's levels, `furiosa_compiler.l1`/`l1_launch`, the crate's run_graph)
  reads full inputs restored from the pinned checkpoint into a cache, bit-identical to the old files. Old goldens
  still work. An 8B model's golden no longer stores its 16 GB of weights once per graph.
- Tolerances: where L1's raised graph on the CPU JIT drifts from the GPU golden more than the reference
  implementation does, that drift sets graph and npu (0342c84; qwen3_1_7b 2.69%, as qwen3_0_6b's 2.7%).
- Codegen for scatter-add and segment sum (each with a lit test):
  - A float reduction step a data condition guards adds `select(c, x, -0.0)` (1771109), extending the integer form.
  - An index compared with data, or masked into such a comparison, is a hoisted table (e841e2c); `rem(a, 2^k) == 0`
    is its low bits (1987d98).
  - A column value computed from a column input streams where it meets a row value (06b7a1f).
  - A reduction no class maps collapses its leading loops once the inputs that read part of them are broadcast over
    them, kept only if the result maps (7d5803c).
  - An index list too long for one slice's DM spreads over the slices (c7e8237; it failed on npu3 with an Sram
    allocation error after mapping).
  `heldout3_scatter_add` (rel 9.6e-8) and `heldout3_segment_sum` (rel 0) pass L1 to L4: all ten kernels of the third
  held-out batch pass L4. The matrix's int32/int8 interleave and join now map as one step and pass on npu3.
## 1.4.0 (2026-11-18, release/1.4.0)

Milestone: kernels the compiler was not written for. The second held-out batch (ten standard ML ops written from their
math without reading the compiler) passed L4 at first try in 5 of 10 with 1.3.0 and passes in all 10 now; all 35
corpus and held-out cases pass L0 to L4 and all 907 operator matrix cases pass on npu3.

- A reduction without a Vector Engine reduction (xor, and, or, a product) over a width that is not a power of two pads
  with the combiner's identity to one before its tree: xor_sum at 2x7x33, which d6133de's live-prefix reductions had
  broken, maps again (cc4fe5b).
- Atomics every program applies to the same elements (add, max, min, and, or, xor) raise as one reduction over the
  program ids, folded into the old value in program order (bit-identical to the rounds), instead of a round per
  program recomputing its whole computation; a reduction that starts from a computed value no class maps reduces
  from its identity and combines after. Held-out cross-entropy (45 programs adding into one mean) passes L0 to L4: its
  crate went from 919 kernel modules (a 49-minute build) to 22 (f4a5660).
- Float-to-int casts are exact: the engine's conversion rounds where `fptosi` truncates, so row-wise converts only values
  that are whole by construction (a clamped floor, a round) and truncates the rest first (`math.trunc` expands):
  held-out int8 quantization passes L0 to L4 (fa006c8).
- A buffer a kernel reads before anything wrote it starts as zeros in the crate, as the golden and L1 assume (812e42e).
- A data-dependent read inside a computation (x[r, t[r]] in a loss) becomes a gather step of its own that the computation
  reads as an input; furiosa-opt's gather DMA takes power-of-two row counts, so other counts pad with row 0 (npu3
  rejected 3, 5 or 12 rows): held-out cross-entropy maps (2bb23f5).
- Raise: a reduction whose input is the combiner's identity past a masked prefix reduces over the prefix (no clamp left,
  the read promotes; the dot's live-K analysis shares it): held-out group norm passes L0 to L4 (d6133de).
- Float scalar arguments stay fp32 values: the driver made every scalar a constexpr, so Triton folded float arguments in
  Python's double precision (1 - beta2 at 0.999 became 1e-3; the GPU computes 1 - 0.999f = 0.00099998713). A float
  argument not marked tl.constexpr is now an fp32 argument whose value the launch records and the raise binds and folds
  in fp32: held-out AdamW passes L1 to L4; attention backward's `_scale` is closer to the GPU (50b66a6).
- The step splitter remembers which candidate steps map (by their outlined text): splits byte-identical, about 2.4x
  faster on many-step kernels (2e8c484).
- Raise: store tiles masked to a prefix may step by the prefix (a 4-channel tile masked to a group's 3 channels starts
  every 3): held-out group norm passes L1 (8c925d8).
- Raise: a loop up to a min of bounds (`range(start, min(start + ROWS, M))`) guards each iteration by every term, and a
  masked load past such a trip reads its `other`: held-out rmsnorm backward passes L1 to L4 (d4295da).
- Release binaries on GitHub: `scripts/package.sh` builds a wheel (`furiosa_mlir-<version>-cp312-cp312-linux_x86_64.whl`)
  beside the tarball; neither bundles FuriosaAI's furiosa-opt tools (`install.sh` installs `cargo-furiosa-opt` 0.10.0
  from crates.io with the crates' toolchain); both carry Triton's and LLVM's licenses and copy the targets of the
  source tree's links (the earlier tarballs carried links that resolved only on the build host). The 1.3.0 release
  has both attached, with checksums (c78db8f, b9ae73d).
- A second held-out batch (`heldout2_*`, c08c3bc): ten standard ML ops (layer norm, rmsnorm and softmax backward, RoPE,
  cross-entropy, group norm, int8 quantization, AdamW, a matmul epilogue, a direct conv) written from their math
  without reading the compiler. First try at 1.3.0: 5 of 10 pass L4 on npu3. The harness groups `heldout*` cases
  as held-out.
## 1.3.0 (2026-11-18, release/1.3.0)

Milestone: the whole operator matrix on npu3. All 907 operator matrix cases pass (879 in 1.2.0), and all 25 corpus
cases pass L0 to L4.

- tl.sort and tl.topk on npu3, exact for every dtype and shape (24 cases): compare-and-swap keys nest into selects in
  reduction terms (eed201e); a loop read by its floordiv/mod digits splits into them, so the reads are projections
  (9ead0e0); a generic no class maps reads an earlier generic's result instead of recomputing the fused stages
  before it (140d820); an index-keyed select blends bitwise, b ^ ((a ^ b) & m), where no class maps the generic
  (bdeb328); loops every operand reads alike collapse where that makes the generic map (aa90044). Split-steps takes
  0.2 to 2 s per case; tl.flip needs fewer steps (45 to 39 at 1x64x64).
- The tiled path's comparison tags honor ordered and unordered predicates exactly (they took oge/ole/one as the
  complements of olt/ogt/oeq, true on NaN; ord/uno did not map): all 14 cmpf predicates match torch on npu3 over NaN,
  +-inf, +-0 and the bounds' neighbours (b56a07b).
- maxnumf/minnumf of two computed operands are exact too: r = maximumf(clean x, clean y), then r > -inf ? r :
  (x == -inf ? x : y), so a NaN is skipped and only an all-NaN pair gives NaN. The tiled path's last arithmetic blend
  (a select whose branch tile exceeds the 8 KB VRF, a*s + b*(1-s), which made NaN rows of attention) is a bit-level
  select b ^ ((a ^ b) & m). `extra.maximum_nan` exact on npu3; whole-graph run time unchanged (fused attention 0.55 s,
  vgg16 3.02 vs 2.98 s), though Vector Engine passes grow (vgg16's maxpool 4 to 34) (2150081).
- tl.sort and tl.topk compile to a verdict in seconds instead of timing out: the row-wise operand weight is memoized
  over the DAG (it recursed exponentially through sort's stages and hung the mapping oracle) (ae69985), and a split
  only the oracle asks for stands only when its pieces map (pieces split again down to one per element: sort at 1024
  lanes made ~62,000 steps, now 681 in 2.1 s) (b8ee106). They still fail to map: a compare-and-swap stage's key
  XORs two conditions inside a generic that also reduces, over 12 size-2 row dims.
- Steps are named from a symbol table and a running counter: probing every name from 0 with a module scan was cubic
  in the step count, minutes for tl.sort's thousands of steps (b761d74).
- The tiled path's NaN-skipping clean value is exact: two logic ops under one tag ((x | m) & m = m where the tag
  fails, x where it holds) instead of AbsMax, which propagated NaN. Tiled relus and reductions now skip NaN on
  npu3 as on the GPU (relu, min, row and column max of NaN, +-inf and NaN-payload inputs exact); run time
  unchanged (vgg16 3.01 vs 3.02 s, resnet101 2.19 vs 2.20 s) (0a7d88d).
- A data-dependent element gather the 8-byte DMA rule blocks gathers 8-byte rows at the even-aligned address and
  selects the half by the address's parity (exact for any bits); a signed remainder by a power of two expands to
  shifts and masks: `extra.gather_data` on npu3 (832a9e3).

## 1.2.0 (2026-11-18, release/1.2.0)

Milestone: every corpus case on npu3. All 25 corpus cases pass L0 to L4 (the attention backward tutorial was the
last), and 879 of the 907 operator matrix cases pass (801 in 1.1.0). Of the rest, 2 differ on NaN inputs
(`extra.maximum_nan`: the engine's max propagates NaN where the GPU's skips it, see the known issues below) and 26 do
not compile: sort (20) and topk (4) exceed the step splitter's time, and `gather_data` (2) waits for 1.3.0.

- Normalization: reduction loops of extent 1 drop (`tl.expand_dims` with a reduction, int32 and int8); `tl.umulhi`
  computed from 16-bit limbs, exact under the engine's wrapping multiply (cae9784).
- Operator matrix on npu3: masked stores pass in every dtype (10 cases; they failed at 1805f3e).
- Selects keyed by a boolean combination of comparisons nest into selects keyed by one each: `tl.map_elementwise`'s
  three-way compare on npu3 (0ab16b6).
- A widened i1 is a select; a reduction no class maps reads its input broadcast over the loops it lacks:
  `tl.histogram` on npu3 (ccb3f49; c49e615 limits the broadcast to a reduction's sole reduced input, which had
  split fused attention's forward kernel on main).
- A read whose address wraps (a mod or floordiv one loop decides) splits at the wrap into box copies: `tl.gather`
  by a rotated index on npu3 (0af780b; 3c04413 fixes a dangling operand list in it).
- In a generic no class maps, a select one loop decides splits even when its sides are computed: `tl.maximum`'s
  NaN-propagating mode exact on npu3 (ab5dce9).
- Row-wise: reductions over masked (padded) loads keep the pad value; i32 reductions over padded rows fill with their
  identity; bf16 row values; one-column rows; a reduction to a scalar stored through a cast is one step. Every
  reduction (sum, max, min, argmax, argmin, xor_sum) and bf16 abs/cos/sin/rsqrt pass on npu3 for all dtypes and
  shapes, 32 more matrix cases (0e97b8c).
- i8 and i16 ride in i32 lanes with wraps at their width; bf16 bitcasts to i16 through the f32 bits: `tl.flip` on bf16
  bit-exact on npu3 (db8aa18; fp16 stays out, it lives on the device as f32).
- The tiled path's select takes each branch under the tag, bit for bit: a NaN or infinity in the branch not taken
  no longer leaks (it blended a·s + b·(1−s)), and selects with infinite constants map (057b951).
- `tl.dot` on fp8 and int8 operands on npu3: fp8 buffers live as f32 like fp16 (decoded exactly, stored to the
  nearest code); int8 sums run on the f32 path where exact (K <= 1024) (ae033b8).
- Triton's attention backward tutorial on npu3, L0 to L4 (dq rel 0.017): the tiled analysis merges a loop's adjacent
  digits back into one axis; 2^x folds as e^(x ln 2); a reduction guarded by a program-dependent trip count splits
  along the program loops; a guarded load reads the window it spans at an affine address (2f945b0). All 25 corpus
  cases now reach L4 (attention backward was the last).
- `maxnumf`/`minnumf` (`tl.maximum`/`tl.minimum`'s default, `tl.max`/`tl.min`, relu) skip NaN as on the GPU in
  kernels the row-wise class maps: a NaN guard runs before the whole-kernel check, at no measured run-time cost
  (vgg16 3.05 s, resnet101 2.21 s either way) (30e9144). Known issues: in the tiled path the guard's clean leaves
  NaN as NaN, so tiled relus and reductions still propagate NaN as the engine does (a fix is in progress); two
  computed operands (an attention running max) keep the engine's NaN-propagating max; an all-NaN row reduces to
  -inf. No corpus or matrix golden holds NaN, so no result changes.
- Row-wise row tiles run as one device loop (a 1x256x256 cumsum was a 350 KB kernel past furiosa-opt's 128 dynamic
  indices); the Splitter refuses a step past 256 DMA statements (furiosa-opt's compile time grows superlinearly in
  them). cumsum and cumprod at 1x256x256, cumprod at 1024 lanes, flip at 4x32x128 and 1x256x256 pass on npu3;
  the matrix at 8f7ce01 plus this change: 879 of 907 (5a5c85f).
- Row-wise: a reduction of a computed term pads its input with the reduction's identity when the term keeps it
  (MLIR's folders evaluate it): bf16 max at 2x7x33 maps again under the NaN guard (d15e419).

## 1.1.0 (2026-11-17, release/1.1.0)

Milestone: Triton's tutorials on npu3 except the attention backward pass; every model; the operator matrix's
data-movement and atomic operators. On npu3: 24 of the 25 corpus cases pass L0 to L4, and 801 of the 907 operator
matrix cases pass (741 at 1805f3e), none with a wrong result.

- Triton tutorials newly on npu3: fused attention (forward), extern functions (libdevice `asin`).
- Operator matrix newly on npu3: `tl.cat`, `tl.join`, `tl.interleave` (every dtype and shape); `tl.flip` (32-bit
  and int8, 64x64 and 1024 lanes); atomics `and`, `or`, `xor`, `max`, `min` (int32, float32) and those returning
  old values (`atomic_add`, `atomic_xchg`, `atomic_cas`); rows that are not a multiple of 16 (padded packets);
  int8 in i32 lanes.
- Codegen: asin, acos, atan and atan2 by MLIR's polynomial approximations; copysign; unsigned i32 min/max;
  selects keyed per row (bitwise); 32-bit bitcasts and the logical right shift; Vector Engine ops fused into
  passes (`furiosa-ve-fuse`).
- Steps: a generic that only chooses between two reads by an index splits into copies; transposed reads in
  elementwise generics are relaid out; an elementwise generic reading its init reads it as an input; returned
  buffers are copies; a step never takes a buffer it reads as its destination; outputs another step reads stay
  outputs (no merge counts a stale group as a reader).
- Fixes: int32 sort no longer re-walks unmappable values exponentially in the row-wise fold (900 s to 12 s); a
  kernel launched from a script in the binary package uses the package's pinned furiosa-opt tools.
- Tools: `furiosa compile | run | trace`, the binary package (`scripts/package.sh`), the Docker playground,
  `docs/usage.md`; `HARNESS_TRITON` runs the harness against a copy of the Triton package.

## 1.0.0 (2026-11-17, release/1.0.0)

First release: Triton 3.9 TTIR raised to linalg, structured, split into steps, tiled and mapped to furiosa-opt for
npu3. On npu3: LeNet-5, AlexNet, VGG16, ResNet-101, MobileNetV3-Large, Qwen3-0.6B; Triton's vector add, fused
softmax, matmul, layer norm, seeded dropout and persistent matmul tutorials; six held-out kernels. Operator matrix
(907 cases modeled on SOTA-PNU/lowlevel-api-test): 905 pass the raise stage.

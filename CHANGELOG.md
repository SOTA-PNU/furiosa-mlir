# Changelog

main carries the next version as `X.Y.Z.dev0`. A release is a `release/X.Y.Z` branch cut from a verified commit,
with the commit "Release X.Y.Z" and the tag `vX.Y.Z`. "On npu3" means harness level L4 (or the operator matrix's
npu stage) against GPU goldens.

## 1.4.0 (unreleased, main)

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

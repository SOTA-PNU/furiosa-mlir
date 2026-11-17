# Changelog

main carries the next version as `X.Y.Z.dev0`. A release is a `release/X.Y.Z` branch cut from a verified commit,
with the commit "Release X.Y.Z" and the tag `vX.Y.Z`. "On npu3" means harness level L4 (or the operator matrix's
npu stage) against GPU goldens.

## 1.2.0 (unreleased, main)

- Normalization: reduction loops of extent 1 drop (`tl.expand_dims` with a reduction, int32 and int8); `tl.umulhi`
  computed from 16-bit limbs, exact under the engine's wrapping multiply (cae9784).
- Operator matrix on npu3: masked stores pass in every dtype (10 cases; they failed at 1805f3e).
- Selects keyed by a boolean combination of comparisons nest into selects keyed by one each: `tl.map_elementwise`'s
  three-way compare on npu3 (0ab16b6).
- A widened i1 is a select; a reduction no class maps reads its input broadcast over the loops it lacks:
  `tl.histogram` on npu3 (ccb3f49).
- A read whose address wraps (a mod or floordiv one loop decides) splits at the wrap into box copies: `tl.gather`
  by a rotated index on npu3 (0af780b; 3c04413 fixes a dangling operand list in it).
- In a generic no class maps, a select one loop decides splits even when its sides are computed: `tl.maximum`'s
  NaN-propagating mode exact on npu3 (ab5dce9).

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

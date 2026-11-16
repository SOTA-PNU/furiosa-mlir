//! harness/cases/rmsnorm.py: y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w for R rows of H, bf16 in and
//! out, fp32 math.
//!
//! Mapping a compiler derives: one row per slice (`Slice = [R, 1 # 32]`, the rest padding), so the
//! row reduction is an intra-slice reduce over H's time and packet; the weight is broadcast to every
//! row's slice. Three passes, because the Vector Engine uses each ALU once per pass and has no rsqrt:
//! (1) sum of squares / H + eps, (2) sqrt, (3) x * w / rms with w and rms as VRF operands.

use furiosa_opt_std::prelude::*;

axes![R = 8, H = 1024];

pub type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![R, 1 # 32];

const EPS: f32 = 1e-6;

#[device(chip = 1)]
pub fn rmsnorm_kernel(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![R, H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
) -> HbmTensor<bf16, Chip, m![R, H]> {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut device.tdma);

    let ms: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(1024.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let weight: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = weight.to_dm(&mut device.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![H]> = device
        .sub
        .begin(weight.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let y: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

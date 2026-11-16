//! harness/cases/matmul.py: y = x @ w^T for x [M, K], w [N, K] (nn.Linear layout), bf16 operands,
//! fp32 accumulation, bf16 out.
//!
//! Mapping a compiler derives (the decode-projection pattern of furiosa-opt's Qwen3 example, with M
//! rows added in time): N splits over slices (`N / 32`, 8-way padded) and lanes (`N % 8`); each slice
//! holds its 32 weight rows in the TRF and the whole x, broadcast. K is contracted in the 32-element
//! packet (tree) and over time (accumulator); M and the remaining `N / 8 % 4` stay in time.

use furiosa_opt_std::prelude::*;

axes![M = 8, K = 1024, N = 1024];

pub type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![N / 32, 1 # 8];

#[device(chip = 1)]
pub fn matmul_kernel(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![M, K]>,
    w: &HbmTensor<bf16, Chip, m![N, K]>,
) -> HbmTensor<bf16, Chip, m![M, N]> {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![M, K]> = x.to_dm(&mut device.tdma);
    let w: DmTensor<bf16, Chip, Cluster, Slice, m![N % 32, K]> = w.to_dm(&mut device.tdma);
    let w_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![N % 8], m![N / 8 % 4, K]> = device
        .sub
        .begin(w.view())
        .fetch::<m![N % 8, N / 8 % 4, K / 16], m![K % 16]>()
        .collect::<m![N % 8, N / 8 % 4, K / 16], m![K % 16]>()
        .to_trf();

    let y: DmTensor<bf16, Chip, Cluster, Slice, m![M, N % 32]> = device
        .main
        .begin(x.view())
        .fetch::<m![M, K / 16], m![K % 16]>()
        .collect::<m![M, K / 16], m![K % 16]>()
        .contract_outer::<m![M, K / 32, N / 8 % 4], m![K % 32], _, _, _>(&w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![M, N / 8 % 4]>()
        .contract_lane::<m![M, N / 8 % 4], m![N % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![N % 8 # 16]>()
        .commit_trim::<m![N % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

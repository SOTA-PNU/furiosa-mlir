//! harness/cases/embedding.py: out[t] = table[ids[t]] for T i32 token ids and a bf16 table [V, H].
//!
//! Mapping a compiler derives: the Triton kernel's data-dependent `table_ptr + ids[t] * H` load is a
//! DMA gather with V as the indexed axis and H as the payload. The ids are positions, so they go to
//! SPM unchanged (`by_positions`): no id-to-byte-offset arithmetic on the device or the host.

use furiosa_opt_std::prelude::*;

axes![T = 8, V = 1024, H = 1024];

pub type Chip = m![1];
type Cluster = m![1 # 2];

#[device(chip = 1)]
pub fn embedding_kernel(
    device: &mut Device,
    ids: &HbmTensor<i32, Chip, m![T]>,
    table: &HbmTensor<bf16, Chip, m![V, H]>,
) -> HbmTensor<bf16, Chip, m![T, H]> {
    let ids: DmTensor<i32, Chip, Cluster, m![1 # 256], m![T]> = ids.to_dm(&mut device.tdma);
    let ids: SpmTensor<i32, Chip, Cluster, m![1 # 4], m![T]> = ids.to_spm(&mut device.tdma);
    let out: DmTensor<bf16, Chip, Cluster, m![T, 1 # 32], m![H]> = table
        .gather::<m![V], m![H]>()
        .by_positions::<m![T]>(&ids)
        .to_dm(&mut device.tdma);
    out.to_hbm(&mut device.tdma)
}

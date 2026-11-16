//! harness/cases/vector_add.py: out = x + y over N fp32 elements.
//!
//! Mapping a compiler derives: the one axis N is split so that a 32-byte flit (8 x f32) is the
//! packet and the rest spreads over the 256 slices of cluster 0. The two operands stream through one
//! pass interleaved (`I`), and the Vector Engine's clip stage adds the pair.

use furiosa_opt_std::prelude::*;

axes![N = 2048, I = 2];

pub type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![N / 8];

#[device(chip = 1)]
pub fn vector_add_kernel(
    device: &mut Device,
    x: &HbmTensor<f32, Chip, m![N]>,
    y: &HbmTensor<f32, Chip, m![N]>,
) -> HbmTensor<f32, Chip, m![N]> {
    let x: DmTensor<f32, Chip, Cluster, Slice, m![N % 8]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, Chip, Cluster, Slice, m![N % 8]> = y.to_dm(&mut device.tdma);
    let out: DmTensor<f32, Chip, Cluster, Slice, m![N % 8]> = device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(x.view(), y.view())
        .fetch::<m![I], m![N % 8]>()
        .collect::<m![I], m![N % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![N % 8]>()
        .commit();
    out.to_hbm(&mut device.tdma)
}

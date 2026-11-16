//! One kernel per Vector Engine f32 unary function: y = f(x) over R x H values, rows on slices.
#![allow(clippy::type_complexity)] // Mapping expressions.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

axes![R = 256, H = 1024];

pub type Values = HbmTensor<f32, m![1], m![R, H]>;

#[device(chip = 1)]
pub fn ve_exp(device: &mut Device, x: &Values) -> Values {
    let x_dm: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn ve_log(device: &mut Device, x: &Values) -> Values {
    let x_dm: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_unary(FpUnaryOp::Log)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn ve_sqrt(device: &mut Device, x: &Values) -> Values {
    let x_dm: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn ve_sin(device: &mut Device, x: &Values) -> Values {
    let x_dm: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_unary(FpUnaryOp::Sin)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn ve_cos(device: &mut Device, x: &Values) -> Values {
    let x_dm: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_unary(FpUnaryOp::Cos)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn ve_tanh(device: &mut Device, x: &Values) -> Values {
    let x_dm: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_unary(FpUnaryOp::Tanh)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn ve_sigmoid(device: &mut Device, x: &Values) -> Values {
    let x_dm: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

#[device(chip = 1)]
pub fn ve_erf(device: &mut Device, x: &Values) -> Values {
    let x_dm: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = x.to_dm(&mut device.tdma);
    let y: DmTensor<f32, m![1], m![1 # 2], m![R], m![H]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.to_hbm(&mut device.tdma)
}

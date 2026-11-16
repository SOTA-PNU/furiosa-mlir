//! Measures each Vector Engine f32 unary function against an f64 reference over fixed input ranges and
//! prints one JSON line per (function, range): the largest error in f32 ulps, relative and absolute.
//!
//!     cargo furiosa-opt build --release --bin characterize && FURIOSA_VISIBLE_CHIPS=3 <exe>

use characterize::*;
use furiosa_opt_std::prelude::*;

axes![SR = 256, SH = 1024];

const N: usize = 256 * 1024;

/// A range of inputs: `n` evenly spaced points in [lo, hi], or log-spaced when `log`.
struct Range {
    lo: f64,
    hi: f64,
    log: bool,
}

const fn lin(lo: f64, hi: f64) -> Range {
    Range { lo, hi, log: false }
}

const fn log(lo: f64, hi: f64) -> Range {
    Range { lo, hi, log: true }
}

fn samples(ranges: &[Range]) -> Vec<f32> {
    let per = N / ranges.len();
    let mut v = Vec::with_capacity(N);
    for (i, r) in ranges.iter().enumerate() {
        let n = if i + 1 == ranges.len() { N - per * i } else { per };
        for k in 0..n {
            let t = k as f64 / (n - 1).max(1) as f64;
            let x = if r.log { (r.lo.ln() + t * (r.hi.ln() - r.lo.ln())).exp() } else { r.lo + t * (r.hi - r.lo) };
            v.push(x as f32);
        }
    }
    v
}

/// The spacing of f32 values at |x|.
fn ulp(x: f64) -> f64 {
    let f = (x as f32).abs();
    if f == 0.0 || !f.is_finite() {
        return f64::from(f32::from_bits(1));
    }
    f64::from(f32::from_bits(f.to_bits() + 1) - f)
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let mut device = Device::new(Topology { chips: 1, pes: 8 })?;
    let pi = std::f64::consts::PI;
    let functions: Vec<(&str, Vec<Range>, fn(f64) -> f64)> = vec![
        ("exp", vec![lin(-87.0, 0.0), lin(0.0, 88.0), lin(-1.0, 1.0)], f64::exp),
        ("log", vec![log(1e-30, 1.0), log(1.0, 1e30)], f64::ln),
        ("sqrt", vec![log(1e-30, 1.0), log(1.0, 1e30)], f64::sqrt),
        ("sin", vec![lin(-pi, pi), lin(-100.0, 100.0), lin(-1e4, 1e4)], f64::sin),
        ("cos", vec![lin(-pi, pi), lin(-100.0, 100.0), lin(-1e4, 1e4)], f64::cos),
        ("tanh", vec![lin(-1.0, 1.0), lin(-10.0, 10.0)], f64::tanh),
        ("sigmoid", vec![lin(-1.0, 1.0), lin(-30.0, 30.0)], |x| 1.0 / (1.0 + (-x).exp())),
        ("erf", vec![lin(-1.0, 1.0), lin(-6.0, 6.0)], libm::erf),
    ];
    for (name, ranges, f) in functions {
        let xs = samples(&ranges);
        let x = HostTensor::<f32, m![SR, SH]>::from_vec(xs.clone())
            .to_hbm::<m![1], m![SR, SH]>(&mut device.pdma)
            .await?
            .into_buffer();
        let x = Values::try_from(x.alias(..)?)?;
        let y = match name {
            "exp" => launch(ve_exp, (&mut device, &x)).await?,
            "log" => launch(ve_log, (&mut device, &x)).await?,
            "sqrt" => launch(ve_sqrt, (&mut device, &x)).await?,
            "sin" => launch(ve_sin, (&mut device, &x)).await?,
            "cos" => launch(ve_cos, (&mut device, &x)).await?,
            "tanh" => launch(ve_tanh, (&mut device, &x)).await?,
            "sigmoid" => launch(ve_sigmoid, (&mut device, &x)).await?,
            "erf" => launch(ve_erf, (&mut device, &x)).await?,
            _ => unreachable!(),
        }
        .into_buffer();
        let ys = HbmTensor::<f32, m![1], m![SR, SH]>::try_from(y.alias(..)?)?
            .to_host::<m![SR, SH]>(&mut device.pdma)
            .await?
            .into_vec();
        let per = N / ranges.len();
        for (i, r) in ranges.iter().enumerate() {
            let end = if i + 1 == ranges.len() { N } else { per * (i + 1) };
            let (mut ulps, mut rel, mut abs, mut at) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
            let (mut nans, mut nan_from) = (0usize, f32::INFINITY);
            for k in per * i..end {
                let want = f(f64::from(xs[k]));
                let got = f64::from(ys[k]);
                let err = (got - want).abs();
                if !want.is_finite() || (want as f32).is_infinite() {
                    continue;
                }
                if !got.is_finite() {
                    nans += 1;
                    nan_from = nan_from.min(xs[k].abs());
                    continue;
                }
                let u = err / ulp(want);
                if u > ulps {
                    ulps = u;
                    at = xs[k];
                }
                if want != 0.0 {
                    rel = rel.max(err / want.abs());
                }
                abs = abs.max(err);
            }
            println!(
                "{{\"function\": \"{name}\", \"range\": [{:e}, {:e}], \"spacing\": \"{}\", \"max_ulp\": {ulps:.3}, \"max_rel\": {rel:.3e}, \"max_abs\": {abs:.3e}, \"worst_x\": {at:e}, \"non_finite\": {nans}, \"non_finite_from_abs_x\": {nan_from:e}}}",
                r.lo,
                r.hi,
                if r.log { "log" } else { "linear" }
            );
        }
    }
    Ok(())
}

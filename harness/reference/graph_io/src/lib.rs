//! The file side of the harness crate contract (harness/README.md):
//!
//! ```text
//! run_graph <graph-index> <inputs.safetensors> <outputs.safetensors>
//! ```
//!
//! Tensors are named by graph buffer and stored in the graph's dtype and row-major shape.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use eyre::{Context, ensure, eyre};
use furiosa_opt_std::prelude::*;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

/// `run_graph`'s arguments: the graph index and the input and output file paths.
pub fn args() -> eyre::Result<(usize, PathBuf, PathBuf)> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: {} <graph-index> <inputs.safetensors> <outputs.safetensors>",
        args[0]
    );
    let graph = args[1].parse().wrap_err_with(|| format!("graph index {:?}", args[1]))?;
    Ok((graph, args[2].clone().into(), args[3].clone().into()))
}

/// The safetensors dtype a host element type is stored as.
pub trait Stored: Copy {
    const DTYPE: Dtype;
}

impl Stored for f32 {
    const DTYPE: Dtype = Dtype::F32;
}

impl Stored for bf16 {
    const DTYPE: Dtype = Dtype::BF16;
}

impl Stored for i32 {
    const DTYPE: Dtype = Dtype::I32;
}

/// A graph's inputs file: every input, weight and state buffer by name.
pub struct Inputs(Vec<u8>);

impl Inputs {
    pub fn load(path: &Path) -> eyre::Result<Self> {
        Ok(Self(
            std::fs::read(path).wrap_err_with(|| format!("reading {}", path.display()))?,
        ))
    }

    /// The buffer `name` as a host tensor; its stored shape and dtype must be `E`'s and `D`'s.
    pub fn host<D: Stored + ScalarBytes + MaterializableScalar, E: M>(
        &self,
        name: &str,
    ) -> eyre::Result<HostTensor<D, E>> {
        let file = SafeTensors::deserialize(&self.0)?;
        let view = file.tensor(name).wrap_err_with(|| format!("buffer {name}"))?;
        ensure!(
            view.dtype() == D::DTYPE,
            "buffer {name} is {:?}, expected {:?}",
            view.dtype(),
            D::DTYPE
        );
        HostTensor::from_safetensors(&view).map_err(|e| eyre!("buffer {name}: {e:?}"))
    }
}

/// A graph's outputs file: every output and state buffer by name.
#[derive(Default)]
pub struct Outputs(HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>);

impl Outputs {
    /// Stores `tensor` as buffer `name` with the graph's row-major `shape`.
    pub fn put<D: Stored + MaterializableScalar, E: M>(
        &mut self,
        name: &str,
        shape: &[usize],
        tensor: HostTensor<D, E>,
    ) {
        let bytes: Vec<u8> = tensor.into_buffer().into();
        assert_eq!(
            bytes.len(),
            shape.iter().product::<usize>() * size_of::<D>(),
            "{name}: shape {shape:?}"
        );
        self.0.insert(name.to_owned(), (D::DTYPE, shape.to_vec(), bytes));
    }

    pub fn save(&self, path: &Path) -> eyre::Result<()> {
        let views = self
            .0
            .iter()
            .map(|(name, (dtype, shape, bytes))| Ok((name.clone(), TensorView::new(*dtype, shape.clone(), bytes)?)))
            .collect::<eyre::Result<Vec<_>>>()?;
        safetensors::serialize_to_file(views, &None, path).wrap_err_with(|| format!("writing {}", path.display()))
    }
}

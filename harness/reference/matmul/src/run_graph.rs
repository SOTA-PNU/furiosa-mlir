//! The harness crate contract for matmul (harness/README.md).

use furiosa_opt_std::prelude::*;
use graph_io::{Inputs, Outputs};
use matmul::kernel::matmul_kernel::{K, M, N, matmul_kernel};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let (graph, inputs, outputs) = graph_io::args()?;
    eyre::ensure!(graph == 0, "matmul has one graph, got index {graph}");
    let inputs = Inputs::load(&inputs)?;
    let mut device = Device::new(matmul_kernel.topology())?;

    let x = inputs.host::<bf16, m![M, K]>("x")?.to_hbm(&mut device.pdma).await?;
    let w = inputs
        .host::<bf16, m![N, K]>("weight")?
        .to_hbm(&mut device.pdma)
        .await?;
    let y = launch(matmul_kernel, (&mut device, &x, &w)).await?;

    let mut files = Outputs::default();
    files.put("y", &[M::SIZE, N::SIZE], y.to_host::<m![M, N]>(&mut device.pdma).await?);
    files.save(&outputs)
}

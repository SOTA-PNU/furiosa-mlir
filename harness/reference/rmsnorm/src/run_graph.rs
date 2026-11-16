//! The harness crate contract for rmsnorm (harness/README.md).

use furiosa_opt_std::prelude::*;
use graph_io::{Inputs, Outputs};
use rmsnorm::kernel::rmsnorm_kernel::{H, R, rmsnorm_kernel};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let (graph, inputs, outputs) = graph_io::args()?;
    eyre::ensure!(graph == 0, "rmsnorm has one graph, got index {graph}");
    let inputs = Inputs::load(&inputs)?;
    let mut device = Device::new(rmsnorm_kernel.topology())?;

    let x = inputs.host::<bf16, m![R, H]>("x")?.to_hbm(&mut device.pdma).await?;
    let weight = inputs.host::<bf16, m![H]>("weight")?.to_hbm(&mut device.pdma).await?;
    let y = launch(rmsnorm_kernel, (&mut device, &x, &weight)).await?;

    let mut files = Outputs::default();
    files.put("y", &[R::SIZE, H::SIZE], y.to_host::<m![R, H]>(&mut device.pdma).await?);
    files.save(&outputs)
}

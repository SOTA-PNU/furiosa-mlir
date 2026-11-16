//! The harness crate contract for vector_add (harness/README.md).

use furiosa_opt_std::prelude::*;
use graph_io::{Inputs, Outputs};
use vector_add::kernel::vector_add_kernel::{N, vector_add_kernel};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let (graph, inputs, outputs) = graph_io::args()?;
    eyre::ensure!(graph == 0, "vector_add has one graph, got index {graph}");
    let inputs = Inputs::load(&inputs)?;
    let mut device = Device::new(vector_add_kernel.topology())?;

    let x = inputs.host::<f32, m![N]>("x")?.to_hbm(&mut device.pdma).await?;
    let y = inputs.host::<f32, m![N]>("y")?.to_hbm(&mut device.pdma).await?;
    let out = launch(vector_add_kernel, (&mut device, &x, &y)).await?;

    let mut files = Outputs::default();
    files.put("out", &[N::SIZE], out.to_host::<m![N]>(&mut device.pdma).await?);
    files.save(&outputs)
}

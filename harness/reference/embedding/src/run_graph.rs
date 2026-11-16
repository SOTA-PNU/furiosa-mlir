//! The harness crate contract for embedding (harness/README.md).

use embedding::kernel::embedding_kernel::{H, T, V, embedding_kernel};
use furiosa_opt_std::prelude::*;
use graph_io::{Inputs, Outputs};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let (graph, inputs, outputs) = graph_io::args()?;
    eyre::ensure!(graph == 0, "embedding has one graph, got index {graph}");
    let inputs = Inputs::load(&inputs)?;
    let mut device = Device::new(embedding_kernel.topology())?;

    let ids = inputs.host::<i32, m![T]>("ids")?.to_hbm(&mut device.pdma).await?;
    let table = inputs.host::<bf16, m![V, H]>("table")?.to_hbm(&mut device.pdma).await?;
    let out = launch(embedding_kernel, (&mut device, &ids, &table)).await?;

    let mut files = Outputs::default();
    files.put(
        "out",
        &[T::SIZE, H::SIZE],
        out.to_host::<m![T, H]>(&mut device.pdma).await?,
    );
    files.save(&outputs)
}

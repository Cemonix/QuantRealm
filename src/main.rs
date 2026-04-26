mod quant;
mod safetensor;

use crate::quant::{find_max_abs_par, find_max_abs_seq};
use crate::safetensor::{HeaderEntry, SafeTensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "data/model.safetensors";
    let st = SafeTensor::load_from_file(model_path)?;

    let (name, weights) = st
        .get_header()
        .iter()
        .filter_map(|(name, entry)| {
            if let HeaderEntry::Tensor(_) = entry {
                st.get_tensor::<f32>(name).ok().map(|w| (name, w))
            } else {
                None
            }
        })
        .max_by_key(|(_, w)| w.len())
        .expect("No tensors found in file");

    println!("Testing on tensor: {} (Elements: {})", name, weights.len());

    let start_seq = Instant::now();
    let max_seq = find_max_abs_seq(weights);
    let duration_seq = start_seq.elapsed();
    println!(
        "Sequential: MaxAbs = {:.6}, Time = {:?}",
        max_seq, duration_seq
    );

    let start_par = Instant::now();
    let max_par = find_max_abs_par(weights);
    let duration_par = start_par.elapsed();
    println!(
        "Parallel:   MaxAbs = {:.6}, Time = {:?}",
        max_par, duration_par
    );

    let speedup = duration_seq.as_secs_f64() / duration_par.as_secs_f64();
    println!("Speedup:    {:.2}x", speedup);

    Ok(())
}

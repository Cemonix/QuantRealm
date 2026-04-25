use rayon::prelude::*;

use crate::safetensor::Weights;

pub fn find_max_abs_seq(weights: &Weights) -> f32 {
    weights.iter().map(|x| x.abs()).fold(0.0, |a, b| a.max(b))
}

pub fn find_max_abs_par(weights: &Weights) -> f32 {
    weights
        .par_iter()
        .map(|x| x.abs())
        .reduce(|| 0.0, |a, b| a.max(b))
}

pub fn quantize(weights: &Weights) -> Vec<i8> {
    let max_abs = find_max_abs_par(weights);

    if max_abs == 0.0 {
        return vec![0i8; weights.len()];
    }

    let invert_scale = 127.0 / max_abs;
    weights
        .par_iter()
        .map(|x| (x * invert_scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

pub fn dequantize(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.par_iter().map(|&x| x as f32 * scale).collect()
}

use rayon::prelude::*;

use crate::safetensor::QuantizationSource;

pub fn find_max_abs_seq<T: QuantizationSource>(weights: &[T]) -> f32 {
    weights
        .iter()
        .map(|x| x.to_f32().abs())
        .fold(0.0, |a, b| a.max(b))
}

pub fn find_max_abs_par<T: QuantizationSource>(weights: &[T]) -> f32 {
    weights
        .par_iter()
        .map(|x| x.to_f32().abs())
        .reduce(|| 0.0, |a, b| a.max(b))
}

pub fn quantize<T: QuantizationSource>(weights: &[T]) -> Vec<i8> {
    let max_abs = find_max_abs_par(weights);

    if max_abs == 0.0 {
        return vec![0i8; weights.len()];
    }

    let invert_scale = 127.0 / max_abs;
    weights
        .par_iter()
        .map(|x| (x.to_f32() * invert_scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

pub fn dequantize<T: QuantizationSource>(quantized: &[i8], scale: f32) -> Vec<T> {
    quantized
        .par_iter()
        .map(|&x| {
            let val_f32 = x as f32 * scale;
            T::from_f32(val_f32)
        })
        .collect()
}

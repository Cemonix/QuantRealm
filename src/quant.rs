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

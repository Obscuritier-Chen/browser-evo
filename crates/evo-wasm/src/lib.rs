use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn normal(seed: u64, index: u64) -> f32 {
    evo_core::normal::normal(seed, index)
}

#[wasm_bindgen]
pub fn fill_perturbation(seed: u64, out: &mut [f32]) {
    evo_core::perturb::fill_perturbation(seed, out);
}

#[wasm_bindgen]
pub fn centered_ranks(fitness: &[f32], out: &mut [f32]) {
    let mut order = vec![0usize; fitness.len()];
    evo_core::rank::centered_ranks(fitness, &mut order, out);
}

#[wasm_bindgen]
pub fn es_gradient(seeds: &[u64], weights: &[f32], population: usize, sigma: f32, grad: &mut [f32]) {
    let pairs: Vec<(u64, f32)> = seeds.iter().copied().zip(weights.iter().copied()).collect();
    evo_core::optim::es_gradient(&pairs, population, sigma, grad);
}

#[wasm_bindgen]
pub fn sgd_step(theta: &mut [f32], grad: &[f32], lr: f32) {
    evo_core::optim::sgd_step(theta, grad, lr);
}

#[wasm_bindgen]
pub fn adam_step(theta: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32], step: u32, lr: f32) {
    let config = evo_core::optim::AdamConfig { lr, ..Default::default() };
    evo_core::optim::adam_step(theta, grad, m, v, step, &config);
}

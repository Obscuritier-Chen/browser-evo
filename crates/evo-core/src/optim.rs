use crate::normal::normal;

pub fn es_gradient(pairs: &[(u64, f32)], population: usize, sigma: f32, grad: &mut [f32]) {
    let scale = 1.0 / (population as f32 * sigma);
    for (index, value) in grad.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for &(seed, weight) in pairs {
            acc += weight * normal(seed, index as u64);
        }
        *value = acc * scale;
    }
}

pub fn sgd_step(theta: &mut [f32], grad: &[f32], lr: f32) {
    for (param, g) in theta.iter_mut().zip(grad) {
        *param -= lr * g;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

pub fn adam_step(
    theta: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    step: u32,
    config: &AdamConfig,
) {
    let t = step as f32;
    let bias1 = 1.0 - libm::powf(config.beta1, t);
    let bias2 = 1.0 - libm::powf(config.beta2, t);
    for i in 0..theta.len() {
        let g = grad[i];
        m[i] = config.beta1 * m[i] + (1.0 - config.beta1) * g;
        v[i] = config.beta2 * v[i] + (1.0 - config.beta2) * g * g;
        let m_hat = m[i] / bias1;
        let v_hat = v[i] / bias2;
        theta[i] -= config.lr * config.weight_decay * theta[i];
        theta[i] -= config.lr * m_hat / (libm::sqrtf(v_hat) + config.eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perturb::fill_perturbation;
    use crate::rank::centered_ranks;

    #[test]
    fn gradient_of_single_pair_matches_perturbation() {
        let mut grad = [0.0f32; 4];
        es_gradient(&[(5, 1.0)], 2, 0.5, &mut grad);
        let mut eps = [0.0f32; 4];
        fill_perturbation(5, &mut eps);
        for i in 0..4 {
            assert!((grad[i] - eps[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_weights_give_zero_gradient() {
        let mut grad = [1.0f32; 4];
        es_gradient(&[(5, 0.0), (6, 0.0)], 4, 0.1, &mut grad);
        assert_eq!(grad, [0.0; 4]);
    }

    #[test]
    fn sgd_moves_against_gradient() {
        let mut theta = [1.0f32, -1.0];
        sgd_step(&mut theta, &[0.5, -0.5], 0.1);
        assert!((theta[0] - 0.95).abs() < 1e-6);
        assert!((theta[1] + 0.95).abs() < 1e-6);
    }

    #[test]
    fn adam_first_step_is_about_lr() {
        let mut theta = [1.0f32];
        let mut m = [0.0f32];
        let mut v = [0.0f32];
        let config = AdamConfig { lr: 0.1, ..AdamConfig::default() };
        adam_step(&mut theta, &[2.0], &mut m, &mut v, 1, &config);
        assert!((theta[0] - 0.9).abs() < 1e-4);
    }

    #[test]
    fn es_maximizes_sphere_fitness() {
        const DIM: usize = 10;
        const PAIRS: usize = 20;
        let mut theta = [1.0f32; DIM];
        let sphere = |x: &[f32]| -> f32 { -x.iter().map(|v| v * v).sum::<f32>() };
        let initial = sphere(&theta);
        let sigma = 0.1f32;
        let lr = 0.05f32;
        for generation in 0..300u64 {
            let mut fitness = [0.0f32; 2 * PAIRS];
            let mut seeds = [0u64; PAIRS];
            let mut eps = [0.0f32; DIM];
            for j in 0..PAIRS {
                let seed = (generation << 32) | j as u64;
                seeds[j] = seed;
                fill_perturbation(seed, &mut eps);
                let mut plus = theta;
                let mut minus = theta;
                for i in 0..DIM {
                    plus[i] += sigma * eps[i];
                    minus[i] -= sigma * eps[i];
                }
                fitness[2 * j] = sphere(&plus);
                fitness[2 * j + 1] = sphere(&minus);
            }
            let mut order = [0usize; 2 * PAIRS];
            let mut ranks = [0.0f32; 2 * PAIRS];
            centered_ranks(&fitness, &mut order, &mut ranks);
            let mut pairs = [(0u64, 0.0f32); PAIRS];
            for j in 0..PAIRS {
                pairs[j] = (seeds[j], ranks[2 * j] - ranks[2 * j + 1]);
            }
            let mut grad = [0.0f32; DIM];
            es_gradient(&pairs, 2 * PAIRS, sigma, &mut grad);
            sgd_step(&mut theta, &grad, -lr);
        }
        assert!(sphere(&theta) > initial);
        assert!(sphere(&theta) > -0.5, "fitness = {}", sphere(&theta));
    }
}


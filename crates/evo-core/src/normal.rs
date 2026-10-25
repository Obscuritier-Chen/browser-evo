use crate::prng::block;

const SCALE: f32 = 1.0 / 16_777_216.0;

pub fn normal(seed: u64, index: u64) -> f32 {
    let [w0, w1, _, _] = block(seed, index);
    let u1 = ((w0 >> 8) as f32 + 1.0) * SCALE;
    let u2 = (w1 >> 8) as f32 * SCALE;
    let radius = libm::sqrtf(-2.0 * libm::logf(u1));
    let angle = core::f32::consts::TAU * u2;
    radius * libm::cosf(angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic() {
        assert_eq!(normal(42, 7), normal(42, 7));
    }

    #[test]
    fn is_finite() {
        for index in 0..10_000 {
            assert!(normal(1, index).is_finite());
        }
    }

    #[test]
    fn moments_are_close_to_standard_normal() {
        let count = 200_000u64;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for index in 0..count {
            let value = normal(123, index) as f64;
            sum += value;
            sum_sq += value * value;
        }
        let mean = sum / count as f64;
        let variance = sum_sq / count as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean = {mean}");
        assert!((variance - 1.0).abs() < 0.02, "variance = {variance}");
    }
}

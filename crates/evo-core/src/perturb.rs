use crate::normal::normal;

pub fn fill_perturbation(seed: u64, out: &mut [f32]) {
    for (index, value) in out.iter_mut().enumerate() {
        *value = normal(seed, index as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_normal_stream() {
        let mut out = [0.0f32; 16];
        fill_perturbation(9, &mut out);
        for (index, value) in out.iter().enumerate() {
            assert_eq!(*value, normal(9, index as u64));
        }
    }

    #[test]
    fn is_deterministic_and_seed_sensitive() {
        let mut first = [0.0f32; 8];
        let mut again = [0.0f32; 8];
        let mut other = [0.0f32; 8];
        fill_perturbation(3, &mut first);
        fill_perturbation(3, &mut again);
        fill_perturbation(4, &mut other);
        assert_eq!(first, again);
        assert_ne!(first, other);
    }
}

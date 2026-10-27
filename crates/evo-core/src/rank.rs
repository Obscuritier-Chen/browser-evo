pub fn centered_ranks(fitness: &[f32], order: &mut [usize], out: &mut [f32]) {
    let count = fitness.len();
    assert_eq!(order.len(), count);
    assert_eq!(out.len(), count);
    if count == 1 {
        out[0] = 0.0;
        return;
    }
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index;
    }
    order.sort_unstable_by(|&a, &b| fitness[a].total_cmp(&fitness[b]).then(a.cmp(&b)));
    let denom = (count - 1) as f32;
    for (rank, &index) in order.iter().enumerate() {
        out[index] = rank as f32 / denom - 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_are_centered_and_ordered() {
        let fitness = [3.0, 1.0, 2.0];
        let mut order = [0usize; 3];
        let mut out = [0.0f32; 3];
        centered_ranks(&fitness, &mut order, &mut out);
        assert_eq!(out, [0.5, -0.5, 0.0]);
    }

    #[test]
    fn ties_are_broken_by_index() {
        let fitness = [1.0, 1.0, 1.0];
        let mut order = [0usize; 3];
        let mut out = [0.0f32; 3];
        centered_ranks(&fitness, &mut order, &mut out);
        assert_eq!(out, [-0.5, 0.0, 0.5]);
    }

    #[test]
    fn single_value_maps_to_zero() {
        let mut order = [0usize; 1];
        let mut out = [1.0f32];
        centered_ranks(&[7.0], &mut order, &mut out);
        assert_eq!(out, [0.0]);
    }

    #[test]
    fn matches_quadratic_reference_on_shuffled_input() {
        let fitness = [0.3, -1.2, 5.0, 0.3, 2.5, -0.7, 0.0, 9.1];
        let count = fitness.len();
        let mut order = [0usize; 8];
        let mut out = [0.0f32; 8];
        centered_ranks(&fitness, &mut order, &mut out);
        for index in 0..count {
            let mut rank = 0usize;
            for other in 0..count {
                let before = fitness[other].total_cmp(&fitness[index]).is_lt()
                    || (fitness[other] == fitness[index] && other < index);
                if before {
                    rank += 1;
                }
            }
            let expected = rank as f32 / (count - 1) as f32 - 0.5;
            assert_eq!(out[index], expected);
        }
    }
}

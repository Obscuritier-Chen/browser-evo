use evo_core::normal::normal;

const FIXTURE: &str = include_str!("golden/normal_bits.txt");

#[test]
fn normal_matches_golden_bits() {
    let mut checked = 0usize;
    for line in FIXTURE.lines() {
        let mut parts = line.split_whitespace();
        let seed = u64::from_str_radix(parts.next().unwrap(), 16).unwrap();
        let index = u64::from_str_radix(parts.next().unwrap(), 16).unwrap();
        let bits = u32::from_str_radix(parts.next().unwrap(), 16).unwrap();
        assert_eq!(normal(seed, index).to_bits(), bits, "seed={seed} index={index}");
        checked += 1;
    }
    assert_eq!(checked, 4 * 4096);
}

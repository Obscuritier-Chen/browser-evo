use evo_core::normal::normal;
use evo_core::optim::{AdamConfig, adam_step, es_gradient};
use std::fs;
use std::path::Path;

const TEXT_FIXTURE: &str = include_str!("golden/normal_bits.txt");
const BIN_FIXTURE: &[u8] = include_bytes!("golden/normal_1e6.bin");
const ES_FIXTURE: &str = include_str!("golden/es_update.txt");

const TEXT_SEEDS: [u64; 4] = [0, 1, 42, 0xDEAD_BEEF_CAFE_BABE];
const TEXT_COUNT: u64 = 4096;
const BIN_SEED: u64 = 42;
const BIN_COUNT: u64 = 1_000_000;

const ES_DIM: usize = 16;
const ES_SEEDS: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const ES_WEIGHTS: [f32; 8] = [0.5, -0.25, 0.125, 0.0, -0.5, 0.75, -0.125, 0.25];
const ES_POPULATION: usize = 16;
const ES_SIGMA: f32 = 0.5;
const ES_LR: f32 = 0.0625;
const ES_STEPS: u32 = 3;

fn es_trajectory() -> Vec<(u32, usize, u32)> {
    let pairs: Vec<(u64, f32)> = ES_SEEDS.iter().copied().zip(ES_WEIGHTS).collect();
    let mut theta: Vec<f32> = (0..ES_DIM).map(|i| (i as f32 - 4.0) * 0.25).collect();
    let mut m = vec![0.0f32; ES_DIM];
    let mut v = vec![0.0f32; ES_DIM];
    let mut grad = vec![0.0f32; ES_DIM];
    let config = AdamConfig { lr: ES_LR, ..AdamConfig::default() };
    let mut rows = Vec::new();
    for step in 1..=ES_STEPS {
        es_gradient(&pairs, ES_POPULATION, ES_SIGMA, &mut grad);
        adam_step(&mut theta, &grad, &mut m, &mut v, step, &config);
        for (index, value) in theta.iter().enumerate() {
            rows.push((step, index, value.to_bits()));
        }
    }
    rows
}

#[test]
fn normal_matches_text_golden_bits() {
    let mut checked = 0usize;
    for line in TEXT_FIXTURE.lines() {
        let mut parts = line.split_whitespace();
        let seed = u64::from_str_radix(parts.next().unwrap(), 16).unwrap();
        let index = u64::from_str_radix(parts.next().unwrap(), 16).unwrap();
        let bits = u32::from_str_radix(parts.next().unwrap(), 16).unwrap();
        assert_eq!(normal(seed, index).to_bits(), bits, "seed={seed} index={index}");
        checked += 1;
    }
    assert_eq!(checked, TEXT_SEEDS.len() * TEXT_COUNT as usize);
}

#[test]
fn normal_matches_million_value_golden_bits() {
    assert_eq!(BIN_FIXTURE.len(), 4 * BIN_COUNT as usize);
    for (index, chunk) in BIN_FIXTURE.chunks_exact(4).enumerate() {
        let bits = u32::from_le_bytes(chunk.try_into().unwrap());
        assert_eq!(normal(BIN_SEED, index as u64).to_bits(), bits, "index={index}");
    }
}

#[test]
fn es_update_matches_golden_bits() {
    let expected: Vec<(u32, usize, u32)> = ES_FIXTURE
        .lines()
        .map(|line| {
            let mut parts = line.split_whitespace();
            let step = parts.next().unwrap().parse().unwrap();
            let index = parts.next().unwrap().parse().unwrap();
            let bits = u32::from_str_radix(parts.next().unwrap(), 16).unwrap();
            (step, index, bits)
        })
        .collect();
    assert_eq!(es_trajectory(), expected);
}

#[test]
#[ignore]
fn regenerate_fixtures() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let mut text = String::new();
    for seed in TEXT_SEEDS {
        for index in 0..TEXT_COUNT {
            text += &format!("{seed:016x} {index:08x} {:08x}\n", normal(seed, index).to_bits());
        }
    }
    fs::write(dir.join("normal_bits.txt"), text).unwrap();
    let bytes: Vec<u8> = (0..BIN_COUNT)
        .flat_map(|index| normal(BIN_SEED, index).to_bits().to_le_bytes())
        .collect();
    fs::write(dir.join("normal_1e6.bin"), bytes).unwrap();
    let mut es = String::new();
    for (step, index, bits) in es_trajectory() {
        es += &format!("{step} {index} {bits:08x}\n");
    }
    fs::write(dir.join("es_update.txt"), es).unwrap();
}

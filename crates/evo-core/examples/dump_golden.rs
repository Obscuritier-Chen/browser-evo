use evo_core::normal::normal;

const SEEDS: [u64; 4] = [0, 1, 42, 0xDEAD_BEEF_CAFE_BABE];
const COUNT: u64 = 4096;

fn main() {
    for seed in SEEDS {
        for index in 0..COUNT {
            println!("{:016x} {:08x} {:08x}", seed, index, normal(seed, index).to_bits());
        }
    }
}

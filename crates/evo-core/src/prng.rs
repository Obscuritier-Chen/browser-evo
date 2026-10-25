const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;
const ROUNDS: usize = 10;

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    ((product >> 32) as u32, product as u32)
}

#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = counter;
    let mut k = key;
    for _ in 0..ROUNDS {
        ctr = round(ctr, k);
        k[0] = k[0].wrapping_add(W0);
        k[1] = k[1].wrapping_add(W1);
    }
    ctr
}

pub fn block(seed: u64, index: u64) -> [u32; 4] {
    let key = [seed as u32, (seed >> 32) as u32];
    let counter = [index as u32, (index >> 32) as u32, 0, 0];
    philox4x32_10(counter, key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_random123_zero_vector() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn matches_random123_ones_vector() {
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
    }

    #[test]
    fn block_is_deterministic() {
        assert_eq!(block(42, 7), block(42, 7));
    }

    #[test]
    fn different_indices_differ() {
        assert_ne!(block(42, 0), block(42, 1));
        assert_ne!(block(42, 0), block(42, 1 << 32));
    }

    #[test]
    fn different_seeds_differ() {
        assert_ne!(block(1, 0), block(2, 0));
    }
}


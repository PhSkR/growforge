//! PCG32, the only source of randomness in growforge.
//!
//! M. E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically
//! Good Algorithms for Random Number Generation", 2014. The generator is a 64
//! bit linear congruential state whose output is an xorshift of the high bits
//! rotated by the top five: 32 bits of output from 64 bits of state, which is
//! what makes it well distributed without a large state.
//!
//! It is written out here rather than pulled in as a dependency for two
//! reasons. A grown structure has to be reproducible from its configuration
//! alone, so the exact stream is part of growforge's contract with the user and
//! may not move under a version bump; and nothing may seed itself from the
//! clock or the operating system's entropy, so a generator that offers such a
//! constructor is a hazard rather than a convenience.

use crate::constants;

/// A deterministic PCG32 generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    /// Stream selector; always odd, which is what makes the underlying LCG
    /// full period.
    increment: u64,
}

impl Pcg32 {
    /// Seed a generator on growforge's fixed stream.
    pub fn new(seed: u64) -> Pcg32 {
        Pcg32::with_stream(seed, constants::PCG32_STREAM)
    }

    /// Seed a generator on an explicit stream, following the reference
    /// `pcg32_srandom_r`: zero the state, install the stream, step, add the
    /// seed, step again.
    pub fn with_stream(seed: u64, stream: u64) -> Pcg32 {
        let mut rng = Pcg32 {
            state: 0,
            increment: (stream << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Draw the next 32 bits.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(constants::PCG32_MULTIPLIER)
            .wrapping_add(self.increment);
        // The output function: xorshift the high half down, then rotate by the
        // top five bits of the *old* state, so the rotation is itself random.
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rotation = (old >> 59) as u32;
        xorshifted.rotate_right(rotation)
    }

    /// Draw a uniform value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / constants::PCG32_F64_DIVISOR
    }

    /// Draw a uniform value in `[low, high)`.
    pub fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference stream of O'Neill's `pcg32-demo` for
    /// `pcg32_srandom_r(&rng, 42u, 54u)`. Nothing about the generator may drift:
    /// the same configuration has to grow the same structure in every future
    /// version, and this vector is what says so.
    const PCG32_SEED_42_STREAM_54: [u32; 6] = [
        0xa15c_02b7,
        0x7b47_f409,
        0xba1d_3330,
        0x83d2_f293,
        0xbfa4_784b,
        0xcbed_606e,
    ];

    #[test]
    fn the_generator_reproduces_the_published_reference_stream() {
        let mut rng = Pcg32::with_stream(42, 54);
        for (index, expected) in PCG32_SEED_42_STREAM_54.iter().enumerate() {
            let got = rng.next_u32();
            assert_eq!(
                got, *expected,
                "draw {index} is {got:#010x}, not {expected:#010x}"
            );
        }
    }

    #[test]
    fn growforges_own_stream_is_the_reference_one() {
        // The shipped stream selector is the reference demo's, so the vector
        // above also pins the generator the engine actually uses.
        assert_eq!(constants::PCG32_STREAM, 54);
        let mut rng = Pcg32::new(42);
        assert_eq!(rng.next_u32(), PCG32_SEED_42_STREAM_54[0]);
    }

    #[test]
    fn draws_are_reproducible_and_seed_dependent() {
        let draws = |seed| {
            let mut rng = Pcg32::new(seed);
            (0..64).map(|_| rng.next_u32()).collect::<Vec<u32>>()
        };
        assert_eq!(draws(7), draws(7), "the same seed must replay exactly");
        assert_ne!(draws(7), draws(8));
    }

    #[test]
    fn floating_point_draws_stay_in_the_unit_interval() {
        let mut rng = Pcg32::new(constants::DEFAULT_GROWTH_SEED);
        let mut sum = 0.0;
        let count = 100_000;
        for _ in 0..count {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value), "draw {value} left [0, 1)");
            sum += value;
        }
        // A crude uniformity check: the mean of a uniform draw is 1/2, and
        // 100k samples put the standard error near 0.001.
        let mean = sum / count as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean of the draws is {mean}");

        for _ in 0..1000 {
            let value = rng.range(-3.0, 5.0);
            assert!((-3.0..5.0).contains(&value), "ranged draw {value} escaped");
        }
        // A degenerate range collapses instead of producing a NaN.
        assert_eq!(rng.range(2.0, 2.0), 2.0);
    }
}

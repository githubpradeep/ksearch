//! CPU logit sampling matching metal-llm-server (`cap * tanh(x/cap)`, then temperature + min-p).

use ksearch_gguf::f16_to_f32;

/// Softcap + temperature + min-p, then multinomial sample.
///
/// `temperature < 1e-6` is greedy argmax after softcap (same as the GPU kernel).
/// `min_p` keeps tokens with `p >= min_p * p_max` (oracle default 0.05).
pub fn sample_softcap_min_p(
    logits_f16: &[u16],
    cap: f32,
    temperature: f32,
    min_p: f32,
    seed: u32,
) -> u32 {
    let n = logits_f16.len();
    debug_assert!(n > 0);
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    let mut xs = vec![0.0f32; n];
    for (i, &bits) in logits_f16.iter().enumerate() {
        let x = f16_to_f32(bits);
        let y = if cap > 0.0 {
            cap * (x / cap).tanh()
        } else {
            x
        };
        xs[i] = y;
        if y > best_v {
            best_v = y;
            best_i = i as u32;
        }
    }
    if temperature < 1e-6 {
        return best_i;
    }

    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = ((*x - best_v) / temperature).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(1e-30);
    let mut p_max = 0.0f32;
    for x in xs.iter_mut() {
        *x *= inv;
        if *x > p_max {
            p_max = *x;
        }
    }
    let threshold = p_max * min_p.max(0.0);
    let mut filtered = 0.0f32;
    for x in xs.iter_mut() {
        if *x < threshold {
            *x = 0.0;
        } else {
            filtered += *x;
        }
    }
    if filtered <= 1e-9 {
        return best_i;
    }
    let mut rng = seed | 1;
    let r = (splitmix32(&mut rng) as f32 / 4294967296.0) * filtered;
    let mut cdf = 0.0f32;
    for (i, &p) in xs.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i as u32;
        }
    }
    (n - 1) as u32
}

fn splitmix32(state: &mut u32) -> u32 {
    *state = state.wrapping_add(0x9E3779B9);
    let mut z = *state;
    z = (z ^ (z >> 16)).wrapping_mul(0x7feb352d);
    z = (z ^ (z >> 15)).wrapping_mul(0x846ca68b);
    z ^ (z >> 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ksearch_gguf::f32_to_f16;

    fn pack(v: &[f32]) -> Vec<u16> {
        v.iter().copied().map(f32_to_f16).collect()
    }

    #[test]
    fn greedy_picks_argmax_after_softcap() {
        let logits = pack(&[1.0, 9.0, 3.0]);
        assert_eq!(sample_softcap_min_p(&logits, 30.0, 0.0, 0.05, 1), 1);
    }

    #[test]
    fn min_p_keeps_only_the_peak_when_peaked() {
        let logits = pack(&[0.0, 20.0, 0.0]);
        let mut seen = std::collections::HashSet::new();
        for seed in 1..64u32 {
            seen.insert(sample_softcap_min_p(&logits, 30.0, 1.0, 0.05, seed));
        }
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn temperature_samples_near_ties_after_softcap() {
        let logits = pack(&[8.0, 8.05, 0.0]);
        let mut counts = [0u32; 3];
        for seed in 1..256u32 {
            counts[sample_softcap_min_p(&logits, 30.0, 1.0, 0.05, seed) as usize] += 1;
        }
        assert!(counts[0] > 0 && counts[1] > 0, "counts={counts:?}");
        assert_eq!(counts[2], 0);
    }
}

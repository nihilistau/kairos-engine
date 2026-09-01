//! G-SWARM-C2-INDEX (L4 mechanics) — proves the C2 similarity index ranks by Hamming correctly:
//! near sigs rank above far, exact match is distance 0, top-k is monotone. Deterministic (no model).
//! This is the INDEX MECHANICS gate; the SEMANTIC recall of C2 is measured separately (Python,
//! against the L5 similarity ground truth).

use sp_swarm::similarity::{agree, hamming, sig_from_hex, sig_to_hex, C2Index, Sig};

// deterministic splitmix64 so the gate is reproducible (no rand dep)
fn smix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
fn rand_sig(seed: u64) -> Sig { [smix(seed), smix(seed ^ 0x11), smix(seed ^ 0x22), smix(seed ^ 0x33)] }
/// flip exactly `k` bits of `s` (deterministic positions from `seed`).
fn flip(s: &Sig, k: usize, seed: u64) -> Sig {
    let mut out = *s;
    let mut used = std::collections::HashSet::new();
    let mut r = seed | 1;
    let mut done = 0;
    while done < k {
        r = smix(r);
        let bit = (r % 256) as usize;
        if used.insert(bit) {
            out[bit / 64] ^= 1u64 << (bit % 64);
            done += 1;
        }
    }
    out
}

#[test]
fn hex_roundtrip() {
    let s = rand_sig(7);
    assert_eq!(sig_from_hex(&sig_to_hex(&s)).unwrap(), s);
    assert_eq!(hamming(&s, &s), 0);
    assert_eq!(agree(&s, &s), 256);
}

#[test]
fn ranks_near_above_far() {
    let base = rand_sig(42);
    let near = flip(&base, 5, 1);    // very similar
    let mid = flip(&base, 40, 2);
    let far = flip(&base, 120, 3);   // ~random
    let mut idx = C2Index::new();
    // insert far/mid/near out of order + some noise
    idx.insert("far".into(), far);
    idx.insert("noise1".into(), rand_sig(900));
    idx.insert("mid".into(), mid);
    idx.insert("noise2".into(), rand_sig(901));
    idx.insert("near".into(), near);

    let hits = idx.find_similar(&base, 3);
    assert_eq!(hits.len(), 3);
    // nearest first, monotone non-decreasing hamming
    assert_eq!(hits[0].0, "near", "near must rank #1, got {:?}", hits);
    assert!(hits[0].1 <= hits[1].1 && hits[1].1 <= hits[2].1, "top-k not monotone: {:?}", hits);
    assert!(hits[0].1 <= 5, "near hamming should be ~5, got {}", hits[0].1);
    // near+mid must both beat any noise (they're the 2 closest deterministically)
    let names: Vec<&str> = hits.iter().map(|(a, _)| a.as_str()).collect();
    assert!(names.contains(&"near") && names.contains(&"mid"), "near+mid should be in top-3: {names:?}");
    eprintln!("[ranks] {:?}", hits);
}

#[test]
fn exact_match_is_zero() {
    let mut idx = C2Index::new();
    let a = rand_sig(5);
    idx.insert("a".into(), a);
    idx.insert("b".into(), flip(&a, 30, 9));
    let hit = idx.find_similar(&a, 1);
    assert_eq!(hit[0].0, "a");
    assert_eq!(hit[0].1, 0, "exact match must be Hamming 0");
    eprintln!("==== G-SWARM-C2-INDEX: GREEN — Hamming top-k ranking (near>far), exact=0, monotone, hex round-trip ====");
}

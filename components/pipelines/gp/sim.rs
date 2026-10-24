//! Similarity metrics and the LSH hashmap index.
//!
//! The index is the "hashmap whose key encodes similarity": every feature is
//! reduced to a 120-bit SimHash of its (rank-transformed) values; bands of the hash
//! are keys of hashmaps, so near-duplicates collide in O(1) expected time and only
//! colliding entries are verified with the exact metric. Sign-flipped features share
//! a key (canonical sign), and a second "folded" view (|rank-0.5|) makes even-symmetric
//! relationships (x vs x^2, |x|, cos x) collide too -- those are invisible to Pearson.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// |Pearson| (pairwise complete). The classic "uncorrelated" test.
    Pearson,
    /// |Spearman| on rank vectors.
    Spearman,
    /// Linfoot informational coefficient sqrt(1-exp(-2 I)) from binned mutual information:
    /// equals |rho| for Gaussians but also sees non-monotone dependence.
    Info,
    /// max(|Spearman|, Info): redundant if EITHER monotone or non-monotone information is shared.
    Mixed,
}

impl Metric {
    pub fn uses_fold(self) -> bool { matches!(self, Metric::Info | Metric::Mixed) }
    pub fn needs_raw(self) -> bool { matches!(self, Metric::Pearson) }
}

pub const NB: usize = 16; // rank bins for binned MI

/// Compact per-feature representation used for all pairwise comparisons.
#[derive(Clone)]
pub struct FeatVec {
    pub z: Vec<f32>,      // rank-standardised (mean 0, unit L2, NaN->0)
    pub bins: Vec<u8>,    // rank bin index, 255 = invalid
    pub raw: Vec<f32>,    // raw values on verify rows (only kept for Pearson)
}

/// Average ranks of finite values -> fractional rank in (0,1); NaN stays NaN.
pub fn frac_ranks(v: &[f32]) -> Vec<f32> {
    let idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
    let mut order = idx.clone();
    order.sort_unstable_by(|&a, &b| v[a].total_cmp(&v[b]));
    let m = order.len();
    let mut out = vec![f32::NAN; v.len()];
    let mut i = 0;
    while i < m {
        let mut j = i;
        while j + 1 < m && v[order[j + 1]] == v[order[i]] { j += 1; }
        let r = ((i + j) as f64 * 0.5 + 0.5) / m as f64;
        for k in i..=j { out[order[k]] = r as f32; }
        i = j + 1;
    }
    out
}

pub fn make_featvec(v: &[f32], keep_raw: bool) -> FeatVec {
    let r = frac_ranks(v);
    let m = r.iter().filter(|x| x.is_finite()).count().max(1);
    let mean = r.iter().filter(|x| x.is_finite()).map(|&x| x as f64).sum::<f64>() / m as f64;
    let mut z: Vec<f32> = r.iter().map(|&x| if x.is_finite() { (x as f64 - mean) as f32 } else { 0.0 }).collect();
    let nrm = z.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if nrm > 0.0 { for x in z.iter_mut() { *x = (*x as f64 / nrm) as f32; } }
    let bins = r.iter().map(|&x| if x.is_finite() { ((x * NB as f32) as usize).min(NB - 1) as u8 } else { 255 }).collect();
    FeatVec { z, bins, raw: if keep_raw { v.to_vec() } else { Vec::new() } }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    let mut s = [0.0f32; 8];
    let ch = a.len() / 8;
    for c in 0..ch {
        for k in 0..8 { s[k] += a[c * 8 + k] * b[c * 8 + k]; }
    }
    let mut t: f64 = s.iter().map(|&x| x as f64).sum();
    for i in ch * 8..a.len() { t += (a[i] * b[i]) as f64; }
    t
}

fn pearson_pair(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut sa, mut sb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        if a[i].is_finite() && b[i].is_finite() { n += 1.0; sa += a[i] as f64; sb += b[i] as f64; }
    }
    if n < 10.0 { return 0.0; }
    let (ma, mb) = (sa / n, sb / n);
    let (mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        if a[i].is_finite() && b[i].is_finite() {
            let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
            saa += da * da; sbb += db * db; sab += da * db;
        }
    }
    if saa <= 1e-24 * n || sbb <= 1e-24 * n { return 0.0; }
    (sab / (saa * sbb).sqrt()).abs()
}

fn info_pair(a: &[u8], b: &[u8]) -> f64 {
    let mut joint = [0u32; NB * NB];
    let mut n = 0u32;
    for i in 0..a.len() {
        if a[i] != 255 && b[i] != 255 { joint[a[i] as usize * NB + b[i] as usize] += 1; n += 1; }
    }
    if n < 50 { return 0.0; }
    let nf = n as f64;
    let mut pa = [0f64; NB];
    let mut pb = [0f64; NB];
    for i in 0..NB { for j in 0..NB { let c = joint[i * NB + j] as f64; pa[i] += c; pb[j] += c; } }
    let mut mi = 0.0;
    for i in 0..NB { for j in 0..NB {
        let c = joint[i * NB + j] as f64;
        if c > 0.0 { mi += (c / nf) * ((c * nf) / (pa[i] * pb[j])).ln(); }
    } }
    mi -= ((NB - 1) * (NB - 1)) as f64 / (2.0 * nf); // plug-in bias
    if mi <= 0.0 { return 0.0; }
    (1.0 - (-2.0 * mi).exp()).sqrt()
}

/// Similarity in [0,1]; 1 = same information under the chosen metric.
pub fn similarity(metric: Metric, a: &FeatVec, b: &FeatVec) -> f64 {
    match metric {
        Metric::Pearson => pearson_pair(&a.raw, &b.raw),
        Metric::Spearman => dot(&a.z, &b.z).abs(),
        Metric::Info => info_pair(&a.bins, &b.bins),
        Metric::Mixed => dot(&a.z, &b.z).abs().max(info_pair(&a.bins, &b.bins)),
    }
}

// ----------------------------------------------------------------------------
// LSH index
// ----------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);
impl Hasher for IdHasher {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, bytes: &[u8]) { for &b in bytes { self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3); } }
    fn write_u32(&mut self, i: u32) { self.0 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15); }
}
type Table = HashMap<u32, Vec<u32>, BuildHasherDefault<IdHasher>>;

pub struct Sigs { pub a: u128, pub b: u128 }

pub struct LshIndex {
    pub bands: usize,
    pub band_bits: usize,
    pub m_sig: usize,
    planes: Vec<f32>, // nbits x m_sig, +-1
    ta: Vec<Table>,
    tb: Vec<Table>,
    fold: bool,
    stamp: Vec<u32>,
    epoch: u32,
    pub n_verify_calls: u64,
}

impl LshIndex {
    pub fn new(bands: usize, band_bits: usize, m_sig: usize, fold: bool, seed: u64) -> Self {
        assert!(bands * band_bits <= 128 && band_bits <= 32);
        let mut s = seed ^ 0x9E3779B97F4A7C15;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let nbits = bands * band_bits;
        let mut planes = Vec::with_capacity(nbits * m_sig);
        for _ in 0..nbits * m_sig { planes.push(if next() & 1 == 1 { 1.0f32 } else { -1.0 }); }
        LshIndex { bands, band_bits, m_sig, planes,
            ta: (0..bands).map(|_| Table::default()).collect(),
            tb: (0..bands).map(|_| Table::default()).collect(),
            fold, stamp: Vec::new(), epoch: 0, n_verify_calls: 0 }
    }

    /// Signatures from already-subsampled raw values (length m_sig).
    pub fn signatures(&self, sample: &[f32]) -> Sigs {
        let r = frac_ranks(sample);
        let m = r.iter().filter(|x| x.is_finite()).count().max(1) as f64;
        let mean = r.iter().filter(|x| x.is_finite()).map(|&x| x as f64).sum::<f64>() / m;
        let va: Vec<f32> = r.iter().map(|&x| if x.is_finite() { (x as f64 - mean) as f32 } else { 0.0 }).collect();
        let folded: Vec<f32> = r.iter().map(|&x| if x.is_finite() { (x - 0.5).abs() } else { f32::NAN }).collect();
        let fm = folded.iter().filter(|x| x.is_finite()).map(|&x| x as f64).sum::<f64>() / m;
        let vb: Vec<f32> = folded.iter().map(|&x| if x.is_finite() { (x as f64 - fm) as f32 } else { 0.0 }).collect();
        Sigs { a: self.hash(&va), b: if self.fold { self.hash(&vb) } else { 0 } }
    }

    fn hash(&self, v: &[f32]) -> u128 {
        let nbits = self.bands * self.band_bits;
        let mut bits: u128 = 0;
        for b in 0..nbits {
            let p = &self.planes[b * self.m_sig..(b + 1) * self.m_sig];
            let mut acc = 0.0f32;
            for i in 0..self.m_sig { acc += p[i] * v[i]; }
            if acc >= 0.0 { bits |= 1u128 << b; }
        }
        // canonical sign: |corr| semantics -> v and -v hash identically
        let mask: u128 = if nbits >= 128 { u128::MAX } else { (1u128 << nbits) - 1 };
        if bits & 1 == 1 { bits = !bits & mask; }
        bits
    }

    fn key(&self, sig: u128, band: usize) -> u32 {
        ((sig >> (band * self.band_bits)) & ((1u128 << self.band_bits) - 1)) as u32
    }

    pub fn insert(&mut self, id: u32, s: &Sigs) {
        for band in 0..self.bands {
            let ka = self.key(s.a, band);
            self.ta[band].entry(ka).or_default().push(id);
            if self.fold {
                let kb = self.key(s.b, band);
                self.tb[band].entry(kb).or_default().push(id);
            }
        }
    }

    /// Read-only variant (safe to call from many threads).
    pub fn candidates_ro(&self, s: &Sigs) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for band in 0..self.bands {
            if let Some(v) = self.ta[band].get(&self.key(s.a, band)) { out.extend_from_slice(v); }
            if self.fold {
                if let Some(v) = self.tb[band].get(&self.key(s.a, band)) { out.extend_from_slice(v); }
                if let Some(v) = self.ta[band].get(&self.key(s.b, band)) { out.extend_from_slice(v); }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Candidate ids colliding with the signatures in any band/view.
    pub fn candidates(&mut self, s: &Sigs, n_items: usize) -> Vec<u32> {
        if self.stamp.len() < n_items { self.stamp.resize(n_items + 1024, 0); }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 { for x in self.stamp.iter_mut() { *x = 0; } self.epoch = 1; }
        let mut out = Vec::new();
        let ep = self.epoch;
        for band in 0..self.bands {
            let probe = |t: &Table, key: u32, stamp: &mut Vec<u32>, out: &mut Vec<u32>| {
                if let Some(v) = t.get(&key) {
                    for &id in v { if stamp[id as usize] != ep { stamp[id as usize] = ep; out.push(id); } }
                }
            };
            probe(&self.ta[band], self.key(s.a, band), &mut self.stamp, &mut out);
            if self.fold {
                probe(&self.tb[band], self.key(s.a, band), &mut self.stamp, &mut out); // cand ~ fold(archived)
                probe(&self.ta[band], self.key(s.b, band), &mut self.stamp, &mut out); // fold(cand) ~ archived
            }
        }
        out
    }
}

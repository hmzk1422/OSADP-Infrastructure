//! Novelty-driven feature search: recursive enumeration + credit-weighted evolution,
//! with the LSH hashmap rejecting near-duplicates *before* they can multiply.

use crate::expr::*;
use crate::sim::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Gate {
    /// "none" | "persistence" (unsupervised: regime features must be slow/persistent) | "mi" (needs target)
    pub kind: String,
    pub min_persistence: f64,
    pub mi_quantile: f64,
    pub mi_rows: usize,
    pub mi_perms: usize,
    pub block: usize,
    pub rows: usize,
    /// lag (bars) at which persistence is measured; regimes persist, smoothed noise does not
    pub lag: usize,
}
impl Default for Gate {
    fn default() -> Self {
        Gate { kind: "persistence".into(), min_persistence: 0.10, mi_quantile: 0.99, mi_rows: 2000, mi_perms: 100, block: 50, rows: 20000, lag: 60 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Cfg {
    pub windows: Vec<u32>,
    pub unary: Vec<String>,
    pub binary: Vec<String>,
    pub temporal: Vec<String>,
    pub partial: bool,
    pub max_height: u32,
    pub max_nesting: u32,
    pub levels: u32,
    pub metric: Metric,
    pub threshold: f64,
    pub candidates_per_level: usize,
    pub max_new_per_level: usize,
    pub max_archive: usize,
    pub evo_generations: u32,
    pub evo_children: usize,
    pub verify_rows: usize,
    pub sig_rows: usize,
    pub bands: usize,
    pub band_bits: usize,
    pub min_valid: f64,
    pub dedupe_terminals: bool,
    pub select_terminals: bool,
    pub warmup: usize,
    /// similarity is measured on a causal `smooth`-bar mean of each feature (0 = raw). A noisy copy of an
    /// existing feature then collides with it: same slow information, different noise.
    pub smooth: usize,
    pub exact_check_max: usize,
    pub exact_reference: bool,
    pub gate: Gate,
    pub seed: u64,
    pub threads: usize,
    pub chunk: usize,
}
impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            windows: vec![5, 10, 20],
            unary: ["sin", "cos", "sqrt", "pow2", "pow3"].iter().map(|s| s.to_string()).collect(),
            binary: ["add", "sub", "mul", "div"].iter().map(|s| s.to_string()).collect(),
            temporal: ["lag", "diff", "deriv", "rollmean", "rollstd"].iter().map(|s| s.to_string()).collect(),
            partial: true,
            max_height: 6,
            max_nesting: 2,
            levels: 3,
            metric: Metric::Mixed,
            threshold: 0.90,
            candidates_per_level: 12_000,
            max_new_per_level: 150,
            max_archive: 1500,
            evo_generations: 0,
            evo_children: 2000,
            verify_rows: 6000,
            sig_rows: 512,
            bands: 16,
            band_bits: 8,
            min_valid: 0.8,
            dedupe_terminals: true,
            select_terminals: true,
            warmup: 0,
            smooth: 30,
            exact_check_max: 800,
            exact_reference: false,
            gate: Gate::default(),
            seed: 0,
            threads: 0,
            chunk: 128,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Recipe { kind: u8, f: u8, w: u32, a: u32, b: u32 }

impl Recipe {
    fn tag(&self) -> usize {
        match self.kind { 0 => self.f as usize, 1 => 8 + self.f as usize, 2 => 16 + self.f as usize, _ => 24 }
    }
}

#[derive(Clone)]
struct Props { height: u32, nesting: u32, has_partial: bool, has_temporal: bool, nonlinear: bool, n_vars: usize }

struct Feat {
    node: Node,
    values: Vec<f32>,
    fv: FeatVec,
    props: Props,
    level: u32,
    novelty: f64,
    persistence: f64,
    mi_stat: f64,
    selected: bool,
    accepted_children: u32,
}

struct Prepared { fv: FeatVec, sigs: Sigs, persistence: f64, mi_stat: f64, gate_pass: bool }

#[derive(Default, Clone, Debug, Serialize)]
pub struct Stats {
    pub n_rows_search: usize,
    pub n_terminals: usize,
    pub candidates_evaluated: u64,
    pub rejected_invalid: u64,
    pub rejected_duplicate: u64,
    pub admitted: u64,
    pub lsh_verifications: u64,
    pub per_level: Vec<[u64; 3]>, // [evaluated, admitted, duplicate]
    pub evo_evaluated: u64,
    pub evo_admitted: u64,
    pub exact_check_removed: u64,
    pub op_credit: Vec<(String, f64)>,
    pub secs_enumerate: f64,
    pub secs_evolve: f64,
    pub secs_total: f64,
}

pub struct Search {
    pub cfg: Cfg,
    pub names: Vec<String>,
    cols: Vec<Vec<f32>>,
    target: Option<Vec<f32>>,
    n: usize,
    feats: Vec<Feat>,
    index: LshIndex,
    verify_rows: Vec<usize>,
    sig_pos: Vec<usize>,
    seen: HashSet<String>,
    tried: [f64; 32],
    acc: [f64; 32],
    unary: Vec<Un>,
    binary: Vec<Bin>,
    temporal: Vec<Tmp>,
    rng: StdRng,
    pub stats: Stats,
}

const TAG_NAMES: [&str; 25] = ["sin","cos","sqrt","pow2","pow3","abs","slog","neg","add","sub","mul","div","","","","",
    "lag","diff","deriv","rollmean","rollstd","","","","partial"];

fn un_tag(f: Un) -> u8 { match f { Un::Sin=>0, Un::Cos=>1, Un::Sqrt=>2, Un::Pow2=>3, Un::Pow3=>4, Un::Abs=>5, Un::SLog=>6, Un::Neg=>7 } }
fn bin_tag(f: Bin) -> u8 { match f { Bin::Add=>0, Bin::Sub=>1, Bin::Mul=>2, Bin::Div=>3 } }
fn tmp_tag(f: Tmp) -> u8 { match f { Tmp::Lag=>0, Tmp::Diff=>1, Tmp::Deriv=>2, Tmp::RollMean=>3, Tmp::RollStd=>4 } }

impl Search {
    pub fn new(cols: Vec<Vec<f32>>, names: Vec<String>, target: Option<Vec<f32>>, cfg: Cfg) -> Result<Search, String> {
        if cols.is_empty() { return Err("no terminal columns".into()); }
        let n = cols[0].len();
        if cols.iter().any(|c| c.len() != n) { return Err("ragged terminal columns".into()); }
        if cfg.threshold <= 0.0 || cfg.threshold > 1.0 { return Err("threshold must be in (0,1]".into()); }
        let unary = cfg.unary.iter().map(|s| Un::parse(s).ok_or(format!("unknown unary op {s}"))).collect::<Result<Vec<_>, _>>()?;
        let binary = cfg.binary.iter().map(|s| Bin::parse(s).ok_or(format!("unknown binary op {s}"))).collect::<Result<Vec<_>, _>>()?;
        let temporal = cfg.temporal.iter().map(|s| Tmp::parse(s).ok_or(format!("unknown temporal op {s}"))).collect::<Result<Vec<_>, _>>()?;
        let max_w = cfg.windows.iter().copied().max().unwrap_or(1) as usize;
        let warm = (if cfg.warmup > 0 { cfg.warmup } else { max_w * (cfg.max_nesting as usize).max(1) }).max(cfg.smooth);
        if warm + 100 >= n { return Err(format!("not enough rows ({n}) for warmup {warm}")); }
        let m_v = cfg.verify_rows.min(n - warm);
        let verify_rows: Vec<usize> = (0..m_v).map(|i| warm + i * (n - warm) / m_v).collect();
        let m_s = cfg.sig_rows.min(m_v);
        let sig_pos: Vec<usize> = (0..m_s).map(|i| i * m_v / m_s).collect();
        let index = LshIndex::new(cfg.bands, cfg.band_bits, m_s, cfg.metric.uses_fold(), cfg.seed);
        let seed = cfg.seed;
        let mut cfg = cfg;
        cfg.warmup = warm;
        Ok(Search { names, cols, target, n, feats: Vec::new(), index, verify_rows, sig_pos,
            seen: HashSet::new(), tried: [1.0; 32], acc: [1.0; 32], unary, binary, temporal,
            rng: StdRng::seed_from_u64(seed ^ 0xC0FFEE), stats: Stats::default(), cfg })
    }

    fn props_of(node: &Node) -> Props {
        let mut v = Vec::new();
        node.vars(&mut v);
        Props { height: node.height(), nesting: node.temporal_nesting(), has_partial: node.has_partial(),
            has_temporal: node.has_temporal(), nonlinear: node.is_nonlinear(), n_vars: v.len() }
    }

    fn persistence(&self, values: &[f32]) -> f64 {
        let lag = self.cfg.gate.lag.max(1);
        let lo = self.cfg.warmup + lag;
        let hi = (self.cfg.warmup + self.cfg.gate.rows).min(self.n);
        let (mut n, mut sa, mut sb) = (0f64, 0f64, 0f64);
        for i in lo..hi { let (a, b) = (values[i], values[i - lag]); if a.is_finite() && b.is_finite() { n += 1.0; sa += a as f64; sb += b as f64; } }
        if n < 50.0 { return 0.0; }
        let (ma, mb) = (sa / n, sb / n);
        let (mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64);
        for i in lo..hi { let (a, b) = (values[i], values[i - lag]); if a.is_finite() && b.is_finite() {
            let (da, db) = (a as f64 - ma, b as f64 - mb); saa += da * da; sbb += db * db; sab += da * db; } }
        if saa <= 1e-20 || sbb <= 1e-20 { 0.0 } else { sab / (saa * sbb).sqrt() }
    }

    /// Small-sample MI gate with a block-shuffle null (keeps autocorrelation of the feature intact).
    fn mi_gate(&self, values: &[f32], seed: u64) -> (bool, f64) {
        let tgt = match &self.target { Some(t) => t, None => return (true, 0.0) };
        let lo = self.cfg.warmup;
        let hi = (lo + self.cfg.gate.mi_rows).min(self.n);
        let idx: Vec<usize> = (lo..hi).filter(|&i| values[i].is_finite() && tgt[i].is_finite()).collect();
        if idx.len() < 200 { return (false, 0.0); }
        let xs: Vec<f32> = idx.iter().map(|&i| values[i]).collect();
        let ys: Vec<f32> = idx.iter().map(|&i| tgt[i]).collect();
        let bx = make_featvec(&xs, false).bins;
        let by = make_featvec(&ys, false).bins;
        let stat = {
            let mut joint = vec![&bx[..], &by[..]];
            let _ = joint.pop();
            similarity_bins(&bx, &by)
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let blk = self.cfg.gate.block.max(1);
        let nb = (by.len() + blk - 1) / blk;
        let mut null = Vec::with_capacity(self.cfg.gate.mi_perms);
        for _ in 0..self.cfg.gate.mi_perms {
            let mut order: Vec<usize> = (0..nb).collect();
            for i in (1..nb).rev() { let j = rng.gen_range(0..=i); order.swap(i, j); }
            let mut sh = Vec::with_capacity(by.len());
            for &b in &order { let s = b * blk; let e = (s + blk).min(by.len()); sh.extend_from_slice(&by[s..e]); }
            sh.truncate(by.len());
            null.push(similarity_bins(&bx, &sh));
        }
        null.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = ((self.cfg.gate.mi_quantile * null.len() as f64).ceil() as usize).clamp(1, null.len()) - 1;
        (stat > null[k] && stat > 0.0, stat)
    }

    fn prepare(&self, values: &[f32], salt: u64) -> Option<Prepared> {
        let sm = self.cfg.smooth;
        let vv: Vec<f32> = self.verify_rows.iter().map(|&r| {
            if sm <= 1 { return values[r]; }
            let seg = &values[r + 1 - sm..=r];
            let (mut c, mut t) = (0usize, 0f64);
            for &v in seg { if v.is_finite() { c += 1; t += v as f64; } }
            if c * 2 >= sm { (t / c as f64) as f32 } else { f32::NAN }
        }).collect();
        let fin = vv.iter().filter(|x| x.is_finite()).count();
        if (fin as f64) < self.cfg.min_valid * vv.len() as f64 { return None; }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in vv.iter().filter(|x| x.is_finite()) { lo = lo.min(x); hi = hi.max(x); }
        if !(hi - lo > 1e-9 * (1.0 + hi.abs().max(lo.abs()))) { return None; }
        let fv = make_featvec(&vv, self.cfg.metric.needs_raw());
        let sample: Vec<f32> = self.sig_pos.iter().map(|&p| vv[p]).collect();
        let sigs = self.index.signatures(&sample);
        let (mut persistence, mut mi_stat, mut gate_pass) = (0.0, 0.0, true);
        match self.cfg.gate.kind.as_str() {
            "persistence" => { persistence = self.persistence(values); gate_pass = persistence >= self.cfg.gate.min_persistence; }
            "mi" => { let (p, s) = self.mi_gate(values, self.cfg.seed ^ salt); gate_pass = p; mi_stat = s; }
            _ => {}
        }
        Some(Prepared { fv, sigs, persistence, mi_stat, gate_pass })
    }

    /// Sequential admission against the archive through the LSH hashmap.
    fn admit(&mut self, node: Node, values: Vec<f32>, prep: Prepared, level: u32, terminal: bool) -> bool {
        let cands = self.index.candidates(&prep.sigs, self.feats.len());
        let mut max_sim = 0.0f64;
        for id in cands {
            self.stats.lsh_verifications += 1;
            let s = similarity(self.cfg.metric, &prep.fv, &self.feats[id as usize].fv);
            if s > max_sim { max_sim = s; }
            if s >= self.cfg.threshold { self.stats.rejected_duplicate += 1; return false; }
        }
        let id = self.feats.len() as u32;
        self.index.insert(id, &prep.sigs);
        let selected = prep.gate_pass || (terminal && self.cfg.select_terminals);
        let props = Self::props_of(&node);
        self.feats.push(Feat { node, values, fv: prep.fv, props, level, novelty: 1.0 - max_sim,
            persistence: prep.persistence, mi_stat: prep.mi_stat, selected, accepted_children: 0 });
        self.stats.admitted += 1;
        true
    }

    fn build_values(&self, r: &Recipe) -> (Node, Vec<f32>) {
        match r.kind {
            0 => { let f = self.unary[r.f as usize]; let a = &self.feats[r.a as usize];
                (Node::un(f, a.node.clone()), un_k(f, &a.values)) }
            1 => { let f = self.binary[r.f as usize]; let (a, b) = (&self.feats[r.a as usize], &self.feats[r.b as usize]);
                (Node::bin(f, a.node.clone(), b.node.clone()), bin_k(f, &a.values, &b.values)) }
            2 => { let f = self.temporal[r.f as usize]; let a = &self.feats[r.a as usize];
                (Node::tmp(f, r.w, a.node.clone()), tmp_k(f, r.w, &a.values)) }
            _ => { let a = &self.feats[r.a as usize]; let wrt = r.w as usize;
                (Node::partial(wrt, a.node.clone()), eval_dual(&a.node, &self.cols, wrt).1) }
        }
    }

    fn recipe_ok(&self, r: &Recipe) -> bool {
        let cfg = &self.cfg;
        let pa = &self.feats[r.a as usize];
        match r.kind {
            0 => { let f = self.unary[r.f as usize];
                pa.props.height < cfg.max_height
                    && !(matches!(f, Un::Abs | Un::Neg) && matches!(&pa.node, Node::Un { f: g, .. } if *g == f)) }
            1 => { let pb = &self.feats[r.b as usize]; let f = self.binary[r.f as usize];
                pa.props.height.max(pb.props.height) < cfg.max_height
                    && pa.props.nesting.max(pb.props.nesting) <= cfg.max_nesting
                    && !(r.a == r.b && matches!(f, Bin::Sub | Bin::Div | Bin::Add))
                    && !(f.commutative() && r.a > r.b) }
            2 => { let f = self.temporal[r.f as usize];
                pa.props.height < cfg.max_height && pa.props.nesting < cfg.max_nesting
                    && !(f == Tmp::Lag && matches!(&pa.node, Node::Tmp { f: Tmp::Lag, .. })) }
            _ => cfg.partial && pa.props.height < cfg.max_height && !pa.props.has_temporal && !pa.props.has_partial
                    && pa.props.nonlinear && pa.props.n_vars > 0,
        }
    }

    fn recipe_weight(&self, r: &Recipe) -> f64 {
        let cr = self.acc[r.tag()] / self.tried[r.tag()];
        let nv = |id: u32| 0.25 + self.feats[id as usize].novelty.max(0.0);
        match r.kind { 1 => cr * nv(r.a) * nv(r.b), _ => cr * nv(r.a) }
    }

    fn gen_recipes(&mut self, frontier: &[u32], arch_len: usize, budget: usize) -> Vec<Recipe> {
        let nw = self.cfg.windows.len();
        let mut all: Vec<Recipe> = Vec::new();
        let total_est = frontier.len() * (self.unary.len() + self.temporal.len() * nw + self.binary.len() * arch_len);
        if total_est <= budget * 6 {
            for &a in frontier {
                for f in 0..self.unary.len() { all.push(Recipe { kind: 0, f: f as u8, w: 0, a, b: 0 }); }
                for f in 0..self.temporal.len() { for &w in &self.cfg.windows { all.push(Recipe { kind: 2, f: f as u8, w, a, b: 0 }); } }
                for f in 0..self.binary.len() { for b in 0..arch_len as u32 { all.push(Recipe { kind: 1, f: f as u8, w: 0, a, b }); } }
                if self.cfg.partial {
                    let mut vs = Vec::new(); self.feats[a as usize].node.vars(&mut vs);
                    for v in vs { all.push(Recipe { kind: 3, f: 0, w: v as u32, a, b: 0 }); }
                }
            }
            // binary pairs (a in frontier, b in frontier) appear twice with a>b order variants; recipe_ok drops
            // the commutative mirror, sub/div keep both orders.
            let mut all: Vec<Recipe> = all.into_iter().filter(|r| self.recipe_ok(r)).collect();
            // when both orders of a commutative pair are generated via (a in F, b in A) and (b in F, a in A), dedupe by key
            let mut seen = HashSet::new();
            all.retain(|r| seen.insert((r.kind, r.f, r.w, r.a, r.b)));
            if all.len() <= budget { return all; }
            // weighted sampling without replacement (Efraimidis-Spirakis)
            let mut keyed: Vec<(f64, Recipe)> = all.into_iter().map(|r| {
                let u: f64 = self.rng.gen::<f64>().max(1e-12);
                (u.ln() / self.recipe_weight(&r).max(1e-9), r)
            }).collect();
            keyed.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            keyed.truncate(budget);
            return keyed.into_iter().map(|x| x.1).collect();
        }
        // huge space: rejection sampling by credit weight
        let mut out = Vec::with_capacity(budget);
        let mut seen = HashSet::new();
        let mut tries = 0usize;
        while out.len() < budget && tries < budget * 40 {
            tries += 1;
            let a = frontier[self.rng.gen_range(0..frontier.len())];
            let c = self.rng.gen_range(0..total_est / frontier.len().max(1));
            let nu = self.unary.len(); let nt = self.temporal.len() * nw;
            let r = if c < nu { Recipe { kind: 0, f: c as u8, w: 0, a, b: 0 } }
                else if c < nu + nt { let k = c - nu; Recipe { kind: 2, f: (k / nw) as u8, w: self.cfg.windows[k % nw], a, b: 0 } }
                else { let k = c - nu - nt; Recipe { kind: 1, f: (k / arch_len) as u8, w: 0, a, b: (k % arch_len) as u32 } };
            let r = if r.kind == 1 && r.a > r.b && self.binary[r.f as usize].commutative() { Recipe { a: r.b, b: r.a, ..r } } else { r };
            if !self.recipe_ok(&r) || !seen.insert((r.kind, r.f, r.w, r.a, r.b)) { continue; }
            // acceptance probability: credit relative to the best operator, times novelty of parents
            let cmax = (0..25).filter(|&t| !TAG_NAMES[t].is_empty()).map(|t| self.acc[t] / self.tried[t]).fold(1e-9, f64::max);
            let cr = (self.acc[r.tag()] / self.tried[r.tag()] / cmax).max(0.05);
            let nv = |id: u32| (0.25 + self.feats[id as usize].novelty.max(0.0)) / 1.25;
            let w = if r.kind == 1 { cr * nv(r.a) * nv(r.b) } else { cr * nv(r.a) };
            if self.rng.gen::<f64>() > w { continue; }
            out.push(r);
        }
        if self.cfg.partial { for &a in frontier {
            let mut vs = Vec::new(); self.feats[a as usize].node.vars(&mut vs);
            for v in vs { let r = Recipe { kind: 3, f: 0, w: v as u32, a, b: 0 }; if self.recipe_ok(&r) && seen.insert((3, 0, v as u32, a, 0)) { out.push(r); } }
        } }
        out
    }

    fn process_chunk(&mut self, items: Vec<(Node, Vec<f32>, u32)>, level: u32, from_evo: bool) -> (u64, u64, u64) {
        // parallel: signatures + gate; sequential: admission
        let this: &Search = &*self;
        let prepared: Vec<Option<Prepared>> = items.par_iter().enumerate()
            .map(|(i, it)| this.prepare(&it.1, (this.feats.len() + i) as u64)).collect();
        let (mut ev, mut ad, mut du) = (0u64, 0u64, 0u64);
        for (it, prep) in items.into_iter().zip(prepared.into_iter()) {
            ev += 1;
            let (node, values, tag_parent) = it;
            let tags = Self::op_tags(&node);
            match prep {
                None => { self.stats.rejected_invalid += 1; }
                Some(p) => {
                    if self.feats.len() >= self.cfg.max_archive { continue; }
                    let before = self.stats.rejected_duplicate;
                    let ok = self.admit(node, values, p, level, false);
                    if ok {
                        ad += 1;
                        for &t in &tags { self.acc[t] += 1.0; self.tried[t] += 1.0; }
                        if from_evo && (tag_parent as usize) < self.feats.len() { self.feats[tag_parent as usize].accepted_children += 1; }
                    } else {
                        du += self.stats.rejected_duplicate - before;
                        for &t in &tags { self.tried[t] += 1.0; }
                    }
                }
            }
        }
        (ev, ad, du)
    }

    fn op_tags(node: &Node) -> Vec<usize> {
        // credit goes to the root operator only (the step that was actually new)
        vec![match node {
            Node::Un { f, .. } => un_tag(*f) as usize,
            Node::Bin { f, .. } => 8 + bin_tag(*f) as usize,
            Node::Tmp { f, .. } => 16 + tmp_tag(*f) as usize,
            Node::Partial { .. } => 24,
            _ => 7,
        }]
    }

    pub fn run(&mut self) -> Result<(), String> {
        let t0 = Instant::now();
        self.stats.n_rows_search = self.n;
        self.stats.n_terminals = self.cols.len();
        // level 0: terminals
        let p = self.cols.len();
        let mut level0 = Vec::new();
        for i in 0..p {
            let node = Node::var(i);
            self.seen.insert(node.formula(&self.names));
            level0.push((node, self.cols[i].clone(), 0u32));
        }
        if self.cfg.dedupe_terminals {
            let mut ad = 0;
            for chunk in level0.chunks(self.cfg.chunk.max(1)) {
                let this: &Search = &*self;
                let prepared: Vec<Option<Prepared>> = chunk.par_iter().enumerate().map(|(i, it)| this.prepare(&it.1, i as u64)).collect();
                for (it, prep) in chunk.iter().zip(prepared.into_iter()) {
                    if let Some(pr) = prep { if self.admit(it.0.clone(), it.1.clone(), pr, 0, true) { ad += 1; } else { self.stats.rejected_duplicate += 0; } }
                    else { self.stats.rejected_invalid += 1; }
                }
            }
            self.stats.per_level.push([p as u64, ad, p as u64 - ad]);
        } else {
            for (i, it) in level0.into_iter().enumerate() {
                if let Some(pr) = self.prepare(&it.1, i as u64) { self.admit(it.0, it.1, pr, 0, true); }
            }
        }
        if self.feats.is_empty() { return Err("no valid terminal features".into()); }
        let mut frontier: Vec<u32> = (0..self.feats.len() as u32).collect();

        for level in 1..=self.cfg.levels {
            let arch_len = self.feats.len();
            let recipes = self.gen_recipes(&frontier, arch_len, self.cfg.candidates_per_level);
            let admitted_before = self.feats.len();
            let (mut tev, mut tdu) = (0u64, 0u64);
            // ---- phase 1: score every candidate against the archive snapshot (read-only, parallel) ----
            let mut scored: Vec<(usize, f64, bool)> = Vec::new();
            for (ci, chunk) in recipes.chunks(self.cfg.chunk.max(1)).enumerate() {
                let this: &Search = &*self;
                let res: Vec<Option<(String, f64, bool)>> = chunk.par_iter().enumerate().map(|(i, r)| {
                    let (node, vals) = this.build_values(r);
                    if !node.valid(this.cfg.max_height, this.cfg.max_nesting, this.cols.len()) { return None; }
                    let key = node.formula(&this.names);
                    let prep = this.prepare(&vals, (ci * 977 + i) as u64)?;
                    let mut max_sim = 0.0f64;
                    for id in this.index.candidates_ro(&prep.sigs) {
                        let s = similarity(this.cfg.metric, &prep.fv, &this.feats[id as usize].fv);
                        if s > max_sim { max_sim = s; }
                        if s >= this.cfg.threshold { break; }
                    }
                    Some((key, max_sim, prep.gate_pass))
                }).collect();
                for (i, o) in res.into_iter().enumerate() {
                    let r = &chunk[i];
                    match o {
                        None => { self.stats.rejected_invalid += 1; }
                        Some((key, sim, gate)) => {
                            if !self.seen.insert(key) { continue; }
                            tev += 1;
                            let tag = r.tag();
                            self.tried[tag] += 1.0;
                            if sim >= self.cfg.threshold { tdu += 1; self.stats.rejected_duplicate += 1; }
                            else if !gate { }
                            else { self.acc[tag] += 1.0; }
                            if sim < self.cfg.threshold { scored.push((ci * self.cfg.chunk.max(1) + i, sim, gate)); }
                        }
                    }
                }
            }
            // ---- phase 2: admit best-first (gate-passers, then most novel) with exact re-check ----
            scored.sort_by(|a, b| b.2.cmp(&a.2).then(a.1.partial_cmp(&b.1).unwrap()));
            let mut pos = 0usize;
            let mut tad = 0u64;
            while pos < scored.len() && self.feats.len() - admitted_before < self.cfg.max_new_per_level && self.feats.len() < self.cfg.max_archive {
                let end = (pos + 64).min(scored.len());
                let this: &Search = &*self;
                let built: Vec<Option<(Node, Vec<f32>, Prepared)>> = scored[pos..end].par_iter().map(|&(ri, _, _)| {
                    let (node, vals) = this.build_values(&recipes[ri]);
                    let prep = this.prepare(&vals, ri as u64)?;
                    Some((node, vals, prep))
                }).collect();
                for b in built.into_iter().flatten() {
                    if self.feats.len() - admitted_before >= self.cfg.max_new_per_level || self.feats.len() >= self.cfg.max_archive { break; }
                    if self.admit(b.0, b.1, b.2, level, false) { tad += 1; } else { tdu += 1; }
                }
                pos = end;
            }
            self.stats.candidates_evaluated += tev;
            self.stats.per_level.push([tev, tad, tdu]);
            frontier = (admitted_before as u32..self.feats.len() as u32).collect();
            if frontier.is_empty() { break; }
        }
        self.stats.secs_enumerate = t0.elapsed().as_secs_f64();

        // ---- credit-weighted evolutionary phase ("path seeking") ----
        let t1 = Instant::now();
        for _g in 0..self.cfg.evo_generations {
            if self.feats.len() >= self.cfg.max_archive { break; }
            let kids = self.make_children(self.cfg.evo_children);
            let this: &Search = &*self;
            let built: Vec<(Node, Vec<f32>, u32)> = kids.par_iter().map(|(node, parent)| (node.clone(), eval(node, &this.cols), *parent)).collect();
            let level = self.cfg.levels + 1;
            for chunk in built.chunks(self.cfg.chunk.max(1)) {
                let (ev, ad, _) = self.process_chunk(chunk.to_vec(), level, true);
                self.stats.evo_evaluated += ev; self.stats.evo_admitted += ad;
            }
        }
        self.stats.secs_evolve = t1.elapsed().as_secs_f64();
        self.stats.lsh_verifications = self.stats.lsh_verifications.max(self.index.n_verify_calls);

        // ---- exact audit of the LSH: remove any pair the hash missed ----
        let sel: Vec<usize> = (0..self.feats.len()).filter(|&i| self.feats[i].selected).collect();
        if sel.len() <= self.cfg.exact_check_max && !sel.is_empty() {
            let mut kept: Vec<usize> = Vec::new();
            for &i in &sel {
                let this: &Search = &*self;
                let dup = kept.par_iter().any(|&k| similarity(this.cfg.metric, &this.feats[i].fv, &this.feats[k].fv) >= this.cfg.threshold);
                if dup { self.feats[i].selected = false; self.stats.exact_check_removed += 1; } else { kept.push(i); }
            }
        }
        self.stats.op_credit = (0..25).filter(|&t| !TAG_NAMES[t].is_empty())
            .map(|t| (TAG_NAMES[t].to_string(), self.acc[t] / self.tried[t])).collect();
        self.stats.secs_total = t0.elapsed().as_secs_f64();
        Ok(())
    }

    fn rand_tree(&mut self, depth: u32, allow_temporal: bool, allow_partial: bool) -> Node {
        let p = self.cols.len();
        if depth <= 1 || self.rng.gen::<f64>() < 0.25 { return Node::var(self.rng.gen_range(0..p)); }
        let cr = |t: usize, s: &Search| s.acc[t] / s.tried[t];
        let w_un: f64 = self.unary.iter().map(|&f| cr(un_tag(f) as usize, self)).sum();
        let w_bin: f64 = self.binary.iter().map(|&f| cr(8 + bin_tag(f) as usize, self)).sum();
        let w_tmp: f64 = if allow_temporal { self.temporal.iter().map(|&f| cr(16 + tmp_tag(f) as usize, self)).sum() } else { 0.0 };
        let w_par: f64 = if allow_partial && self.cfg.partial { 0.15 * (w_un + w_bin) / 2.0 } else { 0.0 };
        let mut x = self.rng.gen::<f64>() * (w_un + w_bin + w_tmp + w_par);
        if x < w_bin {
            let mut cum = 0.0;
            let mut pick = self.binary[0];
            for &f in &self.binary { cum += cr(8 + bin_tag(f) as usize, self); if x < cum { pick = f; break; } }
            let a = self.rand_tree(depth - 1, allow_temporal, allow_partial);
            let b = self.rand_tree(depth - 1, allow_temporal, allow_partial);
            return Node::bin(pick, a, b);
        }
        x -= w_bin;
        if x < w_un {
            let mut cum = 0.0; let mut pick = self.unary[0];
            for &f in &self.unary { cum += cr(un_tag(f) as usize, self); if x < cum { pick = f; break; } }
            return Node::un(pick, self.rand_tree(depth - 1, allow_temporal, allow_partial));
        }
        x -= w_un;
        if x < w_tmp {
            let mut cum = 0.0; let mut pick = self.temporal[0];
            for &f in &self.temporal { cum += cr(16 + tmp_tag(f) as usize, self); if x < cum { pick = f; break; } }
            let w = self.cfg.windows[self.rng.gen_range(0..self.cfg.windows.len())];
            return Node::tmp(pick, w, self.rand_tree(depth - 1, true, allow_partial));
        }
        let child = self.rand_tree(depth - 1, false, false);
        let mut vs = Vec::new(); child.vars(&mut vs);
        if child.is_nonlinear() && !vs.is_empty() { let v = vs[self.rng.gen_range(0..vs.len())]; Node::partial(v, child) } else { child }
    }

    fn make_children(&mut self, count: usize) -> Vec<(Node, u32)> {
        let k = self.feats.len();
        let weights: Vec<f64> = self.feats.iter().map(|f| 1.0 + f.accepted_children as f64).collect();
        let total: f64 = weights.iter().sum();
        let mut out = Vec::with_capacity(count);
        let mut tries = 0;
        while out.len() < count && tries < count * 8 {
            tries += 1;
            let mut x = self.rng.gen::<f64>() * total;
            let mut pi = k - 1;
            for (i, w) in weights.iter().enumerate() { if x < *w { pi = i; break; } x -= w; }
            let parent = self.feats[pi].node.clone();
            let mut paths = Vec::new(); parent.paths(&mut Vec::new(), &mut paths);
            let pp = paths[self.rng.gen_range(0..paths.len())].clone();
            let rep = if self.rng.gen::<f64>() < 0.6 {
                let d = self.rng.gen_range(1..=3);
                self.rand_tree(d, true, true)
            } else {
                let oi = self.rng.gen_range(0..k);
                let other = &self.feats[oi].node;
                let mut op = Vec::new(); other.paths(&mut Vec::new(), &mut op);
                other.at(&op[self.rng.gen_range(0..op.len())]).clone()
            };
            let child = parent.replace(&pp, &rep);
            if !child.valid(self.cfg.max_height, self.cfg.max_nesting, self.cols.len()) { continue; }
            if !self.seen.insert(child.formula(&self.names)) { continue; }
            out.push((child, pi as u32));
        }
        out
    }

    pub fn library_json(&self) -> String {
        #[derive(Serialize)]
        struct Item<'a> { id: usize, name: String, formula: String, tree: &'a Node, level: u32, size: u32, height: u32,
            novelty: f64, persistence: f64, mi_stat: f64 }
        #[derive(Serialize)]
        struct Lib<'a> { schema_version: u32, terminals: &'a [String], cfg: &'a Cfg, stats: &'a Stats, features: Vec<Item<'a>> }
        let mut items = Vec::new();
        for (i, f) in self.feats.iter().enumerate().filter(|(_, f)| f.selected) {
            items.push(Item { id: i, name: format!("gp_{:04}", i), formula: f.node.formula(&self.names), tree: &f.node,
                level: f.level, size: f.node.size(), height: f.props.height, novelty: f.novelty, persistence: f.persistence, mi_stat: f.mi_stat });
        }
        serde_json::to_string(&Lib { schema_version: 2, terminals: &self.names, cfg: &self.cfg, stats: &self.stats, features: items }).unwrap()
    }

    /// Dedupe-only entry point: admitted indices of the supplied columns (in order).
    pub fn admitted_terminal_ids(&self) -> Vec<usize> {
        self.feats.iter().filter(|f| f.level == 0).filter_map(|f| if let Node::Var { idx } = f.node { Some(idx) } else { None }).collect()
    }

    /// Exhaustive (O(K^2)) reference with the same metric, for measuring LSH recall.
    pub fn exact_reference_ids(&self) -> Vec<usize> {
        let mut kept: Vec<usize> = Vec::new();
        let mut fvs: Vec<Option<FeatVec>> = Vec::new();
        for i in 0..self.cols.len() {
            let vv: Vec<f32> = self.verify_rows.iter().map(|&r| self.cols[i][r]).collect();
            let fin = vv.iter().filter(|x| x.is_finite()).count();
            if (fin as f64) < self.cfg.min_valid * vv.len() as f64 { fvs.push(None); continue; }
            fvs.push(Some(make_featvec(&vv, self.cfg.metric.needs_raw())));
        }
        for i in 0..self.cols.len() {
            let fv = match &fvs[i] { Some(f) => f, None => continue };
            let dup = kept.par_iter().any(|&k| similarity(self.cfg.metric, fv, fvs[k].as_ref().unwrap()) >= self.cfg.threshold);
            if !dup { kept.push(i); }
        }
        kept
    }

    pub fn sample_trees(&mut self, count: usize, depth: u32) -> Vec<Node> {
        let mut out = Vec::new();
        while out.len() < count {
            let t = self.rand_tree(depth, true, true);
            if t.valid(self.cfg.max_height, self.cfg.max_nesting, self.cols.len()) { out.push(t); }
        }
        out
    }
}

fn similarity_bins(a: &[u8], b: &[u8]) -> f64 {
    let fa = FeatVec { z: Vec::new(), bins: a.to_vec(), raw: Vec::new() };
    let fb = FeatVec { z: Vec::new(), bins: b.to_vec(), raw: Vec::new() };
    similarity(Metric::Info, &fa, &fb)
}

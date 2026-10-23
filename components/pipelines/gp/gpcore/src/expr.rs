//! Expression trees, causal kernels, and forward-mode partial derivatives.
//! Everything here is pure Rust (no JIT, no Python): the same crate can be linked
//! into the live engine so offline formulas and live formulas cannot drift apart.

use serde::{Deserialize, Serialize};

pub const CLIP: f64 = 1.0e6;
const DIV_EPS: f64 = 1.0e-9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Un { Sin, Cos, Sqrt, Pow2, Pow3, Abs, SLog, Neg }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bin { Add, Sub, Mul, Div }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tmp { Lag, Diff, Deriv, RollMean, RollStd }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Node {
    Var { idx: usize },
    Const { value: f64 },
    Un { f: Un, a: Box<Node> },
    Bin { f: Bin, a: Box<Node>, b: Box<Node> },
    Tmp { f: Tmp, w: u32, a: Box<Node> },
    Partial { wrt: usize, a: Box<Node> },
}

impl Un {
    pub fn name(self) -> &'static str {
        match self { Un::Sin => "sin", Un::Cos => "cos", Un::Sqrt => "sqrt", Un::Pow2 => "pow2",
            Un::Pow3 => "pow3", Un::Abs => "abs", Un::SLog => "slog", Un::Neg => "neg" }
    }
    pub fn parse(s: &str) -> Option<Un> {
        Some(match s { "sin" => Un::Sin, "cos" => Un::Cos, "sqrt" => Un::Sqrt, "pow2" => Un::Pow2,
            "pow3" => Un::Pow3, "abs" => Un::Abs, "slog" => Un::SLog, "neg" => Un::Neg, _ => return None })
    }
}
impl Bin {
    pub fn name(self) -> &'static str {
        match self { Bin::Add => "+", Bin::Sub => "-", Bin::Mul => "*", Bin::Div => "/" }
    }
    pub fn parse(s: &str) -> Option<Bin> {
        Some(match s { "add" => Bin::Add, "sub" => Bin::Sub, "mul" => Bin::Mul, "div" => Bin::Div, _ => return None })
    }
    pub fn commutative(self) -> bool { matches!(self, Bin::Add | Bin::Mul) }
}
impl Tmp {
    pub fn name(self) -> &'static str {
        match self { Tmp::Lag => "lag", Tmp::Diff => "diff", Tmp::Deriv => "deriv",
            Tmp::RollMean => "rollmean", Tmp::RollStd => "rollstd" }
    }
    pub fn parse(s: &str) -> Option<Tmp> {
        Some(match s { "lag" => Tmp::Lag, "diff" => Tmp::Diff, "deriv" => Tmp::Deriv,
            "rollmean" => Tmp::RollMean, "rollstd" => Tmp::RollStd, _ => return None })
    }
}

impl Node {
    pub fn var(idx: usize) -> Node { Node::Var { idx } }
    pub fn un(f: Un, a: Node) -> Node { Node::Un { f, a: Box::new(a) } }
    pub fn bin(f: Bin, a: Node, b: Node) -> Node { Node::Bin { f, a: Box::new(a), b: Box::new(b) } }
    pub fn tmp(f: Tmp, w: u32, a: Node) -> Node { Node::Tmp { f, w, a: Box::new(a) } }
    pub fn partial(wrt: usize, a: Node) -> Node { Node::Partial { wrt, a: Box::new(a) } }

    pub fn height(&self) -> u32 {
        match self {
            Node::Var { .. } | Node::Const { .. } => 1,
            Node::Un { a, .. } | Node::Tmp { a, .. } | Node::Partial { a, .. } => 1 + a.height(),
            Node::Bin { a, b, .. } => 1 + a.height().max(b.height()),
        }
    }
    pub fn size(&self) -> u32 {
        match self {
            Node::Var { .. } | Node::Const { .. } => 1,
            Node::Un { a, .. } | Node::Tmp { a, .. } | Node::Partial { a, .. } => 1 + a.size(),
            Node::Bin { a, b, .. } => 1 + a.size() + b.size(),
        }
    }
    /// Maximum number of temporal operators on any root-to-leaf path.
    pub fn temporal_nesting(&self) -> u32 {
        match self {
            Node::Var { .. } | Node::Const { .. } => 0,
            Node::Un { a, .. } | Node::Partial { a, .. } => a.temporal_nesting(),
            Node::Tmp { a, .. } => 1 + a.temporal_nesting(),
            Node::Bin { a, b, .. } => a.temporal_nesting().max(b.temporal_nesting()),
        }
    }
    pub fn has_partial(&self) -> bool {
        match self {
            Node::Var { .. } | Node::Const { .. } => false,
            Node::Partial { .. } => true,
            Node::Un { a, .. } | Node::Tmp { a, .. } => a.has_partial(),
            Node::Bin { a, b, .. } => a.has_partial() || b.has_partial(),
        }
    }
    pub fn has_temporal(&self) -> bool { self.temporal_nesting() > 0 }
    pub fn is_nonlinear(&self) -> bool {
        match self {
            Node::Var { .. } | Node::Const { .. } => false,
            Node::Un { f, a } => !matches!(f, Un::Neg) || a.is_nonlinear(),
            Node::Bin { f, a, b } => matches!(f, Bin::Mul | Bin::Div) || a.is_nonlinear() || b.is_nonlinear(),
            Node::Tmp { f, a, .. } => matches!(f, Tmp::RollStd) || a.is_nonlinear(),
            Node::Partial { a, .. } => a.is_nonlinear(),
        }
    }
    pub fn vars(&self, out: &mut Vec<usize>) {
        match self {
            Node::Var { idx } => { if !out.contains(idx) { out.push(*idx) } }
            Node::Const { .. } => {}
            Node::Un { a, .. } | Node::Tmp { a, .. } | Node::Partial { a, .. } => a.vars(out),
            Node::Bin { a, b, .. } => { a.vars(out); b.vars(out) }
        }
    }
    pub fn formula(&self, names: &[String]) -> String {
        match self {
            Node::Var { idx } => names.get(*idx).cloned().unwrap_or_else(|| format!("x{}", idx)),
            Node::Const { value } => format!("{}", value),
            Node::Un { f, a } => format!("{}({})", f.name(), a.formula(names)),
            Node::Bin { f, a, b } => format!("({} {} {})", a.formula(names), f.name(), b.formula(names)),
            Node::Tmp { f, w, a } => format!("{}[{}]({})", f.name(), w, a.formula(names)),
            Node::Partial { wrt, a } => format!("partial[{}]({})",
                names.get(*wrt).cloned().unwrap_or_else(|| format!("x{}", wrt)), a.formula(names)),
        }
    }
    /// Structural validity: depth, nesting limits, partial only over pure arithmetic.
    pub fn valid(&self, max_height: u32, max_nesting: u32, n_vars: usize) -> bool {
        if self.height() > max_height || self.temporal_nesting() > max_nesting { return false; }
        fn ok(n: &Node, n_vars: usize, in_partial: bool) -> bool {
            match n {
                Node::Var { idx } => *idx < n_vars,
                Node::Const { value } => value.is_finite(),
                Node::Un { a, .. } => ok(a, n_vars, in_partial),
                Node::Bin { a, b, .. } => ok(a, n_vars, in_partial) && ok(b, n_vars, in_partial),
                Node::Tmp { w, a, .. } => *w >= 1 && !in_partial && ok(a, n_vars, in_partial),
                Node::Partial { wrt, a } => !in_partial && *wrt < n_vars && !a.has_temporal() && ok(a, n_vars, true),
            }
        }
        ok(self, n_vars, false)
    }
    /// Collect paths (child index sequences) to every node.
    pub fn paths(&self, prefix: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
        out.push(prefix.clone());
        match self {
            Node::Var { .. } | Node::Const { .. } => {}
            Node::Un { a, .. } | Node::Tmp { a, .. } | Node::Partial { a, .. } => {
                prefix.push(0); a.paths(prefix, out); prefix.pop();
            }
            Node::Bin { a, b, .. } => {
                prefix.push(0); a.paths(prefix, out); prefix.pop();
                prefix.push(1); b.paths(prefix, out); prefix.pop();
            }
        }
    }
    pub fn at(&self, path: &[u8]) -> &Node {
        if path.is_empty() { return self; }
        match (self, path[0]) {
            (Node::Un { a, .. }, 0) | (Node::Tmp { a, .. }, 0) | (Node::Partial { a, .. }, 0) | (Node::Bin { a, .. }, 0) => a.at(&path[1..]),
            (Node::Bin { b, .. }, 1) => b.at(&path[1..]),
            _ => self,
        }
    }
    pub fn replace(&self, path: &[u8], rep: &Node) -> Node {
        if path.is_empty() { return rep.clone(); }
        match (self, path[0]) {
            (Node::Un { f, a }, 0) => Node::Un { f: *f, a: Box::new(a.replace(&path[1..], rep)) },
            (Node::Tmp { f, w, a }, 0) => Node::Tmp { f: *f, w: *w, a: Box::new(a.replace(&path[1..], rep)) },
            (Node::Partial { wrt, a }, 0) => Node::Partial { wrt: *wrt, a: Box::new(a.replace(&path[1..], rep)) },
            (Node::Bin { f, a, b }, 0) => Node::Bin { f: *f, a: Box::new(a.replace(&path[1..], rep)), b: b.clone() },
            (Node::Bin { f, a, b }, 1) => Node::Bin { f: *f, a: a.clone(), b: Box::new(b.replace(&path[1..], rep)) },
            _ => self.clone(),
        }
    }
}

// ----------------------------------------------------------------------------
// Kernels (all causal; NaN-propagating; results bounded to +-1e6)
// ----------------------------------------------------------------------------

#[inline(always)]
fn san(v: f64) -> f32 {
    if v.is_finite() { v.clamp(-CLIP, CLIP) as f32 } else { f32::NAN }
}

pub fn un_k(f: Un, x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| {
        let v = v as f64;
        san(match f {
            Un::Sin => v.sin(),
            Un::Cos => v.cos(),
            Un::Sqrt => v.abs().sqrt(),
            Un::Pow2 => v * v,
            Un::Pow3 => v * v * v,
            Un::Abs => v.abs(),
            Un::SLog => v.signum() * v.abs().ln_1p(),
            Un::Neg => -v,
        })
    }).collect()
}

pub fn bin_k(f: Bin, x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&a, &b)| {
        let (a, b) = (a as f64, b as f64);
        san(match f {
            Bin::Add => a + b,
            Bin::Sub => a - b,
            Bin::Mul => a * b,
            Bin::Div => if b.abs() >= DIV_EPS { a / b } else { f64::NAN },
        })
    }).collect()
}

fn lagged(x: &[f32], w: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![f32::NAN; n];
    if w < n { out[w..].copy_from_slice(&x[..n - w]); }
    out
}

/// Window statistics over the last `w` rows including the current one; NaN unless all finite.
fn rolling(x: &[f32], w: usize, std: bool) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![f32::NAN; n];
    let mut bad_run = 0usize; // consecutive finite count
    for i in 0..n {
        if x[i].is_finite() { bad_run += 1 } else { bad_run = 0 }
        if i + 1 < w || bad_run < w { continue; }
        let seg = &x[i + 1 - w..=i];
        let m = seg.iter().map(|&v| v as f64).sum::<f64>() / w as f64;
        out[i] = if std {
            let var = seg.iter().map(|&v| { let d = v as f64 - m; d * d }).sum::<f64>() / w as f64;
            san(var.sqrt())
        } else { san(m) };
    }
    out
}

pub fn tmp_k(f: Tmp, w: u32, x: &[f32]) -> Vec<f32> {
    let w = w as usize;
    match f {
        Tmp::Lag => lagged(x, w),
        Tmp::Diff | Tmp::Deriv => {
            let l = lagged(x, w);
            let s = if f == Tmp::Deriv { 1.0 / w as f64 } else { 1.0 };
            x.iter().zip(l.iter()).map(|(&a, &b)| san((a as f64 - b as f64) * s)).collect()
        }
        Tmp::RollMean => rolling(x, w, false),
        Tmp::RollStd => rolling(x, w, true),
    }
}

// ----------------------------------------------------------------------------
// Evaluation (values) and forward-mode dual evaluation (value + d/d x_wrt)
// ----------------------------------------------------------------------------

pub fn eval(node: &Node, cols: &[Vec<f32>]) -> Vec<f32> {
    match node {
        Node::Var { idx } => cols[*idx].clone(),
        Node::Const { value } => vec![san(*value); cols[0].len()],
        Node::Un { f, a } => un_k(*f, &eval(a, cols)),
        Node::Bin { f, a, b } => bin_k(*f, &eval(a, cols), &eval(b, cols)),
        Node::Tmp { f, w, a } => tmp_k(*f, *w, &eval(a, cols)),
        Node::Partial { wrt, a } => eval_dual(a, cols, *wrt).1,
    }
}

pub fn eval_dual(node: &Node, cols: &[Vec<f32>], wrt: usize) -> (Vec<f32>, Vec<f32>) {
    let n = cols[0].len();
    match node {
        Node::Var { idx } => (cols[*idx].clone(), vec![if *idx == wrt { 1.0 } else { 0.0 }; n]),
        Node::Const { value } => (vec![san(*value); n], vec![0.0; n]),
        Node::Partial { .. } => (vec![f32::NAN; n], vec![f32::NAN; n]), // nested partials are invalid
        Node::Un { f, a } => {
            let (x, dx) = eval_dual(a, cols, wrt);
            let val = un_k(*f, &x);
            let tan = x.iter().zip(dx.iter()).map(|(&xv, &dv)| {
                let (x, d) = (xv as f64, dv as f64);
                san(match f {
                    Un::Sin => x.cos() * d,
                    Un::Cos => -x.sin() * d,
                    Un::Sqrt => { let s = x.abs().sqrt(); if s > 1e-9 { x.signum() * d / (2.0 * s) } else { f64::NAN } }
                    Un::Pow2 => 2.0 * x * d,
                    Un::Pow3 => 3.0 * x * x * d,
                    Un::Abs => x.signum() * d,
                    Un::SLog => d / (1.0 + x.abs()),
                    Un::Neg => -d,
                })
            }).collect();
            (val, tan)
        }
        Node::Bin { f, a, b } => {
            let (x, dx) = eval_dual(a, cols, wrt);
            let (y, dy) = eval_dual(b, cols, wrt);
            let val = bin_k(*f, &x, &y);
            let tan = (0..n).map(|i| {
                let (xv, yv, dxv, dyv) = (x[i] as f64, y[i] as f64, dx[i] as f64, dy[i] as f64);
                san(match f {
                    Bin::Add => dxv + dyv,
                    Bin::Sub => dxv - dyv,
                    Bin::Mul => dxv * yv + xv * dyv,
                    Bin::Div => if yv.abs() >= DIV_EPS { (dxv * yv - xv * dyv) / (yv * yv) } else { f64::NAN },
                })
            }).collect();
            (val, tan)
        }
        Node::Tmp { f, w, a } => {
            let (x, dx) = eval_dual(a, cols, wrt);
            match f {
                Tmp::Lag | Tmp::Diff | Tmp::Deriv | Tmp::RollMean => (tmp_k(*f, *w, &x), tmp_k(*f, *w, &dx)),
                Tmp::RollStd => {
                    let xdx: Vec<f32> = x.iter().zip(dx.iter()).map(|(&a, &b)| san(a as f64 * b as f64)).collect();
                    let s = rolling(&x, *w as usize, true);
                    let m_x = rolling(&x, *w as usize, false);
                    let m_d = rolling(&dx, *w as usize, false);
                    let m_xd = rolling(&xdx, *w as usize, false);
                    let tan = (0..n).map(|i| {
                        let sv = s[i] as f64;
                        if sv > 1e-9 { san((m_xd[i] as f64 - m_x[i] as f64 * m_d[i] as f64) / sv) } else { f32::NAN }
                    }).collect();
                    (s, tan)
                }
            }
        }
    }
}

//! gpcore: pure-Rust GP feature search exposed to Python via PyO3.
pub mod expr;
pub mod search;
pub mod sim;

use expr::Node;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use search::{Cfg, Search};

fn cols_of(m: &PyReadonlyArray2<f64>) -> Vec<Vec<f32>> {
    let a = m.as_array();
    (0..a.ncols()).map(|j| a.column(j).iter().map(|&v| v as f32).collect()).collect()
}

fn pool(threads: usize) -> PyResult<rayon::ThreadPool> {
    let b = rayon::ThreadPoolBuilder::new();
    (if threads > 0 { b.num_threads(threads) } else { b })
        .build()
        .map_err(|error| PyValueError::new_err(format!("build GP thread pool: {error}")))
}

fn parse_cfg(s: &str) -> PyResult<Cfg> {
    serde_json::from_str::<Cfg>(s).map_err(|e| PyValueError::new_err(format!("bad config: {e}")))
}

fn normalized_matrix(a: numpy::ndarray::ArrayView2<'_, f64>, window: usize,
                     min_periods: usize, clip_z: f64) -> numpy::ndarray::Array2<f64> {
    let (n, p) = a.dim();
    let mut out = numpy::ndarray::Array2::<f64>::from_elem((n, p), f64::NAN);
    for j in 0..p {
        let col: Vec<f64> = a.column(j).to_vec();
        let shift = col.iter().copied().find(|v| v.is_finite()).unwrap_or(0.0);
        let (mut cnt, mut s1, mut s2) = (0usize, 0f64, 0f64);
        for t in 0..n {
            if cnt >= min_periods {
                let m = s1 / cnt as f64;
                let var = (s2 / cnt as f64 - m * m).max(0.0);
                let sd = var.sqrt();
                if col[t].is_finite() && sd > 1e-12 {
                    out[[t, j]] = ((col[t] - shift - m) / sd).clamp(-clip_z, clip_z);
                }
            }
            if col[t].is_finite() {
                let v = col[t] - shift;
                cnt += 1;
                s1 += v;
                s2 += v * v;
            }
            if t >= window {
                let old = col[t - window];
                if old.is_finite() {
                    let v = old - shift;
                    cnt -= 1;
                    s1 -= v;
                    s2 -= v * v;
                }
            }
        }
    }
    out
}

fn forbidden_terminal(name: &str) -> bool {
    let lower = name.to_lowercase();
    ["future_", "target", "label", "barrier", "latent", "trade_allowed", "embedding"]
        .iter().any(|part| lower.contains(part))
        || lower.contains("timestamp")
        || lower.ends_with("_id")
}

fn price_level_terminal(name: &str) -> bool {
    let parts: Vec<String> = name.to_lowercase().replace('-', "_")
        .split('_').map(str::to_string).collect();
    if parts.iter().any(|part| matches!(part.as_str(),
        "return" | "returns" | "ret" | "change" | "pct" | "percent" | "qty" |
        "quantity" | "size" | "volume" | "depth" | "imbalance" | "flow" |
        "count" | "ratio")) {
        return false;
    }
    ["price", "close", "mid", "last"].iter().any(|part| parts.iter().any(|p| p == part))
        || matches!(parts.last().map(String::as_str), Some("bid" | "ask"))
}

fn ranked(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).filter(|&i| values[i].is_finite()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![f64::NAN; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[start]] == values[order[end]] {
            end += 1;
        }
        let rank = (start + end - 1) as f64 / 2.0;
        for &i in &order[start..end] {
            ranks[i] = rank;
        }
        start = end;
    }
    ranks
}

fn pairwise_corr(x: &[f64], y: &[f64]) -> f64 {
    let (mut count, mut sx, mut sy, mut sxx, mut syy, mut sxy) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            count += 1.0;
            sx += a;
            sy += b;
            sxx += a * a;
            syy += b * b;
            sxy += a * b;
        }
    }
    if count < 2.0 {
        return f64::NAN;
    }
    let vx = count * sxx - sx * sx;
    let vy = count * syy - sy * sy;
    let denom = (vx * vy).sqrt();
    if denom <= 1e-12 {
        f64::NAN
    } else {
        ((count * sxy - sx * sy) / denom).clamp(-1.0, 1.0)
    }
}

/// Greedy exact pairwise correlation pruning over evenly sampled verification rows.
#[pyfunction]
#[pyo3(signature = (values, threshold, metric, verify_rows=12000, threads=0))]
fn correlation_prune(
    py: Python<'_>,
    values: PyReadonlyArray2<'_, f64>,
    threshold: f64,
    metric: &str,
    verify_rows: usize,
    threads: usize,
) -> PyResult<Vec<usize>> {
    if !threshold.is_finite() || threshold <= 0.0 || threshold > 1.0 {
        return Err(PyValueError::new_err("threshold must be in (0, 1]"));
    }
    let rank_metric = match metric {
        "pearson" => false,
        "spearman" => true,
        _ => return Err(PyValueError::new_err("metric must be pearson or spearman")),
    };
    let matrix = values.as_array();
    if matrix.ncols() == 0 || matrix.nrows() < 2 {
        return Err(PyValueError::new_err("correlation pruning needs at least two rows and one column"));
    }
    let sample_count = verify_rows.min(matrix.nrows());
    if sample_count < 2 {
        return Err(PyValueError::new_err("verify_rows must be at least 2"));
    }
    let positions: Vec<usize> = (0..sample_count)
        .map(|i| i * matrix.nrows() / sample_count)
        .collect();
    let mut columns: Vec<Vec<f64>> = (0..matrix.ncols())
        .map(|column| positions.iter().map(|&row| matrix[[row, column]]).collect())
        .collect();
    if rank_metric {
        columns = columns.iter().map(|column| ranked(column)).collect();
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(if threads == 0 { rayon::current_num_threads() } else { threads })
        .build()
        .map_err(|e| PyValueError::new_err(format!("cannot create correlation worker pool: {e}")))?;

    let kept = py.allow_threads(move || {
        let mut kept: Vec<usize> = Vec::new();
        for candidate in 0..columns.len() {
            if !pairwise_corr(&columns[candidate], &columns[candidate]).is_finite() {
                continue;
            }
            let redundant = pool.install(|| {
                kept.par_iter().any(|&selected| {
                    pairwise_corr(&columns[candidate], &columns[selected]).abs() >= threshold
                })
            });
            if !redundant {
                kept.push(candidate);
            }
        }
        kept
    });
    Ok(kept)
}

/// prepare_terminals(raw[n x p], names, input_columns, window, min_periods, clip_z)
/// -> (causal normalized float64 matrix, selected terminal names).
#[pyfunction]
#[pyo3(signature = (raw, names, input_columns=None, window=500, min_periods=50, clip_z=8.0))]
fn prepare_terminals<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'py, f64>,
    names: Vec<String>,
    input_columns: Option<Vec<String>>,
    window: usize,
    min_periods: usize,
    clip_z: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    if window < 2 || min_periods < 2 || min_periods > window {
        return Err(PyValueError::new_err("need 2 <= min_periods <= window"));
    }
    if !clip_z.is_finite() || clip_z <= 0.0 {
        return Err(PyValueError::new_err("clip_z must be finite and positive"));
    }
    let values = raw.as_array();
    if names.len() != values.ncols() {
        return Err(PyValueError::new_err("names/columns mismatch"));
    }
    let selected: Vec<usize> = if let Some(requested) = input_columns {
        requested.iter().map(|name| {
            names.iter().position(|candidate| candidate == name)
                .ok_or_else(|| PyValueError::new_err(format!("missing GP input column {name:?}")))
        }).collect::<PyResult<_>>()?
    } else {
        names.iter().enumerate().filter_map(|(i, name)| {
            (!forbidden_terminal(name)).then_some(i)
        }).collect()
    };
    if selected.is_empty() {
        return Err(PyValueError::new_err("no eligible numeric GP input columns"));
    }

    let mut raw_terminals = numpy::ndarray::Array2::<f64>::from_elem(
        (values.nrows(), selected.len()), f64::NAN);
    let terminal_names: Vec<String> = selected.iter().map(|&i| names[i].clone()).collect();
    for (out_col, &source_col) in selected.iter().enumerate() {
        let source = values.column(source_col);
        if price_level_terminal(&names[source_col]) {
            for row in 1..values.nrows() {
                let previous = source[row - 1];
                let current = source[row];
                if previous.is_finite() && current.is_finite() && previous > 0.0 && current > 0.0 {
                    raw_terminals[[row, out_col]] = current.ln() - previous.ln();
                }
            }
        } else {
            for row in 0..values.nrows() {
                raw_terminals[[row, out_col]] = source[row];
            }
        }
    }
    let normalized = normalized_matrix(raw_terminals.view(), window, min_periods, clip_z);
    Ok((normalized.into_pyarray_bound(py), terminal_names))
}

#[pyfunction]
fn default_config_json() -> PyResult<String> {
    serde_json::to_string(&Cfg::default()).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// search(terminals[n x p] float64, names, cfg_json, target=None) -> library_json
#[pyfunction]
#[pyo3(name = "search", signature = (terminals, names, cfg_json, target=None))]
fn run_search(py: Python<'_>, terminals: PyReadonlyArray2<f64>, names: Vec<String>, cfg_json: &str,
          target: Option<PyReadonlyArray1<f64>>) -> PyResult<String> {
    let cfg = parse_cfg(cfg_json)?;
    let cols = cols_of(&terminals);
    if names.len() != cols.len() { return Err(PyValueError::new_err("names/columns mismatch")); }
    let tgt = target.map(|t| t.as_array().iter().map(|&v| v as f32).collect::<Vec<f32>>());
    let threads = cfg.threads;
    py.allow_threads(move || {
        let mut s = Search::new(cols, names, tgt, cfg).map_err(PyValueError::new_err)?;
        pool(threads)?.install(|| s.run()).map_err(PyValueError::new_err)?;
        s.library_json().map_err(|error| PyValueError::new_err(error.to_string()))
    })
}

/// transform(terminals, library_json, threads=0) -> float32 matrix [n x K] of selected features
#[pyfunction]
#[pyo3(signature = (terminals, library_json, threads=0))]
fn transform<'py>(py: Python<'py>, terminals: PyReadonlyArray2<f64>, library_json: &str, threads: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let cols = cols_of(&terminals);
    let n = cols[0].len();
    let v: serde_json::Value = serde_json::from_str(library_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let trees: Vec<Node> = v["features"].as_array().ok_or_else(|| PyValueError::new_err("no features"))?
        .iter().map(|f| serde_json::from_value::<Node>(f["tree"].clone()).map_err(|e| PyValueError::new_err(e.to_string())))
        .collect::<Result<_, _>>()?;
    let k = trees.len();
    let out: Vec<Vec<f32>> = py.allow_threads(|| pool(threads).map(|pool| pool.install(|| trees.par_iter().map(|t| expr::eval(t, &cols)).collect())))?;
    let mut flat = vec![0f32; n * k];
    for (j, col) in out.iter().enumerate() { for i in 0..n { flat[i * k + j] = col[i]; } }
    let array = numpy::ndarray::Array2::from_shape_vec((n, k), flat)
        .map_err(|error| PyValueError::new_err(format!("build transformed array: {error}")))?;
    Ok(array.into_pyarray_bound(py))
}

/// eval_tree(tree_json, terminals) -> float64 vector (used to cross-check against the legacy evaluator)
#[pyfunction]
fn eval_tree<'py>(py: Python<'py>, tree_json: &str, terminals: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let node: Node = serde_json::from_str(tree_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cols = cols_of(&terminals);
    let v = expr::eval(&node, &cols);
    Ok(v.iter().map(|&x| x as f64).collect::<Vec<f64>>().into_pyarray_bound(py))
}

/// eval_trees(list_of_tree_json, terminals) -> total checksum + count (timing helper, no per-tree Python overhead)
#[pyfunction]
fn eval_trees(py: Python<'_>, trees: Vec<String>, terminals: PyReadonlyArray2<f64>, threads: usize) -> PyResult<f64> {
    let cols = cols_of(&terminals);
    let nodes: Vec<Node> = trees.iter().map(|t| serde_json::from_str(t).map_err(|e| PyValueError::new_err(e.to_string()))).collect::<Result<_, _>>()?;
    py.allow_threads(|| pool(threads).map(|pool| pool.install(|| nodes.par_iter().map(|t| {
        let v = expr::eval(t, &cols);
        v.iter().filter(|x| x.is_finite()).map(|&x| x as f64).sum::<f64>()
    }).sum())) )
}

/// sample_trees(names, n_trees, depth, cfg_json, seed) -> list of tree JSON strings
#[pyfunction]
fn sample_trees(n_vars: usize, count: usize, depth: u32, cfg_json: &str) -> PyResult<Vec<String>> {
    let cfg = parse_cfg(cfg_json)?;
    let n = 400usize.max(cfg.windows.iter().copied().max().unwrap_or(5) as usize * 4 + 200);
    let cols: Vec<Vec<f32>> = (0..n_vars).map(|j| (0..n).map(|i| ((i * (j + 3)) % 17) as f32 + 1.0).collect()).collect();
    let names: Vec<String> = (0..n_vars).map(|j| format!("x{j}")).collect();
    let mut s = Search::new(cols, names, None, cfg).map_err(PyValueError::new_err)?;
    s.sample_trees(count, depth).iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// dedupe(matrix[n x K], cfg_json) -> (kept_indices, exact_reference_indices_or_empty, stats_json)
/// Level-0 only: runs the LSH hashmap over the supplied columns in order.
#[pyfunction]
fn dedupe(py: Python<'_>, values: PyReadonlyArray2<f64>, cfg_json: &str) -> PyResult<(Vec<usize>, Vec<usize>, String)> {
    let mut cfg = parse_cfg(cfg_json)?;
    cfg.levels = 0;
    cfg.exact_check_max = 0;
    let cols = cols_of(&values);
    let names: Vec<String> = (0..cols.len()).map(|j| format!("c{j}")).collect();
    let want_ref = cfg.exact_reference;
    let threads = cfg.threads;
    py.allow_threads(move || {
        let mut s = Search::new(cols, names, None, cfg).map_err(PyValueError::new_err)?;
        pool(threads)?.install(|| s.run()).map_err(PyValueError::new_err)?;
        let kept = s.admitted_terminal_ids();
        let reference = if want_ref { s.exact_reference_ids() } else { Vec::new() };
        let stats = serde_json::to_string(&s.stats).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((kept, reference, stats))
    })
}

/// normalize(matrix[n x p], window, min_periods, clip_z) -> causal rolling z-score using preceding rows only.
#[pyfunction]
fn normalize<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>, window: usize, min_periods: usize, clip_z: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if window < 2 || min_periods < 2 || min_periods > window { return Err(PyValueError::new_err("need 2 <= min_periods <= window")); }
    let out = normalized_matrix(x.as_array(), window, min_periods, clip_z);
    Ok(out.into_pyarray_bound(py))
}

#[pymodule]
fn gpcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prepare_terminals, m)?)?;
    m.add_function(wrap_pyfunction!(default_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_prune, m)?)?;
    m.add_function(wrap_pyfunction!(run_search, m)?)?;
    m.add_function(wrap_pyfunction!(transform, m)?)?;
    m.add_function(wrap_pyfunction!(eval_tree, m)?)?;
    m.add_function(wrap_pyfunction!(eval_trees, m)?)?;
    m.add_function(wrap_pyfunction!(sample_trees, m)?)?;
    m.add_function(wrap_pyfunction!(dedupe, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{pairwise_corr, price_level_terminal, ranked};

    #[test]
    fn correlation_pruning_math_handles_linear_and_tied_data() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        assert!((pairwise_corr(&x, &y) - 1.0).abs() < 1e-12);
        assert_eq!(ranked(&[2.0, 2.0, 5.0]), vec![0.5, 0.5, 2.0]);
    }

    #[test]
    fn terminal_name_rules_distinguish_prices_from_sizes() {
        assert!(price_level_terminal("best_bid_price"));
        assert!(!price_level_terminal("best_bid_qty"));
        assert!(!price_level_terminal("close_return"));
    }
}

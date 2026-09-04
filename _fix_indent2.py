#!/usr/bin/env python3
"""Rewrite load_regime_gate function with correct indentation."""

p = r"uets_rs/live-trade/src/model.rs"
lines = open(p, encoding="utf-8").read().split("\n")

# Find the function boundaries
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if line.startswith("pub fn load_regime_gate"):
        start_idx = i
    elif start_idx is not None and line.startswith("fn nodes_are_consistent"):
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print(f"ERROR: start={start_idx} end={end_idx}")
    exit(1)

new_func = '''pub fn load_regime_gate(symbol: &str) -> Option<engine::RegimeGate> {
    let per_sym = format!("build/regime_gate_{symbol}.json");
    let path = find_model_path(&[per_sym.as_str(), "build/regime_gate.json"])?;
    let text = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let features: Vec<String> = v
        .get("features")?
        .as_array()?
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
    let nodes: Vec<engine::GateNode> = v
        .get("nodes")?
        .as_array()?
        .iter()
        .map(|x| {
            let arr = x.as_array()?;
            if arr.len() != 4 {
                return None;
            }
            Some(engine::GateNode {
                feature: arr[0].as_i64().unwrap_or(0) as i32,
                threshold: arr[1].as_f64().unwrap_or(0.0),
                left: arr[2].as_i64().unwrap_or(-1) as i32,
                right: arr[3].as_i64().unwrap_or(-1) as i32,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let favorable: Vec<bool> = v
        .get("favorable")?
        .as_array()?
        .iter()
        .map(|x| x.as_bool().unwrap_or(false))
        .collect();
    if features.len() != 15 {
        eprintln!(
            "[regime] {symbol}: expected 15 features in export, found {} — gate OPEN",
            features.len()
        );
        return None;
    }
    if nodes.is_empty() || favorable.is_empty() || !nodes_are_consistent(&nodes, favorable.len()) {
        eprintln!("[regime] {symbol}: empty or inconsistent nodes in {} — gate OPEN", path.display());
        return None;
    }
    println!(
        "[regime] {symbol}: regime gate loaded from {} ({} nodes, {} favorable)",
        path.display(),
        nodes.len(),
        favorable.iter().filter(|&&t| t).count()
    );
    Some(engine::RegimeGate {
        features,
        nodes,
        favorable,
    })
}'''

new_lines = lines[:start_idx] + new_func.split("\n") + lines[end_idx:]
open(p, "w", encoding="utf-8").write("\n".join(new_lines))
print(f"Replaced lines {start_idx+1}-{end_idx} with correct function")



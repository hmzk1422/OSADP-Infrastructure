use pyo3::prelude::*;

mod live;

#[pymodule]
fn moe_live(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register live module or exported functions here
    Ok(())
}
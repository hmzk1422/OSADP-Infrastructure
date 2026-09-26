# Stress tests

Run the Python parity tests with:

```text
python -m unittest discover components/rust/researcher-toolkit/stresstest
```

The tests intentionally use exact decimal-friendly inputs and compare the
same feature graph with a small reference evaluator.  They are also examples
for adding a Rust-side golden-vector test when a new operator is introduced.

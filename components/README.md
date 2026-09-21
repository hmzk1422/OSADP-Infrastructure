# Components Folder
Last updated since v1.0.0

Components folder contains isolated, reusable executors or libraries responsible for one well-defined task.

# Root File Structure
Executors/libraries are separated based on the workload and appropriate languages.
- Python: Useful for research, statistical inferences, hypothesis, diagnosis, etc.
- Rust: Useful for heavy workload such as rebuilding orderbook, managing DSA-based problems, etc.
```
# Introduction to Writing Python/Rust Libraries
Each component must own one clearly defined responsibility and live in its own directory.
```bash
Python
|_ Name_Of_Library
    |_ main.py      # Optional (If headless call is needed)
    |_ ...

Rust
|_ Name_Of_Library
   |_ main.rs       # Optional (If headless call is needed)
   |_ ...
```
Note:
1. Keep main.py at root directory of library. Every python file can be accessed using `python -m components.python.{library name}.main` whereas with rust, add in cargo.toml a new bin for it, `then run cargo run --release --bin {bin_name}`
    ```bash
    # example of cargo.toml
    [[bin]]
    name = "lob_builder"
    path = "rust/lob_builder/main.rs"
    ```
2. Always add a README.md and proper comments on each library. Include the following:
    - Version of OSADP-core it is added in/last updated in
    - Name of developer who contributed to this library
    - Documentation on how to use the library

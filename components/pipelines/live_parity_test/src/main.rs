//! Generic parity runner for Python and Rust strategies.
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, ExitCode, Stdio};

struct Config {
    data_path: String,
    python_program: String,
    python_args: Vec<String>,
    python_env: Vec<(String, String)>,
    rust_program: String,
    rust_args: Vec<String>,
    rust_env: Vec<(String, String)>,
    python_out: String,
    rust_out: String,
    timestamp_col: String,
    feature_cols: Vec<String>,
    trade_cols: Vec<String>,
    threshold: f64,
    max_mae: f64,
    symbol: String,
    tag: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_path: String::new(),
            python_program: String::new(),
            python_args: Vec::new(),
            python_env: Vec::new(),
            rust_program: String::new(),
            rust_args: Vec::new(),
            rust_env: Vec::new(),
            python_out: "build\\live_parity_test\\python.csv".into(),
            rust_out: "build\\live_parity_test\\rust.csv".into(),
            timestamp_col: "timestamp_ms".into(),
            feature_cols: Vec::new(),
            trade_cols: Vec::new(),
            threshold: 0.9999,
            max_mae: 1e-5,
            symbol: String::new(),
            tag: String::new(),
        }
    }
}

struct Metric {
    dot: f64,
    norm_a_squared: f64,
    norm_b_squared: f64,
    absolute_error: f64,
    mismatches: usize,
}

impl Metric {
    fn new() -> Self {
        Self {
            dot: 0.0,
            norm_a_squared: 0.0,
            norm_b_squared: 0.0,
            absolute_error: 0.0,
            mismatches: 0,
        }
    }

    fn add(&mut self, a: f64, b: f64) {
        self.dot += a * b;
        self.norm_a_squared += a * a;
        self.norm_b_squared += b * b;
        self.absolute_error += (a - b).abs();
        if a != b {
            self.mismatches += 1;
        }
    }

    fn cosine_similarity(&self) -> f64 {
        let norm_a = self.norm_a_squared.sqrt();
        let norm_b = self.norm_b_squared.sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            if self.mismatches == 0 {
                1.0
            } else {
                0.0
            }
        } else {
            self.dot / (norm_a * norm_b)
        }
    }

    fn mean_absolute_error(&self, rows: usize) -> f64 {
        if rows == 0 {
            0.0
        } else {
            self.absolute_error / rows as f64
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(passed) => {
            if passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("Live parity test failed: {error}");
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, String> {
    let config = parse_args()?;
    if config.data_path.is_empty()
        || config.python_program.is_empty()
        || config.rust_program.is_empty()
        || config.symbol.is_empty()
    {
        return Err(
            "--data-path, --symbol, --python-program, and --rust-program are required".into(),
        );
    }
    if !Path::new(&config.data_path).exists() {
        return Err(format!(
            "input data path does not exist: {}",
            config.data_path
        ));
    }
    if config.feature_cols.is_empty() {
        return Err("--feature-cols must list at least one numeric feature column".into());
    }
    if !config.threshold.is_finite() || !(0.0..=1.0).contains(&config.threshold) {
        return Err("--threshold must be finite and between 0 and 1".into());
    }
    if !config.max_mae.is_finite() || config.max_mae < 0.0 {
        return Err("--max-mae must be finite and non-negative".into());
    }

    for path in [&config.python_out, &config.rust_out] {
        if let Some(parent) = Path::new(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| {
                format!("create output directory {}: {error}", parent.display())
            })?;
        }
    }

    println!("=== Live Python/Rust Strategy Parity ===");
    println!("Input data:  {}", config.data_path);
    println!("Symbol:      {}", config.symbol);
    println!("Tag:         {}", config.tag);
    println!("Features:    {:?}", config.feature_cols);
    println!("Trade path:  {:?}", config.trade_cols);
    println!("Threshold:   {}", config.threshold);

    run_command(
        "Python",
        &config.python_program,
        &config.python_args,
        &config.python_env,
        &config,
    )?;
    run_command(
        "Rust",
        &config.rust_program,
        &config.rust_args,
        &config.rust_env,
        &config,
    )?;

    compare_outputs(&config)
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let args: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let value = |index: usize, option: &str| {
            args.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{option} requires a value"))
        };
        match args[index].as_str() {
            "--data-path" => config.data_path = value(index, "--data-path")?,
            "--python-program" => config.python_program = value(index, "--python-program")?,
            "--python-arg" => config.python_args.push(value(index, "--python-arg")?),
            "--python-env" => config
                .python_env
                .push(parse_env(&value(index, "--python-env")?)?),
            "--rust-program" => config.rust_program = value(index, "--rust-program")?,
            "--rust-arg" => config.rust_args.push(value(index, "--rust-arg")?),
            "--rust-env" => config
                .rust_env
                .push(parse_env(&value(index, "--rust-env")?)?),
            "--python-out" => config.python_out = value(index, "--python-out")?,
            "--rust-out" => config.rust_out = value(index, "--rust-out")?,
            "--timestamp-col" => config.timestamp_col = value(index, "--timestamp-col")?,
            "--feature-cols" => {
                config.feature_cols = parse_columns(&value(index, "--feature-cols")?);
            }
            "--trade-cols" => {
                config.trade_cols = parse_columns(&value(index, "--trade-cols")?);
            }
            "--threshold" => {
                config.threshold = value(index, "--threshold")?
                    .parse()
                    .map_err(|error| format!("invalid --threshold: {error}"))?;
            }
            "--max-mae" => {
                config.max_mae = value(index, "--max-mae")?
                    .parse()
                    .map_err(|error| format!("invalid --max-mae: {error}"))?;
            }
            "--symbol" => config.symbol = value(index, "--symbol")?,
            "--tag" => config.tag = value(index, "--tag")?,
            "--help" | "-h" => {
                print_usage();
                return Err("help requested".into());
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 2;
    }
    Ok(config)
}

fn parse_env(value: &str) -> Result<(String, String), String> {
    let (key, value) = value
        .split_once('=')
        .ok_or_else(|| "environment overrides must use KEY=VALUE".to_string())?;
    if key.is_empty() {
        return Err("environment override keys must not be empty".into());
    }
    Ok((key.to_string(), value.to_string()))
}

fn parse_columns(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|column| !column.is_empty())
        .map(str::to_string)
        .collect()
}

fn print_usage() {
    eprintln!(concat!(
        "Usage: live_parity_test --data-path <matched-data-or-input-path>\n",
        "  --python-program <executable> [--python-arg <arg> ...] [--python-env KEY=VALUE]\n",
        "  --rust-program <executable> [--rust-arg <arg> ...] [--rust-env KEY=VALUE]\n",
        "  --feature-cols <col1,col2,...> [--trade-cols <col1,col2,...>]\n",
        "  [--timestamp-col <numeric-column>] [--python-out <csv>] [--rust-out <csv>]\n",
        "  [--threshold <cosine>] [--max-mae <value>] [--symbol <symbol>] [--tag <tag>]\n",
        "Child processes receive PARITY_DATA_PATH, PARITY_SYMBOL, PARITY_TAG, ",
        "PARITY_PYTHON_OUT, and PARITY_RUST_OUT."
    ));
}

fn run_command(
    label: &str,
    program: &str,
    args: &[String],
    environment: &[(String, String)],
    config: &Config,
) -> Result<(), String> {
    println!("[{}] {} {:?}", label, program, args);
    let status = Command::new(program)
        .args(args)
        .env("PARITY_DATA_PATH", &config.data_path)
        .env("PARITY_SYMBOL", &config.symbol)
        .env("PARITY_TAG", &config.tag)
        .env("PARITY_PYTHON_OUT", &config.python_out)
        .env("PARITY_RUST_OUT", &config.rust_out)
        .envs(environment.iter().map(|(key, value)| (key, value)))
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("start {label} process {program:?}: {error}"))?;
    if !status.success() {
        return Err(format!(
            "{label} process exited unsuccessfully with status {status}"
        ));
    }
    Ok(())
}

fn compare_outputs(config: &Config) -> Result<bool, String> {
    let python = File::open(&config.python_out)
        .map_err(|error| format!("open Python output {}: {error}", config.python_out))?;
    let rust = File::open(&config.rust_out)
        .map_err(|error| format!("open Rust output {}: {error}", config.rust_out))?;
    let mut python = BufReader::new(python);
    let mut rust = BufReader::new(rust);
    let mut python_header = String::new();
    let mut rust_header = String::new();
    python
        .read_line(&mut python_header)
        .map_err(|error| format!("read Python CSV header: {error}"))?;
    rust.read_line(&mut rust_header)
        .map_err(|error| format!("read Rust CSV header: {error}"))?;
    let python_header = parse_csv_record(python_header.trim_end())?;
    let rust_header = parse_csv_record(rust_header.trim_end())?;
    let python_indices = column_indices(
        &python_header,
        &config.timestamp_col,
        &config.feature_cols,
        &config.trade_cols,
        "Python",
    )?;
    let rust_indices = column_indices(
        &rust_header,
        &config.timestamp_col,
        &config.feature_cols,
        &config.trade_cols,
        "Rust",
    )?;
    let mut feature_metrics: Vec<Metric> = (0..config.feature_cols.len())
        .map(|_| Metric::new())
        .collect();
    let mut trade_metrics: Vec<Metric> = (0..config.trade_cols.len())
        .map(|_| Metric::new())
        .collect();
    let mut python_line = String::new();
    let mut rust_line = String::new();
    let mut rows = 0usize;
    let mut skipped_rust_prefix = 0usize;
    let mut skipped_python = 0usize;
    let mut skipped_rust = 0usize;
    let mut python_row = read_csv_row(&mut python, &mut python_line, "Python")?;
    let mut rust_row = read_csv_row(&mut rust, &mut rust_line, "Rust")?;
    loop {
        match (&python_row, &rust_row) {
            (None, None) => break,
            (Some(_), None) => {
                skipped_python += 1;
                python_row = read_csv_row(&mut python, &mut python_line, "Python")?;
            }
            (None, Some(_)) => {
                skipped_rust += 1;
                rust_row = read_csv_row(&mut rust, &mut rust_line, "Rust")?;
            }
            (Some(python_values), Some(rust_values)) => {
                let python_timestamp =
                    parse_numeric(python_values, python_indices.timestamp, "timestamp")?;
                let rust_timestamp =
                    parse_numeric(rust_values, rust_indices.timestamp, "timestamp")?;
                if python_timestamp < rust_timestamp {
                    skipped_python += 1;
                    python_row = read_csv_row(&mut python, &mut python_line, "Python")?;
                    continue;
                }
                if rust_timestamp < python_timestamp {
                    if rows == 0 {
                        skipped_rust_prefix += 1;
                    } else {
                        skipped_rust += 1;
                    }
                    rust_row = read_csv_row(&mut rust, &mut rust_line, "Rust")?;
                    continue;
                }
                for index in 0..feature_metrics.len() {
                    let a = parse_numeric(
                        python_values,
                        python_indices.features[index],
                        &config.feature_cols[index],
                    )?;
                    let b = parse_numeric(
                        rust_values,
                        rust_indices.features[index],
                        &config.feature_cols[index],
                    )?;
                    feature_metrics[index].add(a, b);
                }
                for index in 0..trade_metrics.len() {
                    let a = parse_numeric(
                        python_values,
                        python_indices.trades[index],
                        &config.trade_cols[index],
                    )?;
                    let b = parse_numeric(
                        rust_values,
                        rust_indices.trades[index],
                        &config.trade_cols[index],
                    )?;
                    trade_metrics[index].add(a, b);
                }
                rows += 1;
                python_row = read_csv_row(&mut python, &mut python_line, "Python")?;
                rust_row = read_csv_row(&mut rust, &mut rust_line, "Rust")?;
            }
        }
    }
    if rows == 0 {
        return Err("Python and Rust outputs have no matching timestamps".into());
    }

    println!(
        "\n=== Feature-wise parity ({rows} aligned rows; Rust warm-up rows={skipped_rust_prefix}, unmatched Python rows={skipped_python}, unmatched Rust rows={skipped_rust}) ==="
    );
    let mut all_pass = skipped_python == 0 && skipped_rust == 0;
    let mut total_feature_cosine = 0.0;
    for (column, metric) in config.feature_cols.iter().zip(&feature_metrics) {
        let cosine = metric.cosine_similarity();
        let mae = metric.mean_absolute_error(rows);
        let pass = cosine >= config.threshold && mae <= config.max_mae;
        all_pass &= pass;
        total_feature_cosine += cosine;
        println!(
            "  {column}: cosine={cosine:.8}, MAE={mae:.8}, {}",
            if pass { "MATCH" } else { "DIVERGED" }
        );
    }
    let average_feature_cosine = total_feature_cosine / feature_metrics.len() as f64;

    if !config.trade_cols.is_empty() {
        println!("\n=== Trade-wise parity ===");
        for (column, metric) in config.trade_cols.iter().zip(&trade_metrics) {
            let cosine = metric.cosine_similarity();
            let mae = metric.mean_absolute_error(rows);
            let pass = metric.mismatches == 0;
            all_pass &= pass;
            println!(
                "  {column}: cosine={cosine:.8}, MAE={mae:.8}, exact_mismatches={}, {}",
                metric.mismatches,
                if pass { "MATCH" } else { "DIVERGED" }
            );
        }
    }

    println!("\nAverage feature cosine similarity: {average_feature_cosine:.8}");
    println!(
        "RESULT: {}",
        if all_pass {
            "PASS"
        } else {
            "FAIL — divergence detected"
        }
    );
    Ok(all_pass)
}

fn read_csv_row(
    reader: &mut BufReader<File>,
    line: &mut String,
    label: &str,
) -> Result<Option<Vec<String>>, String> {
    line.clear();
    let bytes = reader
        .read_line(line)
        .map_err(|error| format!("read {label} CSV row: {error}"))?;
    if bytes == 0 {
        return Ok(None);
    }
    parse_csv_record(line.trim_end())
        .map(Some)
        .map_err(|error| format!("{label} CSV row: {error}"))
}

struct ColumnIndices {
    timestamp: usize,
    features: Vec<usize>,
    trades: Vec<usize>,
}

fn column_indices(
    header: &[String],
    timestamp: &str,
    features: &[String],
    trades: &[String],
    label: &str,
) -> Result<ColumnIndices, String> {
    let find = |name: &str| {
        header
            .iter()
            .position(|column| column == name)
            .ok_or_else(|| format!("{label} output is missing column {name:?}"))
    };
    Ok(ColumnIndices {
        timestamp: find(timestamp)?,
        features: features
            .iter()
            .map(|name| find(name))
            .collect::<Result<_, _>>()?,
        trades: trades
            .iter()
            .map(|name| find(name))
            .collect::<Result<_, _>>()?,
    })
}

fn parse_numeric(row: &[String], index: usize, column: &str) -> Result<f64, String> {
    let value = row
        .get(index)
        .ok_or_else(|| format!("row does not contain column index {index} ({column})"))?;
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid number in {column:?}: {value:?}: {error}"))?;
    if !parsed.is_finite() {
        return Err(format!("non-finite value in {column:?}: {value:?}"));
    }
    Ok(parsed)
}

fn parse_csv_record(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut characters = line.chars().peekable();
    let mut quoted = false;
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                field.push('"');
                characters.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(character),
        }
    }
    if quoted {
        return Err("unterminated quoted CSV field".into());
    }
    fields.push(field);
    if fields
        .first()
        .is_some_and(|name| name.starts_with('\u{feff}'))
    {
        fields[0] = fields[0].trim_start_matches('\u{feff}').to_string();
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::{parse_csv_record, Metric};

    #[test]
    fn parses_quoted_csv_fields() {
        assert_eq!(
            parse_csv_record("timestamp,\"feat,ure\",\"say \"\"hi\"\"\"").unwrap(),
            vec!["timestamp", "feat,ure", "say \"hi\""]
        );
    }

    #[test]
    fn reports_exact_similarity_for_zero_columns() {
        let mut metric = Metric::new();
        metric.add(0.0, 0.0);
        assert_eq!(metric.cosine_similarity(), 1.0);
    }
}

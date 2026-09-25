//! CLI args for `event_merger`.

use std::path::PathBuf;

pub struct MergeArgs {
    pub root: PathBuf,
    pub symbol: String,
    pub source: String,    // "ingest_parquet" | "matched"
    pub date: String,      // YYYY-MM-DD
    pub hour: Option<u32>, // None means the whole UTC date (all 24 outputs)
    pub matched_lob_root: Option<PathBuf>,
    pub lob_levels: usize,
}

pub type BoxError = Box<dyn std::error::Error>;

fn next_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, BoxError> {
    let value = args
        .get(*i + 1)
        .ok_or_else(|| format!("{flag} requires a value"))?;
    *i += 2;
    Ok(value.as_str())
}

pub fn parse_args(args: &[String]) -> Result<MergeArgs, BoxError> {
    let mut root = None;
    let mut symbol = String::from("BTCUSDT");
    let mut source = String::from("ingest_parquet");
    let mut date = None;
    let mut hour = None;
    let mut matched_lob_root = None;
    let mut lob_levels = 20usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => root = Some(PathBuf::from(next_value(args, &mut i, "--root")?)),
            "--symbol" => {
                symbol = next_value(args, &mut i, "--symbol")?.to_uppercase();
            }
            "--source" => {
                source = next_value(args, &mut i, "--source")?.to_string();
            }
            "--date" => {
                date = Some(next_value(args, &mut i, "--date")?.to_string());
            }
            "--hour" => {
                let raw = next_value(args, &mut i, "--hour")?;
                if raw.eq_ignore_ascii_case("all") || raw == "-1" {
                    hour = None;
                } else {
                    let value = raw.parse::<u32>()?;
                    if value > 23 {
                        return Err("--hour must be 0..23, all, or -1".into());
                    }
                    hour = Some(value);
                }
            }
            "--matched-lob-root" => {
                matched_lob_root = Some(PathBuf::from(next_value(
                    args,
                    &mut i,
                    "--matched-lob-root",
                )?));
            }
            "--lob-levels" => {
                lob_levels = next_value(args, &mut i, "--lob-levels")?.parse()?;
            }
            "--keep-inputs" | "--delete-inputs" => {
                i += 1;
            }
            other => return Err(format!("unknown flag `{other}`").into()),
        }
    }

    let date = date.ok_or("--date YYYY-MM-DD is required")?;
    if date.len() != 10 {
        return Err(format!("bad --date `{date}`, expected YYYY-MM-DD").into());
    }
    if lob_levels == 0 {
        return Err("--lob-levels must be greater than zero".into());
    }
    if source != "ingest_parquet" && source != "matched" {
        return Err(format!("bad --source `{source}`, expected ingest_parquet|matched").into());
    }

    Ok(MergeArgs {
        root: root.ok_or("--root market_data dir is required")?,
        symbol,
        source,
        date,
        hour,
        matched_lob_root,
        lob_levels,
    })
}

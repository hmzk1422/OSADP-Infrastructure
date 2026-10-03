//! Input/output layout + UTC hour-window math.

use std::path::PathBuf;

use crate::args::MergeArgs;

pub type BoxError = Box<dyn std::error::Error>;

/// Millis of UTC midnight for a YYYY-MM-DD (days-from-civil, Hinnant).
pub fn midnight_ms(date: &str) -> Result<i64, BoxError> {
    let y: i64 = date[0..4].parse()?;
    let m: i64 = date[5..7].parse()?;
    let d: i64 = date[8..10].parse()?;
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Ok(days * 86_400_000)
}

pub struct HourInputs {
    pub lob_paths: Vec<PathBuf>,
    pub agg_path: PathBuf,
    pub depth_path: PathBuf,
    pub output_dir: PathBuf,
    pub day_start_ns: i64,
    pub day_end_ns: i64,
}

fn first_matching_file(root: &std::path::Path, date: &str, hour: u32) -> Option<std::path::PathBuf> {
    if root.is_file() {
        return Some(root.to_path_buf());
    }
    let hh = format!("{hour:02}");
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|value| value.to_str()) != Some("parquet") {
                continue;
            }
            let name = path.file_name().and_then(|value| value.to_str()).unwrap_or_default();
            if name == format!("{hh}.parquet")
                || name == format!("{hh}_every_event.parquet")
                || name.ends_with("_every_event.parquet")
                || name.contains(&format!("{date}_{hh}"))
                || name.contains(&format!("{date}-{hh}"))
                || name.contains(&format!("{date}_{date}_{hh}-{hh}"))
            {
                return Some(path);
            }
        }
    }
    None
}

fn matched_lob_path(
    work: &std::path::Path,
    ticker_root: &std::path::Path,
    ingest_root: &std::path::Path,
    symbol: &str,
    date: &str,
    hour: u32,
) -> std::path::PathBuf {
    let hh = format!("{hour:02}");
    let candidates = [
        work.join("orderflow")
            .join(date)
            .join(&hh)
            .join(format!("{hh}_every_event.parquet")),
        work.join("orderflow")
            .join(date)
            .join(&hh)
            .join(format!("{hh}.parquet")),
        work.join(date)
            .join(&hh)
            .join(format!("{hh}_every_event.parquet")),
        work.join(date)
            .join(&hh)
            .join(format!("{hh}.parquet")),
        work.join(&hh).join(format!("{hh}_every_event.parquet")),
        work.join(&hh).join(format!("{hh}.parquet")),
        work.join(format!("{hh}_every_event.parquet")),
        work.join("lob_state")
            .join(date)
            .join(format!("{hh}.parquet")),
        work.join(date).join(format!("{hh}.parquet")),
        work.join(format!("{hh}.parquet")),
        work.join("source")
            .join("orderbook")
            .join(date)
            .join(format!("{hh}.parquet")),
        work.join("source").join("orderbook").join(format!("{hh}.parquet")),
        ingest_root
            .join("orderbook")
            .join(date)
            .join(format!("{hh}.parquet")),
        ticker_root.join(format!("{symbol}_{date}_{date}_{hh}-{hh}.parquet")),
        ticker_root
            .join("snapshots")
            .join(format!("{date}_{hh}.parquet")),
        ticker_root
            .join("snapshots")
            .join(date)
            .join(format!("{hh}.parquet")),
    ];
    candidates
        .iter()
        .find(|path| path.exists())
        .cloned()
        .or_else(|| first_matching_file(work, date, hour))
        .or_else(|| first_matching_file(ticker_root, date, hour))
        .unwrap_or_else(|| candidates[0].clone())
}

pub fn hour_inputs(args: &MergeArgs) -> Result<HourInputs, BoxError> {
    let day_start_ms = midnight_ms(&args.date)?;
    let day_start_ns = day_start_ms.saturating_mul(1_000_000);
    let day_end_ns = day_start_ns.saturating_add(86_400_000i64.saturating_mul(1_000_000));
    let ticker_root = args.root.join("matched").join(&args.symbol);
    let work = ticker_root.join("_pipeline");
    let ingest_root = args.root.join("ingest_parquet").join(&args.symbol);
    let work_agg = work
        .join("source")
        .join("aggTrades")
        .join(format!("{}.parquet", args.date));
    let work_depth = work
        .join("source")
        .join("bookDepth")
        .join(format!("{}.parquet", args.date));
    let (agg_path, depth_path) = if work_agg.exists() && work_depth.exists() {
        (work_agg, work_depth)
    } else {
        (
            ingest_root
                .join("aggTrades")
                .join(format!("{}.parquet", args.date)),
            ingest_root
                .join("bookDepth")
                .join(format!("{}.parquet", args.date)),
        )
    };
    let lob_work = args
        .matched_lob_root
        .as_deref()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| work.clone());
    let lob_paths = if lob_work.is_file() {
        vec![lob_work]
    } else {
        match args.hour {
            Some(hour) => {
                let path = matched_lob_path(
                    &lob_work,
                    &ticker_root,
                    &ingest_root,
                    &args.symbol,
                    &args.date,
                    hour,
                );
                if path.exists() { vec![path] } else { Vec::new() }
            }
            None => (0..24)
                .map(|hour| {
                    matched_lob_path(
                        &lob_work,
                        &ticker_root,
                        &ingest_root,
                        &args.symbol,
                        &args.date,
                        hour,
                    )
                })
                .filter(|path| path.exists())
                .collect(),
        }
    };
    let output_dir = args
        .root
        .join("matched")
        .join(&args.symbol)
        .join("sparse");

    Ok(HourInputs {
        lob_paths,
        agg_path,
        depth_path,
        output_dir,
        day_start_ns,
        day_end_ns,
    })
}

//! Bounded three-way chronological merge driver.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use crate::args::{BoxError, MergeArgs};
use crate::layout::hour_inputs;
use crate::model::{Cursor, Event, Source};
use crate::stream::{next_depth, next_lob, next_trade};
use crate::writer::{self, HourWriter, SourceSchema};

pub fn run_merge(args: &MergeArgs) -> Result<(), BoxError> {
    let inputs = hour_inputs(args)?;
    if !inputs.agg_path.exists() {
        return Err(format!("aggTrades input not found: {}", inputs.agg_path.display()).into());
    }
    if !inputs.depth_path.exists() {
        return Err(format!("bookDepth input not found: {}", inputs.depth_path.display()).into());
    }
    let mut lob = Vec::new();
    for path in inputs.lob_paths.iter().filter(|p| p.exists()) {
        lob.push(Cursor::open(path)?);
    }
    if lob.is_empty() {
        return Err("no LOB book-state input found".into());
    }
    let mut trade = Cursor::open(&inputs.agg_path)?;
    let mut depth = Cursor::open(&inputs.depth_path)?;
    let mut depth_state = BTreeMap::new();
    let mut sequence = 0u64;
    let mut pending: Vec<Option<Event>> = (0..lob.len() + 2).map(|_| None).collect();
    let mut heap: BinaryHeap<Reverse<(i64, u8, usize)>> = BinaryHeap::new();

    let selected_start = args
        .hour
        .map(|hour| inputs.day_start_ns + hour as i64 * 3_600_000_000_000);
    let selected_end = selected_start.map(|start| start + 3_600_000_000_000);
    let in_selected_window = |timestamp: i64| {
        selected_start.is_none_or(|start| {
            timestamp >= start && timestamp < selected_end.expect("selected end")
        })
    };

    for stream in 0..lob.len() {
        loop {
            let Some(event) = next_lob(&mut lob[stream], args.lob_levels, 0)? else {
                break;
            };
            if !in_selected_window(event.timestamp) {
                if selected_end.is_some_and(|end| event.timestamp >= end) {
                    break;
                }
                continue;
            }
            heap.push(Reverse((event.timestamp, 0, stream)));
            pending[stream] = Some(event);
            break;
        }
    }
    loop {
        let Some(event) = next_trade(&mut trade, 0)? else {
            break;
        };
        if !in_selected_window(event.timestamp) {
            if selected_end.is_some_and(|end| event.timestamp >= end) {
                break;
            }
            continue;
        }
        heap.push(Reverse((event.timestamp, 1, lob.len())));
        pending[lob.len()] = Some(event);
        break;
    }
    loop {
        let Some(event) = next_depth(&mut depth, 0)? else {
            break;
        };
        if !in_selected_window(event.timestamp) {
            if selected_end.is_some_and(|end| event.timestamp >= end) {
                break;
            }
            continue;
        }
        heap.push(Reverse((event.timestamp, 2, lob.len() + 1)));
        pending[lob.len() + 1] = Some(event);
        break;
    }

    let hours: Vec<u32> = args
        .hour
        .map(|h| vec![h])
        .unwrap_or_else(|| (0..24).collect());
    let mut source_fields = lob
        .iter()
        .map(|cursor| SourceSchema {
            source: Source::Lob,
            schema: cursor.schema(),
        })
        .collect::<Vec<_>>();
    source_fields.push(SourceSchema {
        source: Source::Trade,
        schema: trade.schema(),
    });
    source_fields.push(SourceSchema {
        source: Source::Depth,
        schema: depth.schema(),
    });
    let (merged_schema, source_outputs) = writer::source_schema(args.lob_levels, &source_fields)?;
    let mut writers: BTreeMap<u32, HourWriter> = BTreeMap::new();
    while let Some(Reverse((timestamp, _kind, stream))) = heap.pop() {
        if !in_selected_window(timestamp) {
            if selected_end.is_some_and(|end| timestamp >= end) {
                break;
            }
            continue;
        }
        let hour =
            (((timestamp - inputs.day_start_ns) / 3_600_000_000_000) as i32).clamp(0, 23) as u32;
        let Some(mut event) = pending[stream].take() else {
            continue;
        };
        event.sequence = sequence;
        sequence += 1;
        if event.source == Source::Depth {
            if let Some(value) = event.depth.as_ref() {
                if let Some(percentage) = value.percentage {
                    depth_state.insert(percentage, (value.depth, value.notional));
                }
            }
            event.depth_state = Some(depth_state.clone());
        }
        if hours.contains(&hour) {
            let output = inputs.output_dir.join(format!("{hour:02}.parquet"));
            if !writers.contains_key(&hour) {
                writers.insert(
                    hour,
                    HourWriter::new(
                        &output,
                        args.lob_levels,
                        merged_schema.clone(),
                        source_outputs.clone(),
                    )?,
                );
            }
            writers
                .get_mut(&hour)
                .expect("writer exists")
                .push(event.clone())?;
        }
        let source = event.source;
        loop {
            let next = match source {
                Source::Lob => next_lob(&mut lob[stream], args.lob_levels, 0)?,
                Source::Trade => next_trade(&mut trade, 0)?,
                Source::Depth => next_depth(&mut depth, 0)?,
            };
            let Some(next) = next else {
                break;
            };
            if !in_selected_window(next.timestamp) {
                if selected_end.is_some_and(|end| next.timestamp >= end) {
                    break;
                }
                continue;
            }
            let next_kind = match next.source {
                Source::Lob => 0,
                Source::Trade => 1,
                Source::Depth => 2,
            };
            heap.push(Reverse((next.timestamp, next_kind, stream)));
            pending[stream] = Some(next);
            break;
        }
    }
    for (_, writer) in writers {
        writer.finish()?;
    }
    println!(
        "MERGED source={} date={} hours={} output={}",
        args.source,
        args.date,
        hours.len(),
        inputs.output_dir.display()
    );
    Ok(())
}

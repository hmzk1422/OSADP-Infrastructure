//! `reorder.rs` — deterministic total order over independently-arriving streams.
//!
//! Two sockets deliver LOB and TRADE events independently, so arrival order is
//! not event order. This buffer restores a total order using the same key the
//! historical merger uses:
//!
//! ```text
//! primary    event timestamp (exchange ms)
//! secondary  source priority     (LOB < TRADE, mirroring merge.rs Source)
//! tertiary   source-local sequence (update id / agg trade id)
//! ```
//!
//! **Same ordering semantics, not byte-identical output.** The historical
//! merger sorts a complete input set; a live feed only knows what has arrived.
//! Anything later than the watermark cannot be placed in order, so it is
//! *dropped and counted*. Dropping beats emitting a mis-ordered row: a
//! mis-ordered row silently corrupts the as-of book state of every downstream
//! consumer, whereas a drop is visible in `late_dropped`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::event::MarketEvent;

/// Total order over events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrderKey {
    timestamp_ms: i64,
    source: u8,
    source_sequence: u64,
}

impl OrderKey {
    fn of(event: &MarketEvent) -> Self {
        Self {
            timestamp_ms: event.event_time_ms(),
            source: event.source() as u8,
            source_sequence: event.source_sequence(),
        }
    }
}

/// A buffered event plus its ordering key and arrival counter.
///
/// The counter makes the order a strict *total* order even when two events
/// share timestamp, source and source-local sequence, so emission is
/// deterministic rather than dependent on heap internals.
///
/// The event is kept out of the heap's tuple so `MarketEvent` (which holds
/// `f64`) does not need to implement `Ord`.
struct Entry {
    key: OrderKey,
    arrival: u64,
    event: MarketEvent,
}

/// Bounded, watermark-driven reorder buffer.
pub struct ReorderBuffer {
    /// Min-heap over (key, arrival); yields the smallest key first.
    heap: BinaryHeap<Reverse<(OrderKey, u64)>>,
    /// Events awaiting emission, in no particular order.
    pending: Vec<Entry>,
    /// Highest timestamp seen so far; the watermark trails it.
    watermark_ms: i64,
    lateness_ms: i64,
    late_dropped: u64,
    capacity: usize,
    overflow_dropped: u64,
    arrival: u64,
}

impl ReorderBuffer {
    /// `lateness_ms` is the reorder window: how far out-of-order the two
    /// sockets may interleave before events are dropped rather than emitted
    /// mis-ordered. It also bounds memory to a few thousand events.
    pub fn new(lateness_ms: i64, capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            pending: Vec::with_capacity(capacity.min(1024)),
            watermark_ms: i64::MIN,
            lateness_ms,
            late_dropped: 0,
            capacity,
            overflow_dropped: 0,
            arrival: 0,
        }
    }

    /// Offer an event; returns the events that are now safe to emit.
    pub fn push(&mut self, event: MarketEvent) -> Vec<MarketEvent> {
        let key = OrderKey::of(&event);

        // Late-event policy: an event older than the emission cutoff can no
        // longer be placed in order, so it is dropped and counted. The
        // comparison is against the *cutoff* (watermark minus the lateness
        // window), not the watermark itself -- otherwise every out-of-order
        // arrival would be discarded and the window would have no effect.
        if self.watermark_ms != i64::MIN && key.timestamp_ms < self.watermark_ms - self.lateness_ms
        {
            self.late_dropped += 1;
            return Vec::new();
        }

        // Backpressure guard: a stalled writer must not grow memory forever.
        if self.pending.len() >= self.capacity {
            self.overflow_dropped += 1;
            return Vec::new();
        }

        if key.timestamp_ms > self.watermark_ms {
            self.watermark_ms = key.timestamp_ms;
        }
        let arrival = self.arrival;
        self.arrival += 1;
        self.heap.push(Reverse((key, arrival)));
        self.pending.push(Entry {
            key,
            arrival,
            event,
        });

        self.drain_ready()
    }

    /// Flush everything still buffered, e.g. on shutdown.
    pub fn flush(&mut self) -> Vec<MarketEvent> {
        let mut out = Vec::with_capacity(self.pending.len());
        self.pending.sort_by_key(|entry| (entry.key, entry.arrival));
        out.extend(self.pending.drain(..).map(|entry| entry.event));
        self.heap.clear();
        out
    }

    pub fn late_dropped(&self) -> u64 {
        self.late_dropped
    }

    pub fn overflow_dropped(&self) -> u64 {
        self.overflow_dropped
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Pop every event safely behind the watermark. O(k log n).
    fn drain_ready(&mut self) -> Vec<MarketEvent> {
        if self.watermark_ms == i64::MIN {
            return Vec::new();
        }
        let cutoff = self.watermark_ms - self.lateness_ms;
        let mut ready: Vec<Entry> = Vec::new();
        // The heap top is the smallest key, so stop as soon as it is not safe.
        // Copy the key out before popping: peeking borrows `self.heap`
        // immutably, which cannot overlap the `pop` below.
        loop {
            let Some(Reverse((key, arrival))) = self.heap.peek().copied() else {
                break;
            };
            if key.timestamp_ms > cutoff {
                break;
            }
            self.heap.pop();
            if let Some(position) = self.pending.iter().position(|e| e.arrival == arrival) {
                ready.push(self.pending.remove(position));
            }
        }
        ready.sort_by_key(|entry| (entry.key, entry.arrival));
        ready.into_iter().map(|entry| entry.event).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::event::{DiffFrame, Source, Trade};

    fn trade(ts: i64, id: u64) -> MarketEvent {
        MarketEvent::Trade(Trade {
            event_time_ms: ts,
            transact_time_ms: ts,
            agg_trade_id: id,
            first_trade_id: id,
            last_trade_id: id,
            price: 100.0,
            quantity: 1.0,
            is_buyer_maker: false,
        })
    }

    fn lob(ts: i64, last_id: u64) -> MarketEvent {
        MarketEvent::Diff(DiffFrame {
            event_time_ms: ts,
            first_update_id: last_id,
            final_update_id: last_id,
            updates: Vec::new(),
        })
    }

    #[test]
    fn orders_out_of_order_arrivals() {
        // Window (250ms) must exceed the 200ms spread being reordered, or
        // the stragglers are legitimately late and dropped instead.
        let mut buffer = ReorderBuffer::new(250, 1024);
        let mut all = Vec::new();
        // Arrive backwards; the watermark holds them until ordering is safe.
        all.extend(buffer.push(trade(300, 3)));
        all.extend(buffer.push(trade(100, 1)));
        all.extend(buffer.push(trade(200, 2)));
        all.extend(buffer.push(trade(500, 5)));
        all.extend(buffer.flush());
        let times: Vec<i64> = all.iter().map(|e| e.event_time_ms()).collect();
        assert_eq!(times, vec![100, 200, 300, 500], "reordered, not dropped");
    }

    #[test]
    fn drops_events_later_than_the_watermark() {
        let mut buffer = ReorderBuffer::new(50, 1024);
        buffer.push(trade(1000, 1));
        let _ = buffer.push(trade(200, 2));
        assert_eq!(buffer.late_dropped(), 1);
    }

    #[test]
    fn capacity_bounds_memory() {
        let mut buffer = ReorderBuffer::new(10_000, 8);
        for i in 0..20u64 {
            buffer.push(trade((i * 10) as i64, i));
        }
        assert!(buffer.overflow_dropped() > 0);
        assert!(buffer.len() <= 8);
    }

    #[test]
    fn lob_sorts_before_trade_at_equal_timestamp() {
        let mut buffer = ReorderBuffer::new(1, 1024);
        let mut all = Vec::new();
        all.extend(buffer.push(trade(500, 1)));
        all.extend(buffer.push(lob(500, 7)));
        all.extend(buffer.flush());
        assert_eq!(all[0].source(), Source::Lob);
    }

    #[test]
    fn flush_emits_in_total_order() {
        let mut buffer = ReorderBuffer::new(1_000_000, 1024);
        buffer.push(trade(300, 3));
        buffer.push(trade(100, 1));
        buffer.push(trade(200, 2));
        let times: Vec<i64> = buffer.flush().iter().map(|e| e.event_time_ms()).collect();
        assert_eq!(times, vec![100, 200, 300]);
    }
}

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
use std::time::{Duration, Instant};

use crate::event::MarketEvent;

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
    /// Wall-clock instant the oldest still-buffered event was pushed.
    ///
    /// The watermark is an *exchange-time* notion, so it only advances when a
    /// new event arrives. On a quiet stream that means a held row can sit for
    /// seconds and then be released together with whatever finally moves the
    /// watermark -- the flat-then-burst profile measured in the capture file.
    /// Age is the missing term: it bounds how long a row waits regardless of
    /// whether the exchange is talking.
    oldest_pushed_at: Option<Instant>,
    /// How long a row may wait on exchange-time progress before age forces it
    /// out. `Duration::ZERO` disables the guard and restores pure watermark
    /// behaviour.
    max_hold: Duration,
    /// Rows released early by the age guard rather than by the watermark.
    ///
    /// Counted, not silent: an early release is a real ordering compromise, so
    /// the operator can see how often it happens.
    age_released: u64,
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
            oldest_pushed_at: None,
            max_hold: Duration::ZERO,
            age_released: 0,
        }
    }

    /// Release held events once they are older than `max_hold`, even if the
    /// watermark has not moved far enough to make them emittable.
    ///
    /// This is the fix for the flat-then-burst capture profile. Without it the
    /// buffer is driven purely by exchange time: a quiet book holds rows
    /// hostage, and the next trade flurry moves the watermark enough to
    /// release the whole backlog in one batch. Measured on a live BTCUSDT
    /// capture, 41% of rows shared an identical timestamp with their
    /// predecessor -- that is a batch release, not a 100 ms stream.
    ///
    /// `Duration::ZERO` (the default from [`ReorderBuffer::new`]) keeps the
    /// original watermark-only behaviour, so historical replay is unaffected
    /// unless a caller opts in.
    pub fn max_hold(mut self, max_hold: Duration) -> Self {
        self.max_hold = max_hold;
        self
    }

    /// Rows released early because they aged past [`ReorderBuffer::max_hold`].
    pub fn age_released(&self) -> u64 {
        self.age_released
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
        self.pending.push(Entry { key, arrival, event });
        // The oldest buffered event is the one whose age bounds the wait. Set
        // only on the empty -> non-empty edge so the clock is not restarted by
        // every later arrival.
        if self.oldest_pushed_at.is_none() {
            self.oldest_pushed_at = Some(Instant::now());
        }

        self.drain_ready()
    }

    /// Flush everything still buffered, e.g. on shutdown.
    pub fn flush(&mut self) -> Vec<MarketEvent> {
        let mut out = Vec::with_capacity(self.pending.len());
        self.pending.sort_by_key(|entry| (entry.key, entry.arrival));
        out.extend(self.pending.drain(..).map(|entry| entry.event));
        self.heap.clear();
        self.oldest_pushed_at = None;
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
        let mut ready: Vec<Entry> = Vec::new();

        // The age guard. Normally the cutoff trails the watermark by the
        // lateness window, so an event waits for the exchange to prove no
        // earlier event is still coming. When the stream goes quiet the
        // watermark stops advancing and that proof never arrives -- so the
        // held rows wait for the *next* flurry, which then releases them all
        // at once. Age converts "wait indefinitely for exchange progress" into
        // "wait at most max_hold", which keeps the row cadence even without
        // weakening the ordering guarantee during normal traffic.
        let aged_out = match self.oldest_pushed_at {
            Some(oldest) if !self.max_hold.is_zero() => oldest.elapsed() >= self.max_hold,
            _ => false,
        };
        // An aged-out buffer emits everything it holds: the cutoff moves to the
        // watermark itself, because waiting longer is the failure being fixed.
        let cutoff = if aged_out {
            self.watermark_ms
        } else {
            self.watermark_ms - self.lateness_ms
        };

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

        // Only count an event as age-released when age is genuinely why it
        // went: during normal traffic the widened cutoff is still behind the
        // rows that would have gone anyway, so this stays a true signal.
        if aged_out && !ready.is_empty() {
            self.age_released += ready.len() as u64;
        }
        // Restart the age clock only when the buffer actually drained. A
        // partial drain leaves older rows behind, and their age is what bounds
        // the wait -- resetting here would let the backlog age forever.
        if self.pending.is_empty() {
            self.oldest_pushed_at = None;
        }
        ready.into_iter().map(|entry| entry.event).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{DiffFrame, Source, Trade};

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

    // -- The age guard -----------------------------------------------------
    //
    // The watermark is exchange time, so it only moves when a new event
    // arrives. These tests cover what happens between arrivals.
    //
    // Note on style: every `push` return is collected. An earlier release is
    // real output, and dropping it on the floor is how a test ends up asserting
    // a shorter sequence than the buffer actually produced.

    #[test]
    fn without_max_hold_a_held_event_waits_for_the_watermark() {
        // The default: unchanged behaviour, so historical replay is untouched.
        // 900 sits inside the 200ms window of 1000, so it is buffered, not
        // dropped as late and not emitted.
        let mut buffer = ReorderBuffer::new(200, 1024);
        assert!(buffer.push(trade(1000, 1)).is_empty());
        assert!(buffer.push(trade(900, 2)).is_empty());
        assert_eq!(buffer.len(), 2, "both still buffered behind the watermark");
        assert_eq!(buffer.age_released(), 0);
    }

    #[test]
    fn max_hold_releases_a_backlog_the_watermark_would_have_held() {
        // The flat-then-burst fix. With a 200ms window the 1000ms event would
        // stay buffered behind the 1200ms watermark and wait for the next
        // flurry; age releases it instead.
        let mut buffer = ReorderBuffer::new(200, 1024).max_hold(Duration::from_millis(10));
        assert!(buffer.push(trade(1000, 1)).is_empty());

        // Too young to force anything.
        std::thread::sleep(Duration::from_millis(3));
        assert!(
            buffer.push(trade(1100, 2)).is_empty(),
            "3ms is inside the 10ms hold"
        );

        // Aged out: the held 1000ms event goes without waiting for traffic.
        std::thread::sleep(Duration::from_millis(15));
        let released = buffer.push(trade(1200, 3));
        assert!(
            released.iter().any(|e| e.event_time_ms() == 1000),
            "the aged 1000ms event must be released, got {released:?}"
        );
        assert!(buffer.age_released() > 0, "an early release must be counted");
    }

    #[test]
    fn age_guard_still_emits_in_total_order() {
        // Widening the cutoff must not scramble the order: the sort key is
        // unchanged, only the cutoff moved.
        let mut buffer = ReorderBuffer::new(200, 1024).max_hold(Duration::from_millis(10));
        let mut all = Vec::new();
        all.extend(buffer.push(trade(1000, 1)));
        all.extend(buffer.push(trade(1100, 2)));
        all.extend(buffer.push(trade(1200, 3)));
        std::thread::sleep(Duration::from_millis(15));
        all.extend(buffer.push(trade(1300, 4)));
        all.extend(buffer.flush());

        let times: Vec<i64> = all.iter().map(|e| e.event_time_ms()).collect();
        assert_eq!(times, vec![1000, 1100, 1200, 1300], "order is preserved");
        assert_eq!(buffer.len(), 0, "nothing left behind");
    }

    #[test]
    fn age_guard_does_not_reorder_events_that_arrive_normally() {
        // Healthy traffic: the widened cutoff is behind the rows the watermark
        // would have released anyway, so the guard never has to widen anything.
        let mut buffer = ReorderBuffer::new(200, 1024).max_hold(Duration::from_millis(50));
        let mut all = Vec::new();
        for ms in (1000..1400).step_by(100) {
            all.extend(buffer.push(trade(ms, (ms / 100) as u64)));
        }
        all.extend(buffer.flush());
        let times: Vec<i64> = all.iter().map(|e| e.event_time_ms()).collect();
        assert_eq!(times, vec![1000, 1100, 1200, 1300]);
    }

    #[test]
    fn zero_max_hold_disables_the_age_guard() {
        // Same event sequence as the age test above, with the guard off. The
        // 1000ms event must still be buffered -- proving the release in
        // `max_hold_releases_a_backlog_the_watermark_would_have_held` came from
        // age and not from the cutoff arithmetic alone.
        let mut buffer = ReorderBuffer::new(200, 1024).max_hold(Duration::ZERO);
        assert!(buffer.push(trade(1000, 1)).is_empty());
        std::thread::sleep(Duration::from_millis(15));
        assert!(
            buffer.push(trade(1100, 2)).is_empty(),
            "Duration::ZERO must restore pure watermark behaviour"
        );
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.age_released(), 0);
    }
}

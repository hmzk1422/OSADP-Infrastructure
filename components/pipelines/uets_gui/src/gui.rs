use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::Line;
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, ListState, Paragraph,
};
use ratatui::{Frame, Terminal};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Statistics {
    pub starting_equity: Option<f64>,
    pub sortino: Option<f64>,
    pub expectancy_rr: Option<f64>,
    pub expectancy_trades: Option<usize>,
    pub max_drawdown: Option<f64>,
    pub cumulative_return: Option<f64>,
    pub absolute_return: Option<f64>,
    pub cash: Option<f64>,
    pub equity: Option<f64>,
    pub realized_pnl: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    pub trades: Option<usize>,
    pub open_positions: Option<usize>,
    pub stop_loss_pct: Option<f64>,
    pub take_profit_pct: Option<f64>,
    pub price_vs_entry_pct: Option<f64>,
    pub win_rate: Option<f64>,
    pub exposure: Option<f64>,
    pub status: Option<String>,
    pub last_error: Option<String>,
    pub latest_market_data: Option<String>,
    pub latest_decision: Option<String>,
    pub exchange_latency_ms: Option<f64>,
    pub processing_latency_ms: Option<f64>,
    pub latency_p05_ms: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub latency_p99_ms: Option<f64>,
    pub clock_skew_mean_ms: Option<f64>,
    pub clock_skew_max_ms: Option<f64>,
    pub decision_to_order_ms: Option<f64>,
    pub order_to_fill_ms: Option<f64>,
    pub tick_to_trade_ms: Option<f64>,
    pub memory_mb: Option<f64>,
    pub cpu_percent: Option<f64>,
    pub persistence_queue: Option<usize>,
    pub persistence_write_ms: Option<f64>,
    pub persistence_errors: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct InstanceSnapshot {
    pub strategy_id: String,
    pub asset_id: String,
    pub statistics: Statistics,
    pub equity_history: Vec<f64>,
}

impl InstanceSnapshot {
    pub fn identity(&self) -> String {
        format!("{}@{}", self.strategy_id, self.asset_id)
    }
}

#[derive(Clone, Debug, Default)]
pub struct DashboardSnapshot {
    pub timestamp: Option<String>,
    pub aggregate: Statistics,
    pub aggregate_equity_history: Vec<f64>,
    pub instances: Vec<InstanceSnapshot>,
}

pub fn aggregate_instances(
    instances: &[InstanceSnapshot],
    equity_history: &mut Vec<f64>,
    aggregate_starting_equity: &mut f64,
) -> Statistics {
    let mut aggregate = Statistics::default();
    if instances.is_empty() {
        equity_history.clear();
        *aggregate_starting_equity = 0.0;
        return aggregate;
    }
    let eligible = instances
        .iter()
        .filter(|instance| {
            instance.statistics.starting_equity.is_some() && instance.statistics.equity.is_some()
        })
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        equity_history.clear();
        *aggregate_starting_equity = 0.0;
        return aggregate;
    }
    let mut starting_equity = 0.0;
    let mut equity = 0.0;
    let mut cash = 0.0;
    let mut realized_pnl = 0.0;
    let mut unrealized_pnl = 0.0;
    let mut exposure = 0.0;
    let mut open_positions = 0usize;
    let mut trades = 0usize;
    let mut wins = 0.0;
    let mut expectancy_sum = 0.0;
    let mut expectancy_trades = 0usize;
    let mut active = false;
    let mut errored = false;
    for instance in &eligible {
        let stats = &instance.statistics;
        starting_equity += stats.starting_equity.unwrap_or_default();
        equity += stats.equity.unwrap_or_default();
        cash += stats.cash.unwrap_or_default();
        realized_pnl += stats.realized_pnl.unwrap_or_default();
        unrealized_pnl += stats.unrealized_pnl.unwrap_or_default();
        exposure += stats.exposure.unwrap_or_default();
        open_positions += stats.open_positions.unwrap_or_default();
        let instance_trades = stats.trades.unwrap_or_default();
        trades += instance_trades;
        wins += stats.win_rate.unwrap_or_default() * instance_trades as f64;
        let rr_trades = stats.expectancy_trades.unwrap_or(instance_trades);
        expectancy_sum += stats.expectancy_rr.unwrap_or_default() * rr_trades as f64;
        expectancy_trades += rr_trades;
        active |= stats.status.as_deref() == Some("enabled");
        errored |= stats.status.as_deref() == Some("error");
    }
    let starting_delta = starting_equity - *aggregate_starting_equity;
    if starting_delta != 0.0 {
        for previous_equity in equity_history.iter_mut() {
            *previous_equity += starting_delta;
        }
        *aggregate_starting_equity = starting_equity;
    }
    equity_history.push(equity);
    if equity_history.len() > 10_000 {
        equity_history.drain(..equity_history.len() - 10_000);
    }
    aggregate.starting_equity = Some(starting_equity);
    aggregate.equity = Some(equity);
    aggregate.cash = Some(cash);
    aggregate.realized_pnl = Some(realized_pnl);
    aggregate.unrealized_pnl = Some(unrealized_pnl);
    aggregate.exposure = Some(exposure);
    aggregate.open_positions = Some(open_positions);
    aggregate.trades = Some(trades);
    aggregate.win_rate = Some(if trades > 0 {
        wins / trades as f64
    } else {
        0.0
    });
    aggregate.expectancy_trades = Some(expectancy_trades);
    aggregate.expectancy_rr = Some(if expectancy_trades > 0 {
        expectancy_sum / expectancy_trades as f64
    } else {
        0.0
    });
    aggregate.cumulative_return = Some(if starting_equity > 0.0 {
        (equity - starting_equity) / starting_equity
    } else {
        0.0
    });
    aggregate.absolute_return = Some(equity - starting_equity);
    aggregate.max_drawdown = Some(max_drawdown(equity_history));
    aggregate.sortino = Some(sortino(equity_history));
    aggregate.status = Some(
        if errored {
            "error"
        } else if active {
            "enabled"
        } else {
            "disabled"
        }
        .to_string(),
    );
    // Operational telemetry is produced per worker. Preserve it in the
    // aggregate view instead of silently dropping it when summing P&L.
    aggregate.latest_market_data = eligible
        .iter()
        .filter_map(|instance| instance.statistics.latest_market_data.as_deref())
        .filter_map(|value| value.parse::<u64>().ok())
        .max()
        .map(|timestamp| timestamp.to_string());
    aggregate.latest_decision = eligible
        .iter()
        .rev()
        .find_map(|instance| instance.statistics.latest_decision.clone());
    let mean = |select: fn(&Statistics) -> Option<f64>| {
        let values = eligible
            .iter()
            .filter_map(|instance| select(&instance.statistics))
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    aggregate.exchange_latency_ms = mean(|s| s.exchange_latency_ms);
    aggregate.processing_latency_ms = mean(|s| s.processing_latency_ms);
    aggregate.latency_p05_ms = mean(|s| s.latency_p05_ms);
    aggregate.latency_p95_ms = mean(|s| s.latency_p95_ms);
    aggregate.latency_p99_ms = mean(|s| s.latency_p99_ms);
    aggregate.decision_to_order_ms = mean(|s| s.decision_to_order_ms);
    aggregate.order_to_fill_ms = mean(|s| s.order_to_fill_ms);
    aggregate.clock_skew_mean_ms = mean(|s| s.clock_skew_mean_ms);
    aggregate.clock_skew_max_ms = eligible
        .iter()
        .filter_map(|instance| instance.statistics.clock_skew_max_ms)
        .reduce(f64::max);
    aggregate.memory_mb = mean(|s| s.memory_mb);
    aggregate.cpu_percent = mean(|s| s.cpu_percent);
    aggregate.tick_to_trade_ms = mean(|s| s.tick_to_trade_ms);
    aggregate.persistence_write_ms = mean(|s| s.persistence_write_ms);
    aggregate.persistence_queue = eligible.iter().filter_map(|i| i.statistics.persistence_queue).max();
    aggregate.persistence_errors = eligible.iter().filter_map(|i| i.statistics.persistence_errors).reduce(|a, b| a.saturating_add(b));
    // Bracket and mark-to-entry telemetry is per position; surface the mean
    // across workers holding a position instead of dropping it in the
    // aggregate view.
    aggregate.stop_loss_pct = mean(|s| s.stop_loss_pct);
    aggregate.take_profit_pct = mean(|s| s.take_profit_pct);
    aggregate.price_vs_entry_pct = mean(|s| s.price_vs_entry_pct);
    aggregate
}

fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut worst: f64 = 0.0;
    for value in equity.iter().copied().filter(|value| value.is_finite()) {
        peak = peak.max(value);
        if peak > 0.0 {
            worst = worst.max((peak - value) / peak);
        }
    }
    worst
}

fn sortino(equity: &[f64]) -> f64 {
    let returns = equity
        .windows(2)
        .filter_map(|pair| (pair[0] > 0.0).then_some((pair[1] - pair[0]) / pair[0]))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if returns.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside = (returns
        .iter()
        .map(|value| value.min(0.0).powi(2))
        .sum::<f64>()
        / returns.len() as f64)
        .sqrt();
    if downside > 0.0 {
        mean / downside
    } else if mean > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

/// The producer should publish already-calculated snapshots. `snapshot` must
/// return quickly; the GUI polls it independently at the display cadence.
pub trait SnapshotSource {
    fn snapshot(&mut self) -> DashboardSnapshot;

    fn control(&mut self, _command: StrategyCommand) -> Result<(), String> {
        Err("strategy management is unavailable for this snapshot source".to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrategyCommand {
    Add(String),
    Toggle(String),
    Remove(String),
}

#[derive(Debug, Error)]
pub enum MonitorError {
    #[error("terminal I/O failed: {0}")]
    Io(#[from] io::Error),
}

pub struct Monitor<S> {
    source: S,
    selected: Option<usize>,
    interval: Duration,
    add_input: Option<String>,
    message: Option<String>,
}

impl<S: SnapshotSource> Monitor<S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            selected: None,
            interval: Duration::from_millis(100),
            add_input: None,
            message: None,
        }
    }
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval.max(Duration::from_millis(10));
        self
    }

    pub fn run(mut self) -> Result<(), MonitorError> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        let result = self.event_loop(&mut terminal);
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        result
    }

    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> Result<(), MonitorError> {
        loop {
            let started = Instant::now();
            let snapshot = self.source.snapshot();
            if self
                .selected
                .is_some_and(|index| index >= snapshot.instances.len())
            {
                self.selected = None;
            }
            terminal.draw(|frame| {
                render(
                    frame,
                    &snapshot,
                    self.selected,
                    self.add_input.as_deref(),
                    self.message.as_deref(),
                )
            })?;
            let timeout = self.interval.saturating_sub(started.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if let Some(input) = &mut self.add_input {
                        match key.code {
                            crossterm::event::KeyCode::Esc => self.add_input = None,
                            crossterm::event::KeyCode::Enter => {
                                let identity = std::mem::take(input);
                                self.add_input = None;
                                self.control(StrategyCommand::Add(identity));
                            }
                            crossterm::event::KeyCode::Backspace => {
                                input.pop();
                            }
                            crossterm::event::KeyCode::Char(character) => input.push(character),
                            _ => {}
                        }
                        continue;
                    }
                    match crate::controls::action(key) {
                        crate::controls::Action::Quit => return Ok(()),
                        crate::controls::Action::Up => self.move_selection(&snapshot, -1),
                        crate::controls::Action::Down => self.move_selection(&snapshot, 1),
                        crate::controls::Action::Select => {
                            if !snapshot.instances.is_empty() {
                                self.selected = Some(
                                    self.selected.unwrap_or(0).min(snapshot.instances.len() - 1),
                                );
                            }
                        }
                        crate::controls::Action::Clear => self.selected = None,
                        crate::controls::Action::Add => {
                            self.add_input = Some(String::new());
                            self.message = Some("Enter strategy@asset, e.g. ofmr@btcusdt".into());
                        }
                        crate::controls::Action::Toggle => {
                            self.control_selected(&snapshot, StrategyCommand::Toggle);
                        }
                        crate::controls::Action::Remove => {
                            self.control_selected(&snapshot, StrategyCommand::Remove);
                        }
                        crate::controls::Action::None => {}
                    }
                }
            }
        }
    }

    fn control(&mut self, command: StrategyCommand) {
        self.message = Some(match self.source.control(command) {
            Ok(()) => "Strategy configuration updated".to_string(),
            Err(error) => format!("Control failed: {error}"),
        });
    }

    fn control_selected(
        &mut self,
        snapshot: &DashboardSnapshot,
        make: fn(String) -> StrategyCommand,
    ) {
        let Some(instance) = self
            .selected
            .and_then(|index| snapshot.instances.get(index))
        else {
            self.message = Some("Select a strategy first".to_string());
            return;
        };
        self.control(make(instance.identity()));
    }

    fn move_selection(&mut self, snapshot: &DashboardSnapshot, delta: isize) {
        if snapshot.instances.is_empty() {
            self.selected = None;
            return;
        }
        let current = self.selected.unwrap_or(0) as isize;
        self.selected =
            Some((current + delta).clamp(0, snapshot.instances.len() as isize - 1) as usize);
    }
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;

    fn instance(
        asset: &str,
        starting: f64,
        equity: f64,
        trades: usize,
        win_rate: f64,
        rr: f64,
        rr_trades: usize,
    ) -> InstanceSnapshot {
        InstanceSnapshot {
            strategy_id: "ofmr".into(),
            asset_id: asset.into(),
            statistics: Statistics {
                starting_equity: Some(starting),
                equity: Some(equity),
                cash: Some(equity),
                realized_pnl: Some(equity - starting),
                unrealized_pnl: Some(0.0),
                exposure: Some(0.0),
                trades: Some(trades),
                win_rate: Some(win_rate),
                expectancy_rr: Some(rr),
                expectancy_trades: Some(rr_trades),
                status: Some("enabled".into()),
                ..Statistics::default()
            },
            equity_history: vec![starting, equity],
        }
    }

    #[test]
    fn aggregates_account_values_and_weights_trade_metrics() {
        let instances = [
            instance("btcusdt", 100.0, 110.0, 2, 0.5, 1.0, 2),
            instance("ethusdt", 200.0, 180.0, 1, 0.0, -2.0, 1),
        ];
        let mut history = Vec::new();
        let mut starting_equity = 0.0;
        let stats = aggregate_instances(&instances, &mut history, &mut starting_equity);
        assert_eq!(stats.starting_equity, Some(300.0));
        assert_eq!(stats.equity, Some(290.0));
        assert_eq!(stats.absolute_return, Some(-10.0));
        assert!((stats.cumulative_return.unwrap() + 1.0 / 30.0).abs() < 1e-12);
        assert_eq!(stats.trades, Some(3));
        assert!((stats.win_rate.unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(stats.expectancy_rr, Some(0.0));
        assert_eq!(history, vec![290.0]);
        assert_eq!(stats.status.as_deref(), Some("enabled"));
    }

    #[test]
    fn aggregate_preserves_worker_operational_telemetry() {
        let mut item = instance("btcusdt", 100.0, 101.0, 0, 0.0, 0.0, 0);
        item.statistics.latest_market_data = Some("1792249435161".into());
        item.statistics.latest_decision = Some("Hold".into());
        item.statistics.exchange_latency_ms = Some(12.0);
        item.statistics.processing_latency_ms = Some(0.25);
        item.statistics.latency_p95_ms = Some(0.4);
        item.statistics.clock_skew_mean_ms = Some(-3.0);
        item.statistics.clock_skew_max_ms = Some(5.0);
        item.statistics.memory_mb = Some(18.0);
        let mut history = Vec::new();
        let mut starting_equity = 0.0;
        let stats = aggregate_instances(&[item], &mut history, &mut starting_equity);
        assert_eq!(stats.latest_market_data.as_deref(), Some("1792249435161"));
        assert_eq!(stats.latest_decision.as_deref(), Some("Hold"));
        assert_eq!(stats.exchange_latency_ms, Some(12.0));
        assert_eq!(stats.processing_latency_ms, Some(0.25));
        assert_eq!(stats.latency_p95_ms, Some(0.4));
        assert_eq!(stats.clock_skew_mean_ms, Some(-3.0));
        assert_eq!(stats.clock_skew_max_ms, Some(5.0));
        assert_eq!(stats.memory_mb, Some(18.0));
    }

    #[test]
    fn aggregate_drawdown_and_sortino_use_aggregate_curve() {
        let instances = [instance("btcusdt", 100.0, 90.0, 0, 0.0, 0.0, 0)];
        let mut history = vec![100.0, 120.0];
        let mut starting_equity = 100.0;
        let stats = aggregate_instances(&instances, &mut history, &mut starting_equity);
        assert_eq!(stats.max_drawdown, Some(0.25));
        assert!(
            (stats.sortino.unwrap() - (-0.025 / (0.25_f64.powi(2) / 2.0).sqrt())).abs() < 1e-12
        );
    }

    #[test]
    fn adding_an_instance_rebases_history_without_creating_fake_returns() {
        let mut history = Vec::new();
        let mut starting_equity = 0.0;
        let one = [instance("btcusdt", 100.0, 110.0, 0, 0.0, 0.0, 0)];
        aggregate_instances(&one, &mut history, &mut starting_equity);
        let two = [
            instance("btcusdt", 100.0, 110.0, 0, 0.0, 0.0, 0),
            instance("ethusdt", 200.0, 200.0, 0, 0.0, 0.0, 0),
        ];
        let stats = aggregate_instances(&two, &mut history, &mut starting_equity);
        assert_eq!(starting_equity, 300.0);
        assert_eq!(history, vec![310.0, 310.0]);
        assert_eq!(stats.cumulative_return, Some(10.0 / 300.0));
        assert_eq!(stats.max_drawdown, Some(0.0));
    }
    #[test]
    fn statistics_serializes_stop_loss_take_profit_and_price_vs_entry_pct() {
        let stats = Statistics {
            stop_loss_pct: Some(0.5),
            take_profit_pct: Some(1.0),
            price_vs_entry_pct: Some(2.5),
            ..Statistics::default()
        };
        let json = serde_json::to_string(&stats).unwrap();
        let back: Statistics = serde_json::from_str(&json).unwrap();
        assert_eq!(back.stop_loss_pct, Some(0.5));
        assert_eq!(back.take_profit_pct, Some(1.0));
        assert_eq!(back.price_vs_entry_pct, Some(2.5));

        // Older payloads without the new keys deserialize to None.
        let back: Statistics = serde_json::from_str("{}" ).unwrap();
        assert_eq!(back.stop_loss_pct, None);
        assert_eq!(back.take_profit_pct, None);
        assert_eq!(back.price_vs_entry_pct, None);
    }

    #[test]
    fn aggregate_preserves_stop_loss_take_profit_and_price_vs_entry_pct() {
        let mut first = instance("btcusdt", 100.0, 101.0, 0, 0.0, 0.0, 0);
        first.statistics.stop_loss_pct = Some(0.4);
        first.statistics.take_profit_pct = Some(0.8);
        first.statistics.price_vs_entry_pct = Some(2.0);
        let mut second = instance("ethusdt", 100.0, 100.0, 0, 0.0, 0.0, 0);
        second.statistics.stop_loss_pct = Some(0.6);
        second.statistics.take_profit_pct = Some(1.2);
        second.statistics.price_vs_entry_pct = Some(-1.0);
        // Instances without an open position report None and are skipped.
        let third = instance("solusdt", 100.0, 100.0, 0, 0.0, 0.0, 0);
        let instances = [first, second, third];
        let mut history = Vec::new();
        let mut starting_equity = 0.0;
        let stats = aggregate_instances(&instances, &mut history, &mut starting_equity);
        assert!((stats.stop_loss_pct.unwrap() - 0.5).abs() < 1e-12);
        assert!((stats.take_profit_pct.unwrap() - 1.0).abs() < 1e-12);
        assert!((stats.price_vs_entry_pct.unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn instance_detail_table_shows_bracket_and_price_vs_entry_columns() {
        let mut item = instance("btcusdt", 10_000.0, 10_000.0, 1, 0.5, 1.0, 1);
        item.statistics.stop_loss_pct = Some(0.5);
        item.statistics.take_profit_pct = Some(1.0);
        item.statistics.price_vs_entry_pct = Some(2.5);
        item.statistics.status = Some("enabled".into());
        let snapshot = DashboardSnapshot {
            timestamp: None,
            aggregate: Statistics::default(),
            aggregate_equity_history: Vec::new(),
            instances: vec![item],
        };
        let backend = ratatui::backend::TestBackend::new(160, 44);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, &snapshot, Some(0), None, None))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let mut screen = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                screen.push_str(buffer[(x, y)].symbol());
            }
        }
        assert!(screen.contains("SL%"), "missing SL% column:\n{screen}");
        assert!(screen.contains("TP%"), "missing TP% column:\n{screen}");
        assert!(screen.contains("PvE%"), "missing PvE% column:\n{screen}");
        assert!(screen.contains("0.50%"), "missing stop distance:\n{screen}");
        assert!(screen.contains("1.00%"), "missing target distance:\n{screen}");
        assert!(screen.contains("2.50%"), "missing price vs entry:\n{screen}");
    }
}

fn render(
    frame: &mut Frame<'_>,
    snapshot: &DashboardSnapshot,
    selected: Option<usize>,
    add_input: Option<&str>,
    message: Option<&str>,
) {
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage(40),
                Constraint::Length(10),
                Constraint::Min(7),
            ]
            .as_ref(),
        )
        .split(frame.area());
    let (identity, stats, equity) = selected
        .and_then(|i| snapshot.instances.get(i))
        .map(|instance| {
            (
                instance.identity(),
                &instance.statistics,
                &instance.equity_history,
            )
        })
        .unwrap_or_else(|| {
            (
                "Portfolio aggregate".to_string(),
                &snapshot.aggregate,
                &snapshot.aggregate_equity_history,
            )
        });
    render_chart(
        frame,
        areas[0],
        &identity,
        equity,
        snapshot.timestamp.as_deref(),
    );
    render_statistics(frame, areas[1], stats, &identity);
    render_instances(frame, areas[2], snapshot, selected);
    if let Some(value) = add_input {
        let popup = centered_rect(60, 20, frame.area());
        frame.render_widget(ratatui::widgets::Clear, popup);
        frame.render_widget(
            Paragraph::new(format!(
                "Add disabled strategy: {value}_\nEnter confirms · Esc cancels"
            ))
            .block(
                Block::default()
                    .title(" Add strategy ")
                    .borders(Borders::ALL),
            ),
            popup,
        );
    } else if let Some(message) = message {
        let area = frame.area();
        let footer =
            ratatui::layout::Rect::new(area.x, area.bottom().saturating_sub(1), area.width, 1);
        frame.render_widget(
            Paragraph::new(message).style(Style::default().fg(Color::Yellow)),
            footer,
        );
    }
}

fn centered_rect(
    width_percent: u16,
    height_percent: u16,
    area: ratatui::layout::Rect,
) -> ratatui::layout::Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_percent) / 2),
            Constraint::Percentage(height_percent),
            Constraint::Percentage((100 - height_percent) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_percent) / 2),
            Constraint::Percentage(width_percent),
            Constraint::Percentage((100 - width_percent) / 2),
        ])
        .split(vertical[1])[1]
}

fn render_chart(
    frame: &mut Frame<'_>,
    area: ratatui::layout::Rect,
    identity: &str,
    equity: &[f64],
    timestamp: Option<&str>,
) {
    let points: Vec<(f64, f64)> = equity
        .iter()
        .enumerate()
        .map(|(i, value)| (i as f64, *value))
        .collect();
    let x_max = points.last().map(|p| p.0.max(1.0)).unwrap_or(1.0);
    let (y_min, y_max) = equity
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    let bounds = if y_min.is_finite() && y_max.is_finite() {
        (y_min, if y_min == y_max { y_max + 1.0 } else { y_max })
    } else {
        (0.0, 1.0)
    };
    let dataset = Dataset::default()
        .name("Equity")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Cyan))
        .data(&points);
    let title = format!(
        " {identity} · Equity{} ",
        timestamp.map(|t| format!(" · {t}")).unwrap_or_default()
    );
    let chart = Chart::new(vec![dataset])
        .block(Block::default().title(title).borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(vec![Line::from("oldest"), Line::from("latest")]),
        )
        .y_axis(Axis::default().bounds([bounds.0, bounds.1]).labels(vec![
            Line::from(format!("{:.2}", bounds.0)),
            Line::from(format!("{:.2}", bounds.1)),
        ]));
    frame.render_widget(chart, area);
    if equity.is_empty() {
        let empty = ratatui::layout::Rect::new(
            area.x.saturating_add(2),
            area.y.saturating_add(area.height / 2),
            area.width.saturating_sub(4),
            1,
        );
        frame.render_widget(
            Paragraph::new("No equity history received yet".to_string())
                .style(Style::default().fg(Color::DarkGray)),
            empty,
        );
    }
}

fn metric(name: &str, value: Option<String>) -> String {
    format!("{name}: {}", value.unwrap_or_else(|| "—".into()))
}
fn number(value: Option<f64>, percent: bool) -> Option<String> {
    value.map(|v| {
        if v.is_nan() {
            "—".to_string()
        } else if v == f64::INFINITY {
            "∞".to_string()
        } else if v == f64::NEG_INFINITY {
            "−∞".to_string()
        } else if percent {
            format!("{:.2}%", v * 100.0)
        } else {
            format!("{v:.3}")
        }
    })
}

fn telemetry_number(value: Option<f64>) -> Option<String> {
    Some(number(value, false).unwrap_or_else(|| "n/a".to_string()))
}

fn telemetry_text(value: Option<String>) -> Option<String> {
    Some(value.unwrap_or_else(|| "n/a".to_string()))
}

fn render_statistics(
    frame: &mut Frame<'_>,
    area: ratatui::layout::Rect,
    stats: &Statistics,
    identity: &str,
) {
    let values = [
        metric("Sortino", number(stats.sortino, false)),
        metric("Expectancy R", number(stats.expectancy_rr, false)),
        metric("MDD", number(stats.max_drawdown, true)),
        metric("Return", number(stats.cumulative_return, true)),
        metric("Equity", number(stats.equity, false)),
        metric("Cash", number(stats.cash, false)),
        metric("Realized", number(stats.realized_pnl, false)),
        metric("Unrealized", number(stats.unrealized_pnl, false)),
        metric("Exposure", number(stats.exposure, false)),
        metric(
            "Open positions",
            stats.open_positions.map(|v| v.to_string()),
        ),
        metric("Win rate", number(stats.win_rate, true)),
        metric("Stop loss %", telemetry_number(stats.stop_loss_pct)),
        metric("Take profit %", telemetry_number(stats.take_profit_pct)),
        metric("Price vs entry %", telemetry_number(stats.price_vs_entry_pct)),
        metric("Status", stats.status.clone()),
        metric("Error", stats.last_error.clone()),
        metric(
            "Market data",
            telemetry_text(stats.latest_market_data.clone()),
        ),
        metric("Decision", telemetry_text(stats.latest_decision.clone())),
        metric("Strategy ms", telemetry_number(stats.processing_latency_ms)),
        metric(
            "Exchange→local ms",
            telemetry_number(stats.exchange_latency_ms),
        ),
        metric("Latency p05 ms", telemetry_number(stats.latency_p05_ms)),
        metric("Latency p95 ms", telemetry_number(stats.latency_p95_ms)),
        metric("Latency p99 ms", telemetry_number(stats.latency_p99_ms)),
        metric(
            "Clock skew mean ms",
            telemetry_number(stats.clock_skew_mean_ms),
        ),
        metric(
            "Clock skew max ms",
            telemetry_number(stats.clock_skew_max_ms),
        ),
        metric(
            "Decision→order ms",
            telemetry_number(stats.decision_to_order_ms),
        ),
        metric("Order→fill ms", telemetry_number(stats.order_to_fill_ms)),
        metric("Tick→trade ms", telemetry_number(stats.tick_to_trade_ms)),
        metric("Memory MB", telemetry_number(stats.memory_mb)),
        metric("CPU %", telemetry_number(stats.cpu_percent)),
        metric(
            "Persist queue",
            Some(
                stats
                    .persistence_queue
                    .map_or_else(|| "n/a".to_string(), |v| v.to_string()),
            ),
        ),
        metric("Write ms", telemetry_number(stats.persistence_write_ms)),
        metric(
            "Storage errors",
            Some(
                stats
                    .persistence_errors
                    .map_or_else(|| "n/a".to_string(), |v| v.to_string()),
            ),
        ),
        metric("Return $", number(stats.absolute_return, false)),
        metric("Trades", stats.trades.map(|v| v.to_string())),
    ];
    let rows = values
        .chunks(4)
        .take(8)
        .map(|row| row.join("   │   "))
        .collect::<Vec<_>>()
        .join("\n");
    frame.render_widget(
        Paragraph::new(rows).block(
            Block::default()
                .title(format!(" {identity} · Performance "))
                .borders(Borders::ALL),
        ),
        area,
    );
}

fn render_instances(
    frame: &mut Frame<'_>,
    area: ratatui::layout::Rect,
    snapshot: &DashboardSnapshot,
    selected: Option<usize>,
) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(32), Constraint::Percentage(68)].as_ref())
        .split(area);
    let entries = snapshot
        .instances
        .iter()
        .map(|instance| {
            let stats = &instance.statistics;
            let label = format!(
                "{}  {}  ret {}  Sortino {}",
                instance.identity(),
                stats.status.as_deref().unwrap_or("unknown"),
                number(stats.cumulative_return, true).unwrap_or_else(|| "—".into()),
                number(stats.sortino, false).unwrap_or_else(|| "—".into())
            );
            ListItem::new(label)
        })
        .collect::<Vec<_>>();
    let list = List::new(entries)
        .block(
            Block::default()
                .title(" Strategies · j/k move · Enter inspect · a add · e enable/disable · x delete · q quit ")
                .borders(Borders::ALL),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );
    let mut state = ListState::default();
    state.select(selected);
    frame.render_stateful_widget(list, chunks[0], &mut state);
    let rows = snapshot
        .instances
        .iter()
        .map(|i| {
            let s = &i.statistics;
            let pct = |value: Option<f64>| {
                value
                    .map(|v| format!("{v:.2}%"))
                    .unwrap_or_else(|| "—".into())
            };
            format!(
                "{:<20} {:>8} {:>12} {:>12} {:>9} {:>7} {:>7} {:>7} {:>7} {:>7}",
                i.identity(),
                s.status.as_deref().unwrap_or("—"),
                number(s.equity, false).unwrap_or_else(|| "—".into()),
                number(s.absolute_return, false).unwrap_or_else(|| "—".into()),
                number(s.win_rate, true).unwrap_or_else(|| "—".into()),
                s.trades
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "—".into()),
                s.open_positions
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "—".into()),
                pct(s.stop_loss_pct),
                pct(s.take_profit_pct),
                pct(s.price_vs_entry_pct)
            )
        })
        .collect::<Vec<_>>();
    let table = std::iter::once(format!(
        "{:<20} {:>8} {:>12} {:>12} {:>9} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "INSTANCE",
        "STATUS",
        "EQUITY",
        "PNL",
        "WIN%",
        "TRADES",
        "POS",
        "SL%",
        "TP%",
        "PvE%"
    ))
    .chain(rows)
    .collect::<Vec<_>>()
    .join("\n");
    frame.render_widget(
        Paragraph::new(table).block(
            Block::default()
                .title(" Instance detail · Esc aggregate · q quit ")
                .borders(Borders::ALL),
        ),
        chunks[1],
    );
}

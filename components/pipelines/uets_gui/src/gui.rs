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
    pub sortino: Option<f64>,
    pub expectancy_rr: Option<f64>,
    pub max_drawdown: Option<f64>,
    pub cumulative_return: Option<f64>,
    pub absolute_return: Option<f64>,
    pub cash: Option<f64>,
    pub equity: Option<f64>,
    pub realized_pnl: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    pub trades: Option<usize>,
    pub open_positions: Option<usize>,
    pub win_rate: Option<f64>,
    pub exposure: Option<f64>,
    pub status: Option<String>,
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

/// The producer should publish already-calculated snapshots. `snapshot` must
/// return quickly; the GUI polls it independently at the display cadence.
pub trait SnapshotSource {
    fn snapshot(&mut self) -> DashboardSnapshot;
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
}

impl<S: SnapshotSource> Monitor<S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            selected: None,
            interval: Duration::from_millis(100),
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
            terminal.draw(|frame| render(frame, &snapshot, self.selected))?;
            let timeout = self.interval.saturating_sub(started.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
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
                        crate::controls::Action::None => {}
                    }
                }
            }
        }
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

fn render(frame: &mut Frame<'_>, snapshot: &DashboardSnapshot, selected: Option<usize>) {
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
}

fn metric(name: &str, value: Option<String>) -> String {
    format!("{name}: {}", value.unwrap_or_else(|| "—".into()))
}
fn number(value: Option<f64>, percent: bool) -> Option<String> {
    value.map(|v| {
        if percent {
            format!("{:.2}%", v * 100.0)
        } else {
            format!("{v:.3}")
        }
    })
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
        metric("Status", stats.status.clone()),
        metric("Market data", stats.latest_market_data.clone()),
        metric("Decision", stats.latest_decision.clone()),
        metric("Strategy ms", number(stats.processing_latency_ms, false)),
        metric(
            "Exchange→local ms",
            number(stats.exchange_latency_ms, false),
        ),
        metric("Latency p05 ms", number(stats.latency_p05_ms, false)),
        metric("Latency p95 ms", number(stats.latency_p95_ms, false)),
        metric("Latency p99 ms", number(stats.latency_p99_ms, false)),
        metric(
            "Clock skew mean ms",
            number(stats.clock_skew_mean_ms, false),
        ),
        metric("Clock skew max ms", number(stats.clock_skew_max_ms, false)),
        metric(
            "Decision→order ms",
            number(stats.decision_to_order_ms, false),
        ),
        metric("Order→fill ms", number(stats.order_to_fill_ms, false)),
        metric("Memory MB", number(stats.memory_mb, false)),
        metric("CPU %", number(stats.cpu_percent, false)),
        metric(
            "Persist queue",
            stats.persistence_queue.map(|v| v.to_string()),
        ),
        metric("Write ms", number(stats.persistence_write_ms, false)),
        metric(
            "Storage errors",
            stats.persistence_errors.map(|v| v.to_string()),
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
                .title(" Strategies · j/k · Enter ")
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
            format!(
                "{:<20} {:>8} {:>12} {:>12} {:>9} {:>7} {:>7}",
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
                    .unwrap_or_else(|| "—".into())
            )
        })
        .collect::<Vec<_>>();
    let table = std::iter::once(
        "INSTANCE             STATUS       EQUITY       PNL       WIN% TRADES POS".to_string(),
    )
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

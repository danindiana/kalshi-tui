use chrono::{DateTime, Utc};
use rusqlite::Connection;
use tokio::sync::mpsc;

use crate::api::{fetch_markets, Balance, Market, Order, Position};
use crate::db;
use crate::model::ModelOutput;
use crate::sysinfo::StatusData;

// ── Commands from the background fetcher ────────────────────────────────────

pub enum AppCommand {
    MarketsUpdated {
        series: String,
        markets: Vec<Market>,
        fetched_at: DateTime<Utc>,
    },
    SpotUpdated {
        btc_usd: f64,
    },
    ModelUpdated(ModelOutput),
    PortfolioUpdated {
        balance: Option<Balance>,
        positions: Vec<Position>,
        orders: Vec<Order>,
        fetched_at: DateTime<Utc>,
    },
    StatusUpdated(Box<StatusData>),
    FetchError(String),
}

// ── Prediction outcome ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    Pending,
    Win,
    Loss,
}

impl Outcome {
    pub fn label(self) -> &'static str {
        match self {
            Outcome::Pending => "…",
            Outcome::Win => "WIN",
            Outcome::Loss => "LOSS",
        }
    }
}

// ── A user-entered prediction record ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Prediction {
    pub id: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub series: String,
    pub strike: f64,
    pub side: PredictionSide,
    pub price_at_entry: f64,
    pub target_price: f64,
    pub note: String,
    pub outcome: Outcome,
    pub resolved_price: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionSide {
    Yes,
    No,
}

impl std::fmt::Display for PredictionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictionSide::Yes => write!(f, "YES"),
            PredictionSide::No => write!(f, "NO"),
        }
    }
}

// ── Auto-trade types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeOutcome {
    Win,
    Loss,
    Open,
}

impl TradeOutcome {
    pub fn label(self) -> &'static str {
        match self {
            TradeOutcome::Win => "WIN",
            TradeOutcome::Loss => "LOSS",
            TradeOutcome::Open => "OPEN",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoTradeRecord {
    pub id: i64,
    pub placed_at: String,
    pub ticker: String,
    pub strike: f64,
    pub side: String,
    pub count: i64,
    pub price_cents: i64,
    pub cost_dollars: f64,
    pub edge: Option<f64>,
    pub outcome: TradeOutcome,
    pub pnl: Option<f64>,
    pub dry_run: bool,
}

#[derive(Debug, Default)]
pub struct TradeSummary {
    pub total: usize,
    pub live_count: usize,
    pub dry_count: usize,
    pub wins: usize,
    pub losses: usize,
    pub open_count: usize,
    pub total_deployed: f64,
    pub net_pnl: f64,
}

// ── Tabs ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tab {
    Markets,
    Predictions,
    Analysis,
    Portfolio,
    Status,
    AutoTrades,
    PnlChart,
}

impl Tab {
    pub const ALL: &'static [Tab] = &[
        Tab::Markets,
        Tab::Predictions,
        Tab::Analysis,
        Tab::Portfolio,
        Tab::Status,
        Tab::AutoTrades,
        Tab::PnlChart,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Markets => "Markets",
            Tab::Predictions => "My Predictions",
            Tab::Analysis => "Edge Analysis",
            Tab::Portfolio => "Portfolio",
            Tab::Status => "System",
            Tab::AutoTrades => "Auto Trades",
            Tab::PnlChart => "P&L Chart",
        }
    }
}

// ── Input / confirm state ─────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub enum InputMode {
    #[default]
    Normal,
    /// Entering target BTC price for a new prediction
    EnterPrice { buf: String },
    /// Confirming an outcome for the selected prediction
    ConfirmOutcome { idx: usize, buf: String },
}

fn compute_trade_summary(trades: &[AutoTradeRecord]) -> TradeSummary {
    let mut s = TradeSummary::default();
    s.total = trades.len();
    for t in trades {
        if t.dry_run { s.dry_count += 1; } else { s.live_count += 1; }
        match t.outcome {
            TradeOutcome::Win  => s.wins += 1,
            TradeOutcome::Loss => s.losses += 1,
            TradeOutcome::Open => s.open_count += 1,
        }
        s.total_deployed += t.cost_dollars;
        if let Some(pnl) = t.pnl {
            s.net_pnl += pnl;
        }
    }
    s
}

// ── Main application state ────────────────────────────────────────────────────

pub struct App {
    pub tab: Tab,
    pub scroll: usize,
    pub selected: usize,
    pub btc_markets: Vec<Market>,
    pub eth_markets: Vec<Market>,
    pub predictions: Vec<Prediction>,
    pub last_fetched: Option<DateTime<Utc>>,
    pub btc_spot: Option<f64>,
    pub model: Option<ModelOutput>,
    pub model_loading: bool,
    pub balance: Option<Balance>,
    pub positions: Vec<Position>,
    pub orders: Vec<Order>,
    pub portfolio_auth_error: Option<String>,
    pub status: String,
    pub input_mode: InputMode,
    pub status_data: Option<StatusData>,
    pub status_loading: bool,
    pub auto_trades: Vec<AutoTradeRecord>,
    pub trade_summary: TradeSummary,
    tx: mpsc::Sender<AppCommand>,
    pub db: Connection,
    trader_db: Option<Connection>,
}

impl App {
    pub fn new(tx: mpsc::Sender<AppCommand>) -> anyhow::Result<Self> {
        let db = db::open()?;
        let predictions = db::load_predictions(&db).unwrap_or_default();
        let trader_db = db::try_open_trader_db();
        let (auto_trades, trade_summary) = if let Some(ref tdb) = trader_db {
            let trades = db::load_auto_trades(tdb).unwrap_or_default();
            let summary = compute_trade_summary(&trades);
            (trades, summary)
        } else {
            (Vec::new(), TradeSummary::default())
        };
        Ok(Self {
            tab: Tab::Markets,
            scroll: 0,
            selected: 0,
            btc_markets: Vec::new(),
            eth_markets: Vec::new(),
            predictions,
            last_fetched: None,
            btc_spot: None,
            model: None,
            model_loading: true,
            balance: None,
            positions: Vec::new(),
            orders: Vec::new(),
            portfolio_auth_error: None,
            status: "Fetching markets & running model…".into(),
            input_mode: InputMode::Normal,
            status_data: None,
            status_loading: false,
            auto_trades,
            trade_summary,
            tx,
            db,
            trader_db,
        })
    }

    // ── Tab / scroll navigation ───────────────────────────────────────────────

    pub fn next_tab(&mut self) {
        let idx = Tab::ALL.iter().position(|&t| t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + 1) % Tab::ALL.len()];
        self.scroll = 0;
        self.selected = 0;
    }

    pub fn prev_tab(&mut self) {
        let idx = Tab::ALL.iter().position(|&t| t == self.tab).unwrap_or(0);
        self.tab = Tab::ALL[(idx + Tab::ALL.len() - 1) % Tab::ALL.len()];
        self.scroll = 0;
        self.selected = 0;
    }

    pub fn scroll_down(&mut self) {
        let max = match self.tab {
            Tab::Predictions => self.predictions.len().saturating_sub(1),
            Tab::AutoTrades => self.auto_trades.len().saturating_sub(1),
            Tab::Markets | Tab::Analysis | Tab::Portfolio | Tab::Status | Tab::PnlChart => 200,
        };
        if self.selected < max {
            self.selected += 1;
        }
        self.scroll = self.scroll.saturating_add(1);
    }

    pub fn scroll_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
        self.scroll = self.scroll.saturating_sub(1);
    }

    // ── Background data handling ──────────────────────────────────────────────

    pub fn handle_command(&mut self, cmd: AppCommand) {
        match cmd {
            AppCommand::MarketsUpdated { series, markets, fetched_at } => {
                match series.as_str() {
                    "KXBTCD" => self.btc_markets = markets,
                    "KXETH" => self.eth_markets = markets,
                    _ => {}
                }
                self.last_fetched = Some(fetched_at);
                self.status = format!("Updated {}", fetched_at.format("%H:%M:%S UTC"));
            }
            AppCommand::SpotUpdated { btc_usd } => {
                self.btc_spot = Some(btc_usd);
            }
            AppCommand::ModelUpdated(output) => {
                self.model_loading = false;
                self.status = format!(
                    "Model: ${:.0} → ${:.0} ± ${:.0}  |  Updated {}",
                    output.current_price,
                    output.model_prediction,
                    output.mae_current,
                    output.timestamp.get(11..19).unwrap_or("?"),
                );
                self.model = Some(output);
            }
            AppCommand::PortfolioUpdated { balance, positions, orders, fetched_at } => {
                self.balance = balance;
                self.positions = positions;
                self.orders = orders;
                self.status = format!("Portfolio updated {}", fetched_at.format("%H:%M:%S UTC"));
            }
            AppCommand::StatusUpdated(data) => {
                self.status_loading = false;
                self.status_data = Some(*data);
            }
            AppCommand::FetchError(e) => {
                if e.starts_with("portfolio:") || e.contains("key_id") {
                    self.portfolio_auth_error = Some(e.clone());
                }
                self.status = format!("Error: {e}");
            }
        }
    }

    pub fn reload_auto_trades(&mut self) {
        if let Some(ref tdb) = self.trader_db {
            let trades = db::load_auto_trades(tdb).unwrap_or_default();
            self.trade_summary = compute_trade_summary(&trades);
            self.auto_trades = trades;
        }
    }

    pub fn request_refresh(&mut self) {
        self.reload_auto_trades();
        let tx = self.tx.clone();
        self.status = "Refreshing markets…".into();
        tokio::spawn(async move {
            for series in &["KXBTCD", "KXETH"] {
                match fetch_markets(series).await {
                    Ok(markets) => {
                        let _ = tx.send(AppCommand::MarketsUpdated {
                            series: series.to_string(),
                            markets,
                            fetched_at: Utc::now(),
                        }).await;
                    }
                    Err(e) => {
                        let _ = tx.send(AppCommand::FetchError(format!("{series}: {e}"))).await;
                    }
                }
            }
        });
    }

    /// Kick off an immediate model inference run (takes ~20s)
    pub fn request_model_refresh(&mut self) {
        let tx = self.tx.clone();
        self.model_loading = true;
        self.status = "Running model inference…".into();
        tokio::spawn(async move {
            match crate::api::poll_model().await {
                Ok(output) => { let _ = tx.send(AppCommand::ModelUpdated(output)).await; }
                Err(e)     => { let _ = tx.send(AppCommand::FetchError(format!("model: {e}"))).await; }
            }
        });
    }

    /// Refresh the system status tab (services, DB, log tail).
    pub fn request_status_refresh(&mut self) {
        if self.status_loading {
            return;
        }
        self.status_loading = true;
        let tx = self.tx.clone();
        tokio::spawn(async move {
            match crate::sysinfo::collect().await {
                Ok(data) => { let _ = tx.send(AppCommand::StatusUpdated(Box::new(data))).await; }
                Err(e)   => { let _ = tx.send(AppCommand::FetchError(format!("sysinfo: {e}"))).await; }
            }
        });
    }

    // ── Prediction input ──────────────────────────────────────────────────────

    pub fn is_inputting(&self) -> bool {
        !matches!(self.input_mode, InputMode::Normal)
    }

    pub fn open_prediction_input(&mut self) {
        self.input_mode = InputMode::EnterPrice { buf: String::new() };
        self.status = "Enter target BTC price → Enter to save, Esc to cancel".into();
    }

    /// Mark the currently selected prediction as WIN or LOSS.
    /// Press 'w' for win, 'l' for loss.
    pub fn resolve_selected(&mut self, outcome: Outcome) {
        if self.tab != Tab::Predictions || self.predictions.is_empty() {
            return;
        }
        let idx = self.selected.min(self.predictions.len() - 1);
        let pred = &mut self.predictions[idx];
        if pred.outcome != Outcome::Pending {
            self.status = "Already resolved.".into();
            return;
        }
        let price = self.btc_spot.unwrap_or(pred.strike);
        pred.outcome = outcome;
        pred.resolved_price = Some(price);

        if let Some(id) = pred.id {
            let _ = db::resolve_prediction(&self.db, id, &outcome, price);
        }

        self.status = format!(
            "Prediction {} marked as {} at ${:.2}",
            idx + 1,
            outcome.label(),
            price
        );
    }

    pub fn push_char(&mut self, c: char) {
        if let InputMode::EnterPrice { buf } = &mut self.input_mode {
            if c.is_ascii_digit() || (c == '.' && !buf.contains('.')) {
                buf.push(c);
            }
        }
    }

    pub fn pop_char(&mut self) {
        if let InputMode::EnterPrice { buf } = &mut self.input_mode {
            buf.pop();
        }
    }

    pub fn cancel_input(&mut self) {
        self.input_mode = InputMode::Normal;
        self.status = "Cancelled.".into();
    }

    pub fn confirm_input(&mut self) {
        let InputMode::EnterPrice { buf } = self.input_mode.clone() else {
            return;
        };
        self.input_mode = InputMode::Normal;

        let Ok(target_price) = buf.parse::<f64>() else {
            self.status = "Invalid price — prediction discarded.".into();
            return;
        };

        let markets = &self.btc_markets;
        if markets.is_empty() {
            self.status = "No BTC markets loaded yet.".into();
            return;
        }

        let closest = markets
            .iter()
            .min_by(|a, b| {
                (a.strike - target_price)
                    .abs()
                    .partial_cmp(&(b.strike - target_price).abs())
                    .unwrap()
            })
            .unwrap();

        let side = if target_price >= closest.strike {
            PredictionSide::Yes
        } else {
            PredictionSide::No
        };

        let mut pred = Prediction {
            id: None,
            created_at: Utc::now(),
            series: "KXBTCD".into(),
            strike: closest.strike,
            side,
            price_at_entry: closest.yes_price,
            target_price,
            note: String::new(),
            outcome: Outcome::Pending,
            resolved_price: None,
        };

        match db::insert_prediction(&self.db, &pred) {
            Ok(id) => {
                pred.id = Some(id);
                self.status = format!(
                    "Saved #{id}: {} {} at strike ${:.0} (entry YES={:.0}¢)",
                    pred.side, pred.series, pred.strike, pred.price_at_entry * 100.0
                );
            }
            Err(e) => {
                self.status = format!("DB error: {e}");
            }
        }

        self.predictions.push(pred);
        self.tab = Tab::Predictions;
        self.selected = self.predictions.len() - 1;
        self.scroll = self.selected;
    }

    // ── Edge analysis helpers ─────────────────────────────────────────────────

    pub fn current_edge(&self, pred: &Prediction) -> Option<(f64, f64)> {
        let markets = match pred.series.as_str() {
            "KXBTCD" => &self.btc_markets,
            "KXETH" => &self.eth_markets,
            _ => return None,
        };
        let m = markets.iter().find(|m| (m.strike - pred.strike).abs() < 1.0)?;
        let current_yes = m.yes_price;
        let edge = match pred.side {
            PredictionSide::Yes => current_yes - pred.price_at_entry,
            PredictionSide::No => (1.0 - current_yes) - (1.0 - pred.price_at_entry),
        };
        Some((current_yes, edge))
    }
}

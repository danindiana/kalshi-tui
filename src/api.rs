use anyhow::{Context, Result};
use chrono::Utc;
use serde::Deserialize;
use tokio::{process::Command, sync::mpsc, time};
use std::process::Stdio;

use crate::app::AppCommand;
use crate::auth::KalshiSigner;
use crate::model::ModelOutput;

const KALSHI_BASE: &str = "https://api.elections.kalshi.com/trade-api/v2";
const GEMINI_TICKER: &str = "https://api.gemini.com/v1/pubticker/btcusd";
const ETH_SERIES: &str = "KXETH";

// ── Kalshi API response shapes ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct MarketsResponse {
    markets: Vec<RawMarket>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct RawMarket {
    pub ticker: String,
    pub title: String,
    pub floor_strike: Option<f64>,
    // Prices are string-formatted dollar amounts in 0.0000–1.0000 range
    // (already probability; 0.0100 = 1% = "1¢")
    pub yes_ask_dollars: Option<String>,
    pub no_ask_dollars: Option<String>,
    pub volume_24h_fp: Option<String>,
    pub open_interest_fp: Option<String>,
    pub close_time: Option<String>,
}

// ── Normalised market data ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Market {
    pub ticker: String,
    pub title: String,
    pub strike: f64,
    /// yes price in dollars (0.00–1.00 implied probability)
    pub yes_price: f64,
    pub no_price: f64,
    pub volume_24h: f64,
    pub open_interest: f64,
    pub close_time: Option<String>,
}

fn parse_price(s: Option<String>) -> f64 {
    s.and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0)
}

impl TryFrom<RawMarket> for Market {
    type Error = ();
    fn try_from(r: RawMarket) -> Result<Self, ()> {
        let strike = r.floor_strike.ok_or(())?;
        Ok(Market {
            ticker: r.ticker,
            title: r.title,
            strike,
            // Values are already 0.0–1.0 (e.g. "0.0100" = 1% probability)
            yes_price: parse_price(r.yes_ask_dollars),
            no_price: parse_price(r.no_ask_dollars),
            volume_24h: parse_price(r.volume_24h_fp),
            open_interest: parse_price(r.open_interest_fp),
            close_time: r.close_time,
        })
    }
}

// ── Gemini spot response ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GeminiTicker {
    last: String,
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────

pub async fn fetch_markets(series_ticker: &str) -> Result<Vec<Market>> {
    let url = format!(
        "{}/markets?status=open&series_ticker={}&limit=100",
        KALSHI_BASE, series_ticker
    );
    let resp: MarketsResponse = reqwest::Client::new()
        .get(&url)
        .header("Accept", "application/json")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let mut markets: Vec<Market> = resp
        .markets
        .into_iter()
        .filter_map(|r| Market::try_from(r).ok())
        .collect();
    markets.sort_by(|a, b| a.strike.partial_cmp(&b.strike).unwrap());
    Ok(markets)
}

pub async fn fetch_btc_spot() -> Result<f64> {
    let resp: GeminiTicker = reqwest::Client::new()
        .get(GEMINI_TICKER)
        .header("Accept", "application/json")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(resp.last.parse::<f64>()?)
}

// ── Classifier model subprocess ───────────────────────────────────────────────
//
// The TUI shells out to a Python classifier script for inference. Both paths
// are overridable via environment:
//   KALSHI_TUI_PYTHON_BIN     — path to python interpreter (venv recommended)
//   KALSHI_TUI_PREDICT_SCRIPT — path to predict_classifier_json.py
// Defaults assume the sibling gemini_trader layout; adjust or set env vars
// to match your setup.

fn python_bin() -> String {
    std::env::var("KALSHI_TUI_PYTHON_BIN")
        .unwrap_or_else(|_| "python3".to_string())
}

fn predict_script() -> String {
    std::env::var("KALSHI_TUI_PREDICT_SCRIPT")
        .unwrap_or_else(|_| "predict_classifier_json.py".to_string())
}

pub async fn poll_model() -> Result<ModelOutput> {
    let py = python_bin();
    let script = predict_script();
    let out = Command::new(&py)
        .arg(&script)
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| {
            format!(
                "failed to spawn `{py} {script}` — set KALSHI_TUI_PYTHON_BIN and \
                 KALSHI_TUI_PREDICT_SCRIPT if these paths are wrong"
            )
        })?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("{script} exited {} — {}", out.status, stderr.trim());
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    // Surface Python-level exceptions (script exits 0 but prints {"error":"..."})
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(stdout.trim()) {
        if let Some(err) = val.get("error").and_then(|e| e.as_str()) {
            anyhow::bail!("classifier: {err}");
        }
    }
    let output: ModelOutput = serde_json::from_str(stdout.trim())?;
    Ok(output)
}

// ── Portfolio API (authenticated) ─────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Balance {
    pub balance: i64,           // cents
    pub payout: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position {
    pub ticker: String,
    pub position: i64,          // contracts held (+yes, -no)
    pub market_exposure: i64,   // cents
    pub realized_pnl: i64,      // cents
    pub total_cost: i64,        // cents
    pub fees_paid: i64,
    pub resting_orders_count: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PositionsResponse {
    pub market_positions: Vec<Position>,
    pub event_positions: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Order {
    pub order_id: String,
    pub ticker: String,
    pub action: String,         // "buy" | "sell"
    pub side: String,           // "yes" | "no"
    pub status: String,         // "resting" | "filled" | "canceled"
    pub yes_price: i64,         // cents
    pub no_price: i64,          // cents
    pub count: i64,
    pub filled_count: Option<i64>,
    pub remaining_count: Option<i64>,
    pub created_time: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OrdersResponse {
    orders: Vec<Order>,
}

fn signed_client(signer: &KalshiSigner, method: &str, url: &str) -> Result<reqwest::RequestBuilder> {
    let headers = signer.auth_headers(method, url)?;
    let mut builder = reqwest::Client::new()
        .request(method.parse()?, url)
        .header("Accept", "application/json");
    for (k, v) in &headers {
        builder = builder.header(k.as_str(), v.as_str());
    }
    Ok(builder)
}

pub async fn fetch_balance(signer: &KalshiSigner) -> Result<Balance> {
    let url = format!("{KALSHI_BASE}/portfolio/balance");
    let resp = signed_client(signer, "GET", &url)?
        .send().await?
        .error_for_status()?
        .json::<Balance>().await?;
    Ok(resp)
}

pub async fn fetch_positions(signer: &KalshiSigner) -> Result<Vec<Position>> {
    let url = format!("{KALSHI_BASE}/portfolio/positions");
    let resp = signed_client(signer, "GET", &url)?
        .send().await?
        .error_for_status()?
        .json::<PositionsResponse>().await?;
    Ok(resp.market_positions)
}

pub async fn fetch_orders(signer: &KalshiSigner) -> Result<Vec<Order>> {
    let url = format!("{KALSHI_BASE}/portfolio/orders?limit=50&status=resting");
    let resp = signed_client(signer, "GET", &url)?
        .send().await?
        .error_for_status()?
        .json::<OrdersResponse>().await?;
    Ok(resp.orders)
}

// ── Background fetcher loop ───────────────────────────────────────────────────

pub async fn fetcher_loop(
    tx: mpsc::Sender<AppCommand>,
    refresh_secs: u64,
    signer: Option<std::sync::Arc<KalshiSigner>>,
) {
    let mut market_interval = time::interval(time::Duration::from_secs(refresh_secs));
    let spot_secs = (refresh_secs / 3).max(10);
    let mut spot_interval = time::interval(time::Duration::from_secs(spot_secs));
    let mut model_interval = time::interval(time::Duration::from_secs(30));
    // Portfolio refresh every 60s (or on-demand)
    let mut portfolio_interval = time::interval(time::Duration::from_secs(60));

    let series = ["KXBTCD", ETH_SERIES];

    loop {
        tokio::select! {
            _ = market_interval.tick() => {
                for s in &series {
                    match fetch_markets(s).await {
                        Ok(markets) => {
                            let _ = tx.send(AppCommand::MarketsUpdated {
                                series: s.to_string(),
                                markets,
                                fetched_at: Utc::now(),
                            }).await;
                        }
                        Err(e) => {
                            let _ = tx.send(AppCommand::FetchError(format!("{s}: {e}"))).await;
                        }
                    }
                }
            }
            _ = spot_interval.tick() => {
                if let Ok(price) = fetch_btc_spot().await {
                    let _ = tx.send(AppCommand::SpotUpdated { btc_usd: price }).await;
                }
            }
            _ = model_interval.tick() => {
                let tx2 = tx.clone();
                tokio::spawn(async move {
                    match poll_model().await {
                        Ok(output) => { let _ = tx2.send(AppCommand::ModelUpdated(output)).await; }
                        Err(e)     => { let _ = tx2.send(AppCommand::FetchError(format!("model: {e}"))).await; }
                    }
                });
            }
            _ = portfolio_interval.tick() => {
                if let Some(sgn) = signer.clone() {
                    let tx2 = tx.clone();
                    tokio::spawn(async move {
                        fetch_and_send_portfolio(&sgn, &tx2).await;
                    });
                }
            }
        }
    }
}

pub async fn fetch_and_send_portfolio(signer: &KalshiSigner, tx: &mpsc::Sender<AppCommand>) {
    let balance = fetch_balance(signer).await;
    let positions = fetch_positions(signer).await;
    let orders = fetch_orders(signer).await;

    let _ = tx.send(AppCommand::PortfolioUpdated {
        balance: balance.ok(),
        positions: positions.unwrap_or_default(),
        orders: orders.unwrap_or_default(),
        fetched_at: Utc::now(),
    }).await;
}

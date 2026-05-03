/// Collects system-status data for the Status tab:
///   - systemd service/timer states (via `systemctl show`)
///   - recent runs + trades from auto_trader.db (read-only)
///   - last N lines of auto_trader.log
///   - balance/HWM/drawdown parsed from the log

use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::Connection;
use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};
use tokio::process::Command;

// ── Public data types ────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct StatusData {
    pub refreshed_at: DateTime<Utc>,

    // ── Services ──────────────────────────────────────────────────────────────
    pub timer_active: bool,
    /// Absolute UTC time of the next timer fire (from NextElapseUSecRealtime).
    pub timer_next: Option<DateTime<Utc>>,
    /// "success" | "core-dump" | "killed" | "failed" | "timeout" | "unknown"
    pub svc_result: String,
    /// Raw timestamp string "HH:MM CDT" for last service exit
    pub svc_last_run_str: String,
    pub svc_active: bool,

    pub orderbook_active: bool,
    /// Raw "HH:MM CDT" or "Xd Xh Xm" for uptime start
    pub orderbook_since_str: String,
    /// Absolute UTC of orderbook start (for computing uptime)
    pub orderbook_since: Option<DateTime<Utc>>,

    // ── Balance snapshot (parsed from log tail) ───────────────────────────────
    pub balance: Option<f64>,
    pub hwm: Option<f64>,
    pub drawdown_pct: Option<f64>,
    pub daily_deployed: Option<f64>,
    pub daily_budget: Option<f64>,
    pub dry_run: bool,

    // ── DB rows ───────────────────────────────────────────────────────────────
    pub recent_runs: Vec<RunRow>,
    pub recent_trades: Vec<TradeRow>,

    // ── Settlement-joined analytics (from trade_outcomes view) ────────────────
    pub perf_by_side: Vec<PerfSideRow>,
    pub perf_by_bucket: Vec<PerfBucketRow>,
    pub settled_trades: i64,
    pub unsettled_trades: i64,

    // ── Log tail ──────────────────────────────────────────────────────────────
    pub log_lines: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerfSideRow {
    pub side: String,           // "YES" | "NO"
    pub n: i64,
    pub wins: i64,
    pub win_pct: f64,
    pub avg_cost_win: f64,
    pub avg_cost_loss: f64,
    pub net_pnl: f64,
}

#[derive(Debug, Clone)]
pub struct PerfBucketRow {
    pub label: String,          // e.g. "large YES >$3"
    pub side: String,
    pub bucket: String,         // "small" | "mid" | "large"
    pub n: i64,
    pub win_pct: f64,
    pub net_pnl: f64,
}

#[derive(Debug, Clone)]
pub struct RunRow {
    pub run_at: String,
    pub current_price: Option<f64>,
    pub model_pred: Option<f64>,
    pub trades_placed: Option<i64>,
    pub total_staked: Option<f64>,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct TradeRow {
    pub placed_at: String,
    pub ticker: String,
    pub strike: f64,
    pub side: String,
    pub count: i64,
    pub price_cents: i64,
    pub cost_dollars: f64,
    pub edge: Option<f64>,
    pub status: String,
}

// ── Main entry point ─────────────────────────────────────────────────────────

pub async fn collect() -> Result<StatusData> {
    let (timer_info, svc_info, obc_info) = tokio::join!(
        systemctl_show(
            "kalshi-auto-trader.timer",
            &["ActiveState", "NextElapseUSecRealtime"],
        ),
        systemctl_show(
            "kalshi-auto-trader.service",
            &["ActiveState", "Result", "ExecMainExitTimestamp"],
        ),
        systemctl_show(
            "orderbook-collector.service",
            &["ActiveState", "ActiveEnterTimestamp"],
        ),
    );

    let timer_active = timer_info.get("ActiveState").map(|s| s == "active").unwrap_or(false);
    let timer_next = timer_info
        .get("NextElapseUSecRealtime")
        .and_then(|s| parse_usec_realtime(s));

    let svc_active = svc_info.get("ActiveState").map(|s| s == "active").unwrap_or(false);
    let svc_result = svc_info
        .get("Result")
        .cloned()
        .unwrap_or_else(|| "unknown".into());
    let svc_last_run_str = svc_info
        .get("ExecMainExitTimestamp")
        .map(|s| extract_hhmm(s))
        .unwrap_or_default();

    let orderbook_active = obc_info.get("ActiveState").map(|s| s == "active").unwrap_or(false);
    let orderbook_since_str = obc_info
        .get("ActiveEnterTimestamp")
        .map(|s| extract_hhmm(s))
        .unwrap_or_default();
    let orderbook_since = obc_info
        .get("ActiveEnterTimestamp")
        .and_then(|s| parse_systemd_ts_approx(s));

    let log_path = data_home().join("kalshi-tui").join("auto_trader.log");
    let log_lines = read_log_tail(&log_path, 22);

    // Parse balance/HWM/drawdown/budget from last few log lines
    let (balance, hwm, drawdown_pct, daily_deployed, daily_budget, dry_run) =
        parse_balance_from_log(&log_lines);

    // DB queries (best-effort — if DB is locked or absent we get empty vecs)
    let db_path = data_home().join("kalshi-tui").join("auto_trader.db");
    let (recent_runs, recent_trades, perf_by_side, perf_by_bucket, settled_trades, unsettled_trades) =
        if db_path.exists() {
            match load_db_data(&db_path) {
                Ok(x) => x,
                Err(_) => (vec![], vec![], vec![], vec![], 0, 0),
            }
        } else {
            (vec![], vec![], vec![], vec![], 0, 0)
        };

    Ok(StatusData {
        refreshed_at: Utc::now(),
        timer_active,
        timer_next,
        svc_result,
        svc_last_run_str,
        svc_active,
        orderbook_active,
        orderbook_since_str,
        orderbook_since,
        balance,
        hwm,
        drawdown_pct,
        daily_deployed,
        daily_budget,
        dry_run,
        recent_runs,
        recent_trades,
        perf_by_side,
        perf_by_bucket,
        settled_trades,
        unsettled_trades,
        log_lines,
    })
}

// ── systemd helpers ──────────────────────────────────────────────────────────

async fn systemctl_show(unit: &str, props: &[&str]) -> HashMap<String, String> {
    let props_arg = props.join(",");
    let Ok(out) = Command::new("systemctl")
        .args(["show", unit, "--property", &props_arg, "--no-pager"])
        .output()
        .await
    else {
        return HashMap::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut map = HashMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.to_string(), v.to_string());
        }
    }
    map
}

/// Parse microsecond-precision UNIX timestamp from systemd (NextElapseUSecRealtime).
fn parse_usec_realtime(s: &str) -> Option<DateTime<Utc>> {
    let usec: i64 = s.parse().ok()?;
    if usec == 0 {
        return None;
    }
    let secs = usec / 1_000_000;
    let nsecs = ((usec % 1_000_000) * 1000) as u32;
    use chrono::TimeZone;
    Utc.timestamp_opt(secs, nsecs).single()
}

/// Extract "HH:MM TZ" from a systemd human-readable timestamp
/// "Tue 2026-04-14 08:50:04 CDT" → "08:50 CDT"
fn extract_hhmm(s: &str) -> String {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() >= 4 {
        let time = parts[2]; // "08:50:04"
        let tz = parts[3];   // "CDT"
        let hhmm = &time[..5.min(time.len())]; // "08:50"
        return format!("{hhmm} {tz}");
    }
    s.to_string()
}

/// Parse a systemd human-readable timestamp into approximate UTC.
/// "Tue 2026-04-14 08:50:04 CDT" — CDT = UTC-5, CST = UTC-6.
fn parse_systemd_ts_approx(s: &str) -> Option<DateTime<Utc>> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 4 {
        return None;
    }
    let dt_str = format!("{} {}", parts[1], parts[2]);
    let naive = chrono::NaiveDateTime::parse_from_str(&dt_str, "%Y-%m-%d %H:%M:%S").ok()?;
    let tz = parts[3];
    let offset_secs: i64 = match tz {
        "CDT" => 5 * 3600,
        "CST" => 6 * 3600,
        "EDT" => 4 * 3600,
        "EST" => 5 * 3600,
        "MDT" => 6 * 3600,
        "MST" => 7 * 3600,
        "PDT" => 7 * 3600,
        "PST" => 8 * 3600,
        _ => 0,
    };
    let utc_secs = naive.and_utc().timestamp() + offset_secs;
    use chrono::TimeZone;
    Utc.timestamp_opt(utc_secs, 0).single()
}

// ── Log parsing ──────────────────────────────────────────────────────────────

/// Parse balance, HWM, drawdown, daily_deployed, daily_budget, dry_run
/// from lines like:
///   "[OK]   Balance: $47.35  HWM: $53.56  Drawdown: 11.6%"
///   "[OK]   Daily budget: $47.35 — $22.87 remaining"
///   "[OK]   Limits — daily: $23.68  run: $18.94  stake: $5.68"
///   "  LIVE" or "  DRY-RUN"
fn parse_balance_from_log(
    lines: &[String],
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>, bool) {
    let mut balance = None;
    let mut hwm = None;
    let mut drawdown_pct = None;
    let mut daily_deployed = None;
    let mut daily_budget = None;
    let mut dry_run = false;

    for line in lines.iter().rev() {
        let l = line.trim();

        if l.contains("DRY-RUN") || l.contains("DRY_RUN") {
            dry_run = true;
        }
        if l == "LIVE" {
            dry_run = false;
        }

        // "[OK]   Balance: $47.35  HWM: $53.56  Drawdown: 11.6%"
        if balance.is_none() && l.contains("Balance:") && l.contains("HWM:") {
            balance = extract_dollar(l, "Balance:");
            hwm = extract_dollar(l, "HWM:");
            drawdown_pct = extract_pct(l, "Drawdown:");
        }

        // "[OK]   Daily budget: $47.35 — $22.87 remaining"
        if daily_budget.is_none() && l.contains("Daily budget:") {
            daily_budget = extract_dollar(l, "Daily budget:");
            // remaining is after "—"
            if let Some(rem) = extract_dollar(l, "—") {
                if let (Some(budget), Some(remaining)) = (daily_budget, Some(rem)) {
                    daily_deployed = Some(budget - remaining);
                }
            }
        }

        // Stop once we've found everything we need
        if balance.is_some() && daily_budget.is_some() {
            break;
        }
    }

    (balance, hwm, drawdown_pct, daily_deployed, daily_budget, dry_run)
}

fn extract_dollar(s: &str, prefix: &str) -> Option<f64> {
    let idx = s.find(prefix)?;
    let after = &s[idx + prefix.len()..];
    let after = after.trim_start().trim_start_matches('$');
    let end = after
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != ',')
        .unwrap_or(after.len());
    after[..end].replace(',', "").parse().ok()
}

fn extract_pct(s: &str, prefix: &str) -> Option<f64> {
    let idx = s.find(prefix)?;
    let after = &s[idx + prefix.len()..];
    let after = after.trim_start();
    let end = after
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

// ── File I/O ──────────────────────────────────────────────────────────────────

fn data_home() -> PathBuf {
    std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".local").join("share")
        })
}

fn read_log_tail(path: &std::path::Path, n: usize) -> Vec<String> {
    let Ok(mut f) = std::fs::File::open(path) else {
        return vec!["(log not found)".into()];
    };
    let Ok(len) = f.seek(SeekFrom::End(0)) else {
        return vec![];
    };
    let read_bytes = len.min(24_576); // 24 KB
    let _ = f.seek(SeekFrom::End(-(read_bytes as i64)));
    let mut buf = vec![0u8; read_bytes as usize];
    let _ = f.read(&mut buf);
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].to_vec()
}

// ── DB helpers ────────────────────────────────────────────────────────────────

type DbBundle = (
    Vec<RunRow>,
    Vec<TradeRow>,
    Vec<PerfSideRow>,
    Vec<PerfBucketRow>,
    i64, // settled trades
    i64, // unsettled trades
);

fn load_db_data(db_path: &PathBuf) -> Result<DbBundle> {
    let conn = Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(std::time::Duration::from_secs(1))?;
    let runs = load_recent_runs(&conn)?;
    let trades = load_recent_trades(&conn)?;
    // trade_outcomes is created by auto_trader.py. If it's missing (older DB
    // version) we just return empty analytics — the panel shows a hint.
    let (perf_side, perf_bucket, settled, unsettled) =
        load_outcomes(&conn).unwrap_or((vec![], vec![], 0, 0));
    Ok((runs, trades, perf_side, perf_bucket, settled, unsettled))
}

fn load_outcomes(
    conn: &Connection,
) -> Result<(Vec<PerfSideRow>, Vec<PerfBucketRow>, i64, i64)> {
    let has_view: Option<String> = conn
        .query_row(
            "SELECT name FROM sqlite_master WHERE name='trade_outcomes'",
            [],
            |r| r.get(0),
        )
        .ok();
    if has_view.is_none() {
        return Ok((vec![], vec![], 0, 0));
    }

    let mut side = Vec::new();
    let mut stmt = conn.prepare(
        "SELECT side,
                COUNT(*) n,
                SUM(CASE WHEN outcome='win'  THEN 1 ELSE 0 END) wins,
                100.0 * SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) / COUNT(*) win_pct,
                COALESCE(AVG(CASE WHEN outcome='win'  THEN cost_dollars END), 0.0) avg_cost_win,
                COALESCE(AVG(CASE WHEN outcome='loss' THEN cost_dollars END), 0.0) avg_cost_loss,
                COALESCE(SUM(pnl), 0.0) net_pnl
         FROM trade_outcomes
         WHERE outcome IN ('win','loss')
         GROUP BY side
         ORDER BY side",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(PerfSideRow {
            side: row.get(0)?,
            n: row.get(1)?,
            wins: row.get(2)?,
            win_pct: row.get(3)?,
            avg_cost_win: row.get(4)?,
            avg_cost_loss: row.get(5)?,
            net_pnl: row.get(6)?,
        })
    })?;
    for r in rows {
        side.push(r?);
    }

    let mut bucket = Vec::new();
    let mut stmt = conn.prepare(
        "SELECT CASE
                  WHEN cost_dollars < 1.50           THEN 'small'
                  WHEN cost_dollars BETWEEN 1.50 AND 3.00 THEN 'mid'
                  ELSE 'large'
                END AS bucket,
                side,
                COUNT(*) n,
                100.0 * SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) / COUNT(*) win_pct,
                COALESCE(SUM(pnl), 0.0) net_pnl
         FROM trade_outcomes
         WHERE outcome IN ('win','loss')
         GROUP BY bucket, side
         ORDER BY net_pnl ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        let bkt: String = row.get(0)?;
        let sid: String = row.get(1)?;
        let label = format!(
            "{:<5} {} {}",
            bkt,
            sid,
            match bkt.as_str() {
                "small" => "<$1.50",
                "mid" => "$1.5-3",
                _ => ">$3",
            }
        );
        Ok(PerfBucketRow {
            label,
            side: sid,
            bucket: bkt,
            n: row.get(2)?,
            win_pct: row.get(3)?,
            net_pnl: row.get(4)?,
        })
    })?;
    for r in rows {
        bucket.push(r?);
    }

    let settled: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM trade_outcomes WHERE outcome IN ('win','loss')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let unsettled: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM trade_outcomes WHERE outcome='unsettled'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);

    Ok((side, bucket, settled, unsettled))
}

fn load_recent_runs(conn: &Connection) -> Result<Vec<RunRow>> {
    let mut stmt = conn.prepare(
        "SELECT run_at, current_price, model_prediction, trades_placed, total_staked, dry_run
         FROM runs ORDER BY id DESC LIMIT 8",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(RunRow {
            run_at: row.get::<_, String>(0)?,
            current_price: row.get(1)?,
            model_pred: row.get(2)?,
            trades_placed: row.get(3)?,
            total_staked: row.get(4)?,
            dry_run: row.get::<_, i64>(5).map(|v| v != 0).unwrap_or(false),
        })
    })?;
    rows.map(|r| r.map_err(anyhow::Error::from)).collect()
}

fn load_recent_trades(conn: &Connection) -> Result<Vec<TradeRow>> {
    let mut stmt = conn.prepare(
        "SELECT placed_at, ticker, strike, side, count, price_cents, cost_dollars, edge, status
         FROM trades ORDER BY id DESC LIMIT 12",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(TradeRow {
            placed_at: row.get::<_, String>(0)?,
            ticker: row.get(1)?,
            strike: row.get(2)?,
            side: row.get(3)?,
            count: row.get(4)?,
            price_cents: row.get(5)?,
            cost_dollars: row.get(6)?,
            edge: row.get(7)?,
            status: row.get(8)?,
        })
    })?;
    rows.map(|r| r.map_err(anyhow::Error::from)).collect()
}

// ── Duration formatting ───────────────────────────────────────────────────────

/// Format a duration as "Xd Xh Xm" (for uptime) or "Xm Xs" (for short deltas).
pub fn fmt_uptime(since: DateTime<Utc>) -> String {
    let secs = (Utc::now() - since).num_seconds().max(0) as u64;
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    if days > 0 {
        format!("{days}d {hours}h {mins}m")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m {}s", secs % 60)
    }
}

/// Format delta until a future time as "in Xm Xs".
pub fn fmt_until(then: DateTime<Utc>) -> String {
    let secs = (then - Utc::now()).num_seconds();
    if secs <= 0 {
        return "now".into();
    }
    let mins = secs / 60;
    let s = secs % 60;
    if mins > 0 {
        format!("in {mins}m {s}s")
    } else {
        format!("in {s}s")
    }
}

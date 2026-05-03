use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::path::PathBuf;

use crate::app::{Outcome, Prediction, PredictionSide};

fn db_path() -> PathBuf {
    let base = dirs_next();
    base.join("kalshi-tui").join("predictions.db")
}

fn dirs_next() -> PathBuf {
    // XDG_DATA_HOME or ~/.local/share
    std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".local").join("share")
        })
}

pub fn open() -> Result<Connection> {
    let path = db_path();
    std::fs::create_dir_all(path.parent().unwrap())?;
    let conn = Connection::open(&path)?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         CREATE TABLE IF NOT EXISTS predictions (
             id          INTEGER PRIMARY KEY AUTOINCREMENT,
             created_at  TEXT NOT NULL,
             series      TEXT NOT NULL,
             strike      REAL NOT NULL,
             side        TEXT NOT NULL,
             price_at_entry REAL NOT NULL,
             target_price   REAL NOT NULL,
             note        TEXT NOT NULL DEFAULT '',
             outcome     TEXT,
             resolved_at TEXT,
             resolved_price REAL
         );",
    )?;
    Ok(conn)
}

pub fn insert_prediction(conn: &Connection, pred: &Prediction) -> Result<i64> {
    conn.execute(
        "INSERT INTO predictions
             (created_at, series, strike, side, price_at_entry, target_price, note)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            pred.created_at.to_rfc3339(),
            pred.series,
            pred.strike,
            pred.side.to_string(),
            pred.price_at_entry,
            pred.target_price,
            pred.note,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn resolve_prediction(
    conn: &Connection,
    id: i64,
    outcome: &Outcome,
    resolved_price: f64,
) -> Result<()> {
    let outcome_str = match outcome {
        Outcome::Win => "WIN",
        Outcome::Loss => "LOSS",
        Outcome::Pending => "PENDING",
    };
    conn.execute(
        "UPDATE predictions SET outcome=?1, resolved_at=?2, resolved_price=?3 WHERE id=?4",
        params![
            outcome_str,
            Utc::now().to_rfc3339(),
            resolved_price,
            id,
        ],
    )?;
    Ok(())
}

fn trader_db_path() -> PathBuf {
    dirs_next().join("kalshi-tui").join("auto_trader.db")
}

pub fn try_open_trader_db() -> Option<Connection> {
    let path = trader_db_path();
    if !path.exists() {
        return None;
    }
    Connection::open(&path).ok()
}

pub fn load_auto_trades(conn: &Connection) -> Result<Vec<crate::app::AutoTradeRecord>> {
    use crate::app::TradeOutcome;
    let mut stmt = conn.prepare(
        "SELECT t.id, t.placed_at, t.ticker, t.strike, t.side, t.count,
                t.price_cents, t.cost_dollars, t.edge,
                CASE
                  WHEN s.market_result IS NULL THEN 'open'
                  WHEN (t.side='YES' AND s.market_result='yes') OR
                       (t.side='NO'  AND s.market_result='no') THEN 'win'
                  ELSE 'loss'
                END AS outcome,
                CASE
                  WHEN s.market_result IS NULL THEN NULL
                  WHEN (t.side='YES' AND s.market_result='yes') OR
                       (t.side='NO'  AND s.market_result='no')
                       THEN (CAST(t.count AS REAL)) - t.cost_dollars
                  ELSE -t.cost_dollars
                END AS pnl,
                r.dry_run
         FROM trades t
         JOIN runs r ON r.id = t.run_id
         LEFT JOIN settlements_raw s ON s.ticker = t.ticker
         WHERE t.status = 'placed'
         ORDER BY t.id DESC
         LIMIT 500",
    )?;

    let rows = stmt.query_map([], |row| {
        let outcome_str: String = row.get(9)?;
        let outcome = match outcome_str.as_str() {
            "win"  => TradeOutcome::Win,
            "loss" => TradeOutcome::Loss,
            _      => TradeOutcome::Open,
        };
        Ok(crate::app::AutoTradeRecord {
            id:           row.get(0)?,
            placed_at:    row.get(1)?,
            ticker:       row.get(2)?,
            strike:       row.get(3)?,
            side:         row.get(4)?,
            count:        row.get(5)?,
            price_cents:  row.get(6)?,
            cost_dollars: row.get(7)?,
            edge:         row.get(8)?,
            outcome,
            pnl:          row.get(10)?,
            dry_run:      row.get::<_, i64>(11)? != 0,
        })
    })?;

    let mut trades = Vec::new();
    for row in rows {
        trades.push(row?);
    }
    Ok(trades)
}

pub fn load_predictions(conn: &Connection) -> Result<Vec<Prediction>> {
    let mut stmt = conn.prepare(
        "SELECT id, created_at, series, strike, side, price_at_entry, target_price,
                note, outcome, resolved_price
         FROM predictions ORDER BY id ASC",
    )?;

    let rows = stmt.query_map([], |row| {
        let side_str: String = row.get(4)?;
        let outcome_str: Option<String> = row.get(8)?;
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, f64>(3)?,
            side_str,
            row.get::<_, f64>(5)?,
            row.get::<_, f64>(6)?,
            row.get::<_, String>(7)?,
            outcome_str,
            row.get::<_, Option<f64>>(9)?,
        ))
    })?;

    let mut preds = Vec::new();
    for row in rows {
        let (id, created_at, series, strike, side_str, price_at_entry, target_price, note,
             outcome_str, resolved_price) = row?;

        let side = if side_str == "YES" {
            PredictionSide::Yes
        } else {
            PredictionSide::No
        };

        let outcome = match outcome_str.as_deref() {
            Some("WIN") => Outcome::Win,
            Some("LOSS") => Outcome::Loss,
            _ => Outcome::Pending,
        };

        let created_at: DateTime<Utc> = created_at
            .parse()
            .unwrap_or_else(|_| Utc::now());

        preds.push(Prediction {
            id: Some(id),
            created_at,
            series,
            strike,
            side,
            price_at_entry,
            target_price,
            note,
            outcome,
            resolved_price,
        });
    }
    Ok(preds)
}

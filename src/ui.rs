use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span, Text},
    widgets::{
        Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row,
        Scrollbar, ScrollbarOrientation, ScrollbarState, Table, Tabs, Wrap,
    },
    Frame,
};

use crate::app::{App, AutoTradeRecord, InputMode, Outcome, PredictionSide, Tab, TradeOutcome};
use crate::api::{Market, Order, Position};
use crate::model::{ConsensusSignal, ModelOutput};
use crate::sysinfo::{fmt_uptime, fmt_until, StatusData};

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // tab bar
            Constraint::Min(0),    // body
            Constraint::Length(2), // footer
        ])
        .split(area);

    draw_tabs(f, app, chunks[0]);

    match app.tab {
        Tab::Markets    => draw_markets(f, app, chunks[1]),
        Tab::Predictions => draw_predictions(f, app, chunks[1]),
        Tab::Analysis   => draw_analysis(f, app, chunks[1]),
        Tab::Portfolio  => draw_portfolio(f, app, chunks[1]),
        Tab::Status     => draw_status(f, app, chunks[1]),
        Tab::AutoTrades => draw_trades(f, app, chunks[1]),
        Tab::PnlChart   => draw_pnl_chart(f, app, chunks[1]),
    }

    draw_footer(f, app, chunks[2]);

    if app.is_inputting() {
        draw_input_overlay(f, app, area);
    }
}

// ── Tab bar ──────────────────────────────────────────────────────────────────

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<Line> = Tab::ALL
        .iter()
        .map(|t| Line::from(format!(" {} ", t.title())))
        .collect();

    let selected = Tab::ALL.iter().position(|&t| t == app.tab).unwrap_or(0);

    let spot_label = match app.btc_spot {
        Some(p) => format!("  BTC ${p:>10.2}  "),
        None => "  BTC ----------  ".into(),
    };

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Kalshi Crypto Tracker{spot_label}")),
        )
        .select(selected)
        .style(Style::default().fg(Color::DarkGray))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(tabs, area);
}

// ── Markets tab ───────────────────────────────────────────────────────────────

fn draw_markets(f: &mut Frame, app: &App, area: Rect) {
    // If we have a model, carve out a dashboard header above the tables
    if let Some(model) = &app.model {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(0)])
            .split(area);

        draw_dashboard(f, model, rows[0]);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);

        draw_market_table(f, app, &app.btc_markets, "BTC/USD (KXBTCD)", app.btc_spot, Some(model), cols[0]);
        draw_market_table(f, app, &app.eth_markets, "ETH/USD (KXETH)", None, None, cols[1]);
    } else {
        // Model not yet loaded — show loading notice and full-height tables
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        let loading_text = if app.model_loading {
            " ⏳ Model inference running (classifier + Kalshi)… ~5s "
        } else {
            " Model data unavailable "
        };
        let loading = Paragraph::new(loading_text)
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);
        f.render_widget(loading, rows[0]);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);

        draw_market_table(f, app, &app.btc_markets, "BTC/USD (KXBTCD)", app.btc_spot, None, cols[0]);
        draw_market_table(f, app, &app.eth_markets, "ETH/USD (KXETH)", None, None, cols[1]);
    }
}

// ── Dashboard header ──────────────────────────────────────────────────────────

fn draw_dashboard(f: &mut Frame, model: &ModelOutput, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(35), // Prediction + countdown
            Constraint::Percentage(20), // RSI gauge
            Constraint::Percentage(20), // Volatility gauge
            Constraint::Percentage(25), // Consensus signal
        ])
        .split(area);

    // ── Prediction + countdown ────────────────────────────────────────────────
    let direction_arrow = if model.model_bullish() { "▲" } else { "▼" };
    let direction_color = if model.model_bullish() { Color::Green } else { Color::Red };
    let (countdown_str, countdown_color) = fmt_market_countdown(compute_secs_to_market_close());
    let pred_text = Text::from(vec![
        Line::from(vec![
            Span::styled(
                format!(" {} ${:.0}", direction_arrow, model.model_prediction),
                Style::default().fg(direction_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" ± ${:.0}", model.mae_current),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::raw(format!(" Current: ${:.0}", model.current_price)),
        ]),
        Line::from(vec![
            Span::styled(countdown_str, Style::default().fg(countdown_color)),
        ]),
    ]);
    let pred_block = Paragraph::new(pred_text)
        .block(Block::default().borders(Borders::ALL).title(" 5 PM Target "));
    f.render_widget(pred_block, cols[0]);

    // ── RSI gauge ─────────────────────────────────────────────────────────────
    let rsi = model.indicators.rsi.clamp(0.0, 100.0) as u16;
    let rsi_color = if rsi >= 70 {
        Color::Red
    } else if rsi <= 30 {
        Color::Green
    } else {
        Color::Yellow
    };
    let rsi_label = if rsi >= 70 {
        format!("RSI {rsi} OVERBOUGHT")
    } else if rsi <= 30 {
        format!("RSI {rsi} OVERSOLD")
    } else {
        format!("RSI {rsi} neutral")
    };
    let rsi_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" RSI "))
        .gauge_style(Style::default().fg(rsi_color))
        .percent(rsi)
        .label(rsi_label);
    f.render_widget(rsi_gauge, cols[1]);

    // ── Volatility gauge ──────────────────────────────────────────────────────
    // Volatility is std dev in dollars; clamp to 0-200 for display
    let vol = model.indicators.volatility;
    let vol_pct = ((vol / 200.0) * 100.0).clamp(0.0, 100.0) as u16;
    let vol_color = if vol_pct > 60 { Color::Red } else if vol_pct > 30 { Color::Yellow } else { Color::Green };
    let vol_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Volatility "))
        .gauge_style(Style::default().fg(vol_color))
        .percent(vol_pct)
        .label(format!("σ ${vol:.1}"));
    f.render_widget(vol_gauge, cols[2]);

    // ── Consensus signal ──────────────────────────────────────────────────────
    let (signal_str, signal_color, signal_detail) = match model.consensus() {
        ConsensusSignal::StrongBuy => (
            "[ STRONG BUY ]",
            Color::Green,
            "Model & Market Bullish",
        ),
        ConsensusSignal::StrongSell => (
            "[ STRONG SELL ]",
            Color::Red,
            "Model & Market Bearish",
        ),
        ConsensusSignal::Divergence => (
            "[ DIVERGENCE ]",
            Color::Magenta,
            "Model vs Market conflict",
        ),
        ConsensusSignal::Unknown => (
            "[ --- ]",
            Color::DarkGray,
            "Insufficient data",
        ),
    };

    let best_edge_str = model.best_opportunity().map(|o| {
        format!("Best: {} ${:.0} edge={:+.1}%", o.side, o.strike, o.edge * 100.0)
    }).unwrap_or_default();

    let signal_text = Text::from(vec![
        Line::from(""),
        Line::from(Span::styled(
            format!(" {signal_str}"),
            Style::default().fg(signal_color).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!(" {signal_detail}"),
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            format!(" {best_edge_str}"),
            Style::default().fg(Color::Cyan),
        )),
    ]);
    let signal_block = Paragraph::new(signal_text)
        .block(Block::default().borders(Borders::ALL).title(" Consensus "));
    f.render_widget(signal_block, cols[3]);
}

// ── Market table ──────────────────────────────────────────────────────────────

fn draw_market_table(
    f: &mut Frame,
    app: &App,
    markets: &[Market],
    title: &str,
    spot: Option<f64>,
    model: Option<&ModelOutput>,
    area: Rect,
) {
    let header_style = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);

    let header = Row::new(vec![
        Cell::from("Strike"),
        Cell::from("YES¢"),
        Cell::from("NO¢"),
        Cell::from("Vol 24h"),
        Cell::from("Edge"),
    ])
    .style(header_style)
    .height(1);

    let rows: Vec<Row> = markets
        .iter()
        .skip(app.scroll)
        .map(|m| {
            let yes_pct = m.yes_price * 100.0;
            let no_pct  = m.no_price  * 100.0;

            let yes_color = if yes_pct > 60.0 {
                Color::Green
            } else if yes_pct < 40.0 {
                Color::Red
            } else {
                Color::White
            };

            // Look up model edge for this strike
            let edge_cell = if let Some(mdl) = model {
                if let Some(opp) = mdl.opportunities.iter().find(|o| (o.strike - m.strike).abs() < 1.0) {
                    let edge_pct = opp.edge * 100.0;
                    let edge_color = if edge_pct > 10.0 {
                        Color::Green
                    } else if edge_pct < -10.0 {
                        Color::Red
                    } else {
                        Color::DarkGray
                    };
                    Cell::from(format!("{}{:+.0}%", opp.side.chars().next().unwrap_or('?'), edge_pct))
                        .style(Style::default().fg(edge_color))
                } else {
                    Cell::from("")
                }
            } else {
                Cell::from("")
            };

            // Highlight row nearest to spot
            let row_style = if let Some(s) = spot {
                if (m.strike - s).abs() < 500.0 {
                    Style::default().bg(Color::DarkGray)
                } else {
                    Style::default()
                }
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(format!("${:.0}", m.strike)),
                Cell::from(format!("{:.0}¢", yes_pct)).style(Style::default().fg(yes_color)),
                Cell::from(format!("{:.0}¢", no_pct)),
                Cell::from(format!("{:.0}", m.volume_24h)),
                edge_cell,
            ])
            .style(row_style)
        })
        .collect();

    let widths = [
        Constraint::Length(9),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(8),
    ];

    let block_title = match spot {
        Some(s) => format!(" {title} ({} strikes)  spot=${s:.0} ", markets.len()),
        None => format!(" {title} ({} strikes) ", markets.len()),
    };

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(block_title));

    f.render_widget(table, area);

    let mut sb_state = ScrollbarState::new(markets.len()).position(app.scroll);
    f.render_stateful_widget(
        Scrollbar::new(ScrollbarOrientation::VerticalRight),
        area,
        &mut sb_state,
    );
}

// ── Predictions tab ───────────────────────────────────────────────────────────

fn draw_predictions(f: &mut Frame, app: &App, area: Rect) {
    if app.predictions.is_empty() {
        let msg = Paragraph::new(Text::from(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No predictions yet.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from("  Press  p  to enter a target BTC price and log a prediction."),
        ]))
        .block(Block::default().borders(Borders::ALL).title(" My Predictions "));
        f.render_widget(msg, area);
        return;
    }

    let header_style = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);

    let header = Row::new(vec![
        Cell::from("#"),
        Cell::from("Time (UTC)"),
        Cell::from("Strike"),
        Cell::from("Side"),
        Cell::from("Entry¢"),
        Cell::from("Now¢"),
        Cell::from("Target"),
        Cell::from("Edge"),
        Cell::from("Result"),
    ])
    .style(header_style)
    .height(1);

    let rows: Vec<Row> = app
        .predictions
        .iter()
        .enumerate()
        .skip(app.scroll)
        .map(|(i, pred)| {
            let (now_yes, edge) = app
                .current_edge(pred)
                .unwrap_or((pred.price_at_entry, 0.0));

            let edge_color = if edge > 0.05 { Color::Green } else if edge < -0.05 { Color::Red } else { Color::White };
            let side_color = match pred.side { PredictionSide::Yes => Color::Green, PredictionSide::No => Color::Red };

            let (outcome_str, outcome_color) = match pred.outcome {
                Outcome::Pending => ("…", Color::DarkGray),
                Outcome::Win => ("WIN", Color::Green),
                Outcome::Loss => ("LOSS", Color::Red),
            };

            let row_style = if i == app.selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(format!("{}", i + 1)),
                Cell::from(pred.created_at.format("%H:%M:%S").to_string()),
                Cell::from(format!("${:.0}", pred.strike)),
                Cell::from(pred.side.to_string()).style(Style::default().fg(side_color)),
                Cell::from(format!("{:.0}¢", pred.price_at_entry * 100.0)),
                Cell::from(format!("{:.0}¢", now_yes * 100.0)),
                Cell::from(format!("${:.0}", pred.target_price)),
                Cell::from(format!("{:+.1}¢", edge * 100.0)).style(Style::default().fg(edge_color)),
                Cell::from(outcome_str).style(Style::default().fg(outcome_color)),
            ])
            .style(row_style)
        })
        .collect();

    let widths = [
        Constraint::Length(3),
        Constraint::Length(10),
        Constraint::Length(8),
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Length(6),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" My Predictions ({})  [j/k] select  [w] WIN  [l] LOSS  [p] new ", app.predictions.len())),
        );
    f.render_widget(table, area);

    let mut sb_state = ScrollbarState::new(app.predictions.len()).position(app.scroll);
    f.render_stateful_widget(Scrollbar::new(ScrollbarOrientation::VerticalRight), area, &mut sb_state);
}

// ── Edge Analysis tab ─────────────────────────────────────────────────────────

fn draw_analysis(f: &mut Frame, app: &App, area: Rect) {
    if let Some(model) = &app.model {
        draw_opportunities(f, model, area);
    } else {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        let btc_text = build_sentiment_text(&app.btc_markets, app.btc_spot);
        let btc_block = Paragraph::new(btc_text)
            .block(Block::default().borders(Borders::ALL).title(" BTC Implied Probability "))
            .wrap(Wrap { trim: false });
        f.render_widget(btc_block, layout[0]);

        let eth_text = build_sentiment_text(&app.eth_markets, None);
        let eth_block = Paragraph::new(eth_text)
            .block(Block::default().borders(Borders::ALL).title(" ETH Implied Probability "))
            .wrap(Wrap { trim: false });
        f.render_widget(eth_block, layout[1]);
    }
}

fn draw_opportunities(f: &mut Frame, model: &ModelOutput, area: Rect) {
    // Find the best |edge| to use for magenta highlight
    let best_edge = model
        .best_opportunity()
        .map(|o| o.strike)
        .unwrap_or(-1.0);

    let header_style = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);

    let header = Row::new(vec![
        Cell::from("Strike"),
        Cell::from("Side"),
        Cell::from("Mkt YES"),
        Cell::from("Model %"),
        Cell::from("Edge"),
        Cell::from("Kelly"),
        Cell::from("Signal"),
    ])
    .style(header_style)
    .height(1);

    let bar_max: usize = 20;

    let rows: Vec<Row> = model.opportunities.iter().map(|o| {
        let edge_pct  = o.edge * 100.0;
        let kelly_pct = o.stake_pct * 100.0;

        let (edge_color, action_color) = if edge_pct > 10.0 {
            (Color::Green, Color::Green)
        } else if edge_pct < -10.0 {
            (Color::Red, Color::Red)
        } else {
            (Color::DarkGray, Color::DarkGray)
        };

        let side_color = if o.side == "YES" { Color::Green } else { Color::Red };

        // Bar proportional to |edge| — max bar at 25% edge
        let bar_len = ((o.edge.abs() / 0.25) * bar_max as f64).min(bar_max as f64) as usize;
        let bar = "█".repeat(bar_len);

        let is_best = (o.strike - best_edge).abs() < 1.0;
        let row_style = if is_best {
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        Row::new(vec![
            Cell::from(format!("${:.0}", o.strike)),
            Cell::from(o.side.as_str()).style(Style::default().fg(side_color)),
            Cell::from(format!("{:.0}%", o.market_yes * 100.0)),
            Cell::from(format!("{:.1}%", o.model_prob * 100.0)),
            Cell::from(format!("{:+.1}%", edge_pct)).style(Style::default().fg(edge_color)),
            Cell::from(format!("{:.1}%", kelly_pct)),
            Cell::from(bar).style(Style::default().fg(action_color)),
        ])
        .style(row_style)
    }).collect();

    let widths = [
        Constraint::Length(9),
        Constraint::Length(5),
        Constraint::Length(8),
        Constraint::Length(9),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Min(20),
    ];

    let timestamp = model.timestamp.get(11..19).unwrap_or("?");
    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    " Classifier Edge Analysis  |  spot ${:.0}  |  ref-pred ${:.0} ± ${:.0}  |  T-{:.0}m  |  as of {timestamp} ",
                    model.current_price, model.model_prediction, model.mae_current, model.mins_to_expiry
                )),
        );
    f.render_widget(table, area);
}

fn build_sentiment_text(markets: &[Market], spot: Option<f64>) -> Text<'static> {
    if markets.is_empty() {
        return Text::from("  No markets loaded yet.");
    }
    let bar_width: usize = 28;
    let mut lines: Vec<Line> = Vec::new();
    for m in markets.iter().take(20) {
        let yes_pct = (m.yes_price * 100.0).round() as u64;
        let filled  = (m.yes_price * bar_width as f64).round() as usize;
        let empty   = bar_width.saturating_sub(filled);
        let bar_color = if yes_pct > 60 { Color::Green } else if yes_pct < 40 { Color::Red } else { Color::Yellow };
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
        let spot_tag = if let Some(s) = spot {
            if (m.strike - s).abs() < 500.0 && s >= m.strike {
                Span::styled(" ◀ SPOT", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            } else { Span::raw("") }
        } else { Span::raw("") };
        lines.push(Line::from(vec![
            Span::raw(format!("  ${:>7.0} ", m.strike)),
            Span::styled(bar, Style::default().fg(bar_color)),
            Span::raw(format!("  {yes_pct:>3}%")),
            spot_tag,
        ]));
    }
    Text::from(lines)
}

// ── Portfolio tab ─────────────────────────────────────────────────────────────

fn draw_portfolio(f: &mut Frame, app: &App, area: Rect) {
    // If auth not configured, show a clear setup message
    if let Some(err) = &app.portfolio_auth_error {
        if app.balance.is_none() && app.positions.is_empty() {
            let msg = Paragraph::new(Text::from(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  Portfolio authentication not configured.",
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    format!("  {err}"),
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(""),
                Line::from("  Steps:"),
                Line::from("    1. Open Kalshi portal → Settings → API Keys"),
                Line::from("    2. Copy your Key ID"),
                Line::from("    3. Edit  ~/.config/kalshi-tui/config.toml"),
                Line::from("    4. Set   key_id = \"your-key-id-here\""),
                Line::from("    5. Restart kalshi-tui"),
            ]))
            .block(Block::default().borders(Borders::ALL).title(" Portfolio "));
            f.render_widget(msg, area);
            return;
        }
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // balance summary
            Constraint::Min(0),     // positions + orders split
        ])
        .split(area);

    // ── Balance row ───────────────────────────────────────────────────────────
    let balance_text = if let Some(bal) = &app.balance {
        let avail = bal.balance as f64 / 100.0;
        let payout = bal.payout.unwrap_or(0) as f64 / 100.0;
        let total = avail + payout;
        Text::from(vec![
            Line::from(""),
            Line::from(vec![
                Span::raw("  Available cash: "),
                Span::styled(
                    format!("${avail:.2}"),
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ),
                Span::raw("    Pending payout: "),
                Span::styled(format!("${payout:.2}"), Style::default().fg(Color::Yellow)),
                Span::raw("    Total: "),
                Span::styled(
                    format!("${total:.2}"),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                format!("  {} open positions  |  {} resting orders", app.positions.len(), app.orders.len()),
                Style::default().fg(Color::DarkGray),
            )),
        ])
    } else {
        Text::from("  Fetching balance…")
    };
    let balance_block = Paragraph::new(balance_text)
        .block(Block::default().borders(Borders::ALL).title(" Account Balance "));
    f.render_widget(balance_block, rows[0]);

    // ── Positions + Orders side by side ───────────────────────────────────────
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[1]);

    draw_positions(f, &app.positions, cols[0]);
    draw_orders(f, &app.orders, cols[1]);
}

fn draw_positions(f: &mut Frame, positions: &[Position], area: Rect) {
    let header = Row::new(vec![
        Cell::from("Ticker"),
        Cell::from("Pos"),
        Cell::from("Cost $"),
        Cell::from("Realized P&L"),
        Cell::from("Exposure $"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    .height(1);

    let rows: Vec<Row> = positions.iter().map(|p| {
        let pnl = p.realized_pnl as f64 / 100.0;
        let pnl_color = if pnl > 0.0 { Color::Green } else if pnl < 0.0 { Color::Red } else { Color::White };
        let cost = p.total_cost as f64 / 100.0;
        let exposure = p.market_exposure as f64 / 100.0;
        let pos_color = if p.position > 0 { Color::Green } else { Color::Red };

        Row::new(vec![
            Cell::from(p.ticker.as_str()),
            Cell::from(format!("{:+}", p.position)).style(Style::default().fg(pos_color)),
            Cell::from(format!("${cost:.2}")),
            Cell::from(format!("{:+.2}", pnl)).style(Style::default().fg(pnl_color)),
            Cell::from(format!("${exposure:.2}")),
        ])
    }).collect();

    let widths = [
        Constraint::Min(20),
        Constraint::Length(5),
        Constraint::Length(8),
        Constraint::Length(13),
        Constraint::Length(11),
    ];

    let empty_msg = if positions.is_empty() { " (none)" } else { "" };
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL)
            .title(format!(" Open Positions{empty_msg} ")));
    f.render_widget(table, area);
}

fn draw_orders(f: &mut Frame, orders: &[Order], area: Rect) {
    let header = Row::new(vec![
        Cell::from("Ticker"),
        Cell::from("Side"),
        Cell::from("Price¢"),
        Cell::from("Qty"),
        Cell::from("Status"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    .height(1);

    let rows: Vec<Row> = orders.iter().map(|o| {
        let price = if o.side == "yes" { o.yes_price } else { o.no_price };
        let side_color = if o.side == "yes" { Color::Green } else { Color::Red };
        let status_color = match o.status.as_str() {
            "resting" => Color::Cyan,
            "filled" => Color::Green,
            "canceled" => Color::DarkGray,
            _ => Color::White,
        };

        // Truncate ticker for display
        let short_ticker = o.ticker.split('-').last().unwrap_or(&o.ticker);

        Row::new(vec![
            Cell::from(short_ticker.to_string()),
            Cell::from(o.side.to_uppercase()).style(Style::default().fg(side_color)),
            Cell::from(format!("{price}¢")),
            Cell::from(format!("{}", o.count)),
            Cell::from(o.status.as_str()).style(Style::default().fg(status_color)),
        ])
    }).collect();

    let widths = [
        Constraint::Min(12),
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Length(5),
        Constraint::Length(8),
    ];

    let empty_msg = if orders.is_empty() { " (none)" } else { "" };
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL)
            .title(format!(" Resting Orders{empty_msg} ")));
    f.render_widget(table, area);
}

// ── Footer ────────────────────────────────────────────────────────────────────

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let keybinds = " [Tab] Switch  [j/k] Scroll  [p] Predict  [w/l] Win/Loss  [r] Refresh  [m] Model  [q] Quit";

    let footer = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);

    let keys = Paragraph::new(keybinds)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Left);
    f.render_widget(keys, footer[0]);

    let status_style = if app.model_loading {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Cyan)
    };
    let status = Paragraph::new(format!(" {}", app.status))
        .style(status_style)
        .alignment(Alignment::Right);
    f.render_widget(status, footer[1]);
}

// ── Input overlay ─────────────────────────────────────────────────────────────

fn draw_input_overlay(f: &mut Frame, app: &App, area: Rect) {
    let InputMode::EnterPrice { buf } = &app.input_mode else { return; };

    let popup = centered_rect(50, 7, area);

    let spot_hint = match app.btc_spot {
        Some(p) => format!("  Current spot: ${p:.2}"),
        None => "  (spot unavailable)".into(),
    };
    let model_hint = match &app.model {
        Some(m) => format!("  Model target: ${:.0} ± ${:.0}", m.model_prediction, m.mae_current),
        None => "  (model loading…)".into(),
    };

    let input_text = Text::from(vec![
        Line::from(""),
        Line::from(Span::raw("  Enter your target BTC price (e.g. 71500):")),
        Line::from(Span::styled(spot_hint, Style::default().fg(Color::DarkGray))),
        Line::from(Span::styled(model_hint, Style::default().fg(Color::DarkGray))),
        Line::from(""),
        Line::from(vec![
            Span::raw("  > "),
            Span::styled(
                format!("{buf}▌"),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
        ]),
    ]);

    let block = Paragraph::new(input_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" New Prediction — Enter / Esc ")
                .style(Style::default().fg(Color::Cyan)),
        )
        .alignment(Alignment::Left);

    f.render_widget(ratatui::widgets::Clear, popup);
    f.render_widget(block, popup);
}

// ── System Status tab ─────────────────────────────────────────────────────────

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    if app.status_loading && app.status_data.is_none() {
        let p = Paragraph::new("  Loading system status…")
            .block(Block::default().borders(Borders::ALL).title(" System Status "))
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(p, area);
        return;
    }

    let data: &StatusData = match &app.status_data {
        Some(d) => d,
        None => {
            let p = Paragraph::new("  Press [s] to load system status.")
                .block(Block::default().borders(Borders::ALL).title(" System Status "));
            f.render_widget(p, area);
            return;
        }
    };

    // ── Vertical split: top panel | performance | runs | trades | log ────────
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),  // top panels (services + stats)
            Constraint::Length(9),  // performance by side + size buckets
            Constraint::Length(10), // recent runs
            Constraint::Length(10), // recent trades
            Constraint::Min(5),     // log tail
        ])
        .split(area);

    // ── Top row: services (left) + trading stats (right) ─────────────────────
    let top_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[0]);

    draw_services_panel(f, data, top_cols[0]);
    draw_trading_stats(f, data, top_cols[1]);

    // Performance row: side-summary (left) + size-bucket leak flag (right)
    let perf_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[1]);
    draw_perf_by_side(f, data, perf_cols[0]);
    draw_perf_by_bucket(f, data, perf_cols[1]);

    draw_runs_table(f, data, rows[2]);
    draw_trades_table(f, data, rows[3]);
    draw_log_tail(f, data, rows[4]);
}

fn draw_perf_by_side(f: &mut Frame, data: &StatusData, area: Rect) {
    if data.perf_by_side.is_empty() {
        let p = Paragraph::new(Line::from(vec![
            Span::styled("  no settled trades yet — ", Style::default().fg(Color::DarkGray)),
            Span::raw("run auto_trader once settlements_raw is populated"),
        ]))
        .block(Block::default().borders(Borders::ALL).title(" Performance by Side "));
        f.render_widget(p, area);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            format!("  {:<5}{:>5} {:>6} {:>9} {:>9} {:>10}",
                "Side", "N", "Win%", "Cost(W)", "Cost(L)", "NetPnL"),
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
        ),
    ]));
    for r in &data.perf_by_side {
        let side_col = if r.side == "YES" { Color::Green } else { Color::Red };
        let pnl_col = if r.net_pnl >= 0.0 { Color::Green } else { Color::Red };
        // Flag: is losing-avg-cost > winning-avg-cost? (the hypothesis signal)
        let skew = if r.avg_cost_loss > r.avg_cost_win && r.avg_cost_win > 0.0 {
            let pct = 100.0 * (r.avg_cost_loss / r.avg_cost_win - 1.0);
            if pct > 10.0 {
                format!(" loss+{:.0}%", pct)
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{:<5}", r.side), Style::default().fg(side_col).add_modifier(Modifier::BOLD)),
            Span::raw(format!("{:>5} ", r.n)),
            Span::raw(format!("{:>5.0}% ", r.win_pct)),
            Span::raw(format!("{:>8} ", format!("${:.2}", r.avg_cost_win))),
            Span::raw(format!("{:>8} ", format!("${:.2}", r.avg_cost_loss))),
            Span::styled(format!("{:>+9.2}", r.net_pnl),
                Style::default().fg(pnl_col).add_modifier(Modifier::BOLD)),
            Span::styled(skew, Style::default().fg(Color::Yellow)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            format!("  settled: {}   unsettled: {}",
                data.settled_trades, data.unsettled_trades),
            Style::default().fg(Color::DarkGray),
        ),
    ]));

    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" Performance by Side "));
    f.render_widget(p, area);
}

fn draw_perf_by_bucket(f: &mut Frame, data: &StatusData, area: Rect) {
    if data.perf_by_bucket.is_empty() {
        let p = Paragraph::new("  ")
            .block(Block::default().borders(Borders::ALL).title(" Size × Side (ranked) "));
        f.render_widget(p, area);
        return;
    }

    // Sort ascending by PnL — worst leaks at top so they catch the eye.
    let mut rows = data.perf_by_bucket.clone();
    rows.sort_by(|a, b| a.net_pnl.partial_cmp(&b.net_pnl).unwrap_or(std::cmp::Ordering::Equal));

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            format!("  {:<18} {:>4} {:>6} {:>10}  {}",
                "Bucket × Side", "N", "Win%", "NetPnL", ""),
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
        ),
    ]));
    let max_rows = (area.height as usize).saturating_sub(3).min(rows.len());
    for r in rows.iter().take(max_rows) {
        let side_col = if r.side == "YES" { Color::Green } else { Color::Red };
        let pnl_col = if r.net_pnl >= 0.0 { Color::Green } else { Color::Red };
        // Flag large leaks: bucket='large' and pnl<-$5
        let flag = if r.bucket == "large" && r.net_pnl < -5.0 {
            " ⚠ LEAK"
        } else if r.bucket == "large" && r.net_pnl < 0.0 {
            " ⚠"
        } else {
            ""
        };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{:<18}", r.label),
                Style::default().fg(side_col)),
            Span::raw(format!(" {:>4} ", r.n)),
            Span::raw(format!(" {:>4.0}% ", r.win_pct)),
            Span::styled(format!("{:>+9.2}", r.net_pnl),
                Style::default().fg(pnl_col).add_modifier(Modifier::BOLD)),
            Span::styled(flag.to_string(), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]));
    }

    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" Size × Side (worst first) "));
    f.render_widget(p, area);
}

fn result_style(result: &str) -> Style {
    match result {
        "success" => Style::default().fg(Color::Green),
        "core-dump" | "killed" | "timeout" | "failed" => Style::default().fg(Color::Red),
        _ => Style::default().fg(Color::Yellow),
    }
}

fn draw_services_panel(f: &mut Frame, data: &StatusData, area: Rect) {
    let timer_dot = if data.timer_active { "●" } else { "○" };
    let timer_col = if data.timer_active { Color::Green } else { Color::Red };
    let timer_next_str = data
        .timer_next
        .map(|t| fmt_until(t))
        .unwrap_or_else(|| "unknown".into());

    let svc_dot = if data.svc_active { "●" } else { "○" };
    let svc_col = if data.svc_result == "success" || data.svc_active {
        Color::Green
    } else if data.svc_result == "core-dump" || data.svc_result == "killed" {
        Color::Red
    } else {
        Color::Yellow
    };

    let obc_dot = if data.orderbook_active { "●" } else { "○" };
    let obc_col = if data.orderbook_active { Color::Green } else { Color::Red };
    let obc_uptime = data
        .orderbook_since
        .map(|t| fmt_uptime(t))
        .unwrap_or_else(|| data.orderbook_since_str.clone());

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(timer_dot, Style::default().fg(timer_col)),
            Span::raw(" trader.timer   "),
            Span::styled(
                if data.timer_active { "WAITING" } else { "INACTIVE" },
                Style::default().fg(timer_col),
            ),
        ]),
        Line::from(Span::styled(
            format!("      next fire: {timer_next_str}"),
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(svc_dot, Style::default().fg(svc_col)),
            Span::raw(" trader.service  "),
            Span::styled(&data.svc_result, result_style(&data.svc_result)),
            Span::styled(
                if data.svc_last_run_str.is_empty() {
                    String::new()
                } else {
                    format!("  @ {}", data.svc_last_run_str)
                },
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(obc_dot, Style::default().fg(obc_col)),
            Span::raw(" orderbook.svc   "),
            Span::styled(
                if data.orderbook_active { "RUNNING" } else { "STOPPED" },
                Style::default().fg(obc_col),
            ),
        ]),
        Line::from(Span::styled(
            format!("      uptime:     {obc_uptime}"),
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" Services "))
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn draw_trading_stats(f: &mut Frame, data: &StatusData, area: Rect) {
    let mode_label = if data.dry_run { "DRY-RUN" } else { "LIVE" };
    let mode_col = if data.dry_run { Color::Yellow } else { Color::Green };

    let balance_str = data
        .balance
        .map(|b| format!("${b:.2}"))
        .unwrap_or_else(|| "—".into());
    let hwm_str = data
        .hwm
        .map(|h| format!("${h:.2}"))
        .unwrap_or_else(|| "—".into());
    let dd_str = data
        .drawdown_pct
        .map(|d| format!("{d:.1}%"))
        .unwrap_or_else(|| "—".into());

    let deployed_str = data
        .daily_deployed
        .map(|d| format!("${d:.2}"))
        .unwrap_or_else(|| "—".into());
    let budget_str = data
        .daily_budget
        .map(|b| format!("${b:.2}"))
        .unwrap_or_else(|| "—".into());

    let remaining = match (data.daily_budget, data.daily_deployed) {
        (Some(b), Some(d)) => Some(b - d),
        _ => None,
    };
    let remaining_str = remaining
        .map(|r| format!("${r:.2}"))
        .unwrap_or_else(|| "—".into());
    let remaining_col = match remaining {
        Some(r) if r < 5.0 => Color::Red,
        Some(r) if r < 20.0 => Color::Yellow,
        _ => Color::Green,
    };

    let refreshed = data
        .refreshed_at
        .format("%H:%M:%S UTC")
        .to_string();

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Mode:      "),
            Span::styled(mode_label, Style::default().fg(mode_col).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("  Balance:   "),
            Span::styled(&balance_str, Style::default().fg(Color::Cyan)),
            Span::raw("   HWM: "),
            Span::styled(&hwm_str, Style::default().fg(Color::DarkGray)),
            Span::raw("   Drawdown: "),
            Span::styled(&dd_str, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("  Deployed:  "),
            Span::styled(&deployed_str, Style::default().fg(Color::Yellow)),
            Span::raw(" / "),
            Span::styled(&budget_str, Style::default().fg(Color::DarkGray)),
            Span::raw("  (daily budget)"),
        ]),
        Line::from(vec![
            Span::raw("  Remaining: "),
            Span::styled(&remaining_str, Style::default().fg(remaining_col).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  Refreshed: {refreshed}   [s] to refresh"),
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" Trading "))
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn draw_runs_table(f: &mut Frame, data: &StatusData, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Time").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("BTC Price").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Prediction").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Trades").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Deployed").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Mode").style(Style::default().add_modifier(Modifier::BOLD)),
    ])
    .style(Style::default().fg(Color::DarkGray))
    .height(1);

    let rows: Vec<Row> = data.recent_runs.iter().map(|r| {
        // run_at is ISO8601 UTC — extract HH:MM
        let time = r.run_at.get(11..16).unwrap_or(&r.run_at).to_string();
        let price = r.current_price.map(|p| format!("${p:>9.0}")).unwrap_or_else(|| "—".into());
        let pred  = r.model_pred.map(|p| format!("${p:>9.0}")).unwrap_or_else(|| "—".into());
        let trades = r.trades_placed.map(|t| t.to_string()).unwrap_or_else(|| "—".into());
        let staked = r.total_staked.map(|s| format!("${s:.2}")).unwrap_or_else(|| "—".into());
        let mode = if r.dry_run { "DRY" } else { "LIVE" };
        let mode_col = if r.dry_run { Color::Yellow } else { Color::Green };

        Row::new(vec![
            Cell::from(time),
            Cell::from(price).style(Style::default().fg(Color::Cyan)),
            Cell::from(pred).style(Style::default().fg(Color::DarkGray)),
            Cell::from(trades),
            Cell::from(staked).style(Style::default().fg(Color::Yellow)),
            Cell::from(mode).style(Style::default().fg(mode_col)),
        ])
    }).collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Length(11),
            Constraint::Length(11),
            Constraint::Length(7),
            Constraint::Length(10),
            Constraint::Length(5),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" Recent Runs (last 8) "))
    .column_spacing(1);

    f.render_widget(table, area);
}

fn draw_trades_table(f: &mut Frame, data: &StatusData, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Time").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Strike").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Side").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Qty").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Price").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Cost").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Edge").style(Style::default().add_modifier(Modifier::BOLD)),
        Cell::from("Status").style(Style::default().add_modifier(Modifier::BOLD)),
    ])
    .style(Style::default().fg(Color::DarkGray))
    .height(1);

    let rows: Vec<Row> = data.recent_trades.iter().map(|t| {
        let time   = t.placed_at.get(11..16).unwrap_or(&t.placed_at).to_string();
        let strike = format!("${:.0}", t.strike);
        let price  = format!("{}¢", t.price_cents);
        let cost   = format!("${:.2}", t.cost_dollars);
        let edge   = t.edge.map(|e| format!("{:+.1}%", e * 100.0)).unwrap_or_else(|| "—".into());
        let edge_col = match t.edge {
            Some(e) if e > 0.05 => Color::Green,
            Some(e) if e < 0.0  => Color::Red,
            _ => Color::Yellow,
        };
        let status_col = if t.status == "PLACED" || t.status == "ok" {
            Color::Green
        } else {
            Color::Red
        };
        let side_col = if t.side == "YES" { Color::Green } else { Color::Red };

        Row::new(vec![
            Cell::from(time),
            Cell::from(strike).style(Style::default().fg(Color::Cyan)),
            Cell::from(t.side.clone()).style(Style::default().fg(side_col)),
            Cell::from(t.count.to_string()),
            Cell::from(price),
            Cell::from(cost).style(Style::default().fg(Color::Yellow)),
            Cell::from(edge).style(Style::default().fg(edge_col)),
            Cell::from(t.status.clone()).style(Style::default().fg(status_col)),
        ])
    }).collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Length(9),
            Constraint::Length(5),
            Constraint::Length(4),
            Constraint::Length(6),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" Recent Trades (last 12) "))
    .column_spacing(1);

    f.render_widget(table, area);
}

fn draw_log_tail(f: &mut Frame, data: &StatusData, area: Rect) {
    let lines: Vec<Line> = data.log_lines.iter().map(|l| {
        let col = if l.starts_with("[OK]") || l.contains("PLACED") {
            Color::Green
        } else if l.starts_with("[STOP]") || l.starts_with("[SKIP]") {
            Color::Yellow
        } else if l.starts_with("[ERR]") || l.starts_with("[FAIL]") || l.contains("Error") {
            Color::Red
        } else if l.starts_with("===") || l.starts_with("────") {
            Color::DarkGray
        } else {
            Color::Reset
        };
        Line::from(Span::styled(format!("  {l}"), Style::default().fg(col)))
    }).collect();

    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" auto_trader.log (tail) "))
        .wrap(Wrap { trim: true });
    f.render_widget(p, area);
}

// ── Auto Trades tab ───────────────────────────────────────────────────────────

fn draw_trades(f: &mut Frame, app: &App, area: Rect) {
    if app.auto_trades.is_empty() {
        let msg = Paragraph::new(Text::from(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No auto-trader data found.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from("  auto_trader.db not found or contains no placed trades."),
            Line::from("  Run the auto-trader at least once to populate this view."),
        ]))
        .block(Block::default().borders(Borders::ALL).title(" Auto Trades "));
        f.render_widget(msg, area);
        return;
    }

    let s = &app.trade_summary;
    let settled = s.wins + s.losses;
    let win_rate = if settled > 0 {
        s.wins as f64 / settled as f64 * 100.0
    } else {
        0.0
    };
    let pnl_color = if s.net_pnl >= 0.0 { Color::Green } else { Color::Red };

    let mode_label = match (s.live_count, s.dry_count) {
        (l, 0) if l > 0 => "LIVE",
        (0, d) if d > 0 => "DRY",
        _               => "MIXED",
    };
    let mode_color = match mode_label {
        "LIVE"  => Color::Green,
        "DRY"   => Color::Yellow,
        _       => Color::Cyan,
    };

    let summary_lines = vec![
        Line::from(vec![
            Span::raw("  Mode: "),
            Span::styled(mode_label, Style::default().fg(mode_color).add_modifier(Modifier::BOLD)),
            Span::raw(format!(
                "   Total: {}  (LIVE {} / DRY {})   Win: {}/{} ({:.1}%)   Open: {}",
                s.total, s.live_count, s.dry_count, s.wins, settled, win_rate, s.open_count
            )),
        ]),
        Line::from(vec![
            Span::raw("  Net P&L: "),
            Span::styled(
                format!("{:+.2}", s.net_pnl),
                Style::default().fg(pnl_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("   Deployed: ${:.2}", s.total_deployed)),
            Span::raw("   [r] reload  [j/k] scroll"),
        ]),
    ];

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(0)])
        .split(area);

    let summary = Paragraph::new(summary_lines)
        .block(Block::default().borders(Borders::ALL).title(" Auto Trader Summary "));
    f.render_widget(summary, layout[0]);

    let header = Row::new(vec![
        Cell::from("ID"),
        Cell::from("Time (UTC)"),
        Cell::from("Strike"),
        Cell::from("Side"),
        Cell::from("Qty"),
        Cell::from("¢"),
        Cell::from("Cost$"),
        Cell::from("Edge%"),
        Cell::from("Outcome"),
        Cell::from("Mode"),
    ])
    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    .height(1);

    let rows: Vec<Row> = app
        .auto_trades
        .iter()
        .enumerate()
        .skip(app.scroll)
        .map(|(i, t)| trade_row(t, i == app.selected))
        .collect();

    let widths = [
        Constraint::Length(5),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Length(5),
        Constraint::Length(4),
        Constraint::Length(4),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(8),
        Constraint::Length(5),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(format!(
            " Trades ({} shown of {}) ",
            app.auto_trades.len().min(500),
            s.total
        )));
    f.render_widget(table, layout[1]);

    let mut sb = ScrollbarState::new(app.auto_trades.len()).position(app.scroll);
    f.render_stateful_widget(Scrollbar::new(ScrollbarOrientation::VerticalRight), layout[1], &mut sb);
}

fn trade_row(t: &AutoTradeRecord, selected: bool) -> Row<'static> {
    let side_color = if t.side == "YES" { Color::Green } else { Color::Red };
    let (outcome_str, outcome_color) = match t.outcome {
        TradeOutcome::Win  => ("WIN",  Color::Green),
        TradeOutcome::Loss => ("LOSS", Color::Red),
        TradeOutcome::Open => ("OPEN", Color::DarkGray),
    };
    let edge_str = t.edge
        .map(|e| format!("{:+.1}%", e * 100.0))
        .unwrap_or_else(|| "—".into());
    let edge_color = t.edge.map(|e| {
        if e > 0.05 { Color::Green } else if e < -0.05 { Color::Red } else { Color::White }
    }).unwrap_or(Color::DarkGray);
    let mode_str = if t.dry_run { "DRY" } else { "LIVE" };
    let mode_color = if t.dry_run { Color::Yellow } else { Color::Cyan };
    let time_str = t.placed_at.get(11..19).unwrap_or("?").to_string();

    let row_style = if selected {
        Style::default().add_modifier(Modifier::REVERSED)
    } else {
        Style::default()
    };

    Row::new(vec![
        Cell::from(t.id.to_string()),
        Cell::from(time_str),
        Cell::from(format!("${:.0}", t.strike)),
        Cell::from(t.side.clone()).style(Style::default().fg(side_color)),
        Cell::from(t.count.to_string()),
        Cell::from(t.price_cents.to_string()),
        Cell::from(format!("${:.2}", t.cost_dollars)),
        Cell::from(edge_str).style(Style::default().fg(edge_color)),
        Cell::from(outcome_str).style(Style::default().fg(outcome_color)),
        Cell::from(mode_str).style(Style::default().fg(mode_color)),
    ])
    .style(row_style)
}

fn centered_rect(percent_x: u16, height: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100u16.saturating_sub(height * 4)) / 2),
            Constraint::Length(height),
            Constraint::Percentage((100u16.saturating_sub(height * 4)) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// ── Market-close countdown helpers ───────────────────────────────────────────

/// Seconds until next 5 PM EDT (21:00 UTC) market close, computed live each frame.
fn compute_secs_to_market_close() -> i64 {
    use chrono::{NaiveTime, Utc};
    let now = Utc::now();
    let close_utc = NaiveTime::from_hms_opt(21, 0, 0).unwrap(); // 5 PM EDT = UTC-4 → 21:00 UTC
    let today_close = now.date_naive().and_time(close_utc).and_utc();
    let target = if now < today_close {
        today_close
    } else {
        now.date_naive().succ_opt().unwrap().and_time(close_utc).and_utc()
    };
    (target - now).num_seconds().max(0)
}

fn fmt_market_countdown(secs: i64) -> (String, Color) {
    if secs <= 0 {
        return ("  MARKET CLOSED".into(), Color::DarkGray);
    }
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    let label = if h > 0 {
        format!("  T- {}h {:02}m {:02}s to close", h, m, s)
    } else {
        format!("  T- {}m {:02}s to close", m, s)
    };
    let color = if secs < 600 {
        Color::Red
    } else if secs < 1800 {
        Color::Yellow
    } else {
        Color::Cyan
    };
    (label, color)
}

// ── P&L Chart tab ─────────────────────────────────────────────────────────────

fn draw_pnl_chart(f: &mut Frame, app: &App, area: Rect) {
    let s = &app.trade_summary;
    let settled = s.wins + s.losses;

    // Summary header
    let win_rate = if settled > 0 { s.wins as f64 / settled as f64 * 100.0 } else { 0.0 };
    let pnl_color = if s.net_pnl >= 0.0 { Color::Green } else { Color::Red };
    let mode_label = match (s.live_count, s.dry_count) {
        (l, 0) if l > 0 => "LIVE",
        (0, d) if d > 0 => "DRY",
        _               => "MIXED",
    };
    let mode_color = match mode_label {
        "LIVE" => Color::Green,
        "DRY"  => Color::Yellow,
        _      => Color::Cyan,
    };
    let (cdown_str, cdown_color) = fmt_market_countdown(compute_secs_to_market_close());
    let summary_lines = vec![
        Line::from(vec![
            Span::raw("  Mode: "),
            Span::styled(mode_label, Style::default().fg(mode_color).add_modifier(Modifier::BOLD)),
            Span::raw(format!(
                "   Total: {}  (LIVE {} / DRY {})   Settled: {}/{}   Win Rate: {:.1}%   Open: {}",
                s.total, s.live_count, s.dry_count, settled, s.total, win_rate, s.open_count
            )),
        ]),
        Line::from(vec![
            Span::raw("  Net P&L: "),
            Span::styled(
                format!("{:+.2}", s.net_pnl),
                Style::default().fg(pnl_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("   Deployed: ${:.2}   ", s.total_deployed)),
            Span::styled(cdown_str, Style::default().fg(cdown_color)),
        ]),
    ];

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(0)])
        .split(area);

    let summary = Paragraph::new(summary_lines)
        .block(Block::default().borders(Borders::ALL).title(" P&L Chart — Summary "));
    f.render_widget(summary, layout[0]);

    // Build cumulative P&L series (trades stored DESC; iterate reversed for chronological order)
    let settled_trades: Vec<&AutoTradeRecord> = app.auto_trades.iter()
        .rev()
        .filter(|t| t.pnl.is_some())
        .collect();

    if settled_trades.is_empty() {
        let msg = Paragraph::new(Text::from(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No settled trades to chart yet.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from("  Trades appear here once outcomes are resolved."),
        ]))
        .block(Block::default().borders(Borders::ALL).title(" Cumulative P&L "));
        f.render_widget(msg, layout[1]);
        return;
    }

    let mut all_pts: Vec<(f64, f64)> = Vec::new();
    let mut live_pts: Vec<(f64, f64)> = Vec::new();
    let mut dry_pts: Vec<(f64, f64)> = Vec::new();
    let mut cum_all = 0.0f64;
    let mut cum_live = 0.0f64;
    let mut cum_dry = 0.0f64;

    for (i, t) in settled_trades.iter().enumerate() {
        let pnl = t.pnl.unwrap_or(0.0);
        cum_all += pnl;
        all_pts.push((i as f64, cum_all));
        if t.dry_run {
            cum_dry += pnl;
            dry_pts.push((i as f64, cum_dry));
        } else {
            cum_live += pnl;
            live_pts.push((i as f64, cum_live));
        }
    }

    let n = settled_trades.len() as f64;
    let all_y_vals: Vec<f64> = all_pts.iter().map(|p| p.1)
        .chain(live_pts.iter().map(|p| p.1))
        .chain(dry_pts.iter().map(|p| p.1))
        .collect();
    let y_min = all_y_vals.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0) - 0.5;
    let y_max = all_y_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0) + 0.5;

    let y_min_label = format!("{:.2}", y_min);
    let y_zero_label = " 0.00".to_string();
    let y_max_label = format!("{:.2}", y_max);
    let x_mid_label = format!("{}", (n / 2.0) as usize);
    let x_max_label = format!("{}", settled_trades.len());

    let mut datasets = vec![
        Dataset::default()
            .name("All")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&all_pts),
    ];
    if !live_pts.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Live")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&live_pts),
        );
    }
    if !dry_pts.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Dry")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow))
                .data(&dry_pts),
        );
    }

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(" Cumulative P&L ($) — cyan=All  green=Live  yellow=Dry "))
        .x_axis(
            Axis::default()
                .title("Trades (settled)")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, n])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(x_mid_label),
                    Span::raw(x_max_label),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("P&L ($)")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::styled(y_min_label, Style::default().fg(Color::Red)),
                    Span::raw(y_zero_label),
                    Span::styled(y_max_label, Style::default().fg(Color::Green)),
                ]),
        );
    f.render_widget(chart, layout[1]);
}

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame, Terminal,
};
use serde::Deserialize;
use std::{io::Stdout, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    sync::mpsc,
};
use std::process::Stdio;

const VENV_PYTHON: &str = "/home/jeb/programs/gemini_trader/venv/bin/python";
const PAPER_TRADER: &str =
    "/home/jeb/Documents/claude_creations/kalshi_master_workspace/scripts/paper_trader.py";

// ── State machine ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum WizardStep {
    Welcome,
    Balance,
    Cycles,
    MaxTrades,
    Confirm,
    Running,
    Results,
    Errored,
}

#[derive(Debug, Deserialize)]
struct SessionResult {
    initial_balance: f64,
    final_balance: f64,
    pnl: f64,
    pct_return: f64,
    total_trades: u32,
    wins: u32,
    losses: u32,
    pending: u32,
    session_doc: String,
}

struct WizardState {
    step: WizardStep,
    // input buffers (raw user text)
    balance_buf: String,
    cycles_buf: String,
    max_trades_buf: String,
    // validation error shown below the current input
    field_error: String,
    // confirmed config values
    balance: f64,
    max_cycles: u32,
    max_trades: u32,
    // running state
    log_lines: Vec<String>,
    proc_done: bool,
    error_msg: String,
    // results
    result: Option<SessionResult>,
    // scroll offset for the log view
    log_scroll: usize,
}

impl WizardState {
    fn new() -> Self {
        Self {
            step: WizardStep::Welcome,
            balance_buf: String::new(),
            cycles_buf: String::new(),
            max_trades_buf: String::new(),
            field_error: String::new(),
            balance: 100.0,
            max_cycles: 0,
            max_trades: 0,
            log_lines: Vec::new(),
            proc_done: false,
            error_msg: String::new(),
            result: None,
            log_scroll: 0,
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run_wizard(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
) -> Result<()> {
    let (line_tx, mut line_rx) = mpsc::channel::<Option<String>>(512);
    let mut state = WizardState::new();

    loop {
        terminal.draw(|f| draw_wizard(f, &state))?;

        // Drain subprocess lines
        loop {
            match line_rx.try_recv() {
                Ok(Some(line)) => ingest_line(&mut state, line),
                Ok(None) => {
                    // Subprocess stdout closed
                    state.proc_done = true;
                    if state.step == WizardStep::Running && state.result.is_none() {
                        state.error_msg =
                            "Process exited without producing a result. Check the log.".into();
                        state.step = WizardStep::Errored;
                    }
                }
                Err(_) => break,
            }
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match (key.modifiers, key.code) {
                    // Universal quit / back
                    (KeyModifiers::CONTROL, KeyCode::Char('c')) => return Ok(()),

                    (_, KeyCode::Char('q')) if matches!(
                        state.step,
                        WizardStep::Welcome | WizardStep::Results | WizardStep::Errored
                    ) => return Ok(()),

                    // --- Welcome ---
                    (_, KeyCode::Enter) if state.step == WizardStep::Welcome => {
                        state.step = WizardStep::Balance;
                        state.field_error.clear();
                    }

                    // --- Balance input ---
                    (_, KeyCode::Enter) if state.step == WizardStep::Balance => {
                        match state.balance_buf.trim().parse::<f64>() {
                            Ok(v) if v >= 1.0 => {
                                state.balance = v;
                                state.field_error.clear();
                                state.step = WizardStep::Cycles;
                            }
                            Ok(_) => {
                                state.field_error = "Minimum balance is $1.00".into();
                            }
                            Err(_) => {
                                state.field_error = "Enter a number e.g. 100".into();
                            }
                        }
                    }
                    (_, KeyCode::Esc) if state.step == WizardStep::Balance => {
                        state.step = WizardStep::Welcome;
                        state.field_error.clear();
                    }
                    (_, KeyCode::Char(c)) if state.step == WizardStep::Balance
                        && (c.is_ascii_digit() || c == '.') =>
                    {
                        state.balance_buf.push(c);
                        state.field_error.clear();
                    }
                    (_, KeyCode::Backspace) if state.step == WizardStep::Balance => {
                        state.balance_buf.pop();
                    }

                    // --- Cycles input ---
                    (_, KeyCode::Enter) if state.step == WizardStep::Cycles => {
                        let s = state.cycles_buf.trim().to_string();
                        if s.is_empty() {
                            state.max_cycles = 0;
                            state.field_error.clear();
                            state.step = WizardStep::MaxTrades;
                        } else {
                            match s.parse::<u32>() {
                                Ok(v) => {
                                    state.max_cycles = v;
                                    state.field_error.clear();
                                    state.step = WizardStep::MaxTrades;
                                }
                                Err(_) => {
                                    state.field_error = "Enter a whole number or 0".into();
                                }
                            }
                        }
                    }
                    (_, KeyCode::Esc) if state.step == WizardStep::Cycles => {
                        state.step = WizardStep::Balance;
                        state.field_error.clear();
                    }
                    (_, KeyCode::Char(c))
                        if state.step == WizardStep::Cycles && c.is_ascii_digit() =>
                    {
                        state.cycles_buf.push(c);
                        state.field_error.clear();
                    }
                    (_, KeyCode::Backspace) if state.step == WizardStep::Cycles => {
                        state.cycles_buf.pop();
                    }

                    // --- MaxTrades input ---
                    (_, KeyCode::Enter) if state.step == WizardStep::MaxTrades => {
                        let s = state.max_trades_buf.trim().to_string();
                        if s.is_empty() {
                            state.max_trades = 0;
                            state.field_error.clear();
                            state.step = WizardStep::Confirm;
                        } else {
                            match s.parse::<u32>() {
                                Ok(v) => {
                                    state.max_trades = v;
                                    state.field_error.clear();
                                    state.step = WizardStep::Confirm;
                                }
                                Err(_) => {
                                    state.field_error = "Enter a whole number or 0".into();
                                }
                            }
                        }
                    }
                    (_, KeyCode::Esc) if state.step == WizardStep::MaxTrades => {
                        state.step = WizardStep::Cycles;
                        state.field_error.clear();
                    }
                    (_, KeyCode::Char(c))
                        if state.step == WizardStep::MaxTrades && c.is_ascii_digit() =>
                    {
                        state.max_trades_buf.push(c);
                        state.field_error.clear();
                    }
                    (_, KeyCode::Backspace) if state.step == WizardStep::MaxTrades => {
                        state.max_trades_buf.pop();
                    }

                    // --- Confirm ---
                    (_, KeyCode::Enter) if state.step == WizardStep::Confirm => {
                        state.step = WizardStep::Running;
                        state.log_lines.clear();
                        state.proc_done = false;

                        let python = std::env::var("KALSHI_TUI_PYTHON_BIN")
                            .unwrap_or_else(|_| VENV_PYTHON.into());
                        let script = std::env::var("KALSHI_TUI_PAPER_TRADER")
                            .unwrap_or_else(|_| PAPER_TRADER.into());

                        let balance_arg    = state.balance.to_string();
                        let cycles_arg     = state.max_cycles.to_string();
                        let max_trades_arg = state.max_trades.to_string();

                        match Command::new(&python)
                            .arg(&script)
                            .arg("--balance").arg(&balance_arg)
                            .arg("--max-cycles").arg(&cycles_arg)
                            .arg("--max-trades").arg(&max_trades_arg)
                            .stdout(Stdio::piped())
                            .stderr(Stdio::piped())
                            .spawn()
                        {
                            Ok(mut child) => {
                                let tx = line_tx.clone();
                                if let Some(stdout) = child.stdout.take() {
                                    tokio::spawn(async move {
                                        let reader = BufReader::new(stdout);
                                        let mut lines = reader.lines();
                                        while let Ok(Some(line)) = lines.next_line().await {
                                            if tx.send(Some(line)).await.is_err() {
                                                break;
                                            }
                                        }
                                        let _ = tx.send(None).await;
                                    });
                                }
                                // Merge stderr into log as [ERR] lines
                                let tx_err = line_tx.clone();
                                if let Some(stderr) = child.stderr.take() {
                                    tokio::spawn(async move {
                                        let reader = BufReader::new(stderr);
                                        let mut lines = reader.lines();
                                        while let Ok(Some(line)) = lines.next_line().await {
                                            if !line.trim().is_empty() {
                                                let tagged = format!("[ERR] {}", line);
                                                if tx_err.send(Some(tagged)).await.is_err() {
                                                    break;
                                                }
                                            }
                                        }
                                    });
                                }
                            }
                            Err(e) => {
                                state.error_msg = format!("Failed to launch paper_trader.py: {e}");
                                state.step = WizardStep::Errored;
                            }
                        }
                    }
                    (_, KeyCode::Esc) if state.step == WizardStep::Confirm => {
                        state.step = WizardStep::MaxTrades;
                    }

                    // --- Running (scroll log + interrupt) ---
                    (_, KeyCode::Char('q')) if state.step == WizardStep::Running => {
                        // Can't easily kill the process here; just leave and let it run
                        return Ok(());
                    }
                    (_, KeyCode::Down) | (_, KeyCode::Char('j'))
                        if state.step == WizardStep::Running =>
                    {
                        state.log_scroll = state.log_scroll.saturating_add(1);
                    }
                    (_, KeyCode::Up) | (_, KeyCode::Char('k'))
                        if state.step == WizardStep::Running =>
                    {
                        state.log_scroll = state.log_scroll.saturating_sub(1);
                    }

                    // --- Results / Error ---
                    (_, KeyCode::Enter) if state.step == WizardStep::Results => {
                        return Ok(());
                    }
                    (_, KeyCode::Enter) if state.step == WizardStep::Errored => {
                        // Go back to confirm to retry
                        state.step = WizardStep::Confirm;
                        state.error_msg.clear();
                    }

                    _ => {}
                }
            }
        }
    }
}

// ── Line ingestion ────────────────────────────────────────────────────────────

fn ingest_line(state: &mut WizardState, line: String) {
    if let Some(json_str) = line.strip_prefix("JSON:") {
        if let Ok(result) = serde_json::from_str::<SessionResult>(json_str.trim()) {
            state.result = Some(result);
            state.step   = WizardStep::Results;
            return;
        }
    }
    state.log_lines.push(line);
    // Auto-scroll to bottom as new lines arrive
    let visible = 20usize;
    if state.log_lines.len() > visible {
        state.log_scroll = state.log_lines.len() - visible;
    }
    // Cap memory
    if state.log_lines.len() > 500 {
        state.log_lines.remove(0);
        state.log_scroll = state.log_scroll.saturating_sub(1);
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

fn draw_wizard(f: &mut Frame, state: &WizardState) {
    let area = f.area();
    // Dim background
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Rgb(10, 10, 20))),
        area,
    );

    let modal = centered_modal(60, 75, area);

    match &state.step {
        WizardStep::Welcome      => draw_welcome(f, modal),
        WizardStep::Balance      => draw_balance(f, modal, state),
        WizardStep::Cycles       => draw_cycles(f, modal, state),
        WizardStep::MaxTrades    => draw_max_trades(f, modal, state),
        WizardStep::Confirm      => draw_confirm(f, modal, state),
        WizardStep::Running      => draw_running(f, modal, state),
        WizardStep::Results      => draw_results(f, modal, state),
        WizardStep::Errored      => draw_error(f, modal, state),
    }
}

fn modal_block(title: &str) -> Block<'_> {
    Block::default()
        .title(Span::styled(
            format!(" {title} "),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(Color::Rgb(15, 15, 30)))
}

fn hint(text: &str) -> Paragraph<'_> {
    Paragraph::new(text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
}

// ── Welcome screen ────────────────────────────────────────────────────────────

fn draw_welcome(f: &mut Frame, area: Rect) {
    f.render_widget(Clear, area);
    f.render_widget(modal_block("KALSHI PAPER TRADING WIZARD"), area);

    let inner = inner_area(area, 2, 1);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new("Trade with mock dollars. No real orders placed.")
            .style(Style::default().fg(Color::White))
            .alignment(Alignment::Center),
        chunks[1],
    );
    f.render_widget(
        Paragraph::new("Live Kalshi market data  ·  GPU classifier inference  ·  Full settlement report")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center),
        chunks[2],
    );
    f.render_widget(hint("[Enter] Start    [q] Quit"), chunks[4]);
}

// ── Balance screen ────────────────────────────────────────────────────────────

fn draw_balance(f: &mut Frame, area: Rect, state: &WizardState) {
    f.render_widget(Clear, area);
    f.render_widget(modal_block("Step 1/3 — Starting Balance"), area);

    let inner = inner_area(area, 3, 1);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // prompt
            Constraint::Length(1),
            Constraint::Length(3), // input box
            Constraint::Length(1),
            Constraint::Length(1), // error
            Constraint::Min(0),
            Constraint::Length(1), // hint
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new("How much mock money to trade with?")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center),
        chunks[0],
    );

    let display = if state.balance_buf.is_empty() {
        "100.00".to_string()
    } else {
        state.balance_buf.clone()
    };
    let cursor_marker = if state.balance_buf.is_empty() { "" } else { "_" };
    f.render_widget(
        Paragraph::new(format!("  $ {}{}", display, cursor_marker))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow)),
            )
            .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        chunks[2],
    );

    if !state.field_error.is_empty() {
        f.render_widget(
            Paragraph::new(state.field_error.as_str())
                .style(Style::default().fg(Color::Red))
                .alignment(Alignment::Center),
            chunks[4],
        );
    }

    f.render_widget(hint("[Enter] Next    [Esc] Back"), chunks[6]);
}

// ── Cycles screen ─────────────────────────────────────────────────────────────

fn draw_cycles(f: &mut Frame, area: Rect, state: &WizardState) {
    f.render_widget(Clear, area);
    f.render_widget(modal_block("Step 2/3 — Predict Cycles"), area);

    let inner = inner_area(area, 3, 1);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new("How many GPU inference cycles?")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center),
        chunks[0],
    );
    f.render_widget(
        Paragraph::new("Each cycle is one classifier run (≈ one 10-min timer fire).")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        chunks[1],
    );
    f.render_widget(
        Paragraph::new("Enter 0 or leave blank to run until trades cap or manual stop.")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        chunks[2],
    );

    let display = if state.cycles_buf.is_empty() {
        "0 (unlimited)".to_string()
    } else {
        state.cycles_buf.clone()
    };
    let cursor_marker = if state.cycles_buf.is_empty() { "" } else { "_" };
    f.render_widget(
        Paragraph::new(format!("  Cycles: {}{}", display, cursor_marker))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow)),
            )
            .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        chunks[4],
    );

    if !state.field_error.is_empty() {
        f.render_widget(
            Paragraph::new(state.field_error.as_str())
                .style(Style::default().fg(Color::Red))
                .alignment(Alignment::Center),
            chunks[6],
        );
    }
    f.render_widget(hint("[Enter] Next    [Esc] Back"), chunks[8]);
}

// ── MaxTrades screen ──────────────────────────────────────────────────────────

fn draw_max_trades(f: &mut Frame, area: Rect, state: &WizardState) {
    f.render_widget(Clear, area);
    f.render_widget(modal_block("Step 3/3 — Max Trades"), area);

    let inner = inner_area(area, 3, 1);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new("Stop after how many paper trades total?")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center),
        chunks[0],
    );
    f.render_widget(
        Paragraph::new("Useful to limit how much mock capital is deployed per session.")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        chunks[1],
    );
    f.render_widget(
        Paragraph::new("Enter 0 or leave blank for no cap.")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        chunks[2],
    );

    let display = if state.max_trades_buf.is_empty() {
        "0 (unlimited)".to_string()
    } else {
        state.max_trades_buf.clone()
    };
    let cursor_marker = if state.max_trades_buf.is_empty() { "" } else { "_" };
    f.render_widget(
        Paragraph::new(format!("  Max trades: {}{}", display, cursor_marker))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow)),
            )
            .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        chunks[3],
    );

    if !state.field_error.is_empty() {
        f.render_widget(
            Paragraph::new(state.field_error.as_str())
                .style(Style::default().fg(Color::Red))
                .alignment(Alignment::Center),
            chunks[5],
        );
    }
    f.render_widget(hint("[Enter] Next    [Esc] Back"), chunks[7]);
}

// ── Confirm screen ────────────────────────────────────────────────────────────

fn draw_confirm(f: &mut Frame, area: Rect, state: &WizardState) {
    f.render_widget(Clear, area);
    f.render_widget(modal_block("Ready to Launch"), area);

    let inner = inner_area(area, 3, 1);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    let cycles_str = if state.max_cycles == 0 {
        "unlimited".to_string()
    } else {
        state.max_cycles.to_string()
    };
    let trades_str = if state.max_trades == 0 {
        "unlimited".to_string()
    } else {
        state.max_trades.to_string()
    };

    let kv = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);
    let vv = Style::default().fg(Color::Cyan);

    f.render_widget(
        Paragraph::new(vec![Line::from(vec![
            Span::styled("  Starting balance:   ", kv),
            Span::styled(format!("${:.2}", state.balance), vv),
        ])]),
        chunks[0],
    );
    f.render_widget(
        Paragraph::new(vec![Line::from(vec![
            Span::styled("  Predict cycles:     ", kv),
            Span::styled(cycles_str, vv),
        ])]),
        chunks[1],
    );
    f.render_widget(
        Paragraph::new(vec![Line::from(vec![
            Span::styled("  Max trades:         ", kv),
            Span::styled(trades_str, vv),
        ])]),
        chunks[2],
    );

    f.render_widget(
        Paragraph::new("  Cycles run back-to-back using live market data.")
            .style(Style::default().fg(Color::DarkGray)),
        chunks[4],
    );
    f.render_widget(
        Paragraph::new("  No orders will be placed on Kalshi.")
            .style(Style::default().fg(Color::DarkGray)),
        chunks[5],
    );

    f.render_widget(hint("[Enter] Launch    [Esc] Back"), chunks[8]);
}

// ── Running screen ────────────────────────────────────────────────────────────

fn draw_running(f: &mut Frame, area: Rect, state: &WizardState) {
    f.render_widget(Clear, area);

    let title = if state.proc_done {
        "Paper Trading — Finishing…".to_string()
    } else {
        "Paper Trading — Running".to_string()
    };
    f.render_widget(modal_block(&title), area);

    let inner = inner_area(area, 2, 1);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    // Build visible log slice
    let visible_h = chunks[0].height as usize;
    let total     = state.log_lines.len();
    let start     = state.log_scroll.min(total.saturating_sub(visible_h));
    let end       = (start + visible_h).min(total);

    let lines: Vec<Line> = state.log_lines[start..end]
        .iter()
        .map(|l| {
            let color = if l.starts_with("  [PAPER]") {
                Color::Green
            } else if l.starts_with("[FAIL]") || l.starts_with("[STOP]") || l.starts_with("[ERR]") {
                Color::Red
            } else if l.starts_with("[SKIP]") || l.starts_with("[WARN]") {
                Color::Yellow
            } else if l.starts_with("[OK]") || l.starts_with("[...]") {
                Color::Cyan
            } else if l.starts_with("===") || l.starts_with("───") {
                Color::DarkGray
            } else {
                Color::White
            };
            Line::from(Span::styled(l.as_str(), Style::default().fg(color)))
        })
        .collect();

    f.render_widget(Paragraph::new(lines), chunks[0]);
    f.render_widget(hint("[j/k] Scroll    [q] Quit wizard"), chunks[1]);
}

// ── Results screen ────────────────────────────────────────────────────────────

fn draw_results(f: &mut Frame, area: Rect, state: &WizardState) {
    f.render_widget(Clear, area);
    f.render_widget(modal_block("Session Complete"), area);

    let inner = inner_area(area, 3, 1);

    if let Some(r) = &state.result {
        let pnl_color = if r.pnl >= 0.0 { Color::Green } else { Color::Red };
        let pnl_sign  = if r.pnl >= 0.0 { "+" } else { "" };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(inner);

        let kv  = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);
        let vv  = Style::default().fg(Color::Cyan);

        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Starting balance:  ", kv),
                Span::styled(format!("${:.2}", r.initial_balance), vv),
            ])),
            chunks[0],
        );
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Final balance:     ", kv),
                Span::styled(format!("${:.2}", r.final_balance), Style::default().fg(pnl_color)),
            ])),
            chunks[1],
        );
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Net P&L:           ", kv),
                Span::styled(
                    format!("{}{:.2}  ({}{:.1}%)", pnl_sign, r.pnl, pnl_sign, r.pct_return),
                    Style::default().fg(pnl_color).add_modifier(Modifier::BOLD),
                ),
            ])),
            chunks[2],
        );

        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Trades placed:     ", kv),
                Span::styled(r.total_trades.to_string(), vv),
            ])),
            chunks[4],
        );
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Settled:           ", kv),
                Span::styled(
                    format!(
                        "{}  ({} wins / {} losses)",
                        r.wins + r.losses,
                        r.wins,
                        r.losses
                    ),
                    vv,
                ),
            ])),
            chunks[5],
        );
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Pending:           ", kv),
                Span::styled(format!("{}", r.pending), Style::default().fg(Color::Yellow)),
            ])),
            chunks[6],
        );

        // Show just the filename of the session doc to keep it readable
        let doc_name = std::path::Path::new(&r.session_doc)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&r.session_doc);
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("  Session doc:       ", kv),
                Span::styled(doc_name, Style::default().fg(Color::DarkGray)),
            ])),
            chunks[7],
        );

        f.render_widget(hint("[Enter] Quit"), chunks[9]);
    }
}

// ── Error screen ──────────────────────────────────────────────────────────────

fn draw_error(f: &mut Frame, area: Rect, state: &WizardState) {
    f.render_widget(Clear, area);
    f.render_widget(
        Block::default()
            .title(Span::styled(
                " Error ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Red))
            .style(Style::default().bg(Color::Rgb(15, 15, 30))),
        area,
    );

    let inner = inner_area(area, 3, 2);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    f.render_widget(
        Paragraph::new(state.error_msg.as_str())
            .style(Style::default().fg(Color::Red))
            .alignment(Alignment::Center),
        chunks[0],
    );

    // Show last few log lines for context
    let tail: Vec<Line> = state.log_lines.iter().rev().take(5).rev()
        .map(|l| Line::from(Span::styled(l.as_str(), Style::default().fg(Color::DarkGray))))
        .collect();
    if !tail.is_empty() {
        f.render_widget(Paragraph::new(tail), chunks[1]);
    }

    f.render_widget(hint("[Enter] Go back to confirm    [q] Quit"), chunks[2]);
}

// ── Layout helpers ────────────────────────────────────────────────────────────

fn centered_modal(pct_x: u16, pct_y: u16, area: Rect) -> Rect {
    let vchunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - pct_y) / 2),
            Constraint::Percentage(pct_y),
            Constraint::Percentage((100 - pct_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - pct_x) / 2),
            Constraint::Percentage(pct_x),
            Constraint::Percentage((100 - pct_x) / 2),
        ])
        .split(vchunks[1])[1]
}

fn inner_area(area: Rect, pad_x: u16, pad_y: u16) -> Rect {
    Rect {
        x:      area.x + 1 + pad_x,
        y:      area.y + 1 + pad_y,
        width:  area.width.saturating_sub(2 + pad_x * 2),
        height: area.height.saturating_sub(2 + pad_y * 2),
    }
}

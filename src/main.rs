mod api;
mod app;
mod auth;
mod db;
mod model;
mod sysinfo;
mod ui;
mod wizard;

use anyhow::Result;
use clap::Parser;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{io, sync::Arc, time::Duration};
use tokio::sync::mpsc;

use app::{App, AppCommand, Outcome};
use auth::{KalshiConfig, KalshiSigner};

#[derive(Parser)]
#[command(name = "kalshi-tui", about = "Kalshi crypto prediction market tracker")]
struct Cli {
    /// Market data refresh interval in seconds (default: 30)
    #[arg(short, long, default_value_t = 30)]
    refresh: u64,

    /// Launch the paper trading wizard (simulates trades with mock dollars, no real orders)
    #[arg(long)]
    paper_trade: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Attempt to load auth — silently degrade if key_id not yet filled in
    let signer: Option<Arc<KalshiSigner>> = KalshiConfig::load()
        .and_then(|cfg| KalshiSigner::new(&cfg))
        .map(Arc::new)
        .ok();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    if cli.paper_trade {
        let result = wizard::run_wizard(&mut terminal).await;
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
        terminal.show_cursor()?;
        return result;
    }

    let (tx, rx) = mpsc::channel(64);
    let mut app = App::new(tx.clone())?;

    if signer.is_none() {
        app.portfolio_auth_error = Some(
            "Set key_id in ~/.config/kalshi-tui/config.toml to enable portfolio.".into()
        );
    }

    app.request_refresh();
    app.request_model_refresh();
    app.request_status_refresh();

    // Fire immediate portfolio fetch if authenticated
    if let Some(sgn) = signer.clone() {
        let tx2 = tx.clone();
        tokio::spawn(async move {
            api::fetch_and_send_portfolio(&sgn, &tx2).await;
        });
    }

    tokio::spawn(api::fetcher_loop(tx, cli.refresh, signer));

    let result = run_app(&mut terminal, &mut app, rx).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    mut rx: mpsc::Receiver<AppCommand>,
) -> Result<()> {
    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        while let Ok(cmd) = rx.try_recv() {
            app.handle_command(cmd);
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match (key.modifiers, key.code) {
                    (KeyModifiers::CONTROL, KeyCode::Char('c')) | (_, KeyCode::Char('q'))
                        if !app.is_inputting() => return Ok(()),

                    (_, KeyCode::Tab) if !app.is_inputting() => app.next_tab(),
                    (_, KeyCode::BackTab) if !app.is_inputting() => app.prev_tab(),

                    (_, KeyCode::Down) | (_, KeyCode::Char('j'))
                        if !app.is_inputting() => app.scroll_down(),
                    (_, KeyCode::Up) | (_, KeyCode::Char('k'))
                        if !app.is_inputting() => app.scroll_up(),

                    (_, KeyCode::Char('r')) if !app.is_inputting() => app.request_refresh(),
                    (_, KeyCode::Char('m')) if !app.is_inputting() => app.request_model_refresh(),
                    (_, KeyCode::Char('s')) if !app.is_inputting() => app.request_status_refresh(),
                    (_, KeyCode::Char('p')) if !app.is_inputting() => app.open_prediction_input(),

                    (_, KeyCode::Char('w')) if !app.is_inputting() => {
                        app.resolve_selected(Outcome::Win);
                    }
                    (_, KeyCode::Char('l')) if !app.is_inputting() => {
                        app.resolve_selected(Outcome::Loss);
                    }

                    (_, KeyCode::Esc) => app.cancel_input(),
                    (_, KeyCode::Enter) => app.confirm_input(),

                    (_, KeyCode::Char(c)) if app.is_inputting() => app.push_char(c),
                    (_, KeyCode::Backspace) if app.is_inputting() => app.pop_char(),

                    _ => {}
                }
            }
        }
    }
}

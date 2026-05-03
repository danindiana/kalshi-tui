use serde::Deserialize;

/// Deserialized output from predict_classifier_json.py.
/// Note: `model_prediction` and `mae_current` are legacy fields preserved for
/// UI layout stability — the classifier doesn't emit a scalar price forecast
/// (`model_prediction` equals `current_price`, `mae_current` is 0).
/// Directional bias comes from `best_opportunity().side`, not from a
/// predicted-price comparison.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelOutput {
    pub timestamp: String,
    pub current_price: f64,
    pub model_prediction: f64,
    pub mae_current: f64,
    pub mins_to_expiry: f64,
    pub indicators: Indicators,
    pub opportunities: Vec<Opportunity>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Indicators {
    pub rsi: f64,
    pub volatility: f64,
    pub momentum: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Opportunity {
    pub strike: f64,
    pub market_yes: f64,
    pub model_prob: f64,
    pub edge: f64,
    pub stake_pct: f64,
    pub side: String,
}

/// Three-way consensus signal between model direction and market sentiment
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsensusSignal {
    StrongBuy,
    StrongSell,
    Divergence,
    Unknown,
}

impl ModelOutput {
    /// Model directional bias from the classifier's highest-edge opportunity:
    /// true = bullish (best edge is a YES bet), false = bearish (best edge is NO).
    /// Returns None if there are no opportunities.
    pub fn model_bullish_opt(&self) -> Option<bool> {
        self.best_opportunity().map(|o| o.side == "YES")
    }

    /// Legacy bool-returning variant kept for callers that don't handle None.
    /// Defaults to false (bearish) when no opportunities exist — caller should
    /// prefer `model_bullish_opt()` when "no signal" needs to be distinguished.
    pub fn model_bullish(&self) -> bool {
        self.model_bullish_opt().unwrap_or(false)
    }

    /// Market-implied direction: look at the opportunity closest to current price.
    /// If the market prices YES > 50% at that strike, market is bullish.
    pub fn market_bullish(&self) -> Option<bool> {
        let closest = self
            .opportunities
            .iter()
            .min_by(|a, b| {
                (a.strike - self.current_price)
                    .abs()
                    .partial_cmp(&(b.strike - self.current_price).abs())
                    .unwrap()
            })?;
        Some(closest.market_yes > 0.50)
    }

    pub fn consensus(&self) -> ConsensusSignal {
        match (self.model_bullish_opt(), self.market_bullish()) {
            (Some(model_up), Some(mkt_up)) => match (model_up, mkt_up) {
                (true, true)   => ConsensusSignal::StrongBuy,
                (false, false) => ConsensusSignal::StrongSell,
                _              => ConsensusSignal::Divergence,
            },
            _ => ConsensusSignal::Unknown,
        }
    }

    /// Return the opportunity with the largest absolute edge
    pub fn best_opportunity(&self) -> Option<&Opportunity> {
        self.opportunities
            .iter()
            .max_by(|a, b| a.edge.abs().partial_cmp(&b.edge.abs()).unwrap())
    }
}

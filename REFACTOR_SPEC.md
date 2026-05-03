# 🛠 REFACTOR SPECIFICATION: Gemini + Kalshi Hybrid TUI

## 🎯 Objective
Transform this TUI from a manual tracker into an **AI-Driven Command Center** by integrating the Python-based Gemini Trader LSTM stack.

---

## 1. Data Ingest Overhaul (`src/api.rs`)
- **New Background Task**: Implement a tokio process that calls the existing Python script:
  `$GEMINI_TRADER_DIR/venv/bin/python $GEMINI_TRADER_DIR/predict_json.py 2>/dev/null`
- **JSON Structure**: The script returns `current_price`, `model_prediction`, `mae_current`, `mins_to_expiry`, `indicators` (RSI, Volatility), and `opportunities` (Strike, Edge, Kelly Stake).
- **Frequency**: Poll every 30 seconds.

## 2. "Edge Analysis" Tab Refactor
- **Probability Engine**: Use the `opportunities` array from the JSON to populate the analysis.
- **Kelly Sizing**: Display the `stake_pct` for each strike.
- **Visuals**: Highlight the strike with the highest absolute `edge` in **Magenta**.

## 3. Hybrid Signal Dashboard (New UI Block)
- **Top of Markets Tab**:
    - **Predicted Price**: Show bold 5 PM Target (e.g., "$72,229 ± $622").
    - **Gauges**: Visual indicators for RSI (30/70 thresholds) and Volatility.
    - **Consensus Signal**:
        - **[ STRONG BUY ]**: Model & Market are both Bullish.
        - **[ STRONG SELL ]**: Model & Market are both Bearish.
        - **[ DIVERGENCE ]**: Model and Market disagree (Technical vs Sentiment conflict).

## 4. Time Management
- **Countdown**: Display `mins_to_expiry` prominently.
- **Confidence Scaling**: The MAE should visually expand/contract as the countdown approaches 5 PM (handled by the Python JSON output).

## 5. Maintenance Fixes
- **Strike Precision**: Fix strike matching to handle floating points like `$71,249.99`.
- **Silent Execution**: Ensure all shell calls to Python redirect `stderr` to `/dev/null`.

---
**Data Source**: `$GEMINI_TRADER_DIR/predict_json.py`
**Note**: The Python environment is already configured at `$GEMINI_TRADER_DIR/venv/`.

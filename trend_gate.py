import pandas as pd
import numpy as np

class TrendGate:
    """
    Replaces the RSI Regime Detector. 
    Uses MACD to detect sustained directional trends and suppress opposing bets.
    """
    def __init__(self, short_window=12, long_window=26, signal_window=9):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculates the MACD line and Signal line."""
        if len(prices) < self.long_window:
            return 0.0, 0.0, 0.0
            
        short_ema = prices.ewm(span=self.short_window, adjust=False).mean()
        long_ema = prices.ewm(span=self.long_window, adjust=False).mean()
        
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=self.signal_window, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

    def check_defensive_gate(self, prices: list, is_no_bet: bool) -> bool:
        """
        Returns True if the trade should be ALLOWED.
        Returns False if the trade should be SUPPRESSED (Defensive Gate activated).
        """
        price_series = pd.Series(prices)
        macd, signal, hist = self.calculate_macd(price_series)
        
        # Define a threshold for what constitutes a "sustained trend"
        # This prevents minor fluctuations from triggering the gate
        TREND_THRESHOLD = 50.0 

        is_sustained_uptrend = macd > signal and hist > TREND_THRESHOLD
        is_sustained_downtrend = macd < signal and hist < -TREND_THRESHOLD

        if is_sustained_uptrend and is_no_bet:
            print(f"DEFENSIVE GATE TRIGGERED: Suppressing 'NO' bet during sustained BTC uptrend. (MACD Hist: {hist:.2f})")
            return False
            
        if is_sustained_downtrend and not is_no_bet:
            print(f"DEFENSIVE GATE TRIGGERED: Suppressing 'YES' bet during sustained BTC downtrend. (MACD Hist: {hist:.2f})")
            return False

        return True

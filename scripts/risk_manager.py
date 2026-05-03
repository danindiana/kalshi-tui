import numpy as np
from scipy.stats import norm

def calculate_win_probability(prediction, strike, mae, direction="above"):
    """
    Calculates probability of price being above/below strike given MAE.
    Uses a Normal Distribution centered at the prediction.
    """
    # Standard Deviation approx from Mean Absolute Error (MAE * 1.25 for normal dist)
    std_dev = mae * 1.25
    
    # Calculate Z-score: how many std devs is the strike from our prediction
    z_score = (strike - prediction) / std_dev
    
    # Probability of being BELOW the strike
    prob_below = norm.cdf(z_score)
    
    if direction == "below":
        return prob_below
    else:
        return 1 - prob_below

def kelly_criterion(win_prob, market_price, fractional_kelly=0.25):
    """
    win_prob: Our model's estimated probability (0-1)
    market_price: Price of the contract (0-1) e.g. 0.55
    fractional_kelly: Safety factor (default 1/4 Kelly)
    """
    if win_prob <= market_price:
        return 0.0 # No edge, don't bet
    
    # b = Net odds (Profit / Stake)
    # If price is 0.45, we bet 0.45 to win 0.55. b = 0.55/0.45
    b = (1.0 - market_price) / market_price
    
    # Kelly Formula: f* = (p*b - q) / b
    q = 1 - win_prob
    f_star = (win_prob * b - q) / b
    
    # Apply fractional safety
    return max(0, f_star * fractional_kelly)

def get_recommendation(prediction, strike, mae, market_yes_price, fractional_kelly=0.25):
    """Generates a full stake recommendation."""
    # Determine if we are betting YES (above) or NO (below)
    # If model < strike, we want to buy 'NO'.
    # On Kalshi, buying 'NO' at price X is equivalent to buying 'YES' on the inverse at (1-X)

    if prediction > strike:
        side = "YES"
        prob = calculate_win_probability(prediction, strike, mae, "above")
        price = market_yes_price
    else:
        side = "NO"
        prob = calculate_win_probability(prediction, strike, mae, "below")
        price = 1.0 - market_yes_price # Price of the 'NO' contract

    stake_pct = kelly_criterion(prob, price, fractional_kelly=fractional_kelly)

    return {
        'side': side,
        'model_prob': prob,
        'market_price': price,
        'stake_pct': stake_pct,
        'edge': prob - price
    }

if __name__ == "__main__":
    # Test with current numbers
    # Prediction: 71146, Strike: 71250, MAE: 118, Market Yes: 0.55
    rec = get_recommendation(71146, 71250, 118, 0.55)
    print(f"Side: {rec['side']}, Prob: {rec['model_prob']:.2f}, Stake: {rec['stake_pct']*100:.1f}%")

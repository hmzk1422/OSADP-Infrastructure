"""
Alpha Miner

This executable runs alphaSeeker modules which initiates an Automated Feature Engineering
Process (AFE) by brute forcing all variables in the market towards a target variable to
isolate predictive variables. 

Role:
    - Mainframe to run alphaSeeker and seek alpha in the market.

Key Responsibilities:
    - Feature Combinations: Brute forces all possible features that isn't figured out yet
    - Isolate Predictiveness: Seeks out alpha by testing all possible features through MI, SHAP, 
    multicollinearity tests, LASSO (L1), underlying model
    - Persistency Test: Tests effectiveness of features against low Signal to Noise Ratio environment.
    Returns only the most important and persistent features.

Usage:
    Name CSV file that contains all variables. Variables must be Level 1.
    Example: Open, High, Low, Close, Bid, Ask, RSI, ATR, MA, etc
    Recommendation: Strong, discerned variables only. All combination of Close to RSI would be redundant.

Dependencies:
    - ccxt: Universal crypto trading API framework.
    - pandas: For vectorized data aggregation.
"""

from alphaSeeker.combination import *
from utils.bybit import *

def run():
    client = BybitConnector(api_key="...", api_secret="...", testnet=False)

if __name__ == "__main__":
    run()
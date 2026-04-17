"""
Configuration settings for the trading system.
"""

# Exchange settings
EXCHANGE_NAME = "bybit"
TIMEFRAME = "1h"
SYMBOL = "BTC/USDT"

# Data settings
DATA_STORAGE_PATH = "./data/historical_data.csv"

# Feature engineering settings
FEATURE_WINDOW_RSI = 14
FEATURE_WINDOW_MACD_FAST = 12
FEATURE_WINDOW_MACD_SLOW = 26
FEATURE_WINDOW_MACD_SIGNAL = 9
FEATURE_WINDOW_ATR = 14
FEATURE_WINDOW_BB = 20

# Matching engine settings
KNN_NEIGHBORS = 20
SIMILARITY_THRESHOLD = 0.2  # 20%

# Strategy settings
MAX_STRATEGIES = 5
CONFIDENCE_THRESHOLD = 0.7  # 70%

# Risk management settings
RISK_PER_TRADE = 0.01  # 1% of account per trade
RISK_REWARD_RATIO = 2.0  # 1:2 risk-reward ratio
MAX_CONSECUTIVE_LOSSES = 3
TRADING_DAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
]  # No weekend trading

# Model settings
MODEL_3Y_RETRAIN_INTERVAL = "monthly"
MODEL_1Y_RETRAIN_INTERVAL = "bi-weekly"
MODEL_3M_RETRAIN_INTERVAL = "daily"

# Execution settings
ORDER_TYPE = "market"
PAPER_TRADING = True  # Set to False for live trading

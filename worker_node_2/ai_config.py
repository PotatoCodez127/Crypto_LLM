# ai_config.py
# The AI will modify these parameters to find edge.

FEATURES = [
    'cvd_trend', 
    'atr_14', 
    'close_zscore_50', 
    'volume_zscore_24',
    'rsi_14',        
    'macd_line',     
    'bb_lower',      
    'bb_upper'       
]

# How many candles into the future to predict (e.g., 1 = next hour, 3 = next 3 hours)
TARGET_LOOKAHEAD = 1

# What top percentage of probabilities to take a trade on (e.g., 80 = top 20%, 90 = top 10%)
THRESHOLD_PERCENTILE = 80

MODEL_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1
}
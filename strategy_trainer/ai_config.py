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

TARGET_LOOKAHEAD = 1
THRESHOLD_PERCENTILE = 80

MODEL_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1
}
# ai_config.py
FEATURES = [
    'cvd_trend', 
    'rsi_14', 
    'macd_line', 
    'bb_upper',
    'bb_lower',
    'volume_zscore_24',
    'log_return',
    'atr_14'
]

TARGET_LOOKAHEAD = 2
THRESHOLD_PERCENTILE = 95

MODEL_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 125,
    'reg_alpha': 1.6,
    'reg_lambda': 1.6
}

# --- NEW: EXPOSED RISK MANAGEMENT PARAMETERS ---
SL_ATR_MULTIPLIER=1.5
TP_ATR_MULTIPLIER=3.0
FEATURES=['cvd_trend', 'atr_14', 'close_zscore_50', 'volume_zscore_24', 'rsi_14', 'macd_line', 'bb_lower', 'bb_upper']
TARGET_LOOKAHEAD=1
THRESHOLD_PERCENTILE=70
MODEL_PARAMS={'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 500}

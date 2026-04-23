FEATURES=['cvd_trend', 'atr_14', 'close_zscore_50', 'volume_zscore_24', 'rsi_14', 'macd_line', 'bb_lower', 'bb_upper']
TARGET_LOOKAHEAD=3
THRESHOLD_PERCENTILE=75
MODEL_PARAMS={'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 200}
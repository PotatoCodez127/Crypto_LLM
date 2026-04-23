FEATURES=['cvd_trend', 'rsi_14', 'macd_line', 'atr_14', 'close_zscore_50']
TARGET_LOOKAHEAD=3
THRESHOLD_PERCENTILE=75
MODEL_PARAMS={'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8}

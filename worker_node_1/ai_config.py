FEATURES=['cvd_trend', 'atr_14', 'close_zscore_50', 'volume_zscore_24', 'rsi_14', 'macd_line']
TARGET_LOOKAHEAD=2
THRESHOLD_PERCENTILE=80
MODEL_PARAMS={'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0}

FEATURES=['cvd_trend', 'atr_14', 'rsi_14', 'macd_line', 'bb_lower', 'bb_upper']
TARGET_LOOKAHEAD=2
THRESHOLD_PERCENTILE=72
MODEL_PARAMS={'max_depth': 5, 'learning_rate': 0.15, 'n_estimators': 120, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 3}

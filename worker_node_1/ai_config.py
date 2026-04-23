s, while a moderate THRESHOLD_PERCENTILE=75 increases trade frequency to avoid cowardice. For MODEL_PARAMS, I'll use a balanced setup with moderate depth and learning rate to prevent overfitting while maintaining predictive power.

HYPOTHESIS:
FEATURES=['cvd_trend', 'atr_14', 'rsi_14', 'macd_line', 'bb_lower', 'bb_upper']
TARGET_LOOKAHEAD=1
THRESHOLD_PERCENTILE=75
MODEL_PARAMS={'max_depth': 5, 'learning_rate': 0.08, 'n_estimators': 150}
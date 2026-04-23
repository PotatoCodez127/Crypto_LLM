A simpler model using momentum + mean reversion features with lower threshold will increase trade frequency while maintaining predictive power.

FEATURES=['rsi_14', 'macd_line', 'bb_lower', 'bb_upper']
TARGET_LOOKAHEAD=2
THRESHOLD_PERCENTILE=75
MODEL_PARAMS={'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 80}
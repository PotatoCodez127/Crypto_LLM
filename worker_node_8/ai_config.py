# ai_config.py
# The AI will modify these lists and dictionaries to find edge.

FEATURES = [
    'cvd_trend', 
    'atr_14', 
    'close_zscore_50', 
    'volume_zscore_24'
]

MODEL_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 150,
    'random_state': 42,
    'n_jobs': -1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_weight': 5,
    'gamma': 0.1,
    'scale_pos_weight': 1.0
}

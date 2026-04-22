# ai_config.py
# The AI will modify these lists and dictionaries to find edge.

FEATURES = [
    'cvd_trend', 
    'atr_14', 
    'close_zscore_50', 
    'volume_zscore_24'
]

MODEL_PARAMS = {
    'max_depth': 12,
    'learning_rate': 0.2,
    'n_estimators': 150,
    'random_state': 42,
    'n_jobs': -1
}

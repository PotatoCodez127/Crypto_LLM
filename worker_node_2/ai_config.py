# ai_config.py
# The AI will modify these lists and dictionaries to find edge.

FEATURES = [
    'close_zscore_50',
    'cvd_trend'
]

MODEL_PARAMS = {
    'max_depth': 10,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1
}

# ai_config.py
# The AI will modify these lists and dictionaries to find edge.

FEATURES = [
    'cvd_trend', 
    'close_zscore_50'
]

MODEL_PARAMS = {
    'max_depth': 9,
    'learning_rate': 0.03,
    'n_estimators': 200,
    'random_state': 42,
    'n_jobs': -1
}

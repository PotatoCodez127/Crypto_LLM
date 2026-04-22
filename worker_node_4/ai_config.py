# ai_config.py
# Configuration for mission: max_depth=8, learning_rate=0.1, n_estimators=150
# Using all V2 features: cvd_trend, atr_14, close_zscore_50, volume_zscore_24

FEATURES = [
    'cvd_trend', 
    'atr_14', 
    'close_zscore_50', 
    'volume_zscore_24'
]

MODEL_PARAMS = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 150,
    'random_state': 42,
    'n_jobs': -1
}

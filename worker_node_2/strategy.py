import pandas as pd
import numpy as np
import xgboost as xgb

# Import the AI's current settings from the control panel
from ai_config import FEATURES, MODEL_PARAMS, TARGET_LOOKAHEAD, THRESHOLD_PERCENTILE

def get_signals(df):
    data = df.copy()
    data['signal'] = 0
    
    missing_cols = [col for col in FEATURES if col not in data.columns]
    if missing_cols:
        raise ValueError(f"\n❌ FATAL DATA ERROR: Missing {missing_cols}.")
        
    # AI controls how far into the future we predict
    data['target'] = (data['close'].shift(-TARGET_LOOKAHEAD) > data['close']).astype(int)
    clean_data = data.dropna(subset=FEATURES + ['target']).copy()
    
    if len(clean_data) < 100:
        return data
        
    X = clean_data[FEATURES]
    y = clean_data['target']
    
    split_idx = int(len(clean_data) * 0.6)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X)[:, 1]
    
    # AI controls how aggressive the trade threshold is
    threshold = np.percentile(probs, THRESHOLD_PERCENTILE)
    raw_signal = np.where(probs >= threshold, 1, 0)
    
    if len(np.unique(probs)) <= 1 or np.all(raw_signal == 1):
        raw_signal = np.random.choice([0, 1], size=len(raw_signal), p=[0.8, 0.2])
        
    data.loc[X.index, 'signal'] = raw_signal
    return data
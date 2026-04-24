import pandas as pd
import numpy as np
import xgboost as xgb

from ai_config import FEATURES, MODEL_PARAMS, TARGET_LOOKAHEAD, THRESHOLD_PERCENTILE

def get_signals(df):
    data = df.copy()

    # 1. THE SHIELD: Self-Healing Data
    if 'atr_14' not in data.columns:
        data['atr_14'] = (data['high'] - data['low']).rolling(14).mean()
        
    data = data.fillna(0)

    # 2. THE BRAIN: XGBoost Machine Learning Entries
    missing_cols = [col for col in FEATURES if col not in data.columns]
    if missing_cols:
        raise ValueError(f"\n❌ FATAL DATA ERROR: Missing {missing_cols}.")
        
    # CRITICAL ALIGNMENT: 
    # Only label '1' if the forward return exceeds the noise floor (1.0x ATR)
    forward_return = data['close'].shift(-TARGET_LOOKAHEAD) - data['close']
    data['target'] = (forward_return > (1.0 * data['atr_14'])).astype(int)
    
    clean_data = data.dropna(subset=FEATURES + ['target']).copy()
    
    # Require enough data to actually train a model
    if len(clean_data) < 500:
        data['signal'] = 0
        return data
        
    X = clean_data[FEATURES]
    y = clean_data['target']
    
    # Isolate training split to prevent lookahead bias
    split_idx = int(len(clean_data) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    # STRICT OOS GENERATION: 
    # Generate probabilities for Train vs Test independently.
    probs = np.zeros(len(X))
    probs[:split_idx] = model.predict_proba(X_train)[:, 1]
    if len(X_test) > 0:
        probs[split_idx:] = model.predict_proba(X_test)[:, 1]
    
    # Threshold MUST be calculated only on training data
    threshold = np.percentile(probs[:split_idx], THRESHOLD_PERCENTILE)
    ml_entries = np.where(probs >= threshold, 1, 0)
    
    # Map the ML signals back safely
    data['ml_signal'] = 0
    data.loc[X.index, 'ml_signal'] = ml_entries
    full_entries = data['ml_signal'].values

    # 3. THE MERGE: State Machine Risk Management
    n = len(data)
    signals = np.zeros(n)
    
    current_pos = 0
    stop_price = 0
    tp_price = 0

    close_prices = data['close'].values
    high_prices = data['high'].values
    low_prices = data['low'].values
    atr_values = data['atr_14'].values

    for i in range(1, n):
        if current_pos == 1:
            if low_prices[i] <= stop_price or high_prices[i] >= tp_price:
                current_pos = 0 
                stop_price = 0
                tp_price = 0
            signals[i] = current_pos
        else:
            if full_entries[i] == 1:
                current_pos = 1 
                stop_price = close_prices[i] - (1.5 * atr_values[i])
                tp_price = close_prices[i] + (3.0 * atr_values[i])
            signals[i] = current_pos

    data['signal'] = signals
    return data
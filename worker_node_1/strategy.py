import pandas as pd
import numpy as np
import xgboost as xgb

def generate_signals(df):
    """
    XGBoost Machine Learning Strategy Template.
    The AI will modify features and hyperparameters here to find edge.
    """
    data = df.copy()
    data['signal'] = 0
    
    # --- 1. FEATURE SELECTION (AI Tunes This) ---
    features = [
        'cvd_trend', 
        'atr_14', 
        'close_zscore_50', 
        'volume_zscore_24'
    ]
    
    # --- 2. TARGET DEFINITION ---
    # Predicting if the NEXT candle's close will be higher than THIS candle's close
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    # --- 3. DATA CLEANING ---
    clean_data = data.dropna(subset=features + ['target']).copy()
    
    if len(clean_data) < 100:
        return data
        
    X = clean_data[features]
    y = clean_data['target']
    
    # --- 4. CHRONOLOGICAL SPLIT ---
    split_idx = int(len(clean_data) * 0.6)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # --- 5. THE XGBOOST MODEL (AI Tunes This) ---
    model = xgb.XGBClassifier(
        max_depth=15,           
        learning_rate=0.05,    
        n_estimators=300,      
        random_state=42,
        n_jobs=-1              
    )
    
    model.fit(X_train, y_train)
    
    # --- 6. PREDICT & EXECUTE (THE DYNAMIC FIX) ---
    # Instead of hard 1s and 0s, get the raw probability percentages
    probs = model.predict_proba(X_test)[:, 1]
    
    # Find the top 20% highest probability setups in the test window
    # This FORCES the model to take trades, no matter how noisy the data is.
    threshold = np.percentile(probs, 80)
    
    # If the probability is in the top 20%, Buy (1). Otherwise, Hold (0).
    test_indices = X_test.index
    data.loc[test_indices, 'signal'] = np.where(probs >= threshold, 1, 0)
    
    return data

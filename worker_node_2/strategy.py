import pandas as pd
import numpy as np
import xgboost as xgb

def generate_signals(df):
    """
    XGBoost Machine Learning Strategy Template.
    The AI will modify features and hyperparameters here to find edge.
    """
    # Create a safe copy and initialize flat signals
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
        return data # Failsafe if data is too short
        
    X = clean_data[features]
    y = clean_data['target']
    
    # --- 4. CHRONOLOGICAL SPLIT ---
    # Train on the past, predict the future. NO LOOKAHEAD BIAS.
    split_idx = int(len(clean_data) * 0.6)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # --- 5. THE XGBOOST MODEL (AI Tunes This) ---
    model = xgb.XGBClassifier(
        max_depth=10,           # AI tunes how complex the trees are
        learning_rate=0.1,      # AI tunes how fast the model learns
        n_estimators=100,      # AI tunes the number of trees
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # --- 6. PREDICT & EXECUTE ---
    # Generate predictions ONLY on the unseen Out-Of-Sample data
    predictions = model.predict(X_test)
    
    # Map predictions back to the original dataframe (1 = Buy, 0 = Hold/Close)
    test_indices = X_test.index
    data.loc[test_indices, 'signal'] = predictions
    
    return data

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
    
    # Further split training data for early stopping validation
    val_split_idx = int(len(X_train) * 0.8)
    X_train_fit, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_train_fit, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
    
    # --- 5. THE XGBOOST MODEL (AI Tunes This) ---
    # Calculate scale_pos_weight to handle class imbalance
    pos_count = y_train_fit.sum()
    neg_count = len(y_train_fit) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    model = xgb.XGBClassifier(
        max_depth=3,           # Set to 3 as per mission requirements
        learning_rate=0.05,    # Set to 0.05 as per mission requirements
        n_estimators=200,      # Set to 200 as per mission requirements
        subsample=0.8,         # Add regularization to improve OOS performance
        colsample_bytree=0.8,  # Feature sampling per tree
        min_child_weight=5,    # Minimum sum of instance weight needed in a child
        gamma=0.1,             # Minimum loss reduction required to make a split
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        random_state=42,
        n_jobs=-1,             # Use all CPU cores
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    
    # Train the model with early stopping
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # --- 6. PREDICT & EXECUTE ---
    # Generate predictions ONLY on the unseen Out-Of-Sample data
    predictions = model.predict(X_test)
    
    # Map predictions back to the original dataframe (1 = Buy, 0 = Hold/Close)
    test_indices = X_test.index
    data.loc[test_indices, 'signal'] = predictions
    
    return data

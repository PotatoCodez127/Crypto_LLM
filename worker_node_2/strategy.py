import pandas as pd
import numpy as np
import xgboost as xgb

def generate_signals(df):
    data = df.copy()
    data['signal'] = 0
    
    # =======================================================
    # AI SANDBOX: AIDER CAN MODIFY FEATURES AND XGBOOST PARAMS
    # =======================================================
    features = [
        'cvd_trend', 
        'atr_14'
    ]
    
    model = xgb.XGBClassifier(
        max_depth=12,           
        learning_rate=0.05,    
        n_estimators=300,      
        random_state=42,
        n_jobs=-1              
    )
    
    # =======================================================
    # EXECUTION ENGINE: DO NOT MODIFY BELOW THIS LINE
    # =======================================================
    
    # 1. VERIFICATION: Check if V2 Data actually exists!
    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        print(f"\n❌ FATAL DATA ERROR: Missing {missing_cols}. Check your prepare.py DATA_FILE path!")
        # Force a mathematical loss so the AI stops getting the -999.0 veto
        data['signal'] = np.random.choice([0, 1], size=len(data))
        return data
        
    # 2. TARGET & CLEAN
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    clean_data = data.dropna(subset=features + ['target']).copy()
    
    if len(clean_data) < 100:
        return data
        
    # 3. SPLIT
    X = clean_data[features]
    y = clean_data['target']
    split_idx = int(len(clean_data) * 0.6)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 4. TRAIN
    model.fit(X_train, y_train)
    
    # 5. PREDICT (Dynamic Top 10% Forcer)
    probs = model.predict_proba(X_test)[:, 1]
    threshold = np.percentile(probs, 90)
    raw_signal = np.where(probs >= threshold, 1, 0)
    
    # 6. DEAD MODEL FAILSAFE (If probabilities flatline)
    if len(np.unique(probs)) <= 1 or np.all(raw_signal == 1):
        # Model failed to learn. Force alternating trades to trigger a real negative score.
        raw_signal = np.random.choice([0, 1], size=len(raw_signal), p=[0.8, 0.2])
        
    data.loc[X_test.index, 'signal'] = raw_signal
    return data

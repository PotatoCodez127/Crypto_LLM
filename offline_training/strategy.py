import pandas as pd
import numpy as np

def get_signals(df):
    """
    V2 Baseline: Stationary Features, CVD Proxy, and Volatility-Scaled Logic.
    """
    # --- 1. INSTITUTIONAL FEATURE ENGINEERING ---
    
    # A. Log Returns (Stationary price movement)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # B. Volatility Profile
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    
    # C. Cumulative Volume Delta (CVD) Proxy
    # Positive volume if close >= open (buyers aggressive), negative if close < open (sellers aggressive)
    df['candle_dir'] = np.where(df['close'] >= df['open'], 1, -1)
    df['vol_delta'] = df['volume'] * df['candle_dir']
    
    # 20-period rolling CVD (Kept as a rolling window to maintain stationarity)
    df['cvd_20'] = df['vol_delta'].rolling(window=20).sum()
    
    # D. Normalized Momentum (Z-score of price vs 50-SMA)
    sma_50 = df['close'].rolling(window=50).mean()
    std_50 = df['close'].rolling(window=50).std()
    df['z_score_50'] = (df['close'] - sma_50) / (std_50 + 1e-8)
    
    # Fill missing values caused by rolling windows
    df = df.bfill().fillna(0)

    # --- 2. THE EXECUTION LOGIC (For the AI to hack) ---
    df['raw_signal'] = 0
    
    # Compute rolling std of vol_delta for CVD threshold scaling
    df['vol_delta_std_20'] = df['vol_delta'].rolling(window=20).std()
    
    # Initial Hypothesis: Mean reversion with volatility filter.
    # Buy when selling volume is exhausted (negative CVD) but price is moderately undervalued (Z-score < -1.5)
    # and volatility is not too low (above 30% of median volatility)
    long_condition = (df['cvd_20'] < -0.2 * df['vol_delta_std_20']) & (df['z_score_50'] < -1.5) & (df['volatility_20'] > df['vol_median'] * 0.3)
    
    # Sell when buying volume is exhausted (positive CVD) but price is moderately overvalued (Z-score > 1.5)
    # and volatility is not too low
    short_condition = (df['cvd_20'] > 0.2 * df['vol_delta_std_20']) & (df['z_score_50'] > 1.5) & (df['volatility_20'] > df['vol_median'] * 0.3)

    # Cooldown to prevent overtrading but allow more opportunities
    cooldown = 5
    last_signal_idx = -cooldown
    for i in range(len(df)):
        if i < last_signal_idx + cooldown:
            continue
        if long_condition.iloc[i]:
            df.iloc[i, df.columns.get_loc('raw_signal')] = 1
            last_signal_idx = i
        elif short_condition.iloc[i]:
            df.iloc[i, df.columns.get_loc('raw_signal')] = -1
            last_signal_idx = i

    # --- 3. ADVANCED RISK MANAGEMENT (Adaptive ATR) ---
    # (Keeping your original adaptive stop-loss, as the logic is solid)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)

    df['signal'] = 0
    position = 0
    stop_price = 0.0

    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        vol = df['volatility_20'].iloc[i]
        vol_med = df['vol_median'].iloc[i]
        
        if vol_med > 0:
            atr_multiplier = 1.2 * (vol / vol_med)
            atr_multiplier = max(0.8, min(1.8, atr_multiplier))
        else:
            atr_multiplier = 1.2

        if position == 0:
            if raw == 1:
                position = 1
                stop_price = close - atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif raw == -1:
                position = -1
                stop_price = close + atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            new_stop = close - atr_multiplier * atr
            if new_stop > stop_price: stop_price = new_stop
            if close <= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            if new_stop < stop_price: stop_price = new_stop
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1

    return df

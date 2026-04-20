import pandas as pd
import numpy as np

def get_signals(df):
    """
    V2 Baseline: Stationary Features, CVD Proxy, and Volatility-Scaled Logic.
    """
    # A. Log Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # B. Volatility Profile
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    
    # C. Cumulative Volume Delta (CVD) Proxy with shorter window for faster response
    df['candle_dir'] = np.where(df['close'] >= df['open'], 1, -1)
    df['vol_delta'] = df['volume'] * df['candle_dir']
    df['cvd_15'] = df['vol_delta'].rolling(window=15).sum()
    df['cvd_20'] = df['cvd_15']  # Keep for compatibility but use more responsive window
    
    # D. Normalized Momentum (Z-score) with faster window for responsiveness
    sma_30 = df['close'].rolling(window=30).mean()
    std_30 = df['close'].rolling(window=30).std()
    df['z_score_50'] = (df['close'] - sma_30) / (std_30 + 1e-8)
    
    df = df.bfill().fillna(0)

    # --- EXECUTION LOGIC ---
    # Normalized CVD using robust rolling percentiles (IQR) with shorter lookback
    df['cvd_20_q25'] = df['cvd_20'].rolling(window=60).quantile(0.25)
    df['cvd_20_q75'] = df['cvd_20'].rolling(window=60).quantile(0.75)
    df['cvd_iqr'] = df['cvd_20_q75'] - df['cvd_20_q25']
    df['cvd_robust'] = (df['cvd_20'] - df['cvd_20'].rolling(window=60).median()) / (df['cvd_iqr'] + 1e-8)
    
    df['raw_signal'] = 0
    # Relax thresholds to capture more opportunities while maintaining edge
    vol_above_median = df['volatility_20'] > df['vol_median']
    # Use less extreme thresholds: CVD robust ±1.2 (was ±1.5), Z-score ±1.0 (was ±1.2)
    long_condition = (df['cvd_robust'] < -1.2) & (df['z_score_50'] < -1.0) & vol_above_median
    short_condition = (df['cvd_robust'] > 1.2) & (df['z_score_50'] > 1.0) & vol_above_median

    # Reduce cooldown to allow more frequent signals when conditions align
    cooldown = 8
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

    # --- ADVANCED RISK MANAGEMENT (Adaptive ATR) ---
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)

    df['signal'] = 0
    position = 0
    stop_price = 0.0
    entry_price = 0.0

    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        vol = df['volatility_20'].iloc[i]
        vol_med = df['vol_median'].iloc[i]
        
        if vol_med > 0:
            # Adjust scaling to be more responsive to volatility changes
            vol_ratio = vol / vol_med
            # Use linear scaling between 1.2 and 3.5 for tighter stops in normal vol
            # and wider in high vol, but not too extreme
            atr_multiplier = 1.2 + (2.3 * np.tanh(vol_ratio - 1.0))
            atr_multiplier = max(1.2, min(3.5, atr_multiplier))
        else:
            atr_multiplier = 2.0

        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Trailing stop logic with a floor based on entry
            new_stop = close - atr_multiplier * atr
            # Ensure stop never moves below entry - 1.5*ATR (max loss protection)
            max_loss_stop = entry_price - 1.5 * atr
            if new_stop > stop_price and new_stop > max_loss_stop:
                stop_price = new_stop
            elif max_loss_stop > stop_price:
                stop_price = max_loss_stop
            if close <= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            max_loss_stop = entry_price + 1.5 * atr
            if new_stop < stop_price and new_stop < max_loss_stop:
                stop_price = new_stop
            elif max_loss_stop < stop_price:
                stop_price = max_loss_stop
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1

    return df

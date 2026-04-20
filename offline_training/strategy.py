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
    
    # C. Cumulative Volume Delta (CVD) Proxy
    df['candle_dir'] = np.where(df['close'] >= df['open'], 1, -1)
    df['vol_delta'] = df['volume'] * df['candle_dir']
    df['cvd_20'] = df['vol_delta'].rolling(window=20).sum()
    
    # D. Normalized Momentum (Z-score)
    sma_50 = df['close'].rolling(window=50).mean()
    std_50 = df['close'].rolling(window=50).std()
    df['z_score_50'] = (df['close'] - sma_50) / (std_50 + 1e-8)
    
    df = df.bfill().fillna(0)

    # --- EXECUTION LOGIC ---
    # Normalized CVD using robust rolling percentiles (IQR)
    df['cvd_20_q25'] = df['cvd_20'].rolling(window=100).quantile(0.25)
    df['cvd_20_q75'] = df['cvd_20'].rolling(window=100).quantile(0.75)
    df['cvd_iqr'] = df['cvd_20_q75'] - df['cvd_20_q25']
    df['cvd_robust'] = (df['cvd_20'] - df['cvd_20'].rolling(window=100).median()) / (df['cvd_iqr'] + 1e-8)
    
    # Normalized CVD using rolling Z-score (more stable than IQR)
    df['cvd_mean'] = df['cvd_20'].rolling(window=100).mean()
    df['cvd_std'] = df['cvd_20'].rolling(window=100).std()
    df['cvd_z'] = (df['cvd_20'] - df['cvd_mean']) / (df['cvd_std'] + 1e-8)
    
    df['raw_signal'] = 0
    # Volatility filter: require volatility above 80% of median (avoid low-vol whipsaws)
    vol_condition = df['volatility_20'] > (0.8 * df['vol_median'])
    # Adjusted thresholds for more signals while maintaining edge
    long_condition = (df['cvd_z'] < -1.2) & (df['z_score_50'] < -1.0) & vol_condition
    short_condition = (df['cvd_z'] > 1.2) & (df['z_score_50'] > 1.0) & vol_condition

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
            # Less aggressive adaptive multiplier
            vol_ratio = vol / vol_med
            # Keep multiplier between 1.2 and 3.0
            atr_multiplier = 1.2 + (1.8 / (1.0 + np.exp(-vol_ratio + 1.0)))
            atr_multiplier = max(1.2, min(3.0, atr_multiplier))
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
            # Slightly tighter max loss protection
            max_loss_stop = entry_price - 1.2 * atr
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
            max_loss_stop = entry_price + 1.2 * atr
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

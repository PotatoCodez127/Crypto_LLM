import pandas as pd
import numpy as np

def get_signals(df):
    """
    MISSION STRATEGY: Use CVD trend, close z-score, and ATR condition.
    Long when (cvd_trend > 0.8) and (close_zscore_50 < -1.5) and (atr_14 > df['atr_14'].rolling(20).mean()).
    Short when (cvd_trend < -0.8) and (close_zscore_50 > 1.5) and (atr_14 > df['atr_14'].rolling(20).mean()).
    Exit: Fixed stop-loss (1.5*ATR) and take-profit (3*ATR) from entry.
    Position sizing: inverse volatility scaling (0.5 to 1.5).
    Cooldown: 5 bars after exit before new entry.
    """
    # Ensure required columns exist (they should from feature extraction)
    # Calculate rolling mean of atr_14 for condition
    df['atr_14_rolling_mean'] = df['atr_14'].rolling(window=20).mean()
    
    # Entry conditions
    long_entry = (df['cvd_trend'] > 0.8) & (df['close_zscore_50'] < -1.5) & (df['atr_14'] > df['atr_14_rolling_mean'])
    short_entry = (df['cvd_trend'] < -0.8) & (df['close_zscore_50'] > 1.5) & (df['atr_14'] > df['atr_14_rolling_mean'])
    
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # Volatility scaling factor (0.5 to 1.5) using existing volatility column if present, else compute
    if 'volatility' not in df.columns:
        vol_window = 50
        df['volatility'] = df['close'].pct_change().rolling(vol_window).std()
    vol_median = df['volatility'].median()
    df['size_factor'] = np.clip(vol_median / (df['volatility'] + 1e-8), 0.5, 1.5)
    
    # Position management with fixed stop/target and cooldown
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown = 0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr_14'].iloc[i]
        size_factor = df['size_factor'].iloc[i]
        
        # Decrease cooldown
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - 1.5 * atr
                target_price = close + 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * size_factor
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + 1.5 * atr
                target_price = close - 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * size_factor
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Check exit
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown = 5  # bars cooldown
                # No immediate re-entry
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * size_factor
        elif position == -1:
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown = 5
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * size_factor
        else:
            df.iloc[i, df.columns.get_loc('signal')] = 0.0
    
    return df

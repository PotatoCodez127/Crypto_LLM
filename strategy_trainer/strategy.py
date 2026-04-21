import pandas as pd
import numpy as np

def get_signals(df):
    """
    MISSION: Generate long signals when close_zscore_50 < -1.5 and cvd_trend > 0.2 and volume_zscore_24 > 0.8 and atr_14 < df['atr_14'].rolling(50).mean(),
    and short signals when close_zscore_50 > 1.5 and cvd_trend < -0.2 and volume_zscore_24 > 0.8 and atr_14 < df['atr_14'].rolling(50).mean().
    Use existing columns: ['open', 'high', 'low', 'close', 'volume', 'log_return', 'candle_dir', 'volume_delta', 'cvd', 'cvd_trend', 'atr_14', 'close_zscore_50', 'volume_zscore_24'].
    """
    # Ensure required columns exist (they should already)
    required = ['close_zscore_50', 'cvd_trend', 'volume_zscore_24', 'atr_14']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column {col} not found in DataFrame")
    
    # Compute rolling mean of atr_14 for the condition
    df['atr_14_rolling_50'] = df['atr_14'].rolling(window=50).mean()
    
    # Entry conditions
    long_entry = (
        (df['close_zscore_50'] < -1.5) &
        (df['cvd_trend'] > 0.2) &
        (df['volume_zscore_24'] > 0.8) &
        (df['atr_14'] < df['atr_14_rolling_50'])
    )
    short_entry = (
        (df['close_zscore_50'] > 1.5) &
        (df['cvd_trend'] < -0.2) &
        (df['volume_zscore_24'] > 0.8) &
        (df['atr_14'] < df['atr_14_rolling_50'])
    )
    
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # Position management with fixed stop/target and cooldown
    # Use atr_14 for stops/targets
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
        
        # Decrease cooldown
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - 1.5 * atr
                target_price = close + 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0  # size factor removed for simplicity
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + 1.5 * atr
                target_price = close - 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Check exit
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown = 5  # bars cooldown
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
        elif position == -1:
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown = 5
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
        else:
            df.iloc[i, df.columns.get_loc('signal')] = 0.0
    
    # Drop temporary column
    df.drop(columns=['atr_14_rolling_50'], inplace=True, errors='ignore')
    
    return df

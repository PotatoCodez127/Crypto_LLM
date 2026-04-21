import pandas as pd
import numpy as np

def get_signals(df):
    """
    Mission: Create long signal when (cvd_trend > 0.7) & (close_zscore_50 < -1.2) & (volume_zscore_24 > 0.8)
    and short signal when (cvd_trend < -0.7) & (close_zscore_50 > 1.2) & (volume_zscore_24 > 0.8).
    Use pre-calculated columns: ['open','high','low','close','volume','log_return','candle_dir',
    'volume_delta','cvd','cvd_trend','atr_14','close_zscore_50','volume_zscore_24'].
    Exit with fixed stop-loss (1.5*ATR) and take-profit (3*ATR).
    Cooldown: 5 bars after exit before new entry.
    Position sizing: inverse volatility scaling based on ATR.
    """
    # Ensure required columns exist (they should from the data pipeline)
    required_cols = ['cvd_trend', 'close_zscore_50', 'volume_zscore_24', 'atr_14', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame")
    
    # 1. Entry conditions
    long_entry = (df['cvd_trend'] > 0.7) & (df['close_zscore_50'] < -1.2) & (df['volume_zscore_24'] > 0.8)
    short_entry = (df['cvd_trend'] < -0.7) & (df['close_zscore_50'] > 1.2) & (df['volume_zscore_24'] > 0.8)
    
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 2. Volatility scaling factor using atr_14 relative to its rolling median
    # Inverse volatility scaling: higher ATR -> smaller position size
    atr_median = df['atr_14'].rolling(window=100, min_periods=1).median()
    # Avoid division by zero
    df['size_factor'] = np.clip(atr_median / (df['atr_14'] + 1e-8), 0.5, 1.5)
    
    # 3. Position management with fixed stop/target and cooldown
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

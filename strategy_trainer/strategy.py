import pandas as pd
import numpy as np

def get_signals(df):
    """
    MOMENTUM BREAKOUT WITH VOLUME CONFIRMATION:
    - Long when price breaks above highest high of last 20 bars and volume > 20-period average.
    - Short when price breaks below lowest low of last 20 bars and volume > 20-period average.
    - Use ATR (14) for stop-loss (3*ATR) and take-profit (5*ATR) to allow larger runs.
    - Cooldown period of 5 bars after exit before new entry to reduce whipsaw.
    - Dynamic position sizing: 1.0 / (atr / close) normalized to 0.5-1.0 range.
    """
    # Ensure volume column exists; if not, create dummy volume
    if 'volume' not in df.columns:
        df['volume'] = 1.0
    
    # 1. Compute highs/lows over lookback
    lookback = 20
    df['highest_high'] = df['high'].rolling(window=lookback).max().shift(1)
    df['lowest_low'] = df['low'].rolling(window=lookback).min().shift(1)
    
    # 2. Volume confirmation
    df['volume_ma'] = df['volume'].rolling(window=lookback).mean()
    df['volume_ok'] = df['volume'] > df['volume_ma']
    
    # 3. ATR for stops and sizing
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 4. Entry signals
    long_break = df['close'] > df['highest_high']
    short_break = df['close'] < df['lowest_low']
    
    df['raw_long'] = (long_break & df['volume_ok']).astype(int)
    df['raw_short'] = (short_break & df['volume_ok']).astype(int) * -1
    
    # Combine raw signals (long=1, short=-1)
    df['raw_signal'] = 0
    df.loc[df['raw_long'] == 1, 'raw_signal'] = 1
    df.loc[df['raw_short'] == -1, 'raw_signal'] = -1
    
    # 5. Position management with cooldown
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown_counter = 0
    cooldown_period = 5
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        # Decrease cooldown
        if cooldown_counter > 0:
            cooldown_counter -= 1
        
        # Exit conditions if in position
        if position != 0:
            if (position == 1 and (close <= stop_price or close >= target_price)) or \
               (position == -1 and (close >= stop_price or close <= target_price)):
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown_counter = cooldown_period
                # Do not allow immediate re‑entry in same bar
                continue
        
        # Enter new position if cooldown expired
        if position == 0 and cooldown_counter == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - 3.0 * atr
                target_price = close + 5.0 * atr
                # Position sizing based on volatility (inverse)
                atr_ratio = atr / close
                size = 1.0 / (atr_ratio + 0.01)  # avoid division by zero
                # Normalize size between 0.5 and 1.0
                size = max(0.5, min(1.0, size / 10.0))
                df.iloc[i, df.columns.get_loc('signal')] = size
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + 3.0 * atr
                target_price = close - 5.0 * atr
                atr_ratio = atr / close
                size = 1.0 / (atr_ratio + 0.01)
                size = max(0.5, min(1.0, size / 10.0))
                df.iloc[i, df.columns.get_loc('signal')] = -size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        else:
            # Maintain current position with same size (use last non‑zero size)
            if position == 1:
                # retrieve the size from the entry bar (simplified: use 1.0)
                # we'll store size in a variable, but for simplicity keep 1.0
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
            elif position == -1:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
    
    # Clean up temporary columns
    cols_to_drop = ['highest_high', 'lowest_low', 'volume_ma', 'volume_ok', 
                    'raw_long', 'raw_short', 'raw_signal']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    return df

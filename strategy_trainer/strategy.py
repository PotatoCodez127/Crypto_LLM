import pandas as pd
import numpy as np

def get_signals(df):
    """
    ADAPTIVE TREND-FOLLOWING STRATEGY:
    - EMA crossover (fast=8, slow=21) for quicker trend detection.
    - Dynamic RSI thresholds based on recent ATR volatility: lower threshold = 40 - volatility_factor, upper = 60 + volatility_factor.
    - Remove low-volatility filter to capture more opportunities.
    - ADX threshold lowered to 20 for more trend signals.
    - Dynamic exit multiples: stop-loss at 1.5*ATR, take-profit at 2.5*ATR, adjusted by recent win rate (simplified).
    - Add price confirmation: close must be above/below 5-period high/low for entry.
    - Position sizing scaled by volatility: size = min(1.0, 0.5 / (atr_pct + 0.01)).
    """
    # 1. Trend: EMA crossover
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['trend_up'] = df['ema_fast'] > df['ema_slow']
    df['trend_down'] = df['ema_fast'] < df['ema_slow']
    
    # 2. RSI (14) with dynamic thresholds
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. ATR for stops and volatility scaling
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    df['atr_pct'] = df['atr'] / df['close']
    
    # 4. Dynamic RSI thresholds based on recent volatility (20-period ATR pct)
    vol_factor = df['atr_pct'].rolling(window=20).mean() * 100  # scale to ~0-5
    df['rsi_lower'] = 40 - vol_factor * 2
    df['rsi_upper'] = 60 + vol_factor * 2
    # Clip to reasonable bounds
    df['rsi_lower'] = df['rsi_lower'].clip(20, 45)
    df['rsi_upper'] = df['rsi_upper'].clip(55, 80)
    
    # 5. ADX calculation (14 period)
    df['tr'] = tr
    df['atr'] = df['tr'].rolling(window=14).mean().bfill().fillna(0)
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    df['plus_dm'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    df['minus_dm'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr'])
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
    df['adx'] = df['dx'].rolling(window=14).mean()
    df['trend_strong'] = df['adx'] > 20  # lowered threshold
    
    # 6. Price confirmation: close above 5-period high for long, below 5-period low for short
    df['high_5'] = df['high'].rolling(window=5).max()
    df['low_5'] = df['low'].rolling(window=5).min()
    df['close_above_high5'] = df['close'] > df['high_5'].shift(1)
    df['close_below_low5'] = df['close'] < df['low_5'].shift(1)
    
    # 7. Entry conditions
    long_entry = (df['trend_up']) & (df['rsi'] < df['rsi_lower']) & (df['trend_strong']) & (df['close_above_high5'])
    short_entry = (df['trend_down']) & (df['rsi'] > df['rsi_upper']) & (df['trend_strong']) & (df['close_below_low5'])
    
    # 8. Raw signals (no cooldown)
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 9. Position management with dynamic exit multiples and volatility sizing
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    extreme_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    position_size = 1.0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        atr_pct = df['atr_pct'].iloc[i]
        
        # Dynamic exit multiples based on recent volatility (higher volatility -> wider stops)
        # Use inverse relation: if atr_pct is high, reduce risk multiples slightly
        vol_mult = 1.0 + (0.5 - atr_pct * 10)  # adjust multiplier
        vol_mult = max(0.8, min(1.5, vol_mult))
        stop_mult = 1.5 * vol_mult
        target_mult = 2.5 * vol_mult
        
        # Position sizing
        position_size = min(1.0, 0.5 / (atr_pct + 0.01))
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                extreme_price = close
                stop_price = close - stop_mult * atr
                target_price = close + target_mult * atr
                df.iloc[i, df.columns.get_loc('signal')] = position_size
            elif raw == -1:
                position = -1
                entry_price = close
                extreme_price = close
                stop_price = close + stop_mult * atr
                target_price = close - target_mult * atr
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            if close > extreme_price:
                extreme_price = close
                stop_price = extreme_price - stop_mult * atr
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                if raw == 1:
                    position = 1
                    entry_price = close
                    extreme_price = close
                    stop_price = close - stop_mult * atr
                    target_price = close + target_mult * atr
                    df.iloc[i, df.columns.get_loc('signal')] = position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = position_size
        elif position == -1:
            if close < extreme_price:
                extreme_price = close
                stop_price = extreme_price + stop_mult * atr
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                if raw == -1:
                    position = -1
                    entry_price = close
                    extreme_price = close
                    stop_price = close + stop_mult * atr
                    target_price = close - target_mult * atr
                    df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
    
    return df

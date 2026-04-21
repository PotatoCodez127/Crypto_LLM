import pandas as pd
import numpy as np

def get_signals(df):
    """
    ENHANCED ROBUST STRATEGY:
    - Dual moving average crossover (fast=12, slow=36) for trend direction.
    - RSI (14) thresholds 35/65 for entries within trend.
    - Volatility filter: recent volatility below 1.5 * its 100-period median (relaxed).
    - ADX filter: require ADX > 25 to confirm trend strength.
    - Trailing stop-loss at 2*ATR from extreme price since entry, take-profit at 3*ATR.
    - Position sizing: full size (1.0) always.
    """
    # 1. Trend: moving averages
    df['ma_fast'] = df['close'].rolling(window=12).mean()
    df['ma_slow'] = df['close'].rolling(window=36).mean()
    df['trend_up'] = df['ma_fast'] > df['ma_slow']
    df['trend_down'] = df['ma_fast'] < df['ma_slow']
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Volatility filter (relaxed)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    df['low_vol'] = df['volatility_20'] < (df['vol_median'] * 1.5)
    
    # 4. ATR for stops
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 5. ADX calculation (14 period)
    # True Range already computed as tr
    df['tr'] = tr
    df['atr'] = df['tr'].rolling(window=14).mean().bfill().fillna(0)
    # Directional Movement
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    df['plus_dm'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    df['minus_dm'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth DM
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr'])
    # DX and ADX
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
    df['adx'] = df['dx'].rolling(window=14).mean()
    df['trend_strong'] = df['adx'] > 25
    
    # 6. Entry conditions
    # Long: trend up AND RSI < 35 (oversold within uptrend) AND low volatility AND strong trend
    long_entry = (df['trend_up']) & (df['rsi'] < 35) & (df['low_vol']) & (df['trend_strong'])
    # Short: trend down AND RSI > 65 (overbought within downtrend) AND low volatility AND strong trend
    short_entry = (df['trend_down']) & (df['rsi'] > 65) & (df['low_vol']) & (df['trend_strong'])
    
    # 7. Raw signals (no cooldown)
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 8. Position management with trailing stop and fixed target
    df['signal'] = 0
    position = 0
    entry_price = 0.0
    extreme_price = 0.0  # highest close for long, lowest for short
    stop_price = 0.0
    target_price = 0.0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                extreme_price = close
                stop_price = close - 2.0 * atr
                target_price = close + 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
            elif raw == -1:
                position = -1
                entry_price = close
                extreme_price = close
                stop_price = close + 2.0 * atr
                target_price = close - 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Update extreme price
            if close > extreme_price:
                extreme_price = close
                # Adjust trailing stop
                stop_price = extreme_price - 2.0 * atr
            # Check exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                # Allow immediate re-entry if condition still holds
                if raw == 1:
                    position = 1
                    entry_price = close
                    extreme_price = close
                    stop_price = close - 2.0 * atr
                    target_price = close + 3.0 * atr
                    df.iloc[i, df.columns.get_loc('signal')] = 1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
        elif position == -1:
            if close < extreme_price:
                extreme_price = close
                stop_price = extreme_price + 2.0 * atr
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                if raw == -1:
                    position = -1
                    entry_price = close
                    extreme_price = close
                    stop_price = close + 2.0 * atr
                    target_price = close - 3.0 * atr
                    df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
    
    return df

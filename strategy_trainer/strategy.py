import pandas as pd
import numpy as np

def get_signals(df):
    """
    PIVOTED MACD + RSI STRATEGY:
    - MACD (12,26,9) histogram for trend direction.
    - RSI (14) thresholds 30/70 for entries.
    - Volatility filter: ATR below 1.2 * its 100-period median.
    - ADX filter: require ADX > 20 to confirm trend strength.
    - Trailing stop-loss at 1.5*ATR from extreme price, take-profit at 2.5*ATR.
    - Cooldown of 3 bars after exit to reduce whipsaw.
    - Position sizing: full size (1.0) always.
    """
    # 1. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. ATR for stops and volatility filter
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 4. Volatility filter based on ATR
    df['atr_median'] = df['atr'].rolling(window=100).median()
    df['low_vol'] = df['atr'] < (df['atr_median'] * 1.2)
    
    # 5. ADX calculation (14 period)
    df['tr'] = tr
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    df['plus_dm'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    df['minus_dm'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / (df['atr'] + 1e-8))
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / (df['atr'] + 1e-8))
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
    df['adx'] = df['dx'].rolling(window=14).mean()
    df['trend_strong'] = df['adx'] > 20
    
    # 6. Entry conditions
    long_entry = (df['macd_hist'] > 0) & (df['rsi'] < 30) & (df['low_vol']) & (df['trend_strong'])
    short_entry = (df['macd_hist'] < 0) & (df['rsi'] > 70) & (df['low_vol']) & (df['trend_strong'])
    
    # 7. Raw signals (no cooldown yet)
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 8. Position management with trailing stop, target, and cooldown
    df['signal'] = 0
    position = 0
    entry_price = 0.0
    extreme_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown = 0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if cooldown > 0:
            cooldown -= 1
            df.iloc[i, df.columns.get_loc('signal')] = 0
            continue
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                extreme_price = close
                stop_price = close - 1.5 * atr
                target_price = close + 2.5 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
            elif raw == -1:
                position = -1
                entry_price = close
                extreme_price = close
                stop_price = close + 1.5 * atr
                target_price = close - 2.5 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Update extreme price
            if close > extreme_price:
                extreme_price = close
                stop_price = extreme_price - 1.5 * atr
            # Check exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 3  # bars of no new entry
                # Do not re-enter immediately
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
        elif position == -1:
            if close < extreme_price:
                extreme_price = close
                stop_price = extreme_price + 1.5 * atr
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 3
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
    
    return df

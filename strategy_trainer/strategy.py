import pandas as pd
import numpy as np

def get_signals(df):
    """
    SIMPLIFIED ROBUST STRATEGY:
    - Use dual moving average crossover (fast=20, slow=50) for trend direction.
    - Use RSI (14) for overbought/oversold entries within the trend.
    - Volatility filter: only trade when recent volatility is below its 100-period median (avoid high volatility).
    - Fixed stop-loss at 1.5*ATR, take-profit at 3*ATR (risk-reward 1:2).
    - No cooldown periods; allow re-entries immediately after exit.
    - Position sizing: full size (1.0) always.
    """
    # 1. Trend: moving averages
    df['ma_fast'] = df['close'].rolling(window=20).mean()
    df['ma_slow'] = df['close'].rolling(window=50).mean()
    df['trend_up'] = df['ma_fast'] > df['ma_slow']
    df['trend_down'] = df['ma_fast'] < df['ma_slow']
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Volatility filter
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    df['low_vol'] = df['volatility_20'] < df['vol_median']
    
    # 4. ATR for stops
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 5. Entry conditions
    # Long: trend up AND RSI < 40 (oversold within uptrend) AND low volatility
    long_entry = (df['trend_up']) & (df['rsi'] < 40) & (df['low_vol'])
    # Short: trend down AND RSI > 60 (overbought within downtrend) AND low volatility
    short_entry = (df['trend_down']) & (df['rsi'] > 60) & (df['low_vol'])
    
    # 6. Raw signals (no cooldown)
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 7. Position management with fixed stop/target
    df['signal'] = 0
    position = 0
    entry_price = 0.0
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
                stop_price = close - 1.5 * atr
                target_price = close + 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + 1.5 * atr
                target_price = close - 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Check exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                # Allow immediate re-entry if condition still holds
                if raw == 1:
                    position = 1
                    entry_price = close
                    stop_price = close - 1.5 * atr
                    target_price = close + 3.0 * atr
                    df.iloc[i, df.columns.get_loc('signal')] = 1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
        elif position == -1:
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                if raw == -1:
                    position = -1
                    entry_price = close
                    stop_price = close + 1.5 * atr
                    target_price = close - 3.0 * atr
                    df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
    
    return df

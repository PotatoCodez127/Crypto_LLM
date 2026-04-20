import pandas as pd
import numpy as np

def get_signals(df):
    """
    MACD + RSI strategy with adaptive ATR stop-loss.
    - Long when MACD histogram > 0.0005, RSI > 60, close > SMA50, and SMA50 > SMA100
    - Short when MACD histogram < -0.0005, RSI < 40, close < SMA50, and SMA50 < SMA100
    - Stop-loss with dynamic ATR multiplier (1.2-3.0) based on recent volatility.
    Added cooldown period of 5 candles after exit to reduce overtrading.
    """
    # 1. Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    df['macd_hist'] = histogram
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR calculation
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr'] = df['atr'].bfill().fillna(0)

    # SMA for trend filter (use 50 and 100 for stronger trend confirmation)
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma50'] = df['sma50'].bfill().fillna(df['close'])
    # 100-day SMA for longer trend
    df['sma100'] = df['close'].rolling(window=100).mean()
    df['sma100'] = df['sma100'].bfill().fillna(df['close'])
    # Volume filter removed
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ma20'] = df['volume_ma20'].bfill().fillna(df['volume'])

    # Volatility for adaptive stop
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std().bfill().fillna(0)
    df['vol_median'] = df['volatility'].rolling(window=100).median().bfill().fillna(df['volatility'])

    # 2. Generate raw signals with more selective conditions
    df['raw_signal'] = 0
    # No volume filter
    volume_long = True
    volume_short = True
    # Higher MACD threshold to reduce false signals
    macd_threshold = 0.0005
    # Stricter RSI thresholds to require stronger momentum
    long_condition = (df['macd_hist'] > macd_threshold) & (df['rsi'] > 58) & (df['close'] > df['sma50']) & (df['sma50'] > df['sma100']) & volume_long
    short_condition = (df['macd_hist'] < -macd_threshold) & (df['rsi'] < 42) & (df['close'] < df['sma50']) & (df['sma50'] < df['sma100']) & volume_short
    # No cooldown period to allow consecutive signals
    df.loc[long_condition, 'raw_signal'] = 1
    df.loc[short_condition, 'raw_signal'] = -1

    # 3. Apply trailing stop-loss with adaptive ATR multiplier and cooldown period
    df['signal'] = 0
    position = 0  # 0: flat, 1: long, -1: short
    stop_price = 0.0
    cooldown = 0  # candles remaining in cooldown
    # atr_multiplier will be set dynamically per candle

    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        vol = df['volatility'].iloc[i]
        vol_med = df['vol_median'].iloc[i]
        # Dynamic ATR multiplier based on volatility - tighter stops to improve risk management
        if vol_med > 0:
            atr_multiplier = 1.5 * (vol / vol_med)
            atr_multiplier = max(1.2, min(2.5, atr_multiplier))
        else:
            atr_multiplier = 1.8

        if cooldown > 0:
            cooldown -= 1
            df.iloc[i, df.columns.get_loc('signal')] = 0
            continue

        if position == 0:
            if raw == 1:
                position = 1
                stop_price = close - atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif raw == -1:
                position = -1
                stop_price = close + atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Update trailing stop (only move up)
            new_stop = close - atr_multiplier * atr
            if new_stop > stop_price:
                stop_price = new_stop
            # Check stop loss
            if close <= stop_price:
                position = 0
                cooldown = 5  # enter cooldown after stop loss
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            if new_stop < stop_price:
                stop_price = new_stop
            if close >= stop_price:
                position = 0
                cooldown = 5
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1

    return df

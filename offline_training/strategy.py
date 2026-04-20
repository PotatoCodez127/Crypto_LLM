import pandas as pd
import numpy as np

def get_signals(df):
    """
    MACD + RSI strategy with adaptive ATR stop-loss.
    - Long when MACD histogram > 0.0005, RSI > 55, close > SMA200, and SMA50 > SMA200
    - Short when MACD histogram < -0.0005, RSI < 45, close < SMA200, and SMA50 < SMA200
    - Stop-loss with dynamic ATR multiplier (1.0-3.0) based on recent volatility.
    Experiment3: adaptive ATR multiplier.
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
    df['atr'] = df['atr'].fillna(method='bfill').fillna(0)

    # SMA for trend filter
    df['sma200'] = df['close'].rolling(window=200).mean()
    df['sma200'] = df['sma200'].fillna(method='bfill').fillna(df['close'])
    # Momentum filter
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma50'] = df['sma50'].fillna(method='bfill').fillna(df['close'])
    # Volume filter
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ma20'] = df['volume_ma20'].fillna(method='bfill').fillna(df['volume'])

    # Volatility for adaptive stop
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std().fillna(method='bfill').fillna(0)
    df['vol_median'] = df['volatility'].rolling(window=200).median().fillna(method='bfill').fillna(df['volatility'])

    # 2. Generate raw signals
    df['raw_signal'] = 0
    long_condition = (df['macd_hist'] > 0.0005) & (df['rsi'] > 55) & (df['close'] > df['sma200']) & (df['sma50'] > df['sma200'])
    short_condition = (df['macd_hist'] < -0.0005) & (df['rsi'] < 45) & (df['close'] < df['sma200']) & (df['sma50'] < df['sma200'])
    df.loc[long_condition, 'raw_signal'] = 1
    df.loc[short_condition, 'raw_signal'] = -1

    # 3. Apply trailing stop-loss with adaptive ATR multiplier
    df['signal'] = 0
    position = 0  # 0: flat, 1: long, -1: short
    stop_price = 0.0
    # atr_multiplier will be set dynamically per candle

    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        vol = df['volatility'].iloc[i]
        vol_med = df['vol_median'].iloc[i]
        # Dynamic ATR multiplier based on volatility
        if vol_med > 0:
            atr_multiplier = 2.0 * (vol / vol_med)
            atr_multiplier = max(1.0, min(3.0, atr_multiplier))
        else:
            atr_multiplier = 2.0

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
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            if new_stop < stop_price:
                stop_price = new_stop
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1

    return df

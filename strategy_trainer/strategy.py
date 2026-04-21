import pandas as pd
import numpy as np

def get_signals(df):
    """
    ADAPTIVE MOMENTUM STRATEGY:
    - Use EMA for trend (fast=8, slow=21) for quicker response.
    - RSI thresholds widened to 30/70 to capture more opportunities.
    - Volume confirmation: require volume > 20-period average.
    - MACD histogram positive for long, negative for short as additional filter.
    - Volatility filter: only trade when recent volatility < 2 * 100-period median (relaxed).
    - Remove ADX requirement to allow trades in weaker trends.
    - Dynamic position sizing based on volatility: size = min(1.0, 0.5 / (atr_20 / close)).
    - Trailing stop at 1.5*ATR, take-profit at 2.5*ATR.
    """
    # 1. Trend: exponential moving averages
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['trend_up'] = df['ema_fast'] > df['ema_slow']
    df['trend_down'] = df['ema_fast'] < df['ema_slow']
    
    # 2. RSI (14) with widened thresholds
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Volume confirmation
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ok'] = df['volume'] > df['volume_ma']
    else:
        df['volume_ok'] = True  # if volume data missing, ignore
    
    # 4. MACD histogram
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    df['macd_hist'] = histogram
    df['macd_bull'] = df['macd_hist'] > 0
    df['macd_bear'] = df['macd_hist'] < 0
    
    # 5. Volatility filter (relaxed)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    df['vol_ok'] = df['volatility_20'] < (df['vol_median'] * 2.0)
    
    # 6. ATR for stops and position sizing
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 7. Dynamic position sizing
    df['atr_pct'] = df['atr'] / df['close']
    # size = min(1.0, 0.5 / (atr_pct + 1e-8))
    df['position_size'] = np.minimum(1.0, 0.5 / (df['atr_pct'] + 1e-8))
    
    # 8. Entry conditions
    # Long: trend up AND RSI < 30 (oversold) AND volume ok AND volatility ok AND MACD histogram positive
    long_entry = (df['trend_up']) & (df['rsi'] < 30) & (df['volume_ok']) & (df['vol_ok']) & (df['macd_bull'])
    # Short: trend down AND RSI > 70 (overbought) AND volume ok AND volatility ok AND MACD histogram negative
    short_entry = (df['trend_down']) & (df['rsi'] > 70) & (df['volume_ok']) & (df['vol_ok']) & (df['macd_bear'])
    
    # 9. Raw signals (no cooldown)
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 10. Position management with trailing stop and fixed target, incorporating dynamic sizing
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
        size = df['position_size'].iloc[i]
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                extreme_price = close
                stop_price = close - 1.5 * atr
                target_price = close + 2.5 * atr
                position_size = size
                df.iloc[i, df.columns.get_loc('signal')] = position_size
            elif raw == -1:
                position = -1
                entry_price = close
                extreme_price = close
                stop_price = close + 1.5 * atr
                target_price = close - 2.5 * atr
                position_size = size
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Update extreme price
            if close > extreme_price:
                extreme_price = close
                # Adjust trailing stop
                stop_price = extreme_price - 1.5 * atr
            # Check exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                # Allow immediate re-entry if condition still holds
                if raw == 1:
                    position = 1
                    entry_price = close
                    extreme_price = close
                    stop_price = close - 1.5 * atr
                    target_price = close + 2.5 * atr
                    position_size = size
                    df.iloc[i, df.columns.get_loc('signal')] = position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = position_size
        elif position == -1:
            if close < extreme_price:
                extreme_price = close
                stop_price = extreme_price + 1.5 * atr
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                if raw == -1:
                    position = -1
                    entry_price = close
                    extreme_price = close
                    stop_price = close + 1.5 * atr
                    target_price = close - 2.5 * atr
                    position_size = size
                    df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
    
    return df

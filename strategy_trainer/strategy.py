import pandas as pd
import numpy as np

def get_signals(df):
    """
    ADAPTIVE MOMENTUM STRATEGY:
    - EMA crossover (fast=8, slow=21) for trend direction.
    - Adaptive RSI thresholds based on volatility (40-60 dynamic).
    - MACD histogram momentum filter.
    - Volume confirmation via OBV trend.
    - ATR-based stop-loss (1.5*ATR) and take-profit (2.5*ATR) with trailing after 1 ATR move.
    - Max holding period of 10 bars to cut losers.
    - Cooldown period of 3 bars after exit to avoid whipsaw.
    - Position sizing: fixed 1.0 (but can be scaled by volatility).
    """
    # 1. Trend: Exponential Moving Averages
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['trend_up'] = df['ema_fast'] > df['ema_slow']
    df['trend_down'] = df['ema_fast'] < df['ema_slow']
    
    # 2. RSI (14) with adaptive thresholds
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Volatility factor: recent ATR relative to its 50-period median
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    atr_median = df['atr'].rolling(window=50).median()
    vol_factor = np.where(atr_median > 0, df['atr'] / atr_median, 1.0)
    # Dynamic thresholds: widen in high volatility, narrow in low volatility
    oversold = 40 - 5 * vol_factor
    overbought = 60 + 5 * vol_factor
    df['rsi_oversold'] = np.clip(oversold, 30, 45)
    df['rsi_overbought'] = np.clip(overbought, 55, 75)
    
    # 3. MACD histogram for momentum
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    
    # 4. Volume confirmation: OBV trend
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_slope'] = df['obv'].diff().rolling(window=5).mean()
    
    # 5. Volatility filter (relaxed)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    df['low_vol'] = df['volatility_20'] < (df['vol_median'] * 2.0)  # more permissive
    
    # 6. Entry conditions
    # Long: uptrend, RSI below dynamic oversold, MACD histogram positive, OBV rising, low volatility
    long_entry = (
        df['trend_up'] &
        (df['rsi'] < df['rsi_oversold']) &
        (df['macd_hist'] > 0) &
        (df['obv_slope'] > 0) &
        df['low_vol']
    )
    # Short: downtrend, RSI above dynamic overbought, MACD histogram negative, OBV falling, low volatility
    short_entry = (
        df['trend_down'] &
        (df['rsi'] > df['rsi_overbought']) &
        (df['macd_hist'] < 0) &
        (df['obv_slope'] < 0) &
        df['low_vol']
    )
    
    # 7. Raw signals (no cooldown yet)
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 8. Position management with improved exit logic
    df['signal'] = 0
    position = 0
    entry_price = 0.0
    extreme_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    bars_in_trade = 0
    cooldown = 0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        # Apply cooldown
        if cooldown > 0:
            cooldown -= 1
            raw = 0  # ignore signals during cooldown
        
        if position == 0:
            if raw == 1 and cooldown == 0:
                position = 1
                entry_price = close
                extreme_price = close
                stop_price = close - 1.5 * atr
                target_price = close + 2.5 * atr
                bars_in_trade = 1
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
            elif raw == -1 and cooldown == 0:
                position = -1
                entry_price = close
                extreme_price = close
                stop_price = close + 1.5 * atr
                target_price = close - 2.5 * atr
                bars_in_trade = 1
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            bars_in_trade += 1
            # Update extreme price for trailing stop
            if close > extreme_price:
                extreme_price = close
                # Start trailing after 1 ATR move in profit
                if close - entry_price > atr:
                    stop_price = max(stop_price, extreme_price - 1.5 * atr)
            # Exit conditions: stop, target, max bars, opposite signal
            if (close <= stop_price) or (close >= target_price) or (bars_in_trade > 10):
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 3
                # Allow immediate re-entry only if cooldown not applied
                if raw == 1 and cooldown == 0:
                    position = 1
                    entry_price = close
                    extreme_price = close
                    stop_price = close - 1.5 * atr
                    target_price = close + 2.5 * atr
                    bars_in_trade = 1
                    df.iloc[i, df.columns.get_loc('signal')] = 1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
        elif position == -1:
            bars_in_trade += 1
            if close < extreme_price:
                extreme_price = close
                if entry_price - close > atr:
                    stop_price = min(stop_price, extreme_price + 1.5 * atr)
            if (close >= stop_price) or (close <= target_price) or (bars_in_trade > 10):
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 3
                if raw == -1 and cooldown == 0:
                    position = -1
                    entry_price = close
                    extreme_price = close
                    stop_price = close + 1.5 * atr
                    target_price = close - 2.5 * atr
                    bars_in_trade = 1
                    df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
    
    return df

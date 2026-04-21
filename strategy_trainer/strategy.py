import pandas as pd
import numpy as np
from src.features.extractor import FeatureExtractor

def get_signals(df):
    """
    PIVOT: Trend‑following with ADX + MACD, CVD for order flow confirmation.
    Avoid RSI divergence and low‑volatility mean reversion that failed previously.
    Use dynamic position sizing based on volatility regime.
    """
    # 1. Compute volatility (ATR) and log returns
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    df['high_vol_regime'] = df['volatility_20'] > df['vol_median']
    
    # 2. Trend strength (ADX)
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr_s = tr.rolling(window=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / (tr_s + 1e-8))
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / (tr_s + 1e-8))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(window=14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # 3. Momentum (MACD)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. Order flow (CVD approximation using FeatureExtractor)
    # Ensure we have required columns; if not compute simple CVD
    if 'volume' in df.columns and 'close' in df.columns and 'open' in df.columns:
        # Simple CVD: volume signed by close vs open
        df['cvd'] = np.where(df['close'] >= df['open'], df['volume'], -df['volume'])
        df['cvd_cum'] = df['cvd'].cumsum()
        # Normalize CVD using rolling z-score
        cvd_mean = df['cvd_cum'].rolling(window=50).mean()
        cvd_std = df['cvd_cum'].rolling(window=50).std()
        df['cvd_z'] = (df['cvd_cum'] - cvd_mean) / (cvd_std + 1e-8)
    else:
        df['cvd_z'] = 0.0
    
    # 5. Volume profile: volume relative to rolling median
    df['volume_median'] = df['volume'].rolling(window=50).median()
    df['volume_ratio'] = df['volume'] / (df['volume_median'] + 1e-8)
    volume_spike = df['volume_ratio'] > 1.8
    
    # 6. Entry logic
    # Long: ADX > 25 (trending), +DI > -DI (uptrend), MACD histogram > 0, CVD positive, volume spike
    long_condition = (
        (df['adx'] > 25) &
        (df['plus_di'] > df['minus_di']) &
        (df['macd_hist'] > 0) &
        (df['cvd_z'] > 0) &
        volume_spike
    )
    # Short: ADX > 25, -DI > +DI (downtrend), MACD histogram < 0, CVD negative, volume spike
    short_condition = (
        (df['adx'] > 25) &
        (df['minus_di'] > df['plus_di']) &
        (df['macd_hist'] < 0) &
        (df['cvd_z'] < 0) &
        volume_spike
    )
    
    # 7. Cooldown and raw signals (simpler, no complex loops)
    df['raw_signal'] = 0
    cooldown = 10  # Shorter cooldown for trend following
    last_long = -cooldown
    last_short = -cooldown
    for i in range(len(df)):
        if long_condition.iloc[i] and (i - last_long >= cooldown):
            df.iloc[i, df.columns.get_loc('raw_signal')] = 1
            last_long = i
        elif short_condition.iloc[i] and (i - last_short >= cooldown):
            df.iloc[i, df.columns.get_loc('raw_signal')] = -1
            last_short = i
    
    # 8. Position management with dynamic sizing and trailing stop
    df['signal'] = 0
    position = 0
    stop_price = 0.0
    entry_price = 0.0
    position_size = 1.0  # base size
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        high_vol = df['high_vol_regime'].iloc[i]
        
        # Adjust position size based on volatility regime
        if high_vol:
            position_size = 0.7  # reduce size in high volatility
            atr_multiplier = 2.2
        else:
            position_size = 1.0
            atr_multiplier = 1.8
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = position_size  # positive for long
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -position_size  # negative for short
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Update trailing stop
            new_stop = close - atr_multiplier * atr
            # Move stop up only if new_stop is higher than current stop and above breakeven
            if new_stop > stop_price:
                stop_price = new_stop
            # Take profit at 2.5*ATR, then tighten stop to entry + 0.5*ATR
            if close >= entry_price + 2.5 * atr:
                stop_price = max(stop_price, entry_price + 0.5 * atr)
            # Exit condition
            if close <= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = position_size
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            if new_stop < stop_price:
                stop_price = new_stop
            if close <= entry_price - 2.5 * atr:
                stop_price = min(stop_price, entry_price - 0.5 * atr)
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
    
    return df

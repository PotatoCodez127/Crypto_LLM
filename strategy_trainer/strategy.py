import pandas as pd
import numpy as np

def get_signals(df):
    """
    PIVOT APPROACH: Combine trend momentum with mean reversion.
    Use ADX for trend strength, MACD for momentum direction, and RSI for overbought/oversold.
    Entry only when trend is weak (ADX < 25) for mean reversion, or strong trend (ADX > 30) for momentum.
    """
    # 1. Volatility
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    df['high_vol_regime'] = df['volatility_20'] > df['vol_median']
    df['low_vol_regime'] = df['volatility_20'] < df['vol_median'] * 0.7
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_smooth'] = df['rsi'].rolling(window=5).mean()
    
    # 3. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. ADX (simplified)
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(window=14).mean()
    
    # 5. Volume spike
    df['volume_median'] = df['volume'].rolling(window=50).median()
    df['volume_ratio'] = df['volume'] / (df['volume_median'] + 1e-8)
    volume_spike = df['volume_ratio'] > 1.8
    
    # 6. Price distance from moving average
    sma_50 = df['close'].rolling(window=50).mean()
    df['price_dev'] = (df['close'] - sma_50) / (df['close'].rolling(window=50).std() + 1e-8)
    
    # 7. Entry conditions
    # Mean reversion: low ADX (<25), RSI extreme, price deviation extreme
    mean_rev_long = (
        (df['adx'] < 25) &
        (df['rsi_smooth'] < 35) &
        (df['price_dev'] < -1.2) &
        volume_spike &
        df['low_vol_regime']
    )
    mean_rev_short = (
        (df['adx'] < 25) &
        (df['rsi_smooth'] > 65) &
        (df['price_dev'] > 1.2) &
        volume_spike &
        df['low_vol_regime']
    )
    # Momentum: strong ADX (>30), MACD histogram positive/negative, RSI not extreme
    mom_long = (
        (df['adx'] > 30) &
        (df['macd_hist'] > 0) &
        (df['rsi_smooth'] > 40) &
        (df['rsi_smooth'] < 70) &
        volume_spike
    )
    mom_short = (
        (df['adx'] > 30) &
        (df['macd_hist'] < 0) &
        (df['rsi_smooth'] < 60) &
        (df['rsi_smooth'] > 30) &
        volume_spike
    )
    
    long_condition = mean_rev_long | mom_long
    short_condition = mean_rev_short | mom_short
    
    # 8. Cooldown and raw signal
    df['raw_signal'] = 0
    cooldown = 10
    last_signal_idx = -cooldown
    for i in range(len(df)):
        if i < last_signal_idx + cooldown:
            continue
        if long_condition.iloc[i]:
            df.iloc[i, df.columns.get_loc('raw_signal')] = 1
            last_signal_idx = i
        elif short_condition.iloc[i]:
            df.iloc[i, df.columns.get_loc('raw_signal')] = -1
            last_signal_idx = i
    
    # 9. Risk management with dynamic position sizing based on volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    df['signal'] = 0
    position = 0
    stop_price = 0.0
    entry_price = 0.0
    position_size = 1.0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        vol_regime = df['high_vol_regime'].iloc[i]
        
        # Position sizing: smaller size in high volatility
        if vol_regime:
            atr_multiplier = 2.0
            position_size = 0.7
        else:
            atr_multiplier = 1.2
            position_size = 1.0
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = position_size
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Trailing stop
            new_stop = close - atr_multiplier * atr
            # Take profit at 2*ATR, then trail at 1*ATR
            take_profit_level = entry_price + 2.0 * atr
            if close >= take_profit_level:
                stop_price = max(stop_price, entry_price + 1.0 * atr)
            if new_stop > stop_price:
                stop_price = new_stop
            if close <= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = position_size
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            take_profit_level = entry_price - 2.0 * atr
            if close <= take_profit_level:
                stop_price = min(stop_price, entry_price - 1.0 * atr)
            if new_stop < stop_price:
                stop_price = new_stop
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
    
    return df

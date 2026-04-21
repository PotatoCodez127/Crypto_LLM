import pandas as pd
import numpy as np

def get_signals(df):
    """
    TREND FOLLOWING WITH ADAPTIVE EXITS
    Use EMA crossover for trend direction, RSI for overbought/oversold filter,
    volume confirmation, and ATR-based dynamic stops.
    """
    # 1. Trend indicators
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_trend'] = df['ema_fast'] - df['ema_slow']
    
    # 2. RSI for momentum extremes
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Volume confirmation
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    volume_ok = df['volume_ratio'] > 1.0
    
    # 4. Volatility regime (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    df['atr_pct'] = df['atr'] / df['close'].rolling(window=14).mean()
    
    # 5. Market regime filter using volatility bands
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['vol_median'] = df['volatility'].rolling(window=100).median()
    df['high_vol'] = df['volatility'] > df['vol_median']
    df['low_vol'] = df['volatility'] < df['vol_median'] * 0.8
    
    # 6. Entry conditions
    # Long: EMA fast above slow, RSI not overbought (<70), volume confirmation
    long_entry = (
        (df['ema_trend'] > 0) &
        (df['ema_trend'].shift(1) <= 0) &  # crossover up
        (df['rsi'] < 70) &
        volume_ok &
        (~df['high_vol'])  # avoid high volatility regimes
    )
    # Short: EMA fast below slow, RSI not oversold (>30), volume confirmation
    short_entry = (
        (df['ema_trend'] < 0) &
        (df['ema_trend'].shift(1) >= 0) &  # crossover down
        (df['rsi'] > 30) &
        volume_ok &
        (~df['high_vol'])
    )
    
    # 7. Raw signals with cooldown
    df['raw_signal'] = 0
    cooldown = 15
    last_signal_idx = -cooldown
    for i in range(len(df)):
        if i < last_signal_idx + cooldown:
            continue
        if long_entry.iloc[i]:
            df.iloc[i, df.columns.get_loc('raw_signal')] = 1
            last_signal_idx = i
        elif short_entry.iloc[i]:
            df.iloc[i, df.columns.get_loc('raw_signal')] = -1
            last_signal_idx = i
    
    # 8. Position management with adaptive stops and take profit
    df['signal'] = 0
    position = 0
    stop_price = 0.0
    entry_price = 0.0
    position_size = 1.0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        atr_pct = df['atr_pct'].iloc[i]
        high_vol = df['high_vol'].iloc[i]
        low_vol = df['low_vol'].iloc[i]
        
        # Dynamic position sizing based on volatility
        if high_vol:
            position_size = 0.5
            stop_multiplier = 2.5
            profit_multiplier = 3.0
        elif low_vol:
            position_size = 1.2
            stop_multiplier = 1.0
            profit_multiplier = 2.0
        else:
            position_size = 1.0
            stop_multiplier = 1.5
            profit_multiplier = 2.5
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - stop_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = position_size
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + stop_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Trailing stop logic
            new_stop = close - stop_multiplier * atr
            # Take profit level
            take_profit = entry_price + profit_multiplier * atr
            if close >= take_profit:
                # Move stop to breakeven + 0.5*ATR
                stop_price = max(stop_price, entry_price + 0.5 * atr)
            # Update trailing stop if better
            if new_stop > stop_price:
                stop_price = new_stop
            # Exit condition
            if close <= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = position_size
        elif position == -1:
            new_stop = close + stop_multiplier * atr
            take_profit = entry_price - profit_multiplier * atr
            if close <= take_profit:
                stop_price = min(stop_price, entry_price - 0.5 * atr)
            if new_stop < stop_price:
                stop_price = new_stop
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
    
    return df

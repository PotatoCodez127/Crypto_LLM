import pandas as pd
import numpy as np

def get_signals(df):
    """
    PIVOT: Trend-following with ADX, MACD, and Bollinger Bands.
    Avoid RSI divergence and low-vol mean reversion that likely caused past failures.
    Focus on strong trends with momentum confirmation.
    """
    # 1. ADX for trend strength
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([high - low, 
                    (high - close.shift()).abs(), 
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(window=14).mean()
    
    # 2. MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 3. Bollinger Bands
    window = 20
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    df['bb_upper'] = sma + 2 * std
    df['bb_lower'] = sma - 2 * std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    # 4. Volume confirmation (simple)
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
    
    # 5. Entry conditions
    # Long: ADX > 25 (strong trend), price above upper BB (breakout), MACD histogram > 0 (momentum up)
    long_condition = (
        (df['adx'] > 25) &
        (close > df['bb_upper']) &
        (df['macd_hist'] > 0) &
        (df['volume_ratio'] > 1.2)
    )
    # Short: ADX > 25, price below lower BB, MACD histogram < 0
    short_condition = (
        (df['adx'] > 25) &
        (close < df['bb_lower']) &
        (df['macd_hist'] < 0) &
        (df['volume_ratio'] > 1.2)
    )
    
    # 6. Raw signals without cooldown (allow consecutive signals if conditions persist)
    df['raw_signal'] = 0
    df.loc[long_condition, 'raw_signal'] = 1
    df.loc[short_condition, 'raw_signal'] = -1
    
    # 7. Simple position management with fixed stop and take profit
    df['signal'] = 0
    position = 0
    entry_price = 0.0
    stop_loss_pct = 0.02  # 2% stop loss
    take_profit_pct = 0.04  # 4% take profit
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close_price = df['close'].iloc[i]
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close_price
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif raw == -1:
                position = -1
                entry_price = close_price
                df.iloc[i, df.columns.get_loc('signal')] = -1
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Check exit conditions
            if close_price <= entry_price * (1 - stop_loss_pct) or close_price >= entry_price * (1 + take_profit_pct):
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        elif position == -1:
            if close_price >= entry_price * (1 + stop_loss_pct) or close_price <= entry_price * (1 - take_profit_pct):
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1
    
    return df

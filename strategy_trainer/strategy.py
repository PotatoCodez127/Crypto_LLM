import pandas as pd
import numpy as np

def get_signals(df):
    """
    PIVOTED STRATEGY: MACD + Bollinger Bands + Stochastic RSI + Volume confirmation.
    - Use EMA(12) and EMA(26) for MACD, signal line EMA(9).
    - Bollinger Bands (20,2) for volatility and mean reversion.
    - Stochastic RSI (14,3,3) for overbought/oversold.
    - Volume above 20-period average for confirmation.
    - Entry: MACD above signal & price below lower band & StochRSI < 0.2 & volume confirm -> long.
              MACD below signal & price above upper band & StochRSI > 0.8 & volume confirm -> short.
    - Exit: Fixed stop-loss (1.5*ATR) and take-profit (3*ATR) from entry.
    - Position sizing: inverse volatility scaling (0.5 to 1.5).
    - Cooldown: 5 bars after exit before new entry.
    """
    # 1. MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bull'] = df['macd'] > df['macd_signal']
    df['macd_bear'] = df['macd'] < df['macd_signal']
    
    # 2. Bollinger Bands
    bb_window = 20
    bb_std = 2
    df['bb_mid'] = df['close'].rolling(window=bb_window).mean()
    df['bb_std'] = df['close'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * bb_std)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['price_below_lower'] = df['close'] < df['bb_lower']
    df['price_above_upper'] = df['close'] > df['bb_upper']
    
    # 3. Stochastic RSI
    # RSI first
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    # Stochastic of RSI
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-8)
    df['stoch_rsi'] = stoch_rsi.rolling(window=3).mean()  # smooth
    df['stoch_oversold'] = df['stoch_rsi'] < 0.2
    df['stoch_overbought'] = df['stoch_rsi'] > 0.8
    
    # 4. Volume confirmation
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_high'] = df['volume'] > df['volume_sma']
    
    # 5. ATR for stops and sizing
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 6. Volatility scaling factor (0.5 to 1.5)
    vol_window = 50
    df['volatility'] = df['close'].pct_change().rolling(vol_window).std()
    vol_median = df['volatility'].median()
    df['size_factor'] = np.clip(vol_median / (df['volatility'] + 1e-8), 0.5, 1.5)
    
    # 7. Entry conditions
    long_entry = (df['macd_bull']) & (df['price_below_lower']) & (df['stoch_oversold']) & (df['volume_high'])
    short_entry = (df['macd_bear']) & (df['price_above_upper']) & (df['stoch_overbought']) & (df['volume_high'])
    
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 8. Position management with fixed stop/target and cooldown
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown = 0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        size_factor = df['size_factor'].iloc[i]
        
        # Decrease cooldown
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - 1.5 * atr
                target_price = close + 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * size_factor
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + 1.5 * atr
                target_price = close - 3.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * size_factor
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Check exit
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown = 5  # bars cooldown
                # No immediate re-entry
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * size_factor
        elif position == -1:
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown = 5
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * size_factor
        else:
            df.iloc[i, df.columns.get_loc('signal')] = 0.0
    
    return df

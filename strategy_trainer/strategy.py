import pandas as pd
import numpy as np

def get_signals(df):
    """
    ADAPTIVE MOMENTUM STRATEGY WITH BOLLINGER BANDS & MACD
    - MACD histogram for trend momentum (12,26,9).
    - RSI (14) with dynamic thresholds (30/70) for overbought/oversold.
    - Bollinger Bands (20,2) for volatility and exit signals.
    - ATR (14) for dynamic stop-loss and position sizing.
    - Cooldown period of 3 bars after exit to avoid whipsaw.
    - Position sizing based on 2% risk per trade (stop distance = 2.5*ATR).
    - Entry: MACD histogram >0 & RSI<30 & price below lower band for long.
    - Exit: trailing stop (2.5*ATR) or price crossing opposite band.
    """
    # 1. MACD (12,26,9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    df['macd_hist'] = macd_hist
    df['macd_trend_up'] = macd_hist > 0
    df['macd_trend_down'] = macd_hist < 0
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Bollinger Bands (20,2)
    bb_ma = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = bb_ma + 2 * bb_std
    df['bb_lower'] = bb_ma - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_ma
    
    # 4. ATR (14) for stops and sizing
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 5. Entry conditions
    # Long: MACD histogram positive, RSI < 30, price below lower Bollinger Band
    long_entry = (df['macd_hist'] > 0) & (df['rsi'] < 30) & (df['close'] < df['bb_lower'])
    # Short: MACD histogram negative, RSI > 70, price above upper Bollinger Band
    short_entry = (df['macd_hist'] < 0) & (df['rsi'] > 70) & (df['close'] > df['bb_upper'])
    
    # Raw signals
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 6. Position management with cooldown
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    extreme_price = 0.0
    stop_price = 0.0
    cooldown_counter = 0
    # Risk per trade (2% of capital) - we'll compute position size as fraction of capital
    # Assume capital = 1.0, risk per trade = 0.02, stop distance = 2.5 * ATR
    # position_size = risk_per_trade / (stop_distance * entry_price) but we'll keep simple fraction
    # We'll use fixed fractional sizing: 0.5 for normal, 1.0 if volatility low (bb_width < 0.05)
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]
        bb_width = df['bb_width'].iloc[i]
        
        # Cooldown logic
        if cooldown_counter > 0:
            cooldown_counter -= 1
            raw = 0  # ignore signals during cooldown
        
        if position == 0:
            if raw == 1 and cooldown_counter == 0:
                position = 1
                entry_price = close
                extreme_price = close
                stop_price = close - 2.5 * atr
                # Determine position size based on volatility
                pos_size = 0.5 if bb_width > 0.05 else 1.0
                df.iloc[i, df.columns.get_loc('signal')] = pos_size
            elif raw == -1 and cooldown_counter == 0:
                position = -1
                entry_price = close
                extreme_price = close
                stop_price = close + 2.5 * atr
                pos_size = 0.5 if bb_width > 0.05 else 1.0
                df.iloc[i, df.columns.get_loc('signal')] = -pos_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Update extreme price
            if close > extreme_price:
                extreme_price = close
                # Adjust trailing stop
                stop_price = extreme_price - 2.5 * atr
            # Exit conditions: stop loss, or price touches upper band
            if close <= stop_price or close >= bb_upper:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown_counter = 3  # 3-bar cooldown
                # Do not re-enter immediately
            else:
                # Maintain position with same size as entry
                pos_size = 0.5 if bb_width > 0.05 else 1.0
                df.iloc[i, df.columns.get_loc('signal')] = pos_size
        elif position == -1:
            if close < extreme_price:
                extreme_price = close
                stop_price = extreme_price + 2.5 * atr
            if close >= stop_price or close <= bb_lower:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown_counter = 3
            else:
                pos_size = 0.5 if bb_width > 0.05 else 1.0
                df.iloc[i, df.columns.get_loc('signal')] = -pos_size
    
    return df

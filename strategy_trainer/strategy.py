import pandas as pd
import numpy as np

def get_signals(df):
    """
    ENHANCED MOMENTUM STRATEGY:
    - Triple moving average system (8, 21, 55) for trend confirmation
    - MACD (12,26,9) for momentum signals
    - Stochastic RSI (14,3,3) for overbought/oversold levels
    - Volatility-adjusted position sizing: scale position based on ATR/price ratio
    - Dynamic stop-loss: trailing stop at 2*ATR from highest/lowest since entry
    - Take-profit at 2.5*ATR with partial exits
    - Add volume confirmation: require above-average volume for entries
    - Market regime filter: only trade when price above 200-period EMA (bullish regime)
    """
    # 1. Trend: triple moving averages
    df['ma_fast'] = df['close'].rolling(window=8).mean()
    df['ma_medium'] = df['close'].rolling(window=21).mean()
    df['ma_slow'] = df['close'].rolling(window=55).mean()
    df['trend_up'] = (df['ma_fast'] > df['ma_medium']) & (df['ma_medium'] > df['ma_slow'])
    df['trend_down'] = (df['ma_fast'] < df['ma_medium']) & (df['ma_medium'] < df['ma_slow'])
    
    # 2. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bull'] = df['macd'] > df['macd_signal']
    df['macd_bear'] = df['macd'] < df['macd_signal']
    
    # 3. Stochastic RSI
    # First calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Stochastic of RSI
    rsi_min = df['rsi'].rolling(window=14).min()
    rsi_max = df['rsi'].rolling(window=14).max()
    df['stoch_rsi'] = 100 * (df['rsi'] - rsi_min) / (rsi_max - rsi_min + 1e-8)
    df['stoch_rsi_k'] = df['stoch_rsi'].rolling(window=3).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
    
    # 4. ATR for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    df['atr_ratio'] = df['atr'] / df['close']
    
    # 5. Volume confirmation
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_high'] = df['volume'] > df['volume_sma']
    
    # 6. Market regime filter
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['bull_market'] = df['close'] > df['ema_200']
    
    # 7. Entry conditions
    # Long: uptrend confirmed + MACD bullish crossover + Stoch RSI oversold (<20) + volume confirmation + bull market
    long_entry = (
        df['trend_up'] & 
        df['macd_bull'] & 
        (df['stoch_rsi_k'] < 20) & 
        df['volume_high'] & 
        df['bull_market']
    )
    # Short: downtrend confirmed + MACD bearish crossover + Stoch RSI overbought (>80) + volume confirmation
    short_entry = (
        df['trend_down'] & 
        df['macd_bear'] & 
        (df['stoch_rsi_k'] > 80) & 
        df['volume_high']
    )
    
    # 8. Position sizing based on volatility
    df['position_size'] = np.where(df['atr_ratio'] < 0.02, 1.0,
                                   np.where(df['atr_ratio'] < 0.05, 0.7, 0.4))
    
    # 9. Raw signals
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = df.loc[long_entry, 'position_size']
    df.loc[short_entry, 'raw_signal'] = -df.loc[short_entry, 'position_size']
    
    # 10. Advanced position management with trailing stops and partial exits
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    trail_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    atr_at_entry = 0.0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if position == 0:
            if raw > 0:
                position = 1
                entry_price = close
                atr_at_entry = atr
                highest_since_entry = close
                trail_stop = close - 2.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = raw
            elif raw < 0:
                position = -1
                entry_price = close
                atr_at_entry = atr
                lowest_since_entry = close
                trail_stop = close + 2.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = raw
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Update highest price
            highest_since_entry = max(highest_since_entry, close)
            # Update trailing stop
            trail_stop = max(trail_stop, highest_since_entry - 2.0 * atr_at_entry)
            # Check exit conditions
            if close <= trail_stop or close >= entry_price + 2.5 * atr_at_entry:
                # Exit
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                # Check for immediate re-entry
                if raw > 0:
                    position = 1
                    entry_price = close
                    atr_at_entry = atr
                    highest_since_entry = close
                    trail_stop = close - 2.0 * atr
                    df.iloc[i, df.columns.get_loc('signal')] = raw
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * df['position_size'].iloc[i]
        elif position == -1:
            # Update lowest price
            lowest_since_entry = min(lowest_since_entry, close)
            # Update trailing stop
            trail_stop = min(trail_stop, lowest_since_entry + 2.0 * atr_at_entry)
            # Check exit conditions
            if close >= trail_stop or close <= entry_price - 2.5 * atr_at_entry:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                if raw < 0:
                    position = -1
                    entry_price = close
                    atr_at_entry = atr
                    lowest_since_entry = close
                    trail_stop = close + 2.0 * atr
                    df.iloc[i, df.columns.get_loc('signal')] = raw
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * df['position_size'].iloc[i]
    
    return df

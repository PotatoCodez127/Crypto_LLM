import pandas as pd
import numpy as np

def get_signals(df):
    """
    Trend Following + Momentum Strategy:
    - Use dual moving average crossover (fast=20, slow=50) for trend direction.
    - RSI (14) for momentum filter: avoid overbought in uptrend, oversold in downtrend.
    - Dynamic position sizing based on recent volatility (ATR).
    - Trailing stop-loss: 2*ATR from highest close since entry for longs, lowest for shorts.
    - No cooldown to capture more opportunities.
    - Exit on trend reversal or stop hit.
    """
    # 1. Moving averages
    fast_period = 20
    slow_period = 50
    df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
    df['trend_bull'] = df['fast_ma'] > df['slow_ma']
    df['trend_bear'] = df['fast_ma'] < df['slow_ma']
    
    # 2. RSI for momentum
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_overbought'] = df['rsi'] > 70
    df['rsi_oversold'] = df['rsi'] < 30
    
    # 3. ATR for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 4. Position sizing factor (0.5 to 2.0) based on inverse volatility
    vol_window = 50
    df['volatility'] = df['close'].pct_change().rolling(vol_window).std()
    vol_median = df['volatility'].median()
    df['size_factor'] = np.clip(vol_median / (df['volatility'] + 1e-8), 0.5, 2.0)
    
    # 5. Entry conditions
    # Long: trend bullish AND RSI not overbought (<70)
    long_entry = df['trend_bull'] & (~df['rsi_overbought'])
    # Short: trend bearish AND RSI not oversold (>30)
    short_entry = df['trend_bear'] & (~df['rsi_oversold'])
    
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 6. Position management with trailing stops
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    trail_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        size_factor = df['size_factor'].iloc[i]
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                highest_since_entry = close
                trail_stop = close - 2.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * size_factor
            elif raw == -1:
                position = -1
                entry_price = close
                lowest_since_entry = close
                trail_stop = close + 2.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * size_factor
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Update highest close since entry
            if close > highest_since_entry:
                highest_since_entry = close
                trail_stop = highest_since_entry - 2.0 * atr
            # Check exit: stop loss or trend reversal
            if close <= trail_stop or df['trend_bear'].iloc[i]:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * size_factor
        elif position == -1:
            if close < lowest_since_entry:
                lowest_since_entry = close
                trail_stop = lowest_since_entry + 2.0 * atr
            if close >= trail_stop or df['trend_bull'].iloc[i]:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * size_factor
        else:
            df.iloc[i, df.columns.get_loc('signal')] = 0.0
    
    return df

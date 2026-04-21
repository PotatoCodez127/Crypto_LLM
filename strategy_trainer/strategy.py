import pandas as pd
import numpy as np

def get_signals(df):
    """
    ADAPTIVE MOMENTUM STRATEGY:
    - Dual MA crossover (fast=15, slow=40) for trend.
    - RSI (14) with dynamic thresholds based on recent volatility.
    - Volatility filter: allow moderate volatility (below 75th percentile).
    - MACD histogram for momentum confirmation.
    - Trailing stop loss at 2*ATR, no fixed take-profit.
    - Position sizing inverse to volatility (max 1.0, min 0.3).
    - Re-entries allowed immediately.
    """
    # 1. Trend: moving averages
    df['ma_fast'] = df['close'].rolling(window=15).mean()
    df['ma_slow'] = df['close'].rolling(window=40).mean()
    df['trend_up'] = df['ma_fast'] > df['ma_slow']
    df['trend_down'] = df['ma_fast'] < df['ma_slow']
    
    # 2. RSI (14) with dynamic thresholds
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Volatility adaptive filter
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_75'] = df['volatility_20'].rolling(window=100).quantile(0.75)
    df['moderate_vol'] = df['volatility_20'] < df['vol_75']
    
    # 4. MACD for momentum
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bull'] = df['macd_hist'] > 0
    df['macd_bear'] = df['macd_hist'] < 0
    
    # 5. ATR for stops and sizing
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 6. Dynamic RSI thresholds based on volatility
    df['vol_rank'] = df['volatility_20'].rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    # When volatility is high, widen thresholds (more extreme RSI required)
    df['rsi_long_thresh'] = 40 - (df['vol_rank'] * 10)   # ranges 30-40
    df['rsi_short_thresh'] = 60 + (df['vol_rank'] * 10)  # ranges 60-70
    
    # 7. Entry conditions
    long_entry = (df['trend_up']) & (df['rsi'] < df['rsi_long_thresh']) & (df['moderate_vol']) & (df['macd_bull'])
    short_entry = (df['trend_down']) & (df['rsi'] > df['rsi_short_thresh']) & (df['moderate_vol']) & (df['macd_bear'])
    
    # 8. Position sizing based on volatility (inverse)
    df['position_size'] = 1.0 - (df['vol_rank'] * 0.7).clip(0.0, 0.7)  # between 0.3 and 1.0
    
    # 9. Raw signals
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 10. Position management with trailing stop
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    trailing_stop = 0.0
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
                trailing_stop = close - 2.0 * atr
                position_size = size
                df.iloc[i, df.columns.get_loc('signal')] = position_size
            elif raw == -1:
                position = -1
                entry_price = close
                trailing_stop = close + 2.0 * atr
                position_size = size
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Update trailing stop (only move up)
            new_stop = close - 2.0 * atr
            if new_stop > trailing_stop:
                trailing_stop = new_stop
            # Exit if hit stop
            if close <= trailing_stop:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                # Immediate re-entry check
                if raw == 1:
                    position = 1
                    entry_price = close
                    trailing_stop = close - 2.0 * atr
                    position_size = size
                    df.iloc[i, df.columns.get_loc('signal')] = position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = position_size
        elif position == -1:
            new_stop = close + 2.0 * atr
            if new_stop < trailing_stop:
                trailing_stop = new_stop
            if close >= trailing_stop:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                if raw == -1:
                    position = -1
                    entry_price = close
                    trailing_stop = close + 2.0 * atr
                    position_size = size
                    df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
    
    return df

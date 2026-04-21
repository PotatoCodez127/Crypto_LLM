import pandas as pd
import numpy as np

def get_signals(df):
    """
    ADAPTIVE DYNAMIC STRATEGY:
    - Adaptive moving average periods based on recent volatility (fast=10-30, slow=40-80).
    - Dynamic RSI thresholds using rolling percentiles (30th/70th) for regime adaptation.
    - Volatility filter using ATR ratio (current ATR vs median ATR) to avoid extreme volatility.
    - MACD histogram confirmation for trend strength.
    - Trailing stop loss (2*ATR) and take-profit (4*ATR) with breakeven adjustment.
    - Position sizing inversely proportional to volatility (max 1.0, min 0.3).
    - Cooldown of 5 bars after exit to avoid whipsaw.
    """
    # 1. Volatility adaptive periods
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std().fillna(0.01)
    # Normalize volatility to range 0-1 over last 200 bars
    vol_min = df['volatility'].rolling(window=200).min().fillna(0.01)
    vol_max = df['volatility'].rolling(window=200).max().fillna(0.5)
    vol_scaled = (df['volatility'] - vol_min) / (vol_max - vol_min + 1e-8)
    vol_scaled = vol_scaled.clip(0, 1)
    # Fast MA period between 10 and 30, slow between 40 and 80
    df['fast_period'] = (10 + (20 * vol_scaled)).astype(int)
    df['slow_period'] = (40 + (40 * vol_scaled)).astype(int)
    # Compute adaptive MAs
    df['ma_fast'] = df['close'].rolling(window=df['fast_period']).mean()
    df['ma_slow'] = df['close'].rolling(window=df['slow_period']).mean()
    df['trend_up'] = df['ma_fast'] > df['ma_slow']
    df['trend_down'] = df['ma_fast'] < df['ma_slow']
    
    # 2. RSI (14) with dynamic thresholds
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Rolling percentiles for thresholds
    df['rsi_lower'] = df['rsi'].rolling(window=100).quantile(0.30).fillna(30)
    df['rsi_upper'] = df['rsi'].rolling(window=100).quantile(0.70).fillna(70)
    
    # 3. MACD confirmation
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bull'] = df['macd_hist'] > 0
    df['macd_bear'] = df['macd_hist'] < 0
    
    # 4. Volatility filter using ATR ratio
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    df['atr_median'] = df['atr'].rolling(window=100).median()
    df['atr_ratio'] = df['atr'] / (df['atr_median'] + 1e-8)
    df['vol_ok'] = df['atr_ratio'] < 1.5  # avoid extreme volatility
    
    # 5. Entry conditions with multiple confirmations
    long_entry = (df['trend_up']) & (df['rsi'] < df['rsi_lower']) & (df['macd_bull']) & (df['vol_ok'])
    short_entry = (df['trend_down']) & (df['rsi'] > df['rsi_upper']) & (df['macd_bear']) & (df['vol_ok'])
    
    # 6. Raw signals
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 7. Position management with trailing stop and cooldown
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown = 0
    # Position sizing factor
    df['position_size'] = np.where(df['volatility'] > 0, 
                                   np.clip(0.5 / (df['volatility'] * np.sqrt(252)), 0.3, 1.0), 
                                   1.0)
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        size_factor = df['position_size'].iloc[i]
        
        if cooldown > 0:
            cooldown -= 1
            df.iloc[i, df.columns.get_loc('signal')] = 0
            continue
            
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - 2.0 * atr
                target_price = close + 4.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * size_factor
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + 2.0 * atr
                target_price = close - 4.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * size_factor
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Trailing stop: move stop up if price moves favorably
            if close > entry_price:
                # Move stop to breakeven + 0.5*ATR
                stop_price = max(stop_price, entry_price + 0.5 * atr)
            # Check exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 5  # bars cooldown
                # Allow immediate re-entry only after cooldown
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0 * size_factor
        elif position == -1:
            if close < entry_price:
                stop_price = min(stop_price, entry_price - 0.5 * atr)
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 5
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0 * size_factor
    
    return df

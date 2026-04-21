import pandas as pd
import numpy as np

def get_signals(df):
    """
    MOMENTUM REVERSAL STRATEGY:
    - MACD (12,26,9) signal line crossover for trend direction.
    - Stochastic RSI (14,3,3) for overbought/oversold levels.
    - Volume confirmation: require volume above 20-period average.
    - Volatility-adjusted position sizing: size = 1.0 / (volatility_20 * sqrt(20)).
    - Dynamic exit: trailing stop at 1.5*ATR, take-profit at 2.5*ATR.
    - Re-entry only after a cooldown period of 5 bars.
    """
    # 1. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['macd_trend_up'] = df['macd_diff'] > 0
    df['macd_trend_down'] = df['macd_diff'] < 0
    
    # 2. Stochastic RSI
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    # Stochastic of RSI
    rsi_low = rsi.rolling(window=14).min()
    rsi_high = rsi.rolling(window=14).max()
    df['stoch_rsi'] = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-8)
    df['stoch_rsi_k'] = df['stoch_rsi'].rolling(window=3).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
    
    # 3. Volume confirmation
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_high'] = df['volume'] > df['volume_sma']
    
    # 4. Volatility for position sizing
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    # Avoid division by zero
    df['position_size'] = 1.0 / (df['volatility_20'] * np.sqrt(20) + 1e-8)
    df['position_size'] = df['position_size'].clip(upper=2.0, lower=0.25)
    
    # 5. ATR for stops
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 6. Entry conditions
    # Long: MACD bullish crossover (diff > 0) AND Stoch RSI K < 30 AND volume confirmation
    long_entry = (df['macd_trend_up']) & (df['stoch_rsi_k'] < 30) & (df['volume_high'])
    # Short: MACD bearish crossover (diff < 0) AND Stoch RSI K > 70 AND volume confirmation
    short_entry = (df['macd_trend_down']) & (df['stoch_rsi_k'] > 70) & (df['volume_high'])
    
    # 7. Raw signals (with cooldown)
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 8. Position management with dynamic sizing, trailing stop, and cooldown
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    extreme_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown = 0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        pos_size = df['position_size'].iloc[i]
        
        # Decrease cooldown
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0:
            if raw == 1 and cooldown == 0:
                position = 1
                entry_price = close
                extreme_price = close
                stop_price = close - 1.5 * atr
                target_price = close + 2.5 * atr
                df.iloc[i, df.columns.get_loc('signal')] = pos_size
            elif raw == -1 and cooldown == 0:
                position = -1
                entry_price = close
                extreme_price = close
                stop_price = close + 1.5 * atr
                target_price = close - 2.5 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -pos_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
        elif position == 1:
            # Update extreme price
            if close > extreme_price:
                extreme_price = close
                # Adjust trailing stop
                stop_price = extreme_price - 1.5 * atr
            # Check exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown = 5
                # Allow immediate re-entry if condition still holds and cooldown is 0
                if raw == 1 and cooldown == 0:
                    position = 1
                    entry_price = close
                    extreme_price = close
                    stop_price = close - 1.5 * atr
                    target_price = close + 2.5 * atr
                    df.iloc[i, df.columns.get_loc('signal')] = pos_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = pos_size
        elif position == -1:
            if close < extreme_price:
                extreme_price = close
                stop_price = extreme_price + 1.5 * atr
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0.0
                cooldown = 5
                if raw == -1 and cooldown == 0:
                    position = -1
                    entry_price = close
                    extreme_price = close
                    stop_price = close + 1.5 * atr
                    target_price = close - 2.5 * atr
                    df.iloc[i, df.columns.get_loc('signal')] = -pos_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -pos_size
    
    return df

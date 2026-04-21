import pandas as pd
import numpy as np

def get_signals(df):
    """
    ADAPTIVE MOMENTUM-REVERSION HYBRID STRATEGY:
    - Triple moving average system (8, 21, 50) for multi-timeframe trend confirmation.
    - Stochastic RSI (14,3,3) with dynamic thresholds based on recent volatility.
    - Bollinger Bands (20,2) for mean reversion entries during low ADX periods.
    - Volatility-adjusted position sizing (0.5 to 1.5) using ATR/price ratio.
    - Dynamic stop-loss (1.5-3*ATR) and take-profit (2-5*ATR) based on market regime.
    - Cooldown period (5 bars) after exit to avoid whipsaw.
    - Volume confirmation (volume > 20-period average) for entries.
    """
    # 1. Multi-timeframe moving averages
    df['ma_fast'] = df['close'].rolling(window=8).mean()
    df['ma_medium'] = df['close'].rolling(window=21).mean()
    df['ma_slow'] = df['close'].rolling(window=50).mean()
    
    # Trend direction: fast above medium AND medium above slow = uptrend, opposite for downtrend
    df['trend_up'] = (df['ma_fast'] > df['ma_medium']) & (df['ma_medium'] > df['ma_slow'])
    df['trend_down'] = (df['ma_fast'] < df['ma_medium']) & (df['ma_medium'] < df['ma_slow'])
    df['trend_sideways'] = ~(df['trend_up'] | df['trend_down'])
    
    # 2. Stochastic RSI
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    df['stoch_rsi'] = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-8)
    df['stoch_rsi_k'] = df['stoch_rsi'].rolling(window=3).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
    
    # 3. Bollinger Bands
    bb_window = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    df['bb_std'] = df['close'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_middle'] + bb_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - bb_std * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # 4. ATR for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    df['atr_pct'] = df['atr'] / df['close']
    
    # 5. ADX for trend strength
    df['tr'] = tr
    df['atr'] = df['tr'].rolling(window=14).mean().bfill().fillna(0)
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    df['plus_dm'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    df['minus_dm'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr'])
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
    df['adx'] = df['dx'].rolling(window=14).mean()
    
    # 6. Volume confirmation
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_high'] = df['volume'] > df['volume_ma']
    
    # 7. Market regime detection
    df['volatility_regime'] = df['atr_pct'].rolling(window=50).rank(pct=True)
    # High volatility if above 70th percentile, low if below 30th
    df['high_vol'] = df['volatility_regime'] > 0.7
    df['low_vol'] = df['volatility_regime'] < 0.3
    
    # 8. Entry conditions
    # Long entries:
    # Condition A: Trend up + Stoch RSI K < 30 (oversold) + volume confirmation
    long_trend = df['trend_up'] & (df['stoch_rsi_k'] < 30) & df['volume_high']
    # Condition B: Sideways market + price near BB lower band (within 0.5 std) + ADX < 30 (weak trend)
    long_reversion = df['trend_sideways'] & (df['close'] <= df['bb_lower'] + 0.5*df['bb_std']) & (df['adx'] < 30)
    
    # Short entries:
    # Condition A: Trend down + Stoch RSI K > 70 (overbought) + volume confirmation
    short_trend = df['trend_down'] & (df['stoch_rsi_k'] > 70) & df['volume_high']
    # Condition B: Sideways market + price near BB upper band (within 0.5 std) + ADX < 30
    short_reversion = df['trend_sideways'] & (df['close'] >= df['bb_upper'] - 0.5*df['bb_std']) & (df['adx'] < 30)
    
    # Combine conditions
    long_entry = long_trend | long_reversion
    short_entry = short_trend | short_reversion
    
    # 9. Raw signals
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 10. Position management with adaptive sizing and cooldown
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    extreme_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown = 0
    position_size = 1.0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        atr_pct = df['atr_pct'].iloc[i]
        high_vol = df['high_vol'].iloc[i]
        
        # Adaptive position sizing: smaller in high volatility
        if high_vol:
            position_size = 0.5
        else:
            position_size = 1.0
        
        # Adaptive stop and target multipliers
        if high_vol:
            stop_mult = 3.0
            target_mult = 5.0
        else:
            stop_mult = 1.5
            target_mult = 2.0
        
        # Cooldown decrement
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0:
            if raw == 1 and cooldown == 0:
                position = 1
                entry_price = close
                extreme_price = close
                stop_price = close - stop_mult * atr
                target_price = close + target_mult * atr
                df.iloc[i, df.columns.get_loc('signal')] = position_size
            elif raw == -1 and cooldown == 0:
                position = -1
                entry_price = close
                extreme_price = close
                stop_price = close + stop_mult * atr
                target_price = close - target_mult * atr
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Update extreme price
            if close > extreme_price:
                extreme_price = close
                # Adjust trailing stop
                stop_price = max(stop_price, extreme_price - stop_mult * atr)
            # Check exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 5  # 5-bar cooldown
                # No immediate re-entry during cooldown
            else:
                df.iloc[i, df.columns.get_loc('signal')] = position_size
        elif position == -1:
            if close < extreme_price:
                extreme_price = close
                stop_price = min(stop_price, extreme_price + stop_mult * atr)
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 5
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -position_size
    
    return df

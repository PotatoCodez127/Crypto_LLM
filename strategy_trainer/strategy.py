import pandas as pd
import numpy as np

def get_signals(df):
    """
    PIVOT APPROACH: Use regime detection via volatility clustering and mean reversion extremes.
    Avoid repeating previous CVD/z-score combo. Instead, use RSI divergence + volume spike.
    """
    # 1. Log returns for volatility
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['vol_median'] = df['volatility_20'].rolling(window=100).median()
    
    # 2. RSI (14) with smoothing
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Smoothed RSI
    df['rsi_smooth'] = df['rsi'].rolling(window=5).mean()
    
    # 3. Volume spike detection (relative to rolling median)
    df['volume_median'] = df['volume'].rolling(window=50).median()
    df['volume_ratio'] = df['volume'] / (df['volume_median'] + 1e-8)
    volume_spike = df['volume_ratio'] > 2.0
    
    # 4. Price distance from rolling z-score (different window)
    sma_30 = df['close'].rolling(window=30).mean()
    std_30 = df['close'].rolling(window=30).std()
    df['z_30'] = (df['close'] - sma_30) / (std_30 + 1e-8)
    # Robust normalization using rolling IQR
    df['z_median'] = df['z_30'].rolling(window=50).median()
    df['z_q25'] = df['z_30'].rolling(window=50).quantile(0.25)
    df['z_q75'] = df['z_30'].rolling(window=50).quantile(0.75)
    df['z_iqr'] = df['z_q75'] - df['z_q25']
    df['z_norm'] = (df['z_30'] - df['z_median']) / (df['z_iqr'] + 1e-8)
    
    # 5. Regime detection: high volatility + trending vs low volatility + mean reversion
    df['high_vol_regime'] = df['volatility_20'] > df['vol_median']
    df['low_vol_regime'] = df['volatility_20'] < df['vol_median'] * 0.7
    
    # 6. RSI divergence: price makes new low but RSI does not (bullish) or vice versa
    df['price_low_20'] = df['low'].rolling(window=20).min()
    df['price_high_20'] = df['high'].rolling(window=20).max()
    df['rsi_low_20'] = df['rsi_smooth'].rolling(window=20).min()
    df['rsi_high_20'] = df['rsi_smooth'].rolling(window=20).max()
    
    bullish_div = (df['low'] <= df['price_low_20'].shift(1)) & (df['rsi_smooth'] > df['rsi_low_20'].shift(1))
    bearish_div = (df['high'] >= df['price_high_20'].shift(1)) & (df['rsi_smooth'] < df['rsi_high_20'].shift(1))
    
    # 7. Entry conditions
    # Long: RSI oversold (<30) + bullish divergence + volume spike + low volatility regime (mean reversion)
    long_condition = (
        (df['rsi_smooth'] < 30) &
        bullish_div &
        volume_spike &
        df['low_vol_regime'] &
        (df['z_norm'] < -1.0)
    )
    # Short: RSI overbought (>70) + bearish divergence + volume spike + low volatility regime
    short_condition = (
        (df['rsi_smooth'] > 70) &
        bearish_div &
        volume_spike &
        df['low_vol_regime'] &
        (df['z_norm'] > 1.0)
    )
    
    # 8. Cooldown and raw signal generation
    df['raw_signal'] = 0
    cooldown = 15  # Longer cooldown to avoid whipsaw
    last_signal_idx = -cooldown
    for i in range(len(df)):
        if i < last_signal_idx + cooldown:
            continue
        if long_condition.iloc[i]:
            df.iloc[i, df.columns.get_loc('raw_signal')] = 1
            last_signal_idx = i
        elif short_condition.iloc[i]:
            df.iloc[i, df.columns.get_loc('raw_signal')] = -1
            last_signal_idx = i
    
    # 9. Risk management: Adaptive ATR with regime‑based multiplier
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    df['signal'] = 0
    position = 0
    stop_price = 0.0
    entry_price = 0.0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        vol_regime = df['high_vol_regime'].iloc[i]
        
        # ATR multiplier depends on regime
        if vol_regime:
            atr_multiplier = 2.5  # wider stops in high volatility
        else:
            atr_multiplier = 1.5  # tighter stops in low volatility
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Trailing stop with profit lock
            new_stop = close - atr_multiplier * atr
            max_loss_stop = entry_price - 1.8 * atr  # wider max loss in high vol
            take_profit_level = entry_price + 3.0 * atr
            if close >= take_profit_level:
                stop_price = max(stop_price, entry_price + 1.0 * atr)
            if new_stop > stop_price and new_stop > max_loss_stop:
                stop_price = new_stop
            elif max_loss_stop > stop_price:
                stop_price = max_loss_stop
            if close <= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            max_loss_stop = entry_price + 1.8 * atr
            take_profit_level = entry_price - 3.0 * atr
            if close <= take_profit_level:
                stop_price = min(stop_price, entry_price - 1.0 * atr)
            if new_stop < stop_price and new_stop < max_loss_stop:
                stop_price = new_stop
            elif max_loss_stop < stop_price:
                stop_price = max_loss_stop
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1
    
    return df

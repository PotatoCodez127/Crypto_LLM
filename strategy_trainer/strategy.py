import pandas as pd
import numpy as np

def get_signals(df):
    """
    ADAPTIVE MULTI-FACTOR STRATEGY:
    - Hull Moving Average (9) for smooth trend direction.
    - RSI (14) with dynamic thresholds based on recent volatility.
    - ADX (14) filter: require trend strength > 25.
    - Bollinger Bands (20,2) for mean reversion entries within trend.
    - Volume confirmation: OBV divergence detection.
    - Dynamic stop-loss/take-profit based on ATR multiples adjusted by volatility regime.
    - Position sizing using Kelly fraction (risk 1% per trade).
    - Higher timeframe alignment (resampled to 4x period).
    - Trailing stop after 1.5*ATR profit.
    """
    # Ensure index is datetime for resampling
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 1. Hull Moving Average (fast)
    def hull_moving_average(series, window):
        wma_half = series.rolling(window=window//2).apply(lambda x: (x * np.arange(1, len(x)+1)).sum() / np.arange(1, len(x)+1).sum(), raw=False)
        wma_full = series.rolling(window=window).apply(lambda x: (x * np.arange(1, len(x)+1)).sum() / np.arange(1, len(x)+1).sum(), raw=False)
        hull_raw = 2 * wma_half - wma_full
        hull_ma = hull_raw.rolling(window=int(np.sqrt(window))).mean()
        return hull_ma
    df['hma'] = hull_moving_average(df['close'], 9)
    df['trend_up'] = df['close'] > df['hma']
    df['trend_down'] = df['close'] < df['hma']
    
    # 2. RSI with dynamic thresholds based on recent volatility
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Volatility-adjusted thresholds
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['vol_percentile'] = df['volatility'].rolling(100).rank(pct=True)
    # Lower threshold: 30 + 20 * vol_percentile (range 30-50)
    df['rsi_low_th'] = 30 + 20 * df['vol_percentile']
    # Upper threshold: 70 - 20 * vol_percentile (range 50-70)
    df['rsi_high_th'] = 70 - 20 * df['vol_percentile']
    
    # 3. ADX for trend strength
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = -minus_dm
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(14).mean()
    df['strong_trend'] = df['adx'] > 25
    
    # 4. Bollinger Bands
    bb_window = 20
    bb_std = 2.0
    df['bb_mid'] = df['close'].rolling(bb_window).mean()
    bb_std_series = df['close'].rolling(bb_window).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * bb_std_series
    df['bb_lower'] = df['bb_mid'] - bb_std * bb_std_series
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # 5. Volume confirmation (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    # OBV 5-period slope
    df['obv_slope'] = df['obv'].diff(5).rolling(5).mean()
    df['obv_bullish'] = df['obv_slope'] > 0
    df['obv_bearish'] = df['obv_slope'] < 0
    
    # 6. Higher timeframe alignment (4x period)
    # Resample to 4x bars (assuming input is 1h -> 4h)
    resample_factor = 4
    # Create temporary resampled close
    resampled = df['close'].resample(f'{resample_factor}H').ohlc()
    resampled_close = resampled['close']
    # Align back to original index using forward fill
    df['htf_close'] = resampled_close.reindex(df.index, method='ffill')
    df['htf_trend'] = df['htf_close'] > df['htf_close'].rolling(20).mean()
    
    # 7. ATR for stops
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 8. Entry conditions
    # Long: uptrend (HMA), RSI below dynamic low threshold, price near BB lower (within 0.5 std), strong trend, bullish OBV, HTF aligned
    long_entry = (
        df['trend_up'] &
        (df['rsi'] < df['rsi_low_th']) &
        (df['close'] <= (df['bb_lower'] + 0.5 * (df['bb_mid'] - df['bb_lower']))) &
        df['strong_trend'] &
        df['obv_bullish'] &
        df['htf_trend']
    )
    # Short: downtrend, RSI above dynamic high threshold, price near BB upper, strong trend, bearish OBV, HTF downtrend
    short_entry = (
        df['trend_down'] &
        (df['rsi'] > df['rsi_high_th']) &
        (df['close'] >= (df['bb_upper'] - 0.5 * (df['bb_upper'] - df['bb_mid']))) &
        df['strong_trend'] &
        df['obv_bearish'] &
        (~df['htf_trend'])
    )
    
    # 9. Raw signals
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 10. Position management with dynamic stops and trailing
    df['signal'] = 0.0
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    trailing_activated = False
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        volatility = df['volatility'].iloc[i]
        
        # Dynamic risk multiples based on volatility regime
        vol_regime = df['vol_percentile'].iloc[i]
        # Lower risk in high volatility (tighten stops), higher reward in low volatility
        stop_mult = 1.2 + 0.8 * (1 - vol_regime)  # 1.2 to 2.0
        target_mult = 2.5 + 2.0 * vol_regime      # 2.5 to 4.5
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - stop_mult * atr
                target_price = close + target_mult * atr
                trailing_activated = False
                # Position size based on Kelly (simplified: risk 1% per trade)
                risk = abs(close - stop_price) / close
                kelly_fraction = min(0.01 / (risk + 1e-8), 1.0)
                df.iloc[i, df.columns.get_loc('signal')] = kelly_fraction
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + stop_mult * atr
                target_price = close - target_mult * atr
                trailing_activated = False
                risk = abs(close - stop_price) / close
                kelly_fraction = min(0.01 / (risk + 1e-8), 1.0)
                df.iloc[i, df.columns.get_loc('signal')] = -kelly_fraction
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Check for trailing stop activation
            if not trailing_activated and close >= entry_price + 1.5 * atr:
                trailing_activated = True
            if trailing_activated:
                # Update stop to 0.5*ATR below highest close since activation
                # For simplicity, we'll just move stop to close - 0.5*ATR
                stop_price = max(stop_price, close - 0.5 * atr)
            # Exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                # Immediate re-entry allowed
                if raw == 1:
                    position = 1
                    entry_price = close
                    stop_price = close - stop_mult * atr
                    target_price = close + target_mult * atr
                    trailing_activated = False
                    risk = abs(close - stop_price) / close
                    kelly_fraction = min(0.01 / (risk + 1e-8), 1.0)
                    df.iloc[i, df.columns.get_loc('signal')] = kelly_fraction
            else:
                df.iloc[i, df.columns.get_loc('signal')] = df['signal'].iloc[i-1] if i>0 else 0
        elif position == -1:
            if not trailing_activated and close <= entry_price - 1.5 * atr:
                trailing_activated = True
            if trailing_activated:
                stop_price = min(stop_price, close + 0.5 * atr)
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                if raw == -1:
                    position = -1
                    entry_price = close
                    stop_price = close + stop_mult * atr
                    target_price = close - target_mult * atr
                    trailing_activated = False
                    risk = abs(close - stop_price) / close
                    kelly_fraction = min(0.01 / (risk + 1e-8), 1.0)
                    df.iloc[i, df.columns.get_loc('signal')] = -kelly_fraction
            else:
                df.iloc[i, df.columns.get_loc('signal')] = df['signal'].iloc[i-1] if i>0 else 0
    
    return df

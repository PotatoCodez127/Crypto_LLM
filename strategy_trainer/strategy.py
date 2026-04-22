import pandas as pd
import numpy as np

def get_signals(df):
    # 1. Indicators
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    # ATR for dynamic sizing
    high_low = df['high'] - df['low']
    df['atr'] = high_low.rolling(14).mean()

    # 2. Signals
    df['long_signal'] = (df['close'] > df['ema_200']) & (df['ema_9'] > df['ema_21'])
    df['short_signal'] = (df['close'] < df['ema_200']) & (df['ema_9'] < df['ema_21'])

    # 3. Execution (Iterative Loop)
    # Ensure you handle the 'Short' exit differently than the 'Long' exit
    # to account for the 'Dual-Velocity' nature of crypto.
    return df
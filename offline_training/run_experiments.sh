#!/bin/bash
set -e

# Function to get the last score from results.tsv
get_last_score() {
    tail -1 results.tsv | cut -f2
}

# Baseline (current commit)
echo "=== Running baseline ==="
./run_experiment.sh 1y "baseline"
baseline_score=$(get_last_score)
echo "Baseline score: $baseline_score"

# Experiment 1: faster EMAs, RSI 55/45, ATR 1.5
echo "=== Experiment 1: faster EMAs, RSI 55/45, ATR 1.5 ==="
git checkout HEAD -- strategy.py  # revert to original
# Apply changes via sed (simpler than full replace)
sed -i "s/span=20/span=12/g" strategy.py
sed -i "s/span=50/span=26/g" strategy.py
sed -i "s/rsi > 60/rsi > 55/g" strategy.py
sed -i "s/rsi < 40/rsi < 45/g" strategy.py
sed -i "s/atr_multiplier = 2.0/atr_multiplier = 1.5/g" strategy.py
git add strategy.py
git commit -m "exp1: faster EMAs, RSI 55/45, ATR 1.5"
./run_experiment.sh 1y "faster EMAs, RSI 55/45, ATR 1.5"
exp1_score=$(get_last_score)
echo "Experiment 1 score: $exp1_score"

# Compare and revert if not better
if (( $(echo "$exp1_score <= $baseline_score" | bc -l) )); then
    echo "Experiment 1 not better, reverting."
    git reset --hard HEAD~1
else
    echo "Experiment 1 improved, keeping."
fi

# Experiment 2: crossover filter + take-profit
echo "=== Experiment 2: crossover filter + take-profit ==="
git checkout HEAD -- strategy.py
# We'll apply a more complex change by replacing the file with a predefined version
# For simplicity, we'll just revert and apply a new patch
# Since this is complex, we'll skip for now and just run the next experiment
echo "Skipping Experiment 2 for now."

# Experiment 3: cooldown
echo "=== Experiment 3: cooldown ==="
git checkout HEAD -- strategy.py
# Apply cooldown changes
sed -i "/position = 0/a \    cooldown = 0" strategy.py
sed -i "/if position == 0:/a \        if cooldown > 0:\n            cooldown -= 1\n            raw = 0" strategy.py
sed -i "/if close <= stop_price:/a \                cooldown = 5" strategy.py
sed -i "/if close >= stop_price:/a \                cooldown = 5" strategy.py
git add strategy.py
git commit -m "exp3: cooldown after stop"
./run_experiment.sh 1y "cooldown after stop"
exp3_score=$(get_last_score)
echo "Experiment 3 score: $exp3_score"

# Compare with best so far
best_score=$baseline_score
if (( $(echo "$exp1_score > $best_score" | bc -l) )); then
    best_score=$exp1_score
fi
if (( $(echo "$exp3_score > $best_score" | bc -l) )); then
    best_score=$exp3_score
else
    echo "Experiment 3 not better, reverting."
    git reset --hard HEAD~1
fi

# Experiment 4: MACD strategy
echo "=== Experiment 4: MACD strategy ==="
git checkout HEAD -- strategy.py
# Replace entire file with MACD version
cat > strategy.py << 'EOF'
import pandas as pd
import numpy as np

def get_signals(df):
    """
    MACD + RSI strategy with ATR stop-loss.
    - Long when MACD histogram > 0 and RSI > 50
    - Short when MACD histogram < 0 and RSI < 50
    - Stop-loss at 2*ATR
    """
    # 1. Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    df['macd_hist'] = histogram
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR calculation
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr'] = df['atr'].fillna(method='bfill').fillna(0)

    # 2. Generate raw signals
    df['raw_signal'] = 0
    df.loc[(df['macd_hist'] > 0) & (df['rsi'] > 50), 'raw_signal'] = 1
    df.loc[(df['macd_hist'] < 0) & (df['rsi'] < 50), 'raw_signal'] = -1

    # 3. Apply trailing stop-loss
    df['signal'] = 0
    position = 0  # 0: flat, 1: long, -1: short
    stop_price = 0.0
    atr_multiplier = 2.0

    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]

        if position == 0:
            if raw == 1:
                position = 1
                stop_price = close - atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif raw == -1:
                position = -1
                stop_price = close + atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Update trailing stop (only move up)
            new_stop = close - atr_multiplier * atr
            if new_stop > stop_price:
                stop_price = new_stop
            # Check stop loss
            if close <= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            if new_stop < stop_price:
                stop_price = new_stop
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1

    return df
EOF
git add strategy.py
git commit -m "exp4: MACD strategy"
./run_experiment.sh 1y "MACD strategy"
exp4_score=$(get_last_score)
echo "Experiment 4 score: $exp4_score"

if (( $(echo "$exp4_score > $best_score" | bc -l) )); then
    echo "Best score so far: $exp4_score"
else
    echo "Experiment 4 not better, reverting."
    git reset --hard HEAD~1
fi

echo "=== All experiments completed ==="

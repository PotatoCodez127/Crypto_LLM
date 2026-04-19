import re
import sys
import os
from datetime import datetime

def parse_log(log_file='run.log'):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # Extract FINAL_RESULT
    final_match = re.search(r'FINAL_RESULT:([\d\.\-]+)', content)
    final_score = float(final_match.group(1)) if final_match else None
    # Extract metrics from performance report
    sharpe = re.search(r'Sharpe Ratio\s*:\s*([\d\.\-]+)', content)
    sharpe = float(sharpe.group(1)) if sharpe else None
    drawdown = re.search(r'Max Drawdown\s*:\s*([\d\.\-]+)%', content)
    drawdown = float(drawdown.group(1)) if drawdown else None
    trades = re.search(r'Total Trades\s*:\s*(\d+)', content)
    trades = int(trades.group(1)) if trades else None
    raw_score = re.search(r'RAW SCORE\s*:\s*([\d\.\-]+)', content)
    raw_score = float(raw_score.group(1)) if raw_score else None
    penalty = re.search(r'PENALTY MULTI\s*:\s*([\d\.\-]+)', content)
    penalty = float(penalty.group(1)) if penalty else None
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'final_score': final_score,
        'sharpe': sharpe,
        'drawdown': drawdown,
        'trades': trades,
        'raw_score': raw_score,
        'penalty': penalty
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, required=True, help='Change description')
    args = parser.parse_args()
    data = parse_log()
    if data['final_score'] is None:
        print("ERROR: Could not parse FINAL_RESULT")
        sys.exit(1)
    # Append to results.tsv
    with open('results.tsv', 'a', encoding='utf-8') as f:
        f.write(f"{data['timestamp']}\t{args.desc}\t{data['final_score']:.6f}\t{data['sharpe']:.4f}\t{data['drawdown']:.4f}\t{data['trades']}\n")
    print(f"Logged: {args.desc} score={data['final_score']:.6f}")

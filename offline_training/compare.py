import pandas as pd
import sys

def compare():
    try:
        df = pd.read_csv('results.tsv', sep='\t')
        if len(df) < 2:
            print("Need at least two results to compare.")
            return
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last['score'] > prev['score']:
            print(f"IMPROVED: {last['score']} > {prev['score']} - keep changes")
            # keep commit (do nothing)
        else:
            print(f"NO IMPROVEMENT: {last['score']} <= {prev['score']} - reverting")
            # revert the last commit
            import subprocess
            subprocess.run(['git', 'reset', '--hard', 'HEAD~1'], check=True)
            # also remove the last line from results.tsv? We'll keep for record.
    except Exception as e:
        print(f"Error comparing: {e}")

if __name__ == '__main__':
    compare()

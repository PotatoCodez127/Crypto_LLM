"""
AutoResearch system for cryptocurrency trading strategies.
This is the main module that gets modified during autoresearch experiments.
"""

import numpy as np
from autoresearch.strategy_evaluator import StrategyEvaluator, example_strategy


def baseline_strategy(features):
    """
    Baseline strategy based on multiple indicators.
    This is the starting point for autoresearch experiments.

    Args:
        features (dict): Feature vector with keys:
            - rsi: Relative Strength Index (0-100)
            - macd: MACD line value
            - atr: Average True Range (volatility)
            - volume_change: Volume change percentage
            - trend_slope: Price trend slope
            - bb_width: Bollinger Band width

    Returns:
        float: Position size (-1 to 1, where -1 is full short, 1 is full long, 0 is neutral)
    """
    rsi = features.get("rsi", 50)
    macd = features.get("macd", 0)
    trend_slope = features.get("trend_slope", 0)
    bb_width = features.get("bb_width", 0)

    # Start with neutral position
    position = 0.0

    # RSI signals
    if rsi < 30:
        position += 0.5  # Oversold, consider buying
    elif rsi > 70:
        position -= 0.5  # Overbought, consider selling

    # MACD signals
    if macd > 0:
        position += 0.3  # Positive MACD, bullish
    else:
        position -= 0.3  # Negative MACD, bearish

    # Trend signals
    if trend_slope > 0:
        position += 0.2  # Uptrend
    else:
        position -= 0.2  # Downtrend

    # Normalize to [-1, 1]
    position = max(min(position, 1.0), -1.0)

    return position


def run_experiment():
    """
    Run a single experiment with the current strategy.

    Returns:
        dict: Experiment results
    """
    evaluator = StrategyEvaluator()
    results = evaluator.evaluate_strategy(baseline_strategy)
    return results


def print_results(results):
    """
    Print experiment results in the standardized format.

    Args:
        results (dict): Results from strategy evaluation
    """
    print("---")
    print(f"total_return:          {results['total_return']:.6f}")
    print(f"sharpe_ratio:          {results['sharpe_ratio']:.6f}")
    print(f"max_drawdown:          {results['max_drawdown']:.6f}")
    print(f"consistency:           {results['consistency']:.6f}")
    print(f"risk_adjusted_score:   {results['risk_adjusted_score']:.6f}")
    print(f"experiment_time:       {results['eval_time']:.1f}")
    print("---")


if __name__ == "__main__":
    # Run the baseline experiment
    print("Running baseline strategy evaluation...")
    results = run_experiment()
    print_results(results)

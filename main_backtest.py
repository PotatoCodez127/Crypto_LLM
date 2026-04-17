"""
Fixed evaluation framework for trading strategies.
This module should NOT be modified during autoresearch experiments.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from data.handler import DataHandler
from features.extractor import FeatureExtractor


class StrategyEvaluator:
    """Fixed framework for evaluating trading strategies."""

    def __init__(self, data_handler=None, feature_extractor=None):
        """Initialize evaluator with data and feature components."""
        self.data_handler = data_handler or DataHandler()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.data = self.data_handler.load_data()

    def calculate_returns(self, positions, prices):
        """
        Calculate portfolio returns from positions and prices.

        Args:
            positions (list): Position sizes for each period
            prices (list): Asset prices for each period

        Returns:
            list: Period returns
        """
        returns = []
        for i in range(1, len(positions)):
            price_return = (prices[i] - prices[i - 1]) / prices[i - 1]
            portfolio_return = positions[i - 1] * price_return
            returns.append(portfolio_return)
        return returns

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """
        Calculate Sharpe ratio from returns.

        Args:
            returns (list): Period returns
            risk_free_rate (float): Risk-free rate per period

        Returns:
            float: Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = [r - risk_free_rate for r in returns]
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if std_excess == 0:
            return 0.0

        # Annualize (assuming hourly data, 365*24 periods per year)
        annualized_mean = mean_excess * 365 * 24
        annualized_std = std_excess * np.sqrt(365 * 24)

        return annualized_mean / annualized_std if annualized_std != 0 else 0.0

    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve (list): Portfolio equity values over time

        Returns:
            float: Maximum drawdown (positive value)
        """
        if len(equity_curve) == 0:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def calculate_consistency(self, returns):
        """
        Calculate consistency of positive returns.

        Args:
            returns (list): Period returns

        Returns:
            float: Consistency score (0-1)
        """
        if len(returns) == 0:
            return 0.0

        positive_count = sum(1 for r in returns if r > 0)
        return positive_count / len(returns)

    def calculate_risk_adjusted_score(self, total_return, sharpe_ratio, max_drawdown):
        """
        Calculate composite risk-adjusted score.

        Args:
            total_return (float): Total portfolio return
            sharpe_ratio (float): Sharpe ratio
            max_drawdown (float): Maximum drawdown

        Returns:
            float: Risk-adjusted score
        """
        # Penalize heavily for large drawdowns
        drawdown_penalty = 1.0
        if max_drawdown > 0.3:
            drawdown_penalty = 0.5
        elif max_drawdown > 0.5:
            drawdown_penalty = 0.1

        # Combine metrics (normalized)
        score = (
            total_return * 0.4 + sharpe_ratio * 0.4 + (1 - max_drawdown) * 0.2
        ) * drawdown_penalty

        return score

    def evaluate_strategy(self, strategy_function, evaluation_period_days=30):
        """
        Evaluate a trading strategy over historical data.

        Args:
            strategy_function (function): Function that takes features and returns position
            evaluation_period_days (int): Number of days to evaluate

        Returns:
            dict: Evaluation results
        """
        if self.data.empty:
            print("Warning: No data available for evaluation")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 1.0,
                "consistency": 0.0,
                "risk_adjusted_score": 0.0,
                "eval_time": 0.0,
            }

        # Limit evaluation period
        cutoff_date = self.data["timestamp"].max() - pd.Timedelta(
            days=evaluation_period_days
        )
        eval_data = self.data[self.data["timestamp"] >= cutoff_date].copy()

        if len(eval_data) < 2:
            print("Warning: Insufficient data for evaluation")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 1.0,
                "consistency": 0.0,
                "risk_adjusted_score": 0.0,
                "eval_time": 0.0,
            }

        start_time = datetime.now()

        # Extract features for all data points
        feature_data = self.feature_extractor.extract_features(eval_data)

        # Generate positions using strategy function
        positions = []
        prices = feature_data["close"].tolist()

        for i in range(len(feature_data)):
            row_features = feature_data.iloc[i]
            feature_vector = {
                "rsi": row_features["rsi"],
                "macd": row_features["macd"],
                "atr": row_features["atr"],
                "volume_change": row_features["volume_change"],
                "trend_slope": row_features["trend_slope"],
                "bb_width": row_features["bb_width"],
            }

            try:
                position = strategy_function(feature_vector)
                # Clamp position to reasonable limits
                position = max(min(position, 1.0), -1.0)
            except Exception as e:
                print(f"Warning: Strategy function error: {e}")
                position = 0.0

            positions.append(position)

        # Calculate metrics
        if len(positions) > 1 and len(prices) == len(positions):
            returns = self.calculate_returns(positions, prices)
            equity_curve = [1.0]  # Start with 1.0 (100% capital)

            for ret in returns:
                equity_curve.append(equity_curve[-1] * (1 + ret))

            total_return = equity_curve[-1] - 1.0
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(equity_curve)
            consistency = self.calculate_consistency(returns)
            risk_adjusted_score = self.calculate_risk_adjusted_score(
                total_return, sharpe_ratio, max_drawdown
            )
        else:
            total_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 1.0
            consistency = 0.0
            risk_adjusted_score = 0.0

        end_time = datetime.now()
        eval_time = (end_time - start_time).total_seconds()

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "consistency": consistency,
            "risk_adjusted_score": risk_adjusted_score,
            "eval_time": eval_time,
        }


# Example strategy function (for testing)
def example_strategy(features):
    """
    Simple example strategy based on RSI.

    Args:
        features (dict): Feature vector

    Returns:
        float: Position size (-1 to 1)
    """
    rsi = features.get("rsi", 50)

    if rsi < 30:
        return 1.0  # Buy
    elif rsi > 70:
        return -1.0  # Sell
    else:
        return 0.0  # Hold


if __name__ == "__main__":
    # Test the evaluator
    evaluator = StrategyEvaluator()
    results = evaluator.evaluate_strategy(example_strategy)
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

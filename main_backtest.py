"""
Quantitative Strategy Evaluation Framework
Enhanced for parameter optimization and comprehensive metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools

class QuantitativeEvaluator:
    """
    Advanced evaluator for quantitative trading strategies.
    Supports parameter optimization, walk-forward analysis, and comprehensive metrics.
    """
    
    def __init__(self, data_path=None):
        """Initialize evaluator with optional data path."""
        self.data_path = data_path
        self.data = self._load_data()
        
    def _load_data(self):
        """Load market data from CSV file."""
        import os
        if self.data_path and os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df
        else:
            # Try to find data in default location
            default_paths = [
                "data/btc_1h_1y.csv",
                "../data/btc_1h_1y.csv",
                "./data/btc_1h_1y.csv"
            ]
            for path in default_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    return df
            print("Warning: No data found. Returning empty DataFrame.")
            return pd.DataFrame()
    
    def calculate_metrics(self, returns, prices, signals):
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns (pd.Series): Strategy returns
            prices (pd.Series): Price series
            signals (pd.Series): Trading signals (-1, 0, 1)
            
        Returns:
            dict: Performance metrics
        """
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'num_trades': 0,
                'volatility': 0.0
            }
        
        # Equity curve
        equity = (1 + returns).cumprod()
        
        # Total return
        total_return = equity.iloc[-1] - 1
        
        # Sharpe ratio (annualized)
        excess_returns = returns - 0.0  # Assuming 0% risk-free rate
        sharpe = np.sqrt(365*24) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Sortino ratio (annualized, only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = np.sqrt(365*24) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio (return / max drawdown)
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        trade_signals = signals.diff().abs()
        num_trades = trade_signals.sum() / 2  # Each trade has entry and exit
        
        # Win rate and profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        gross_profit = positive_returns.sum()
        gross_loss = abs(negative_returns.sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        # Average trade return
        avg_trade = returns.mean() if len(returns) > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(365*24)
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_trade': float(avg_trade),
            'num_trades': int(num_trades),
            'volatility': float(volatility)
        }
    
    def evaluate_strategy_with_params(self, strategy_func, params=None, 
                                     train_days=180, test_days=30):
        """
        Walk-forward evaluation with train/test splits.
        
        Args:
            strategy_func: Function that takes (data, params) and returns signals
            params: Strategy parameters
            train_days: Days for training/optimization
            test_days: Days for out-of-sample testing
            
        Returns:
            dict: Evaluation results
        """
        if self.data.empty:
            return {"error": "No data available"}
        
        # Ensure data is sorted
        data = self.data.sort_index()
        
        # Split data into train/test periods
        all_signals = pd.Series(index=data.index, dtype=float)
        all_returns = pd.Series(index=data.index, dtype=float)
        
        # Walk-forward analysis
        start_date = data.index.min()
        end_date = data.index.max()
        current_start = start_date
        
        while current_start + timedelta(days=train_days+test_days) <= end_date:
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            
            train_data = data[(data.index >= current_start) & (data.index < train_end)]
            test_data = data[(data.index >= train_end) & (data.index < test_end)]
            
            if len(train_data) < 10 or len(test_data) < 10:
                current_start += timedelta(days=test_days)
                continue
            
            # Generate signals on test data
            test_signals = strategy_func(test_data, params)
            
            if 'signal' not in test_signals.columns:
                current_start += timedelta(days=test_days)
                continue
            
            # Calculate returns
            test_returns = test_signals['signal'].shift(1) * test_data['close'].pct_change()
            test_returns = test_returns.dropna()
            
            # Store results
            all_signals.loc[test_returns.index] = test_signals.loc[test_returns.index, 'signal']
            all_returns.loc[test_returns.index] = test_returns
            
            current_start += timedelta(days=test_days)
        
        # Calculate overall metrics
        all_returns = all_returns.dropna()
        all_signals = all_signals.loc[all_returns.index]
        
        metrics = self.calculate_metrics(all_returns, self.data.loc[all_returns.index, 'close'], all_signals)
        
        # Add parameter information
        metrics['params'] = params
        metrics['total_periods'] = len(all_returns)
        
        return metrics
    
    def _generate_param_combinations(self, param_grid):
        """Generate all combinations of parameters from a grid."""
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return combinations
    
    def optimize_parameters(self, strategy_func, param_grid, train_days=180, test_days=30):
        """
        Grid search for parameter optimization.
        
        Args:
            strategy_func: Function that takes (data, params) and returns signals
            param_grid: Dictionary of parameter ranges to test
            train_days: Days for training
            test_days: Days for testing
            
        Returns:
            dict: Best parameters and their performance
        """
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            metrics = self.evaluate_strategy_with_params(
                strategy_func, params, train_days, test_days
            )
            
            if 'error' in metrics:
                continue
            
            # Score function: weighted combination of metrics
            # Favor higher Sharpe, higher total return, lower drawdown
            score = (
                metrics['sharpe_ratio'] * 0.4 +
                metrics['total_return'] * 0.3 +
                (1 - abs(metrics['max_drawdown'])) * 0.2 +
                metrics['profit_factor'] * 0.1
            )
            
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics
        }

# Example usage
if __name__ == "__main__":
    # Test with a simple strategy
    from offline_training.strategy import get_signals
    
    evaluator = QuantitativeEvaluator("data/btc_1h_1y.csv")
    
    # Define parameter grid for optimization
    param_grid = {
        'ema_short': [10, 20, 30],
        'ema_long': [50, 100],
        'rsi_period': [14, 21],
        'rsi_overbought': [70, 75],
        'rsi_oversold': [25, 30]
    }
    
    # Run optimization
    result = evaluator.optimize_parameters(get_signals, param_grid, train_days=90, test_days=30)
    
    print("Optimization Results:")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Best Score: {result['best_score']}")
    print(f"Metrics: {result['best_metrics']}")

# AI-Powered Cryptocurrency Trading System

An adaptive trading AI that learns from historical crypto market data, detects market conditions, and executes profitable trades automatically.

## System Overview

This system implements a modular approach to cryptocurrency trading with the following components:

- **Data Layer**: Retrieves and stores OHLCV data from Bybit exchange
- **Feature Engineering**: Transforms market data into feature vectors for analysis
- **Matching Engine**: Finds similar historical market conditions using KNN
- **Strategy System**: Implements multiple strategies to capitalize on market conditions
- **Multi-Model System**: Uses three models (3Y, 1Y, 3M) for robust decision making
- **Decision Engine**: Coordinates all components to make trading decisions
- **Risk Management**: Enforces strict risk controls to protect capital
- **Execution Engine**: Interfaces with Bybit API for trade execution
- **AutoResearch**: Continuously improves strategies based on performance

## Project Structure

```
/trading_ai/
│
├── data/           # Data ingestion and storage
├── features/       # Feature engineering pipeline
├── strategies/     # Trading strategies implementations
├── models/         # Machine learning models
├── matching/       # Similarity matching algorithms
├── autoresearch/   # Strategy evolution and improvement
├── risk/           # Risk management components
├── execution/      # Trade execution interface
├── backtest/       # Backtesting framework
├── core/           # Core engine and coordination
└── config/         # Configuration files
```

## Getting Started

1. Install required dependencies
2. Configure API keys in `config/`
3. Run the system with `python main.py`

## Key Features

- Modular design for easy maintenance and extension
- Multiple strategies working in parallel
- Adaptive risk management
- Historical pattern matching for decision making
- Continuous strategy improvement through AutoResearch
- Paper trading support for testing

## Requirements

- Python 3.8+
- ccxt (for exchange APIs)
- pandas (for data manipulation)
- scikit-learn (for machine learning)
- numpy (for numerical computations)
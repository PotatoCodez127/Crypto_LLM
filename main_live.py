"""
Main entry point for the AI-powered cryptocurrency trading system.
"""

from src.core.engine import TradingEngine


def main():
    """Initialize and run the trading engine."""
    engine = TradingEngine()
    engine.run()


if __name__ == "__main__":
    main()

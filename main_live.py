# main_live.py
# Add this import to the top of main_live.py
from src.ai_agent.llm_client import TradingAgentClient
from dotenv import load_dotenv # Also add this to load your .env file
import os

load_dotenv() # Load environment variables on startup
import time
import pandas as pd
from datetime import datetime, timezone

from src.core.logger import trading_logger
from src.data_feed.handler import DataHandler
from src.features.extractor import FeatureExtractor
from src.ai_agent.tape_generator import SemanticTapeGenerator

class LiveTradingEngine:
    """Continuous execution engine for the AI Crypto Bot."""
    
    def __init__(self):
        self.logger = trading_logger
        self.data_handler = DataHandler()
        self.feature_extractor = FeatureExtractor()
        self.tape_generator = SemanticTapeGenerator()
        
        self.ai_client = TradingAgentClient(provider="ollama")

        # Configuration
        self.loop_delay_seconds = 60  # Check market every 60 seconds
        self.tape_lookback = 5        # Number of candles to feed the AI at once

    def initialize(self):
        """Startup sequence."""
        self.logger.info("🟢 Initializing Live Crypto LLM Engine...")
        self.logger.info("Updating historical data to ensure feature accuracy...")
        # Fetch enough data to calculate slow indicators (like MACD/Bollinger Bands)
        self.data_handler.update_historical_data(days_back=3) 
        self.logger.info("Initialization complete. Entering live monitoring loop.")

    def run(self):
        """The 24/7 continuous monitoring loop."""
        self.initialize()

        while True:
            try:
                now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
                print(f"[{now}] 📡 Scanning market...", end='\r')

                # 1. Fetch the latest batch of data (we need a small window for indicator math)
                # Note: Adjust limit based on your longest indicator lookback (e.g., 30 for BBs)
                recent_ohlcv = self.data_handler.fetch_ohlcv(limit=50) 
                
                if not recent_ohlcv or len(recent_ohlcv) == 0:
                    self.logger.warning("No recent data available from exchange.")
                    time.sleep(self.loop_delay_seconds)
                    continue

                df = self.data_handler.ohlcv_to_dataframe(recent_ohlcv)

                # 2. Extract Quantitative Features
                features_df = self.feature_extractor.extract_features(df)

                # 3. Generate Semantic Tape (The AI's "Eyes")
                semantic_tape = self.tape_generator.build_tape(
                    df=df, 
                    features_df=features_df, 
                    lookback=self.tape_lookback
                )

                # Print the tape to console (Eventually, this gets sent to the LLM)
                print(f"\n\n--- NEW MARKET TAPE GENERATED ---")
                print(semantic_tape)
                print("---------------------------------\n")

                # TODO: Pass `semantic_tape` to RAG / LLM for trade decision here!

                # Sleep until the next candle
                time.sleep(self.loop_delay_seconds)

            except KeyboardInterrupt:
                print("\n🛑 Shutting down live engine gracefully.")
                break
            except Exception as e:
                self.logger.error(f"⚠️ Live Feed Error: {e}")
                time.sleep(30) # Brief pause on error before retrying

if __name__ == "__main__":
    engine = LiveTradingEngine()
    engine.run()
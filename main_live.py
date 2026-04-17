# main_live.py
import time
import pandas as pd
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

from src.core.logger import trading_logger
from src.data_feed.handler import DataHandler
from src.features.extractor import FeatureExtractor
from src.ai_agent.tape_generator import SemanticTapeGenerator
from src.ai_agent.llm_client import TradingAgentClient

# Load the environment variables (.env file)
load_dotenv()

class LiveTradingEngine:
    """Continuous execution engine for the AI Crypto Bot."""
    
    def __init__(self):
        self.logger = trading_logger
        self.data_handler = DataHandler()
        self.feature_extractor = FeatureExtractor()
        self.tape_generator = SemanticTapeGenerator()
        
        # Initialize the AI Client
        self.ai_client = TradingAgentClient() 
        
        # Configuration
        self.loop_delay_seconds = 60  
        self.tape_lookback = 5        

    def initialize(self):
        """Startup sequence."""
        # Removed the emoji to prevent Windows encoding errors just in case!
        self.logger.info("Initializing Live Crypto LLM Engine...")
        self.logger.info("Updating historical data to ensure feature accuracy...")
        self.data_handler.update_historical_data(days_back=3) 
        self.logger.info("Initialization complete. Entering live monitoring loop.")

    def run(self):
        """The 24/7 continuous monitoring loop."""
        self.initialize()

        while True:
            try:
                now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
                print(f"[{now}] 📡 Scanning market...", end='\r')

                # 1. Fetch latest data
                recent_ohlcv = self.data_handler.fetch_ohlcv(limit=50) 
                
                if not recent_ohlcv or len(recent_ohlcv) == 0:
                    self.logger.warning("No recent data available from exchange.")
                    time.sleep(self.loop_delay_seconds)
                    continue

                df = self.data_handler.ohlcv_to_dataframe(recent_ohlcv)

                # 2. Extract Quantitative Features
                features_df = self.feature_extractor.extract_features(df)

                # 3. Generate Semantic Tape
                semantic_tape = self.tape_generator.build_tape(
                    df=df, 
                    features_df=features_df, 
                    lookback=self.tape_lookback
                )

                print(f"\n\n--- NEW MARKET TAPE GENERATED ---")
                print(semantic_tape)
                print("---------------------------------\n")

                # 4. 🔥 ASK THE AI BRAIN 🔥
                print("🧠 Sending tape to Ollama Cloud for analysis...")
                ai_decision = self.ai_client.analyze_tape(semantic_tape)

                if ai_decision:
                    decision = ai_decision.get("decision", "NONE")
                    confidence = ai_decision.get("confidence", 0)
                    reasoning = ai_decision.get("reasoning", "No reasoning provided.")

                    print(f"=====================================")
                    print(f"🤖 AI DECISION: {decision} (Confidence: {confidence}%)")
                    print(f"📝 REASONING:   {reasoning}")
                    print(f"=====================================\n")

                    if decision in ["LONG", "SHORT"] and confidence >= 75:
                        print("🔔 HIGH CONFIDENCE SIGNAL DETECTED! (Ready for execution routing)")
                        
                else:
                    print("⚠️ No valid response received from AI.")

                # Sleep until the next candle
                time.sleep(self.loop_delay_seconds)

            except KeyboardInterrupt:
                print("\n🛑 Shutting down live engine gracefully.")
                break
            except Exception as e:
                self.logger.error(f"⚠️ Live Feed Error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    engine = LiveTradingEngine()
    engine.run()
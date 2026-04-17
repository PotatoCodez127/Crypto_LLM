"""
Core trading engine that coordinates all system components.
"""
Core trading engine that coordinates all system components.
"""
from data.handler import DataHandler
from features.extractor import FeatureExtractor
from core.logger import trading_logger

class TradingEngine:
    """Main trading engine that orchestrates all system components."""
    
    def __init__(self):
        """Initialize the trading engine with all required components."""
        self.data_handler = DataHandler()
        self.feature_extractor = FeatureExtractor()
        self.logger = trading_logger
        self.logger.info("Trading Engine initialized")
    
    def initialize_system(self):
        """Initialize all system components."""
        self.logger.info("Initializing system...")
        # Update historical data
        self.logger.info("Updating historical data...")
        self.data_handler.update_historical_data(30)  # Last 30 days
        self.logger.info("System initialization complete")
    
    def process_latest_candle(self):
        """Process the latest market candle and generate signals."""
        self.logger.info("Processing latest candle...")
        
        # Get latest data
        latest_candle = self.data_handler.get_latest_candle()
        if latest_candle.empty:
            self.logger.warning("No recent data available")
            return None
            
        self.logger.info(f"Latest candle: {latest_candle.iloc[0]['timestamp']}")
        
        # Extract features
        features = self.feature_extractor.get_feature_vector(latest_candle)
        if not features:
            self.logger.error("Failed to extract features")
            return None
            
        self.logger.info(f"Features extracted: {features}")
        return features
    
    def run(self):
        """Main execution loop for the trading engine."""
        self.logger.info("Starting Trading Engine...")
        
        try:
            # Initialize system
            self.initialize_system()
            
            # Process latest market data
            features = self.process_latest_candle()
            
            if features:
                self.logger.info("Processing complete. Features ready for strategy evaluation.")
            else:
                self.logger.info("Processing complete. No actionable signals generated.")
                
        except Exception as e:
            self.logger.error(f"Error in trading engine: {e}")

if __name__ == "__main__":
    engine = TradingEngine()
    engine.run()

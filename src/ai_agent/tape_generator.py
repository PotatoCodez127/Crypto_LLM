import pandas as pd
from datetime import datetime
import pytz
import logging

logger = logging.getLogger(__name__)

class TapeGenerator:
    """
    Translates quantitative ML features (OHLCV + Derivatives) into a textual 
    Semantic Tape for LLM ingestion, aware of SAST session timings.
    """
    
    def __init__(self):
        # SAST reference for regime filtering
        self.sast_tz = pytz.timezone('Africa/Johannesburg')

    def generate_tape(self, current_state: pd.Series) -> str:
        """
        Builds the Semantic Tape from the latest feature vector.
        """
        try:
            # Convert execution time to SAST for the LLM's spatial awareness
            utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
            sast_now = utc_now.astimezone(self.sast_tz)
            
            # Extract features safely
            close_price = current_state.get('close', 0.0)
            atr = current_state.get('atr', 0.0)
            atr_norm = current_state.get('atr_normalized', 0.0)
            cvd_trend = current_state.get('cvd_trend', 0.0)
            vol_z = current_state.get('volume_zscore', 0.0)
            ml_signal = current_state.get('ml_signal', 0.0)
            
            # Assess Momentum Sequence
            log_returns = [
                current_state.get('log_return_lag_3', 0.0),
                current_state.get('log_return_lag_2', 0.0),
                current_state.get('log_return_lag_1', 0.0),
                current_state.get('log_return', 0.0)
            ]
            momentum_sum = sum(log_returns)
            
            # Discretize numerical conditions for the LLM
            momentum_state = "Strong Bullish" if momentum_sum > 0.01 else "Strong Bearish" if momentum_sum < -0.01 else "Chopping/Ranging"
            flow_state = "Aggressive Buying" if cvd_trend > 0 else "Aggressive Selling" if cvd_trend < 0 else "Neutral Flow"
            vol_state = "Expanding (Breakout Risk)" if atr_norm > 1.0 else "Compressing (Mean Reversion)" if atr_norm < -1.0 else "Normal Volatility"
            
            tape = f"""--- MARKET MICRO-STRUCTURE TAPE ---
Execution Local Time (SAST): {sast_now.strftime('%Y-%m-%d %H:%M:%S %Z')}

1. PRICE & MOMENTUM CONTEXT
- Current Asset Price: {close_price:.2f}
- Cumulative Momentum (4-Period): {momentum_sum:.5f}
- Regime Assessment: {momentum_state}

2. VOLATILITY & RISK (ATR)
- Absolute Volatility (ATR): {atr:.2f}
- Normalized Volatility (Z-Score): {atr_norm:.2f}
- Volatility State: {vol_state}

3. ORDER FLOW & LIQUIDITY
- CVD Trajectory (Aggressive Flow): {cvd_trend:.2f} ({flow_state})
- Relative Volume (Z-Score): {vol_z:.2f}

4. QUANTITATIVE REFLEX (XGBoost)
- ML Node Signal Fired: {"YES (Confluence Required)" if ml_signal == 1 else "NO"}
-----------------------------------"""
            return tape
            
        except Exception as e:
            logger.error(f"Tape Generation Failed: {str(e)}")
            return f"SYSTEM FAULT: Unable to generate tape. Error: {str(e)}"
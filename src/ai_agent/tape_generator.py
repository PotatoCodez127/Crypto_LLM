# src/ai_agent/tape_generator.py
import pandas as pd

class SemanticTapeGenerator:
    """Translates raw quantitative market data into a semantic story for LLM analysis."""

    def __init__(self):
        pass

    def build_tape(self, df: pd.DataFrame, features_df: pd.DataFrame = None, lookback: int = 5) -> str:
        """
        Generates a textual tape of the last N candles.
        """
        if df.empty or len(df) < lookback:
            return "Insufficient data to build tape."

        # Get the most recent N candles
        recent_tape = df.tail(lookback)
        
        tape_lines = []
        tape_lines.append(f"--- MARKET TAPE (Last {lookback} Periods) ---")

        for i in range(len(recent_tape)):
            row = recent_tape.iloc[i]
            timestamp = row.name if isinstance(recent_tape.index, pd.DatetimeIndex) else row.get('timestamp', f"T-{lookback-i}")
            
            # Formulate timestamp string
            if isinstance(timestamp, pd.Timestamp):
                time_str = timestamp.strftime('%H:%M:%S UTC')
            else:
                time_str = str(timestamp)

            # Extract OHLC
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            
            # Calculate price action metrics
            point_change = c - o
            total_range = h - l
            body = abs(c - o)
            if total_range == 0: total_range = 0.0001 # Prevent division by zero
            
            # Determine Direction
            direction = "Bullish" if point_change > 0 else "Bearish" if point_change < 0 else "Neutral"
            
            # Determine Candle Shape
            if body <= (total_range * 0.25):
                shape = "Indecision/Doji"
            elif body >= (total_range * 0.75):
                shape = "Strong Momentum"
            else:
                shape = "Standard Candle"

            # Base semantic line
            line = f"[{time_str}] Close: {c:.2f} | {direction} ({point_change:+.2f}) | Shape: {shape}"

            # Append Technical Features if provided
            if features_df is not None and len(features_df) == len(df):
                f_row = features_df.tail(lookback).iloc[i]
                rsi = f_row.get('rsi', 50)
                macd = f_row.get('macd', 0)
                
                # Contextualize RSI
                if rsi > 70:
                    rsi_ctx = f"Overbought ({rsi:.1f})"
                elif rsi < 30:
                    rsi_ctx = f"Oversold ({rsi:.1f})"
                else:
                    rsi_ctx = f"Neutral ({rsi:.1f})"
                    
                line += f" | RSI: {rsi_ctx} | MACD: {macd:.2f}"

            tape_lines.append(line)
            
        return "\n".join(tape_lines)
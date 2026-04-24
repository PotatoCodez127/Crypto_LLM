import json
import logging
from typing import Dict, Any
import litellm

logger = logging.getLogger(__name__)

class TradingBrain:
    def __init__(self, model_name: str = "gpt-4-turbo", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        
        self.system_prompt = """You are an elite quantitative trading AI specializing in BTC and US30 market micro-structure. 
You are receiving a 'Semantic Tape' detailing real-time price action, structural levels, and momentum indicators.

ANALYTICAL FRAMEWORK:
1. Regime Identification: Determine if the market is trending, ranging, or chopping.
2. Structural Analysis: Note proximity to key liquidity zones (Daily Open, Order Blocks).
3. Risk Assessment: Evaluate volatility (ATR) and signal confluence. Capital preservation is your primary directive. If signals conflict, you must remain flat.

OUTPUT INSTRUCTIONS:
You must output ONLY a valid JSON object. Do not include markdown blocks or conversational text. 
Use the following schema:
{
    "thought_process": {
        "market_regime": "<brief assessment>",
        "key_levels": "<closest support/resistance>",
        "confluence_check": "<do indicators align with price action?>"
    },
    "decision": "LONG" | "SHORT" | "NONE",
    "confidence": <0-100>,
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "reasoning": "<1-2 sentence final justification>"
}"""

    def _graceful_degradation(self, reason: str) -> Dict[str, Any]:
        """Provides a safe default flat state if the LLM or API fails."""
        logger.error(f"BRAIN FAULT | Degrading to safe state. Reason: {reason}")
        return {
            "thought_process": {
                "market_regime": "UNKNOWN",
                "key_levels": "UNKNOWN",
                "confluence_check": "API Timeout or Parsing Failure"
            },
            "decision": "NONE",
            "confidence": 0,
            "risk_level": "LOW",
            "reasoning": f"System forced flat due to exception: {reason}"
        }

    def analyze_tape(self, semantic_tape: str) -> Dict[str, Any]:
        """
        Submits the semantic tape to the LLM, enforcing the JSON CoT schema.
        """
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"SEMANTIC TAPE:\n{semantic_tape}"}
                ],
                temperature=self.temperature,
                # Force JSON mode on supported models (OpenAI, Anthropic, etc.)
                response_format={ "type": "json_object" }
            )
            
            raw_response = response.choices[0].message.content
            
            # Clean potential markdown fences if the model disobeys prompt bounds
            if raw_response.startswith("```"):
                raw_response = raw_response.strip("```json").strip("```")
                
            decision_json = json.loads(raw_response)
            
            # Schema validation
            required_keys = ["thought_process", "decision", "confidence", "risk_level", "reasoning"]
            if not all(key in decision_json for key in required_keys):
                return self._graceful_degradation("Missing required JSON keys in LLM output.")
                
            return decision_json

        except json.JSONDecodeError as e:
            return self._graceful_degradation(f"JSON Parsing Error: {str(e)}")
        except Exception as e:
            return self._graceful_degradation(f"API/Network Error: {str(e)}")
# src/ai_agent/llm_client.py
import os
import json
from openai import OpenAI
from typing import Dict, Optional

class TradingAgentClient:
    """Handles communication with the LLM to generate trading decisions from the Semantic Tape."""

    def __init__(self, provider: str = "ollama"):
        """
        Initializes the client. Defaults to Ollama.
        """
        self.provider = provider.lower()
        
        # Pull the Ollama Cloud URL from .env, or fallback to local if not found
        base_url = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434/v1")
        
        # We specify the model here, but allow override via .env
        self.model = os.environ.get("OLLAMA_MODEL", "deepseek-v3.2")
        # Pull the real API key from .env (fallback to 'ollama' if local)
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")

        # Initialize the OpenAI SDK client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key 
        )

        self._build_system_prompt()

    def _build_system_prompt(self):
        """The core instructions that tell the AI how to behave and format its output."""
        self.system_prompt = """
        You are an elite cryptocurrency quantitative trading AI. 
        You will be provided with a 'Semantic Tape' showing recent price action, volume, and technical indicators.
        
        Your task is to analyze this tape and make a trading decision.
        
        CRITICAL RULES:
        1. Capital preservation is paramount. If the tape shows chop, indecision, or conflicting signals, return NONE.
        2. You must output ONLY valid, raw JSON. 
        3. Follow this exact JSON schema:
        
        {
            "decision": "LONG" | "SHORT" | "NONE",
            "confidence": <integer from 0 to 100>,
            "reasoning": "<A 1-2 sentence explanation of your logic>",
            "risk_level": "LOW" | "MEDIUM" | "HIGH"
        }
        """

    def analyze_tape(self, semantic_tape: str) -> Optional[Dict]:
        """
        Sends the tape to the LLM and parses the JSON response.
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Here is the latest market tape. Analyze it and return your JSON decision:\n\n{semantic_tape}"}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1, # Keep it low for analytical consistency
                response_format={"type": "json_object"} # Forces Ollama to lock into JSON mode
            )

            raw_response = response.choices[0].message.content.strip()
            
            # Parse JSON
            decision_data = json.loads(raw_response)
            return decision_data

        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse AI JSON response: {e}")
            print(f"Raw Output was: {raw_response}")
            return None
        except Exception as e:
            print(f"⚠️ LLM API Error: {e}")
            return None
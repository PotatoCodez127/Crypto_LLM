# Crypto LLM: AI-Native Quantitative Trading Engine

An advanced cryptocurrency trading bot that bridges the gap between traditional quantitative analysis and Large Language Models (LLMs). 

Unlike standard algorithmic bots that rely purely on hardcoded indicator thresholds, this engine translates live market data into a "Semantic Tape"—a readable narrative of price action and volatility—allowing an AI agent to make context-aware trading decisions augmented by RAG (Retrieval-Augmented Generation) memory.

---

## 🏗️ System Architecture

The codebase is strictly modular, separating the continuous execution loop, quantitative math, AI translation, and offline model training.

    crypto_llm/
    ├── src/
    │   ├── ai_agent/              # 🧠 LLM Prompts & Semantic Tape translation
    │   │   └── tape_generator.py  # Translates OHLCV/indicators into text
    │   ├── config/                # ⚙️ Global variables and API keys
    │   │   └── settings.py
    │   ├── core/                  # 🚀 The execution engine and logging
    │   │   ├── engine.py
    │   │   └── logger.py
    │   ├── data_feed/             # 📡 Exchange connections (ccxt)
    │   │   └── handler.py
    │   ├── database/              # 🗄️ (WIP) Supabase & ChromaDB (RAG)
    │   ├── features/              # 🧮 Technical indicator math
    │   │   └── extractor.py
    │   └── strategy/              # 🎯 (WIP) State machine & time filters
    ├── tests/                     # 🧪 Unit testing
    ├── offline_training/          # 🔬 PyTorch LLM pre-training & research
    ├── main_live.py               # ▶️ The 24/7 continuous execution loop
    ├── main_backtest.py           # ⏪ The historical strategy evaluator
    └── requirements.txt

---

## 📊 Current State

**Version:** 0.1.0 (Architecture Overhaul)
**Status:** Live Engine Active (Read-Only)

- [x] **Modular Architecture:** Successfully decoupled execution, math, and data handling.
- [x] **Continuous Live Loop:** `main_live.py` runs as a 24/7 daemon, fetching market data and updating features without crashing.
- [x] **Quantitative Feature Extraction:** Standard indicators (RSI, MACD, ATR, BB) calculate correctly in real-time.
- [x] **Semantic Tape Generator:** Raw market data is successfully translated into a textual "story" format designed for LLM consumption.

---

## 🗺️ Roadmap & Next Steps

### Phase 1: The AI Brain (In Progress)
- [ ] **LLM Integration (`llm_client.py`):** Pass the generated Semantic Tape to an LLM (Ollama/OpenAI) to receive a structured `LONG`, `SHORT`, or `NONE` decision.
- [ ] **RAG Memory Bank (`ChromaDB`):** Store historical semantic tapes and their actual PnL outcomes. Query the database before every trade so the AI can recall similar past setups.

### Phase 2: State Tracking & Logic
- [ ] **State Machine:** Track specific structural levels (Daily Open, Previous Day High/Low, Order Blocks) rather than relying purely on lagging indicators.
- [ ] **Time/Regime Filters:** Hardcode logic to prevent the AI from trading during notorious low-liquidity periods (e.g., Asian session weekends).

### Phase 3: Execution & Telemetry
- [ ] **Supabase Integration:** Log every trade, including the AI's full "thought process" and RAG context, to a PostgreSQL database for forensic debugging.
- [ ] **Execution Routing:** Connect the engine's buy/sell decisions to actual exchange orders via `ccxt`.

---

## 🚀 How to Run

### Live Monitoring (Daemon)
To start the continuous live market scanner and Semantic Tape generator:
bash
python main_live.py


### Run Unit Tests
To verify the Semantic Tape is generating correctly with mock data:
bash
python tests/test_semantic_tape.py


---

## 📝 Changelog

### [v0.1.0] - Architecture Overhaul
- Completely restructured the repository into a `src/` module format.
- Extracted offline PyTorch training logic into `offline_training/`.
- Created `main_live.py` continuous execution loop.
- Built `SemanticTapeGenerator` to translate quantitative features into LLM-readable text.
- Fixed absolute/relative import routing across all `src` modules.

### [Pre-v0.1.0] - Initial Prototype
- Basic quantitative backtesting framework established.
- `ccxt` data ingestion and localized CSV storage implemented.
- Base technical indicators (RSI, MACD, Bollinger Bands) written in pandas.
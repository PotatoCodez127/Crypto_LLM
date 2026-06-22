# 📈 Crypto_LLM: Autonomous AI Multi-Agent Swarm Trading Engine

An enterprise-grade quantitative research framework that bridges advanced technical indicator feature engineering with LLM reasoning. The system runs autonomous multi-processed simulation loops ("Swarms") to discover, backtest, and optimize algorithmic trading strategies in parallel.

---

## 🧠 Core Architecture Concepts

### 1. The Semantic Tape Generator
Instead of feeding raw, noisy numerical matrix arrays directly to an AI model, this engine leverages a **Semantic Tape Generator**. It processes historical financial data, derives key technical indicators, and translates market states into a structured textual narrative stream designed specifically for Large Language Model (LLM) pattern extraction and cross-horizon contextual comprehension.

### 2. Autonomous Hyperparameter Research Loop
The project features a self-directed optimization machine managed by an AI Lead Quant agent (`strategy_trainer/auto_loop.py`):
- **Hypothesis Generation:** The agent reviews recent historical performance data and queries an internal LLM to generate fresh structural parameter configurations.
- **Direct Code Injection:** Hypotheses are converted into valid Python variables and injected directly into the running strategy code space.
- **Walk-Forward Judge:** The new strategy is passed to a Multi-Timeframe Walk-Forward Optimization matrix that evaluates model returns against strict data slippage and exchange fees.
- **Vectorized RAG Memory Bank:** Loop metrics are committed to a local persistent `ChromaDB` index. Future generations query this memory bank to identify profitable paths ("Winners") and actively avoid historical failure zones ("Landmines").

### 3. Parallel Worker Swarm Core
Utilizing `start_swarm.py`, the engine can spawn decentralized, isolated directory workspaces (`worker_node_1`, `worker_node_2`, etc.) across native operating system subprocesses. This creates a high-throughput simulation swarm where separate nodes can investigate independent strategy iterations simultaneously without memory leaks or process crashes.

---

## 🛠️ Tech Stack
- **Core Quant Stack:** Python, Pandas, NumPy, XGBoost, CCXT
- **AI Core:** LiteLLM (Unified inference layer targeting DeepSeek-V3/OpenAI), ChromaDB (Vectorized RAG State)
- **Infrastructure:** Pytest, Gunicorn, Docker
- **Automation:** GitHub Actions Continuous Integration

---

## 🧹 Code Quality & Standards
Development controls are managed centrally within the root `pyproject.toml` layout:
- **Black:** Enforces rigorous PEP 8 visual compliance across dense mathematical extraction files.
- **Ruff:** Monitors static compilation syntax health, sorts import hierarchies automatically via Isort, and prevents cross-module circular routing anomalies.

```bash
# Execute local quality checks
python -m black .
python -m ruff check . --fix
```

---

## ⚙️ Setup & Execution Guide

### Prerequisites
- Python 3.11+
- LiteLLM proxy instance configured locally or cloud API access credentials.

### Local Installation & Simulation
1. Clone the repository and install the production package sheet:
```bash
git clone [https://github.com/PotatoCodez127/Crypto_LLM.git](https://github.com/PotatoCodez127/Crypto_LLM.git)
cd Crypto_LLM
pip install -r requirements.txt
```

2. Initialize a single-threaded research loop instance:
```bash
python strategy_trainer/auto_loop.py
```

3. Or launch a multi-processed parallel worker swarm:
```bash
python start_swarm.py
```

### Production Container Deployment
The application includes a production-ready container blueprint to guarantee environment isolation across remote cloud nodes.
1. Compile the lightweight container image:
```bash
docker build -t crypto-llm-engine .
```

2. Compile the lightweight container image:
```bash
docker run -d --name live-quant-container --env-file .env crypto-llm-engine
```

---

## 🤖 Continuous Integration (CI/CD)

Automated testing gates are managed natively through GitHub Actions (.github/workflows/ci.yml). Every codebase update triggers an isolated Ubuntu container matrix that runs environment builds, strict Black check verifications, Ruff syntax compilation audits, and parallel Pytest execution trackers to verify absolute pipeline stability on push.

---
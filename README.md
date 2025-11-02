# Quant Trading Projects

A collection of quantitative trading strategies and market microstructure simulations implemented in Python.

## 📊 Projects

### 1. Pairs Trading Strategy
- Statistical arbitrage using cointegration tests and z-score triggers.
- Dynamic pair selection, hedge ratio estimation, and rolling window calibration.
- Backtesting harness with transaction costs, position sizing, and performance attribution.

### 2. Limit Order Book Simulator
- Event-driven market simulation with heap-backed matching engine.
- Aggregated depth and order-book snapshot exports for downstream analytics.
- Market impact experiments, plotting utilities, and reproducible random order generation.

### 3. Intraday Momentum Strategy
- Technical indicator-based entries blended with risk filters and stop management.
- High-frequency mean reversion overlays to reduce whipsaw noise.
- Comprehensive performance analytics, tear-sheet generation, and reporting helpers.

### 4. Portfolio Utilities (`utils/`)
- Reusable data loading helpers, plotting presets, and evaluation metrics shared across projects.
- Experiment tracking hooks and serialization helpers for research notebooks.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/hren0707/Quant-Trading-Projects.git
cd Quant-Trading-Projects

# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest

# Run pairs trading example
python 01_pairs_trading/pairs_trading.py

# Explore Jupyter notebooks
jupyter notebook

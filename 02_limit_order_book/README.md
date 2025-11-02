# Limit Order Book Simulator

This module implements a lightweight limit order book (LOB) with price–time priority matching and analysis utilities.

## Key Features
- Heap-backed best bid/ask retrieval with automatic cleanup of cancelled or filled orders.
- Aggregated depth queries through `LimitOrderBook.get_depth`.
- Tabular snapshots for analytics via `LimitOrderBook.get_order_book_snapshot`.
- Market impact estimation and matplotlib-based visualisations for exploratory research.

## Usage
```python
from limit_order_book import LimitOrderBook

lob = LimitOrderBook()
lob.add_order(order_id=1, side='bid', price=100.0, quantity=50, timestamp=0.0)
lob.add_order(order_id=2, side='ask', price=100.5, quantity=40, timestamp=0.1)

best_bid, best_ask = lob.get_best_bid_ask()
depth = lob.get_depth(levels=3)
snapshot = lob.get_order_book_snapshot()
```

## Tests
```
pytest 02_limit_order_book/tests/test_lob.py
```

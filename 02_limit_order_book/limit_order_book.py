import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

@dataclass
class Order:
    order_id: int
    side: str  # 'bid' or 'ask'
    price: float
    quantity: int
    timestamp: float

class LimitOrderBook:
    def __init__(self, tick_size=0.01):
        self.tick_size = tick_size
        # Using heaps for price-time priority
        self.bids = []  # Max heap for bids (use negative prices)
        self.asks = []  # Min heap for asks
        self.orders = {}  # order_id -> Order
        self.order_queues = {
            'bid': defaultdict(deque),
            'ask': defaultdict(deque),
        }
        self.trade_history = []
        self.mid_price_history = []
        self.spread_history = []

    def _clean_heap(self, heap, is_bid: bool):
        """Remove stale orders from the top of the given heap."""
        while heap:
            price, timestamp, order_id = heap[0]
            if order_id not in self.orders:
                heapq.heappop(heap)
                continue

            order = self.orders[order_id]
            if order.quantity <= 0:
                heapq.heappop(heap)
                self._remove_order_from_queue(order.side, order.price, order_id)
                del self.orders[order_id]
                continue

            # Prices in the bid heap are stored as negative values
            current_price = -price if is_bid else price
            if order.price != current_price:
                heapq.heappop(heap)
                self._remove_order_from_queue(order.side, order.price, order_id)
                del self.orders[order_id]
                continue

            break

    def _remove_order_from_queue(self, side: str, price: float, order_id: int):
        """Safely remove an order id from the queue associated with a price level."""
        price_queue = self.order_queues[side].get(price)
        if not price_queue:
            return

        try:
            price_queue.remove(order_id)
        except ValueError:
            return

        if not price_queue:
            del self.order_queues[side][price]
        
    def add_order(self, order_id: int, side: str, price: float, quantity: int, timestamp: float):
        """Add a new order to the book"""
        # Round price to tick size
        price = round(price / self.tick_size) * self.tick_size
        
        order = Order(order_id, side, price, quantity, timestamp)
        self.orders[order_id] = order
        
        if side == 'bid':
            heapq.heappush(self.bids, (-price, timestamp, order_id))
        else:  # ask
            heapq.heappush(self.asks, (price, timestamp, order_id))
            
        self.order_queues[side][price].append(order_id)
        self._check_for_trades()
        self._record_market_data()
        
    def cancel_order(self, order_id: int):
        """Cancel an existing order"""
        if order_id not in self.orders:
            return

        order = self.orders.pop(order_id)
        self._remove_order_from_queue(order.side, order.price, order_id)

        # Lazily clean heaps to remove stale entries
        self._clean_heap(self.bids, is_bid=True)
        self._clean_heap(self.asks, is_bid=False)
        
    def process_market_order(self, side: str, quantity: int, timestamp: float):
        """Process a market order"""
        remaining_quantity = quantity
        trades = []
        
        while remaining_quantity > 0:
            if side == 'bid' and not self.asks:
                break  # No more asks to hit
            if side == 'ask' and not self.bids:
                break  # No more bids to hit

            if side == 'bid':
                self._clean_heap(self.asks, is_bid=False)
                if not self.asks:
                    break
                # Buying at best ask
                best_ask_price, best_ask_time, best_ask_id = self.asks[0]
                best_order = self.orders[best_ask_id]
            else:
                self._clean_heap(self.bids, is_bid=True)
                if not self.bids:
                    break
                # Selling at best bid (remember bids are stored as negative)
                best_bid_neg, best_bid_time, best_bid_id = self.bids[0]
                best_bid_price = -best_bid_neg
                best_order = self.orders[best_bid_id]
                
            # Determine trade quantity
            trade_quantity = min(remaining_quantity, best_order.quantity)
            trade_price = best_order.price
            
            # Execute trade
            trades.append({
                'timestamp': timestamp,
                'price': trade_price,
                'quantity': trade_quantity,
                'side': side
            })
            
            # Update order quantity
            best_order.quantity -= trade_quantity
            remaining_quantity -= trade_quantity
            
            # Remove order if fully filled
            if best_order.quantity == 0:
                if side == 'bid':
                    heapq.heappop(self.asks)
                else:
                    heapq.heappop(self.bids)
                self._remove_order_from_queue(best_order.side, best_order.price, best_order.order_id)
                del self.orders[best_order.order_id]

        self.trade_history.extend(trades)
        self._record_market_data()
        return trades

    def get_best_bid_ask(self):
        """Get best bid and ask prices"""
        self._clean_heap(self.bids, is_bid=True)
        self._clean_heap(self.asks, is_bid=False)

        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask

    def _aggregate_price_levels(self, side: str) -> List[Dict[str, object]]:
        """Aggregate remaining quantity at each price level for the given side."""
        price_levels: List[Dict[str, object]] = []
        empty_prices = []
        for price, queue in list(self.order_queues[side].items()):
            valid_ids = deque()
            total_quantity = 0

            for order_id in list(queue):
                order = self.orders.get(order_id)
                if not order or order.quantity <= 0 or order.side != side:
                    continue

                valid_ids.append(order_id)
                total_quantity += order.quantity

            if valid_ids:
                self.order_queues[side][price] = valid_ids
                price_levels.append(
                    {
                        'side': side,
                        'price': price,
                        'quantity': total_quantity,
                        'order_ids': list(valid_ids),
                    }
                )
            else:
                empty_prices.append(price)

        for price in empty_prices:
            del self.order_queues[side][price]

        price_levels.sort(key=lambda level: level['price'], reverse=(side == 'bid'))
        return price_levels

    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, int]]]:
        """Return aggregated depth up to the requested number of levels for each side."""
        depth = {
            'bids': [],
            'asks': [],
        }

        bid_levels = self._aggregate_price_levels('bid')
        ask_levels = self._aggregate_price_levels('ask')

        depth['bids'] = [
            (level['price'], level['quantity']) for level in bid_levels[:levels]
        ]
        depth['asks'] = [
            (level['price'], level['quantity']) for level in ask_levels[:levels]
        ]

        return depth

    def get_order_book_snapshot(self, levels: Optional[int] = None) -> pd.DataFrame:
        """Return a tabular snapshot of the current order book state."""

        bid_levels = self._aggregate_price_levels('bid')
        ask_levels = self._aggregate_price_levels('ask')

        if levels is not None:
            bid_levels = bid_levels[:levels]
            ask_levels = ask_levels[:levels]

        snapshot_records = bid_levels + ask_levels
        if not snapshot_records:
            return pd.DataFrame(columns=['side', 'price', 'quantity', 'order_ids'])

        snapshot_df = pd.DataFrame(snapshot_records, columns=['side', 'price', 'quantity', 'order_ids'])
        snapshot_df['side_rank'] = snapshot_df['side'].map({'bid': 0, 'ask': 1})
        snapshot_df['price_sort'] = snapshot_df.apply(
            lambda row: -row['price'] if row['side'] == 'bid' else row['price'], axis=1
        )
        snapshot_df = snapshot_df.sort_values(
            ['side_rank', 'price_sort'], ascending=[True, True]
        ).reset_index(drop=True)
        snapshot_df = snapshot_df.drop(columns=['side_rank', 'price_sort'])
        return snapshot_df

    def get_mid_price(self):
        """Calculate mid price"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def get_spread(self):
        """Calculate bid-ask spread"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def _check_for_trades(self):
        """Check if any crosses occur after order entry"""
        best_bid, best_ask = self.get_best_bid_ask()
        if (
            best_bid is not None
            and best_ask is not None
            and best_bid >= best_ask
        ):
            # Cross occurred - this is simplified
            print(f"Cross detected: bid {best_bid} >= ask {best_ask}")

    def _record_market_data(self):
        """Record market data for analysis"""
        mid_price = self.get_mid_price()
        spread = self.get_spread()

        if mid_price is not None:
            self.mid_price_history.append(mid_price)
        if spread is not None:
            self.spread_history.append(spread)
    
    def analyze_market_impact(self, order_size):
        """Analyze price impact of market orders"""
        initial_mid = self.get_mid_price()
        
        if initial_mid is None:
            return 0
            
        # Simulate buy market order
        trades = self.process_market_order('bid', order_size, 0)
        
        if not trades:
            return 0
            
        # Calculate volume weighted average price
        total_value = sum(trade['price'] * trade['quantity'] for trade in trades)
        total_quantity = sum(trade['quantity'] for trade in trades)
        vwap = total_value / total_quantity if total_quantity > 0 else 0
        
        price_impact = vwap - initial_mid
        return price_impact
    
    def plot_order_book(self):
        """Visualize current order book state"""
        bid_levels = defaultdict(int)
        ask_levels = defaultdict(int)
        
        for order in self.orders.values():
            if order.side == 'bid':
                bid_levels[order.price] += order.quantity
            else:
                ask_levels[order.price] += order.quantity
        
        if not bid_levels and not ask_levels:
            print("Order book is empty")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot bids
        if bid_levels:
            bid_prices = sorted(bid_levels.keys(), reverse=True)
            bid_quantities = [bid_levels[p] for p in bid_prices]
            ax1.barh(bid_prices, bid_quantities, color='green', alpha=0.6)
            ax1.set_title('Bid Side')
            ax1.set_xlabel('Quantity')
            ax1.set_ylabel('Price')
        
        # Plot asks
        if ask_levels:
            ask_prices = sorted(ask_levels.keys())
            ask_quantities = [ask_levels[p] for p in ask_prices]
            ax2.barh(ask_prices, ask_quantities, color='red', alpha=0.6)
            ax2.set_title('Ask Side')
            ax2.set_xlabel('Quantity')
            ax2.set_ylabel('Price')
        
        plt.tight_layout()
        plt.show()

# Market simulation and analysis
class MarketSimulator:
    def __init__(self):
        self.lob = LimitOrderBook()
        self.order_id = 0
        
    def generate_random_orders(self, n_orders=1000):
        """Generate random orders to simulate market activity"""
        prices = np.cumsum(np.random.normal(0, 0.1, n_orders)) + 100
        quantities = np.random.randint(1, 100, n_orders)
        sides = np.random.choice(['bid', 'ask'], n_orders)
        
        for i in range(n_orders):
            self.lob.add_order(
                order_id=self.order_id,
                side=sides[i],
                price=prices[i],
                quantity=quantities[i],
                timestamp=i
            )
            self.order_id += 1
            
            # Occasionally add market orders
            if i % 50 == 0 and i > 0:
                market_side = 'bid' if np.random.random() > 0.5 else 'ask'
                market_quantity = np.random.randint(10, 200)
                self.lob.process_market_order(market_side, market_quantity, i)
    
    def run_simulation(self):
        """Run complete market simulation"""
        print("Running market simulation...")
        self.generate_random_orders(500)
        
        # Analyze results
        print(f"\n=== MARKET ANALYSIS ===")
        print(f"Total trades: {len(self.lob.trade_history)}")
        print(f"Final mid-price: {self.lob.get_mid_price():.2f}")
        print(f"Final spread: {self.lob.get_spread():.2f}")
        
        # Plot market data
        self.plot_simulation_results()
        
        # Market impact analysis
        impacts = []
        for size in [100, 500, 1000]:
            # Create fresh LOB for each test
            test_lob = LimitOrderBook()
            # Add some liquidity
            for i in range(100):
                test_lob.add_order(i, 'bid', 100 - i*0.1, 100, i)
                test_lob.add_order(i+100, 'ask', 100 + i*0.1, 100, i)
                
            impact = test_lob.analyze_market_impact(size)
            impacts.append(impact)
            print(f"Market impact for {size} shares: {impact:.4f}")
    
    def plot_simulation_results(self):
        """Plot simulation results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot mid-price evolution
        if self.lob.mid_price_history:
            ax1.plot(self.lob.mid_price_history)
            ax1.set_title('Mid-Price Evolution')
            ax1.set_ylabel('Price')
            ax1.grid(True)
        
        # Plot spread evolution
        if self.lob.spread_history:
            ax2.plot(self.lob.spread_history)
            ax2.set_title('Bid-Ask Spread Evolution')
            ax2.set_ylabel('Spread')
            ax2.set_xlabel('Time')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Run the complete LOB simulation
if __name__ == "__main__":
    simulator = MarketSimulator()
    simulator.run_simulation()
    
    # Show final order book state
    simulator.lob.plot_order_book()

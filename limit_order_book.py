import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        self.order_queues = defaultdict(deque)  # price -> deque of orders
        self.trade_history = []
        self.mid_price_history = []
        self.spread_history = []
        
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
            
        self.order_queues[price].append(order_id)
        self._check_for_trades()
        self._record_market_data()
        
    def cancel_order(self, order_id: int):
        """Cancel an existing order"""
        if order_id not in self.orders:
            return
            
        order = self.orders[order_id]
        price_queue = self.order_queues[order.price]
        
        # Remove from price queue
        if order_id in price_queue:
            price_queue.remove(order_id)
            
        # Note: In a real system, we'd need to rebuild heaps, but for simplicity...
        del self.orders[order_id]
        
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
                # Buying at best ask
                best_ask_price, best_ask_time, best_ask_id = self.asks[0]
                best_order = self.orders[best_ask_id]
            else:
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
                del self.orders[best_order.order_id]
                
        self.trade_history.extend(trades)
        self._record_market_data()
        return trades
    
    def get_best_bid_ask(self):
        """Get best bid and ask prices"""
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask
    
    def get_mid_price(self):
        """Calculate mid price"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self):
        """Calculate bid-ask spread"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def _check_for_trades(self):
        """Check if any crosses occur after order entry"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask and best_bid >= best_ask:
            # Cross occurred - this is simplified
            print(f"Cross detected: bid {best_bid} >= ask {best_ask}")
    
    def _record_market_data(self):
        """Record market data for analysis"""
        mid_price = self.get_mid_price()
        spread = self.get_spread()
        
        if mid_price:
            self.mid_price_history.append(mid_price)
        if spread:
            self.spread_history.append(spread)
    
    def analyze_market_impact(self, order_size):
        """Analyze price impact of market orders"""
        initial_mid = self.get_mid_price()
        
        if not initial_mid:
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

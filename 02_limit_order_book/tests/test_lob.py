import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from limit_order_book import LimitOrderBook

class TestLimitOrderBook(unittest.TestCase):
    
    def setUp(self):
        self.lob = LimitOrderBook()
    
    def test_add_bid_order(self):
        self.lob.add_order(1, 'bid', 100.0, 100, 1.0)
        best_bid, best_ask = self.lob.get_best_bid_ask()
        self.assertEqual(best_bid, 100.0)
        self.assertEqual(best_ask, None)
    
    def test_market_order_execution(self):
        self.lob.add_order(1, 'ask', 100.0, 100, 1.0)
        trades = self.lob.process_market_order('bid', 50, 2.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['quantity'], 50)

if __name__ == '__main__':
    unittest.main()

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

    def test_cancel_order_removes_from_book(self):
        self.lob.add_order(1, 'bid', 100.0, 100, 1.0)
        self.lob.cancel_order(1)
        best_bid, best_ask = self.lob.get_best_bid_ask()
        self.assertIsNone(best_bid)
        self.assertIsNone(best_ask)

    def test_mid_price_with_zero_price_orders(self):
        self.lob.add_order(1, 'bid', 0.0, 100, 1.0)
        self.lob.add_order(2, 'ask', 0.01, 100, 2.0)
        self.assertAlmostEqual(self.lob.get_mid_price(), 0.005)
        self.assertAlmostEqual(self.lob.get_spread(), 0.01)

    def test_depth_keeps_bid_and_ask_levels_separate(self):
        self.lob.add_order(1, 'bid', 100.0, 50, 1.0)
        self.lob.add_order(2, 'ask', 100.0, 25, 2.0)

        depth = self.lob.get_depth()

        self.assertEqual(depth['bids'], [(100.0, 50)])
        self.assertEqual(depth['asks'], [(100.0, 25)])

    def test_order_book_snapshot_returns_sorted_levels(self):
        self.lob.add_order(1, 'bid', 100.0, 50, 1.0)
        self.lob.add_order(2, 'bid', 99.5, 25, 2.0)
        self.lob.add_order(3, 'ask', 101.0, 10, 3.0)
        self.lob.add_order(4, 'ask', 101.5, 15, 4.0)

        snapshot = self.lob.get_order_book_snapshot(levels=1)

        self.assertEqual(len(snapshot), 2)
        self.assertEqual(snapshot.iloc[0]['side'], 'bid')
        self.assertEqual(snapshot.iloc[0]['price'], 100.0)
        self.assertEqual(snapshot.iloc[0]['quantity'], 50)
        self.assertEqual(snapshot.iloc[1]['side'], 'ask')
        self.assertEqual(snapshot.iloc[1]['price'], 101.0)
        self.assertEqual(snapshot.iloc[1]['quantity'], 10)

    def test_order_book_snapshot_handles_empty_book(self):
        snapshot = self.lob.get_order_book_snapshot()
        self.assertTrue(snapshot.empty)

if __name__ == '__main__':
    unittest.main()

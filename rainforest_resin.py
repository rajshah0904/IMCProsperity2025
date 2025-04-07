from collections import deque
import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        self.product = "RAINFOREST_RESIN"
        self.position_limit = 10
        self.min_spread = 5
        self.price_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        self.fair_value = 10000  # Center point for mean reversion
        self.last_trade_price = None
        self.position = 0

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return self.fair_value

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        
        if best_bid > 0 and best_ask < float('inf'):
            self.fair_value = (best_bid + best_ask) / 2
        elif best_bid > 0:
            self.fair_value = best_bid
        elif best_ask < float('inf'):
            self.fair_value = best_ask

        return self.fair_value

    def calculate_volatility(self) -> float:
        if len(self.price_history) < 2:
            return 0.0
        
        prices = np.array(list(self.price_history))
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    def run(self, state: TradingState):
        """
        Mean reversion strategy for rainforest resin
        Returns: Dict[str, List[Order]], int, str
        """
        result = {}
        order_depth: OrderDepth = state.order_depths.get(self.product, OrderDepth())
        
        # Update position
        self.position = state.position.get(self.product, 0)
        
        # Get best bid and ask
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        
        # Only trade if we have valid prices
        if best_bid > 0 and best_ask < float('inf'):
            orders: List[Order] = []
            
            # If price is below fair value (10000) and we're not at position limit, BUY
            if best_ask < 10000 and self.position < self.position_limit:
                buy_quantity = min(1, self.position_limit - self.position)
                if buy_quantity > 0:
                    orders.append(Order(self.product, best_ask, buy_quantity))
            
            # If price is above fair value (10000) and we have inventory to sell, SELL
            elif best_bid > 10000 and self.position > -self.position_limit:
                sell_quantity = min(1, self.position_limit + self.position)
                if sell_quantity > 0:
                    orders.append(Order(self.product, best_bid, -sell_quantity))

            result[self.product] = orders

        return result, 0, ""  # Return format expected by backtesting environment

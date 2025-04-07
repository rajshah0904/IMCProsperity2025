from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import numpy as np
from collections import deque

class Trader:
    def __init__(self):
        # Market Making Parameters for RAINFOREST_RESIN
        self.product = "RAINFOREST_RESIN"
        self.position_limit = 10
        self.min_spread = 5
        self.price_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        self.fair_value = 10000  # Center point for mean reversion
        self.last_trade_price = None
        self.position = 0

        # Pairs Trading Parameters
        self.pairs_position_limit = 20
        self.pairs_position_size = 5
        self.short_window = 10
        self.long_window = 50
        self.volatility_window = 20
        self.min_profit_threshold = 0.002  # 0.2% minimum profit target
        
        # Price history for pairs
        self.squid_prices = deque(maxlen=self.long_window)
        self.kelp_prices = deque(maxlen=self.long_window)
        self.ratio_history = deque(maxlen=self.long_window)
        self.volatility_history = deque(maxlen=self.volatility_window)
        
        # Position tracking
        self.positions = {"SQUID_INK": 0, "KELP": 0}
        self.position_values = {"SQUID_INK": 0, "KELP": 0}  # Track average position value

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

    def calculate_pair_signals(self, squid_mid: float, kelp_mid: float) -> tuple:
        """Calculate trading signals for pairs trading"""
        # Add latest prices
        self.squid_prices.append(squid_mid)
        self.kelp_prices.append(kelp_mid)
        
        if len(self.squid_prices) < self.long_window:
            return 0, 0, float('inf')
            
        # Calculate ratio and its moving averages
        ratio = squid_mid / kelp_mid
        self.ratio_history.append(ratio)
        
        short_ma = np.mean(list(self.ratio_history)[-self.short_window:])
        long_ma = np.mean(list(self.ratio_history))
        
        # Calculate volatility
        ratios = np.array(list(self.ratio_history)[-self.volatility_window:])
        current_vol = np.std(ratios)
        self.volatility_history.append(current_vol)
        
        # Calculate dynamic threshold based on recent volatility
        vol_multiplier = np.mean(list(self.volatility_history)) / current_vol
        threshold = max(1.5, min(3.0, 2.0 * vol_multiplier))
        
        return short_ma - long_ma, ratio - long_ma, threshold

    def pairs_trade(self, squid_depth: OrderDepth, kelp_depth: OrderDepth) -> List[Order]:
        """Implement pairs trading strategy with dynamic risk management"""
        orders = []
        
        # Get mid prices
        squid_best_bid = max(squid_depth.buy_orders.keys()) if squid_depth.buy_orders else 0
        squid_best_ask = min(squid_depth.sell_orders.keys()) if squid_depth.sell_orders else float('inf')
        kelp_best_bid = max(kelp_depth.buy_orders.keys()) if kelp_depth.buy_orders else 0
        kelp_best_ask = min(kelp_depth.sell_orders.keys()) if kelp_depth.sell_orders else float('inf')
        
        if not all([squid_best_bid, squid_best_ask, kelp_best_bid, kelp_best_ask]):
            return orders
            
        squid_mid = (squid_best_bid + squid_best_ask) / 2
        kelp_mid = (kelp_best_bid + kelp_best_ask) / 2
        
        # Calculate signals
        trend_signal, deviation_signal, threshold = self.calculate_pair_signals(squid_mid, kelp_mid)
        
        # Calculate position sizes based on current positions
        squid_pos = self.positions["SQUID_INK"]
        kelp_pos = self.positions["KELP"]
        
        # Adjust position size based on current positions
        base_size = self.pairs_position_size
        if abs(squid_pos) > self.pairs_position_limit / 2:
            base_size = max(1, base_size // 2)
        
        # Trading logic with profit targets and risk management
        if deviation_signal > threshold and trend_signal > 0:
            # SQUID_INK overvalued - sell SQUID_INK, buy KELP
            if squid_pos > -self.pairs_position_limit and kelp_pos < self.pairs_position_limit:
                orders.append(Order("SQUID_INK", squid_best_bid, -base_size))
                orders.append(Order("KELP", kelp_best_ask, base_size))
                
        elif deviation_signal < -threshold and trend_signal < 0:
            # SQUID_INK undervalued - buy SQUID_INK, sell KELP
            if squid_pos < self.pairs_position_limit and kelp_pos > -self.pairs_position_limit:
                orders.append(Order("SQUID_INK", squid_best_ask, base_size))
                orders.append(Order("KELP", kelp_best_bid, -base_size))
                
        # Mean reversion for profit taking
        elif abs(deviation_signal) < threshold * 0.5:
            # Close positions if profitable
            if squid_pos > 0 and squid_best_bid > self.position_values["SQUID_INK"] * (1 + self.min_profit_threshold):
                orders.append(Order("SQUID_INK", squid_best_bid, -min(base_size, squid_pos)))
            elif squid_pos < 0 and squid_best_ask < self.position_values["SQUID_INK"] * (1 - self.min_profit_threshold):
                orders.append(Order("SQUID_INK", squid_best_ask, min(base_size, -squid_pos)))
                
            if kelp_pos > 0 and kelp_best_bid > self.position_values["KELP"] * (1 + self.min_profit_threshold):
                orders.append(Order("KELP", kelp_best_bid, -min(base_size, kelp_pos)))
            elif kelp_pos < 0 and kelp_best_ask < self.position_values["KELP"] * (1 - self.min_profit_threshold):
                orders.append(Order("KELP", kelp_best_ask, min(base_size, -kelp_pos)))
        
        return orders

    def run(self, state: TradingState):
        """
        Mean reversion strategy for rainforest resin
        Returns: Dict[str, List[Order]], int, str
        """
        result = {}
        
        # Market make for RAINFOREST_RESIN (unchanged)
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

        # Pairs trade SQUID_INK and KELP
        if "SQUID_INK" in state.order_depths and "KELP" in state.order_depths:
            pairs_orders = self.pairs_trade(
                state.order_depths["SQUID_INK"],
                state.order_depths["KELP"]
            )
            # Split orders by product
            squid_orders = [order for order in pairs_orders if order.symbol == "SQUID_INK"]
            kelp_orders = [order for order in pairs_orders if order.symbol == "KELP"]
            
            if squid_orders:
                result["SQUID_INK"] = squid_orders
            if kelp_orders:
                result["KELP"] = kelp_orders
        
        # Update positions and position values
        for product in state.position:
            self.positions[product] = state.position[product]
            if product in state.order_depths:
                depth = state.order_depths[product]
                if depth.buy_orders and depth.sell_orders:
                    mid_price = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
                    if self.positions[product] != 0:
                        self.position_values[product] = mid_price

        return result, 0, ""  # Return format expected by backtesting environment 
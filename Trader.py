from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import numpy as np

class Trader:
    def __init__(self):
        # Base parameters
        self.base_spread = 4
        self.position_limit = 20
        self.base_position_size = 10
        
        # Dynamic spread adjustment parameters
        self.max_spread = 8
        self.min_spread = 2
        self.volatility_window = 20
        self.price_history = []
        
        # Position management
        self.current_position = 0
        self.target_position = 0
        
        # Risk management
        self.max_position_imbalance = 10
        self.aggressive_reversion = False
        
    def calculate_volatility(self) -> float:
        """Calculate price volatility based on recent price history"""
        if len(self.price_history) < 2:
            return 0
        returns = np.diff(self.price_history)
        return np.std(returns) if len(returns) > 0 else 0
    
    def calculate_dynamic_spread(self) -> int:
        """Calculate spread based on volatility and position"""
        volatility = self.calculate_volatility()
        position_impact = abs(self.current_position) / self.position_limit
        
        # Base spread adjustment based on volatility
        spread = self.base_spread + int(volatility * 10)
        
        # Position-based spread adjustment
        if position_impact > 0.5:
            spread = int(spread * (1 + position_impact))
        
        # Ensure spread stays within bounds
        spread = max(self.min_spread, min(self.max_spread, spread))
        return int(spread)
    
    def calculate_fair_value(self, order_depth: OrderDepth) -> int:
        """Calculate fair value based on order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Calculate weighted mid price
        bid_volume = order_depth.buy_orders[best_bid]
        ask_volume = order_depth.sell_orders[best_ask]
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return int(round((best_bid + best_ask) / 2))
            
        weighted_mid = (best_bid * ask_volume + best_ask * bid_volume) / total_volume
        return int(round(weighted_mid))
    
    def generate_orders(self, order_depth: OrderDepth) -> List[Order]:
        """Generate market making orders"""
        orders: List[Order] = []
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        fair_value = self.calculate_fair_value(order_depth)
        spread = self.calculate_dynamic_spread()
        
        # Update price history
        self.price_history.append(fair_value)
        if len(self.price_history) > self.volatility_window:
            self.price_history.pop(0)
        
        # Calculate position-based order sizes
        position_imbalance = self.current_position / self.position_limit
        buy_size = max(1, int(self.base_position_size * (1 - position_imbalance)))
        sell_size = max(1, int(self.base_position_size * (1 + position_imbalance)))
        
        # Place buy orders
        buy_price = fair_value - spread // 2
        if buy_price > max(order_depth.buy_orders.keys()):
            orders.append(Order("RAINFOREST_RESIN", buy_price, buy_size))
            
        # Place sell orders
        sell_price = fair_value + spread // 2
        if sell_price < min(order_depth.sell_orders.keys()):
            orders.append(Order("RAINFOREST_RESIN", sell_price, -sell_size))
            
        return orders
    
    def run(self, state: TradingState):
        result = {}
        
        if "RAINFOREST_RESIN" in state.order_depths:
            # Update current position
            self.current_position = state.position.get("RAINFOREST_RESIN", 0)
            
            # Generate orders
            orders = self.generate_orders(state.order_depths["RAINFOREST_RESIN"])
            result["RAINFOREST_RESIN"] = orders
        
        traderData = ""  # No state to track across rounds yet
        conversions = 0  # No conversion logic used for now
        
        return result, conversions, traderData

import numpy as np
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import Dict, List, Any
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statistics
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import Dict, List, Any
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statistics
from collections import deque

class Product:
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANT = "CROISSANT"
    JAM = "JAM"
    DJEMBE = "DJEMBE"

PARAMS = {
    Product.PICNIC_BASKET1: {
        "fair_value": 0,
        "take_width": 0.5,
        "clear_width": 0.5,
        "min_profit": 1.5,
        "position_limit": 60,
        "position_step": 12,
        "max_spread": 5.0,
        "min_volume": 5,
        "z_score_threshold": 2.0,
        "volatility_window": 20,
        "dynamic_spread_adjustment": 0.1,
        "component_weights": {
            Product.CROISSANT: 6,
            Product.JAM: 3,
            Product.DJEMBE: 1
        }
    },
    Product.PICNIC_BASKET2: {
        "fair_value": 0,
        "take_width": 0.5,
        "clear_width": 0.5,
        "min_profit": 1.5,
        "position_limit": 60,
        "position_step": 12,
        "max_spread": 5.0,
        "min_volume": 5,
        "z_score_threshold": 2.0,
        "volatility_window": 20,
        "dynamic_spread_adjustment": 0.1,
        "component_weights": {
            Product.CROISSANT: 4,
            Product.JAM: 2
        }
    }
}

class Trader:
    def __init__(self):
        # Product definitions with correct names
        self.products = {
            "PICNIC_BASKET1": {
                "components": {
                    "CROISSANTS": 6,
                    "JAMS": 3,
                    "DJEMBES": 1
                },
                "position_limit": 70
            },
            "PICNIC_BASKET2": {
                "components": {
                    "CROISSANTS": 4,
                    "DJEMBES": 2
                },
                "position_limit": 70
            }
        }
        
        # Position limits for individual products
        self.position_limits = {
            "CROISSANTS": 300,
            "JAMS": 300,
            "DJEMBES": 100
        }
        
        # Minimum profit threshold for arbitrage
        self.min_profit = 10
        
        # Price history for each product
        self.price_history = {
            product: deque(maxlen=100) for product in 
            ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]
        }
        
        # Spread history for each basket
        self.spread_history = {
            "PICNIC_BASKET1": deque(maxlen=20),
            "PICNIC_BASKET2": deque(maxlen=20)
        }
        
        # Last known prices for components
        self.last_prices = {
            "CROISSANTS": None,
            "JAMS": None,
            "DJEMBES": None
        }

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Calculate volume-weighted mid price"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = order_depth.sell_orders[best_ask]

        # Handle zero volume cases
        if best_bid_vol == 0 and best_ask_vol == 0:
            return (best_bid + best_ask) / 2
        elif best_bid_vol == 0:
            return best_ask
        elif best_ask_vol == 0:
            return best_bid

        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def calculate_basket_value(self, basket_name: str, order_depths: Dict[str, OrderDepth]) -> float:
        """Calculate the fair value of a basket based on its components"""
        basket = self.products[basket_name]
        total_value = 0.0
        
        for component, quantity in basket["components"].items():
            order_depth = order_depths.get(component, OrderDepth())
            swmid = self.get_swmid(order_depth)
            if swmid is None:
                return None
                
            total_value += swmid * quantity
            self.last_prices[component] = swmid
            
        return total_value

    def calculate_spread_stats(self, basket_name: str) -> tuple:
        """Calculate spread statistics for a basket"""
        if len(self.spread_history[basket_name]) < 2:
            return 0, 1
            
        spread_mean = np.mean(self.spread_history[basket_name])
        spread_std = np.std(self.spread_history[basket_name])
        return spread_mean, spread_std

    def find_arbitrage_opportunity(self, basket_name: str, basket_price: float, 
                                 component_value: float) -> tuple:
        """Determine if there's an arbitrage opportunity and return the direction"""
        if component_value is None:
            return None, 0
            
        price_diff = basket_price - component_value
        
        # Update spread history
        self.spread_history[basket_name].append(price_diff)
        
        # Calculate spread statistics
        spread_mean, spread_std = self.calculate_spread_stats(basket_name)
        z_score = (price_diff - spread_mean) / spread_std if spread_std != 0 else 0
        
        # Adjust minimum profit based on z-score
        adjusted_min_profit = self.min_profit * (1 + abs(z_score) * 0.1)
        
        if abs(price_diff) < adjusted_min_profit:
            return None, 0
            
        if price_diff > 0:
            # Basket is overpriced, sell basket and buy components
            return "SELL", price_diff
        else:
            # Basket is underpriced, buy basket and sell components
            return "BUY", abs(price_diff)

    def execute_arbitrage(self, state: TradingState, basket_name: str, 
                         direction: str, profit: float) -> Dict[str, List[Order]]:
        """Execute arbitrage trades for the given basket"""
        result = {}
        basket = self.products[basket_name]
        position = state.position.get(basket_name, 0)
        
        # Get basket order depth
        basket_depth = state.order_depths.get(basket_name, OrderDepth())
        if not basket_depth.buy_orders or not basket_depth.sell_orders:
            return result
            
        best_bid = max(basket_depth.buy_orders.keys())
        best_ask = min(basket_depth.sell_orders.keys())
        
        # Calculate maximum quantity we can trade
        max_quantity = min(
            basket["position_limit"] - position if direction == "BUY" else basket["position_limit"] + position,
            basket_depth.buy_orders[best_bid] if direction == "SELL" else basket_depth.sell_orders[best_ask]
        )
        
        if max_quantity <= 0:
            return result
            
        # Execute basket trade
        orders = []
        if direction == "BUY":
            orders.append(Order(basket_name, best_ask, max_quantity))
        else:
            orders.append(Order(basket_name, best_bid, -max_quantity))
        result[basket_name] = orders
        
        # Execute component trades
        for component, quantity in basket["components"].items():
            component_position = state.position.get(component, 0)
            component_depth = state.order_depths.get(component, OrderDepth())
            
            if not component_depth.buy_orders or not component_depth.sell_orders:
                continue
                
            component_best_bid = max(component_depth.buy_orders.keys())
            component_best_ask = min(component_depth.sell_orders.keys())
            
            component_max_quantity = min(
                self.position_limits[component] - component_position if direction == "SELL" 
                else self.position_limits[component] + component_position,
                component_depth.buy_orders[component_best_bid] if direction == "SELL" 
                else component_depth.sell_orders[component_best_ask]
            )
            
            component_quantity = min(max_quantity * quantity, component_max_quantity)
            if component_quantity <= 0:
                continue
                
            if direction == "SELL":
                result[component] = [Order(component, component_best_bid, component_quantity)]
            else:
                result[component] = [Order(component, component_best_ask, -component_quantity)]
                
        return result

    def run(self, state: TradingState):
        """
        Arbitrage strategy for picnic baskets
        Returns: Dict[str, List[Order]], int, str
        """
        result = {}
        
        # Check arbitrage opportunities for each basket
        for basket_name in self.products:
            basket_depth = state.order_depths.get(basket_name, OrderDepth())
            basket_swmid = self.get_swmid(basket_depth)
            if basket_swmid is None:
                continue
                
            # Calculate component value
            component_value = self.calculate_basket_value(basket_name, state.order_depths)
            if component_value is None:
                continue
                
            # Find arbitrage opportunity
            direction, profit = self.find_arbitrage_opportunity(
                basket_name, basket_swmid, component_value)
                
            if direction is not None:
                # Execute arbitrage
                arbitrage_orders = self.execute_arbitrage(state, basket_name, direction, profit)
                result.update(arbitrage_orders)
                
        return result, 0, "" 
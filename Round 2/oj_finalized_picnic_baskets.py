from collections import deque
import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

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
        
        # Hardcoded ARIMA coefficients from training
        self.coefficients = {
            "CROISSANTS": {
                'ar_coef': [-0.13778924473364898, -0.017001430663861707, -0.007126079130103736, 0.003956475454297141, 0.0005939005629995958],
                'ma_coef': [0.0, 0.0, 0.0, 0.0, 0.0],
                'intercept': -0.13778924473364898,
                'sigma2': 1.0
            },
            "JAMS": {
                'ar_coef': [-0.07681975802925964, 0.00855102932364836, 0.004110743183574504, -0.001057105898536105, -0.0070153095407396475],
                'ma_coef': [0.0, 0.0, 0.0, 0.0, 0.0],
                'intercept': -0.07681975802925964,
                'sigma2': 1.0
            },
            "DJEMBES": {
                'ar_coef': [0.014983408336735213, 0.009723454198626398, -0.0072454381893885264, 0.0010821369386140355, -0.002663542650568304],
                'ma_coef': [0.0, 0.0, 0.0, 0.0, 0.0],
                'intercept': 0.014983408336735213,
                'sigma2': 1.0
            }
        }
        
        # Standard deviation threshold for trading
        self.std_threshold = 1.5

    def update_price_history(self, product: str, price: float):
        """Update price history for a product"""
        self.price_history[product].append(price)

    def predict_price(self, product: str) -> float:
        """Predict next price using ARIMA coefficients"""
        if len(self.price_history[product]) < 5:  # Need minimum 5 points for AR(5)
            return None
            
        prices = list(self.price_history[product])
        coeffs = self.coefficients[product]
        
        # Calculate prediction using AR terms
        prediction = coeffs['intercept']
        for i in range(5):
            if i < len(prices):
                prediction += coeffs['ar_coef'][i] * prices[-(i+1)]
                
        return prediction

    def find_prediction_opportunity(self, product: str, current_price: float) -> tuple:
        """Find trading opportunity based on ARIMA predictions"""
        prediction = self.predict_price(product)
        if prediction is None:
            return None, 0
            
        # Calculate standard deviation of recent prices
        recent_prices = list(self.price_history[product])[-20:]
        if len(recent_prices) < 2:
            return None, 0
            
        std = np.std(recent_prices)
        
        # Calculate z-score
        z_score = (current_price - prediction) / std
        
        if abs(z_score) > self.std_threshold:
            if z_score > 0:
                # Price is above prediction, sell
                return "SELL", abs(z_score)
            else:
                # Price is below prediction, buy
                return "BUY", abs(z_score)
                
        return None, 0

    def calculate_basket_value(self, basket_name: str, order_depths: Dict[str, OrderDepth]) -> float:
        """Calculate the fair value of a basket based on its components"""
        basket = self.products[basket_name]
        total_value = 0.0
        
        for component, quantity in basket["components"].items():
            order_depth = order_depths.get(component, OrderDepth())
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return None
                
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.update_price_history(component, mid_price)
            total_value += mid_price * quantity
            
        return total_value

    def find_arbitrage_opportunity(self, basket_name: str, basket_price: float, 
                                 component_value: float) -> tuple:
        """Determine if there's an arbitrage opportunity and return the direction"""
        if component_value is None:
            return None, 0
            
        price_diff = basket_price - component_value
        if abs(price_diff) < self.min_profit:
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

    def execute_prediction_trade(self, state: TradingState, product: str, 
                               direction: str, confidence: float) -> Dict[str, List[Order]]:
        """Execute trades based on ARIMA predictions"""
        result = {}
        position = state.position.get(product, 0)
        order_depth = state.order_depths.get(product, OrderDepth())
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Calculate trade quantity based on confidence
        max_quantity = min(
            self.position_limits[product] - position if direction == "BUY" 
            else self.position_limits[product] + position,
            order_depth.sell_orders[best_ask] if direction == "BUY" 
            else order_depth.buy_orders[best_bid]
        )
        
        # Scale quantity based on confidence
        quantity = int(max_quantity * min(1.0, confidence / self.std_threshold))
        
        if quantity <= 0:
            return result
            
        if direction == "BUY":
            result[product] = [Order(product, best_ask, quantity)]
        else:
            result[product] = [Order(product, best_bid, -quantity)]
            
        return result

    def run(self, state: TradingState):
        """
        Combined strategy using ARIMA predictions and basket arbitrage
        Returns: Dict[str, List[Order]], int, str
        """
        result = {}
        
        # Update price histories
        for product in ["CROISSANTS", "JAMS", "DJEMBES"]:
            order_depth = state.order_depths.get(product, OrderDepth())
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                self.update_price_history(product, mid_price)
        
        # Trade individual products based on ARIMA predictions
        for product in ["CROISSANTS", "JAMS", "DJEMBES"]:
            order_depth = state.order_depths.get(product, OrderDepth())
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue
                
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            
            direction, confidence = self.find_prediction_opportunity(product, mid_price)
            if direction is not None:
                prediction_orders = self.execute_prediction_trade(state, product, direction, confidence)
                result.update(prediction_orders)
        
        # Check arbitrage opportunities for each basket
        for basket_name in self.products:
            basket_depth = state.order_depths.get(basket_name, OrderDepth())
            if not basket_depth.buy_orders or not basket_depth.sell_orders:
                continue
                
            best_bid = max(basket_depth.buy_orders.keys())
            best_ask = min(basket_depth.sell_orders.keys())
            basket_mid_price = (best_bid + best_ask) / 2
            
            # Calculate component value
            component_value = self.calculate_basket_value(basket_name, state.order_depths)
            if component_value is None:
                continue
                
            # Find arbitrage opportunity
            direction, profit = self.find_arbitrage_opportunity(
                basket_name, basket_mid_price, component_value)
                
            if direction is not None:
                # Execute arbitrage
                arbitrage_orders = self.execute_arbitrage(state, basket_name, direction, profit)
                result.update(arbitrage_orders)
                
        return result, 0, "" 
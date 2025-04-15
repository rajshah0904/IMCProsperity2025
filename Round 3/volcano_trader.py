import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from volcano_nn import VolcanicRockNN

class Trader:
    def __init__(self):
        self.model = VolcanicRockNN()
        self.price_history = []
        self.time_history = []
        self.position_limits = {
            "VOLCANIC_ROCK": 20,
            "VOLCANIC_ROCK_VOUCHER_9500": 20,
            "VOLCANIC_ROCK_VOUCHER_9750": 20,
            "VOLCANIC_ROCK_VOUCHER_10000": 20,
            "VOLCANIC_ROCK_VOUCHER_10250": 20,
            "VOLCANIC_ROCK_VOUCHER_10500": 20
        }
        self.positions = {product: 0 for product in self.position_limits.keys()}
        self.min_profit = 5  # Minimum profit threshold for trades
        self.confidence_threshold = 0.7  # Minimum confidence for trading
        
    def update_price_history(self, state: TradingState):
        """Update price history with current market data"""
        if "VOLCANIC_ROCK" in state.order_depths:
            order_depth = state.order_depths["VOLCANIC_ROCK"]
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                self.price_history.append(mid_price)
                self.time_history.append(state.timestamp)
                
                # Keep only last 1000 prices
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history[-1000:]
                    self.time_history = self.time_history[-1000:]
    
    def calculate_confidence(self, predictions: List[float]) -> float:
        """Calculate confidence in predictions based on variance"""
        if len(predictions) < 2:
            return 0
        return 1 - (np.std(predictions) / np.mean(predictions))
    
    def find_trading_opportunity(self, state: TradingState) -> Dict[str, List[Order]]:
        """Find trading opportunities based on model predictions"""
        result = {}
        
        if len(self.price_history) < 20:
            return result
            
        # Get model prediction
        predicted_price = self.model.predict(
            np.array(self.price_history),
            np.array(self.time_history)
        )
        
        # Calculate confidence
        recent_predictions = self.model.predict(
            np.array(self.price_history[-20:]),
            np.array(self.time_history[-20:])
        )
        confidence = self.calculate_confidence([recent_predictions])
        
        if confidence < self.confidence_threshold:
            return result
            
        # Get current market data
        if "VOLCANIC_ROCK" not in state.order_depths:
            return result
            
        order_depth = state.order_depths["VOLCANIC_ROCK"]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_price = (best_bid + best_ask) / 2
        
        # Calculate expected price change
        expected_change = predicted_price - current_price
        
        # Trade underlying if significant price movement expected
        if abs(expected_change) > self.min_profit:
            if expected_change > 0:  # Price expected to rise
                # Buy underlying
                if self.positions["VOLCANIC_ROCK"] < self.position_limits["VOLCANIC_ROCK"]:
                    result["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, 1)]
            else:  # Price expected to fall
                # Sell underlying
                if self.positions["VOLCANIC_ROCK"] > -self.position_limits["VOLCANIC_ROCK"]:
                    result["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -1)]
        
        # Trade options based on predicted price
        for voucher in self.position_limits.keys():
            if "VOUCHER" not in voucher:
                continue
                
            strike = int(voucher.split("_")[-1])
            if voucher not in state.order_depths:
                continue
                
            voucher_depth = state.order_depths[voucher]
            if not voucher_depth.buy_orders or not voucher_depth.sell_orders:
                continue
                
            voucher_bid = max(voucher_depth.buy_orders.keys())
            voucher_ask = min(voucher_depth.sell_orders.keys())
            voucher_price = (voucher_bid + voucher_ask) / 2
            
            # Calculate intrinsic value
            intrinsic_value = max(0, predicted_price - strike)
            
            # Calculate mispricing
            mispricing = intrinsic_value - voucher_price
            
            if abs(mispricing) > self.min_profit:
                if mispricing > 0:  # Option undervalued
                    if self.positions[voucher] < self.position_limits[voucher]:
                        result[voucher] = [Order(voucher, voucher_ask, 1)]
                else:  # Option overvalued
                    if self.positions[voucher] > -self.position_limits[voucher]:
                        result[voucher] = [Order(voucher, voucher_bid, -1)]
        
        return result
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """Main trading function"""
        # Update positions
        for product, position in state.position.items():
            if product in self.positions:
                self.positions[product] = position
        
        # Update price history
        self.update_price_history(state)
        
        # Find trading opportunities
        result = self.find_trading_opportunity(state)
        
        return result 
from collections import deque
import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        # Product definitions
        self.underlying = "VOLCANIC_ROCK"
        self.options = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        
        # Position limits
        self.position_limits = {
            self.underlying: 300
        }
        # Add position limits for each option
        for option in self.options:
            self.position_limits[option] = 100
        
        # Price history for all products
        self.price_history = {
            self.underlying: deque(maxlen=100)
        }
        # Add price history for each option
        for option in self.options:
            self.price_history[option] = deque(maxlen=100)
            
        # Trading parameters
        self.confidence_threshold = 0.7  # Minimum confidence in prediction to trade
        self.min_profit = 10  # Minimum expected profit to trade
        self.safety_margin = 0.05  # 5% safety margin for option pricing
        self.base_quantity = 25  # Base position size
        self.max_drawdown = 0.1  # Maximum allowed drawdown (10%)
        self.stop_loss = 0.05  # Stop loss percentage (5%)
        
        # Performance tracking
        self.initial_capital = 0
        self.current_capital = 0
        self.max_capital = 0
        self.last_trade_prices = {}  # Track last trade prices for stop loss

    def update_price_history(self, product: str, price: float):
        """Update price history for a product"""
        self.price_history[product].append(price)

    def calculate_features(self, prices: np.ndarray) -> tuple:
        """Calculate technical features from price history"""
        if len(prices) < 5:
            return 0, 0, 0, 0, 0
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility (5-period)
        volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0
        
        # Momentum (5-period)
        momentum = (prices[-1] - prices[-5])/prices[-5] if len(prices) >= 5 else 0
        
        # Mean reversion (10-period)
        mean_reversion = (prices[-1] - np.mean(prices[-10:]))/np.mean(prices[-10:]) if len(prices) >= 10 else 0
        
        # Recent trend (3-period)
        trend = (prices[-1] - prices[-3])/prices[-3] if len(prices) >= 3 else 0
        
        # Recent accuracy
        recent_accuracy = 1 - np.mean(np.abs(np.diff(prices[-5:])/prices[-5:-1])) if len(prices) >= 5 else 0
        
        return volatility, momentum, mean_reversion, trend, recent_accuracy

    def predict_future_price(self) -> tuple:
        """Predict future price using multiple features"""
        if len(self.price_history[self.underlying]) < 5:
            return None, 0.0
            
        # Get current price sequence
        prices = np.array(list(self.price_history[self.underlying]))
        
        # Calculate features
        volatility, momentum, mean_reversion, trend, recent_accuracy = self.calculate_features(prices)
        
        # Weighted prediction
        # More weight to mean reversion when price deviates significantly
        mean_reversion_weight = min(1.0, abs(mean_reversion) * 2)
        momentum_weight = 1.0 - mean_reversion_weight
        
        # Base prediction on current price
        base_prediction = prices[-1]
        
        # Adjust prediction based on features
        prediction = base_prediction * (
            1 + 
            0.3 * momentum * momentum_weight -  # Momentum component
            0.2 * mean_reversion * mean_reversion_weight -  # Mean reversion component
            0.1 * trend  # Short-term trend
        )
        
        # Calculate confidence
        # Higher confidence when:
        # - Volatility is low
        # - Recent predictions were accurate
        # - Price is not too far from mean
        volatility_factor = 1 - min(1.0, volatility * 10)  # Scale volatility impact
        mean_reversion_factor = 1 - min(1.0, abs(mean_reversion))  # Lower confidence when far from mean
        confidence = volatility_factor * mean_reversion_factor * recent_accuracy
        
        print(f"Prediction: {prediction:.2f}, Confidence: {confidence:.2f}")
        print(f"Features - Vol: {volatility:.4f}, Mom: {momentum:.4f}, MR: {mean_reversion:.4f}, Trend: {trend:.4f}")
        
        return prediction, confidence

    def check_stop_loss(self, product: str, current_price: float) -> bool:
        """Check if stop loss has been triggered"""
        if product not in self.last_trade_prices:
            return False
            
        last_trade_price = self.last_trade_prices[product]
        price_change = (current_price - last_trade_price) / last_trade_price
        
        # Stop loss triggered if price moved against us by more than threshold
        return abs(price_change) > self.stop_loss

    def find_trading_opportunity(self, spot_price: float, option_price: float, strike_price: int, 
                               predicted_price: float, confidence: float) -> tuple:
        """Find trading opportunity based on predicted price with improved risk management"""
        if confidence < self.confidence_threshold:
            return None, 0
            
        # Calculate expected intrinsic value with safety margin
        expected_intrinsic = max(0, predicted_price - strike_price)
        current_intrinsic = max(0, spot_price - strike_price)
        
        # Add safety margin to option pricing
        safety_margin_value = self.safety_margin * spot_price
        price_diff = option_price - (expected_intrinsic - safety_margin_value)
        
        # Check if profit opportunity is significant
        if abs(price_diff) < self.min_profit:
            return None, 0
            
        # Calculate position size based on confidence and price difference
        # Scale down position size when:
        # - Confidence is lower
        # - Price difference is smaller relative to spot price
        # - Volatility is higher
        volatility, _, _, _, _ = self.calculate_features(np.array(list(self.price_history[self.underlying])))
        volatility_factor = 1 - min(1.0, volatility * 5)  # Reduce position size with higher volatility
        
        price_ratio = min(1.0, abs(price_diff) / spot_price)
        scaled_quantity = int(self.base_quantity * confidence * price_ratio * volatility_factor)
        
        # Ensure minimum position size
        scaled_quantity = max(1, scaled_quantity)
        
        print(f"Trading opportunity - Price diff: {price_diff:.2f}, Quantity: {scaled_quantity}")
        
        if price_diff > 0:
            return "SELL_OPTION", scaled_quantity
        else:
            return "BUY_OPTION", scaled_quantity

    def black_scholes_implied_vol(self, St, Vt, K, TTE):
        """Approximate implied volatility using a simple method."""
        # Simple approximation or fixed value
        return 0.2  # Example fixed value for simplicity

    def compute_mt_vt(self, St, K, TTE, Vt):
        """Compute m_t and v_t for given parameters."""
        m_t = np.log(K / St) / np.sqrt(TTE)
        v_t = self.black_scholes_implied_vol(St, Vt, K, TTE)
        return m_t, v_t

    def smooth_vt(self, vt_values):
        """Simple moving average to smooth v_t values."""
        if len(vt_values) < 3:
            return vt_values
        smoothed = []
        for i in range(1, len(vt_values) - 1):
            smoothed.append((vt_values[i-1] + vt_values[i] + vt_values[i+1]) / 3)
        return smoothed

    def run(self, state: TradingState):
        """
        Options trading strategy using improved prediction and risk management
        Returns: Dict[str, List[Order]], int, str
        """
        result = {}
        
        # Update capital tracking
        if self.initial_capital == 0:
            try:
                self.initial_capital = float(state.traderData) if state.traderData else 0
            except (ValueError, TypeError):
                self.initial_capital = 0
        self.current_capital = self.initial_capital  # Since we don't have real-time capital updates
        self.max_capital = max(self.max_capital, self.current_capital)
        
        # Check for maximum drawdown
        drawdown = (self.max_capital - self.current_capital) / self.max_capital if self.max_capital > 0 else 0
        if drawdown > self.max_drawdown:
            print(f"Maximum drawdown reached: {drawdown:.2%}")
            return result, 0, ""
        
        # First get the spot price
        if self.underlying in state.order_depths:
            spot_depth = state.order_depths[self.underlying]
            if spot_depth.buy_orders and spot_depth.sell_orders:
                spot_best_bid = max(spot_depth.buy_orders.keys())
                spot_best_ask = min(spot_depth.sell_orders.keys())
                spot_mid_price = (spot_best_bid + spot_best_ask) / 2
                
                # Update spot price history and get prediction
                self.update_price_history(self.underlying, spot_mid_price)
                predicted_price, confidence = self.predict_future_price()
                
                if predicted_price is not None:
                    vt_values = []
                    for option_name, strike_price in self.options.items():
                        if option_name in state.order_depths:
                            option_depth = state.order_depths[option_name]
                            if option_depth.buy_orders and option_depth.sell_orders:
                                option_best_bid = max(option_depth.buy_orders.keys())
                                option_best_ask = min(option_depth.sell_orders.keys())
                                option_mid_price = (option_best_bid + option_best_ask) / 2
                                
                                # Update option price history
                                self.update_price_history(option_name, option_mid_price)
                                
                                # Compute m_t and v_t
                                TTE = 1  # Placeholder for actual time to expiry
                                m_t, v_t = self.compute_mt_vt(spot_mid_price, strike_price, TTE, option_mid_price)
                                vt_values.append(v_t)
                                
                                # Use m_t and v_t to evaluate opportunities
                                # Example: print or log the values
                                print(f"m_t: {m_t}, v_t: {v_t}")
                                
                                # Check stop loss
                                if self.check_stop_loss(option_name, option_mid_price):
                                    print(f"Stop loss triggered for {option_name}")
                                    continue
                                
                                direction, quantity = self.find_trading_opportunity(
                                    spot_mid_price, option_mid_price, strike_price,
                                    predicted_price, confidence
                                )
                                
                                if direction is not None:
                                    print(f"Found opportunity: {direction} with quantity {quantity}")
                                    # Get current positions
                                    spot_position = state.position.get(self.underlying, 0)
                                    option_position = state.position.get(option_name, 0)
                                    
                                    if direction == "SELL_OPTION":
                                        # Calculate maximum quantities based on position limits
                                        max_option_quantity = min(
                                            self.position_limits[option_name] + option_position,
                                            quantity
                                        )
                                        
                                        max_spot_quantity = min(
                                            self.position_limits[self.underlying] - spot_position,
                                            quantity
                                        )
                                        
                                        # Take the minimum of the two to maintain delta neutrality
                                        final_quantity = min(max_option_quantity, max_spot_quantity)
                                        
                                        if final_quantity > 0:
                                            result[option_name] = [Order(option_name, option_best_bid, -final_quantity)]
                                            result[self.underlying] = [Order(self.underlying, spot_best_ask, final_quantity)]
                                            self.last_trade_prices[option_name] = option_mid_price
                                            
                                    elif direction == "BUY_OPTION":
                                        # Calculate maximum quantities based on position limits
                                        max_option_quantity = min(
                                            self.position_limits[option_name] - option_position,
                                            quantity
                                        )
                                        
                                        max_spot_quantity = min(
                                            self.position_limits[self.underlying] + spot_position,
                                            quantity
                                        )
                                        
                                        # Take the minimum of the two to maintain delta neutrality
                                        final_quantity = min(max_option_quantity, max_spot_quantity)
                                        
                                        if final_quantity > 0:
                                            result[option_name] = [Order(option_name, option_best_ask, final_quantity)]
                                            result[self.underlying] = [Order(self.underlying, spot_best_bid, -final_quantity)]
                                            self.last_trade_prices[option_name] = option_mid_price

                    # Smooth v_t values
                    smoothed_vt = self.smooth_vt(vt_values)
                    print(f"Smoothed v_t: {smoothed_vt}")
        
        return result, 0, str(self.current_capital)  # Return current capital as string 
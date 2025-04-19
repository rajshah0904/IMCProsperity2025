import numpy as np
import math
from collections import deque, defaultdict
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState
import jsonpickle  # if needed for further serialization
import logging

class Trader:
    def __init__(self):
        # ========= COMMON VARIABLES =========
        self.positions = {}
        self.logger = set_up_logger("trader", logging.ERROR)
        self.pattern_products = ["SQUID_INK", "KELP"]

        # ========= PATTERN RECOGNITION VARIABLES =========
        self.position_limit_pattern = 50
        self.iteration_pattern = 0
        
        # Save price history
        self.pattern_price_history = {
            "SQUID_INK": deque(maxlen=50),
            "KELP": deque(maxlen=50)
        }
        
        # Initialize data structures for both products
        self.recent_highs = {"SQUID_INK": deque(maxlen=10), "KELP": deque(maxlen=10)}
        self.recent_lows = {"SQUID_INK": deque(maxlen=10), "KELP": deque(maxlen=10)}
        self.rsi_values = {"SQUID_INK": deque(maxlen=20), "KELP": deque(maxlen=20)}
        self.volatility_values = {"SQUID_INK": deque(maxlen=20), "KELP": deque(maxlen=20)}
        self.positions_pattern = {"SQUID_INK": 0, "KELP": 0}
        self.trade_duration = {"SQUID_INK": 0, "KELP": 0}
        self.market_state = {"SQUID_INK": "neutral", "KELP": "neutral"}
        self.pattern_detected = {"SQUID_INK": None, "KELP": None}
        self.trade_started_at = {"SQUID_INK": 0, "KELP": 0}
        
        # ========= CONVERSION VARIABLES =========
        # ========= Sub-accounts for each strategy ==========
        self.resin_positions: Dict[str, int] = defaultdict(int)
        self.pattern_positions: Dict[str, int] = defaultdict(int)
        self.volcano_positions: Dict[str, int] = defaultdict(int)

        # ========= Resin Market Maker Parameters ==========
        self.alloc_resin = 0.33
        self.base_spread = 4
        self.market_position_limit = int(20 * self.alloc_resin)
        self.base_position_size = int(10 * self.alloc_resin)
        self.max_spread = 8
        self.min_spread = 2
        self.volatility_window = 20
        self.resin_price_history: List[float] = []
        self.current_position_resin = 0

        # ========= Pattern Recognition Parameters ==========
        self.alloc_pattern = 0.34
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volatility_window_pattern = 20
        self.volatility_breakout_threshold = 1.5
        self.base_trade_size = int(3 * self.alloc_pattern)
        self.max_trade_size = int(7 * self.alloc_pattern)
        self.max_trade_duration = 15
        self.historical_pl = 0

        # ========= Volcano Options Parameters ==========
        self.volcano_underlying = "VOLCANIC_ROCK"
        # Focus only on the options that were profitable in previous testing
        self.volcano_options = {
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250
        }
        self.volcano_position_limits = {self.volcano_underlying: 300}
        for option in self.volcano_options:
            self.volcano_position_limits[option] = 100

        self.volcano_price_history = {self.volcano_underlying: deque(maxlen=100)}
        for option in self.volcano_options:
            self.volcano_price_history[option] = deque(maxlen=100)

        self.volcano_confidence_threshold = 0.8  # Increased from 0.7 for higher confidence
        self.volcano_min_profit = 15  # Increased from 10 for higher profit margin
        self.volcano_safety_margin = 0.08  # Increased from 0.05 for safer trades
        self.volcano_base_quantity = 20  # Reduced from 25 for smaller positions
        self.volcano_max_drawdown = 0.1
        self.volcano_stop_loss = 0.05
        self.volcano_initial_capital = 0
        self.volcano_current_capital = 0
        self.volcano_max_capital = 0
        self.volcano_last_trade_prices = {}
        
        # Product performance tracker
        self.product_performance = defaultdict(float)

    # Add a method to track product performance
    def update_product_performance(self, product, profit):
        self.product_performance[product] += profit
        
    # Add a method to check if a product is profitable to trade
    def is_profitable_to_trade(self, product):
        # If we have no performance data, be conservative
        if product not in self.product_performance:
            return False
        
        # Check if the product has been profitable
        return self.product_performance[product] >= 0

    #
    # ========= Utility Methods =========
    #
    def create_sub_state(self, global_state: TradingState, local_positions: Dict[str, int]) -> TradingState:
        """
        Create a new TradingState using all required fields from global_state,
        but with the 'position' replaced by the provided local_positions.
        """
        return TradingState(
            traderData = global_state.traderData,
            timestamp = global_state.timestamp,
            listings = global_state.listings,
            order_depths = global_state.order_depths,
            own_trades = global_state.own_trades,
            market_trades = global_state.market_trades,
            position = local_positions,  # Use the local sub-account
            observations = global_state.observations
        )

    def combine_orders_no_netting(self, orders_list: List[Dict[str, List[Order]]]) -> Dict[str, List[Order]]:
        combined: Dict[str, List[Order]] = {}
        
        # List of losing products from the analysis
        losing_products = [
            "SQUID_INK",
            "VOLCANIC_ROCK_VOUCHER_10500",
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750"
        ]
        
        for strat_orders in orders_list:
            for product, orders in strat_orders.items():
                # Skip products we know are losing money
                if product in losing_products:
                    continue
                    
                # Focus only on products that are profitable
                if product != self.volcano_underlying:  # Don't include orders for volcano underlying
                    if product not in combined:
                        combined[product] = []
                    combined[product].extend(orders)
        return combined

    #
    # ========= RESIN MARKET MAKER METHODS =========
    #
    def resin_calculate_volatility(self) -> float:
        if len(self.resin_price_history) < 2:
            return 0
        rets = np.diff(self.resin_price_history)
        return np.std(rets) if len(rets) > 0 else 0

    def calculate_dynamic_spread(self) -> int:
        vol = self.resin_calculate_volatility()
        spread = self.base_spread + int(vol * 10)
        return max(self.min_spread, min(self.max_spread, spread))

    def calculate_fair_value(self, od: OrderDepth) -> int:
        if not od.buy_orders or not od.sell_orders:
            return 0
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = od.sell_orders[best_ask]
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return (best_bid + best_ask) // 2
        weighted_mid = (best_bid * ask_vol + best_ask * bid_vol) / total_vol
        return int(round(weighted_mid))

    def generate_resin_orders(self, od: OrderDepth) -> List[Order]:
        orders = []
        if not od.buy_orders or not od.sell_orders:
            return orders

        fair_value = self.calculate_fair_value(od)
        spread = self.calculate_dynamic_spread()
        self.resin_price_history.append(fair_value)
        if len(self.resin_price_history) > self.volatility_window:
            self.resin_price_history.pop(0)

        # More aggressive position sizing
        buy_size = max(5, int(self.base_position_size * 1.5))  # Increased size
        sell_size = max(5, int(self.base_position_size * 1.5))  # Increased size

        # Always create both buy and sell orders with tight spread
        buy_price = max(od.buy_orders.keys())  # Use best bid
        orders.append(Order("RAINFOREST_RESIN", buy_price, buy_size))

        sell_price = min(od.sell_orders.keys())  # Use best ask
        orders.append(Order("RAINFOREST_RESIN", sell_price, -sell_size))
                
        return orders

    def run_resin(self, sub_state: TradingState):
        result = {}
        if "RAINFOREST_RESIN" in sub_state.order_depths:
            od = sub_state.order_depths["RAINFOREST_RESIN"]
            orders = self.generate_resin_orders(od)
            result["RAINFOREST_RESIN"] = orders
        return result, 0, ""

    #
    # ========= PATTERN RECOGNITION METHODS =========
    #
    def detect_patterns(self, product, prices):
        if len(prices) < 20:
            return None
        
        # Get recent price data
        recent_prices = list(prices)[-20:]
        
        # Calculate price changes
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        # Update recent highs and lows
        if len(recent_prices) >= 3:
            if recent_prices[-2] > recent_prices[-3] and recent_prices[-2] > recent_prices[-1]:
                self.recent_highs[product].append(recent_prices[-2])
            if recent_prices[-2] < recent_prices[-3] and recent_prices[-2] < recent_prices[-1]:
                self.recent_lows[product].append(recent_prices[-2])
        
        # Calculate short-term and long-term moving averages
        short_ma = sum(recent_prices[-5:]) / 5
        long_ma = sum(recent_prices[-15:]) / 15
        
        # Calculate RSI
        if len(price_changes) >= 14:
            gains = [max(0, change) for change in price_changes[-14:]]
            losses = [abs(min(0, change)) for change in price_changes[-14:]]
            
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            self.rsi_values[product].append(rsi)
        
        # Detect patterns
        if len(self.rsi_values[product]) > 0:
            rsi = self.rsi_values[product][-1]
            
            # Trend following
            if short_ma > long_ma * 1.01:
                return "uptrend"
            elif short_ma < long_ma * 0.99:
                return "downtrend"
            
            # Oversold/overbought conditions
            if rsi < 30:
                return "oversold"
            elif rsi > 70:
                return "overbought"
        
        # Check for double tops/bottoms
        if len(self.recent_highs[product]) >= 2:
            last_two_highs = list(self.recent_highs[product])[-2:]
            if abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] < 0.02:
                return "double_top"
                
        if len(self.recent_lows[product]) >= 2:
            last_two_lows = list(self.recent_lows[product])[-2:]
            if abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] < 0.02:
                return "double_bottom"
        
        return "neutral"

    def run_pattern(self, sub_state: TradingState):
        self.iteration_pattern += 1
        result = {}
        
        for product in self.pattern_products:
            od = sub_state.order_depths.get(product, OrderDepth())
            if not od.buy_orders or not od.sell_orders:
                continue
                
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            
            # Store price history
            self.pattern_price_history[product].append(mid_price)
            
            # Get current position
            current_pos = sub_state.position.get(product, 0)
            self.positions_pattern[product] = current_pos
            
            # Skip if we don't have enough price history
            if len(self.pattern_price_history[product]) < 20:
                continue
            
            # Detect patterns and update market state
            pattern = self.detect_patterns(product, self.pattern_price_history[product])
            self.pattern_detected[product] = pattern
            
            # Trade based on detected pattern
            if pattern == "uptrend" or pattern == "double_bottom":
                # Bullish signals - buy aggressively if we're not already at our position limit
                if current_pos < self.position_limit_pattern:
                    buy_size = min(15, self.position_limit_pattern - current_pos)
                    result[product] = [Order(product, best_ask, buy_size)]
            
            elif pattern == "downtrend" or pattern == "double_top":
                # Bearish signals - sell aggressively if we're not already at our negative position limit
                if current_pos > -self.position_limit_pattern:
                    sell_size = min(15, self.position_limit_pattern + current_pos)
                    result[product] = [Order(product, best_bid, -sell_size)]
            
            elif pattern == "oversold":
                # Oversold - good buying opportunity
                if current_pos < self.position_limit_pattern:
                    buy_size = min(10, self.position_limit_pattern - current_pos)
                    result[product] = [Order(product, best_ask, buy_size)]
            
            elif pattern == "overbought":
                # Overbought - good selling opportunity
                if current_pos > -self.position_limit_pattern:
                    sell_size = min(10, self.position_limit_pattern + current_pos)
                    result[product] = [Order(product, best_bid, -sell_size)]
            
            # Simple mean reversion if no strong patterns detected
            elif pattern == "neutral":
                mean_price = sum(list(self.pattern_price_history[product])[-10:]) / 10
                
                if mid_price > mean_price * 1.01:  # Price is above mean, sell
                    sell_size = min(8, self.position_limit_pattern + current_pos)
                    if sell_size > 0:
                        result[product] = [Order(product, best_bid, -sell_size)]
                elif mid_price < mean_price * 0.99:  # Price is below mean, buy
                    buy_size = min(8, self.position_limit_pattern - current_pos)
                    if buy_size > 0:
                        result[product] = [Order(product, best_ask, buy_size)]
        
        return result, 0, f"Iter: {self.iteration_pattern}"

    def simple_linear_regression(self, x, y):
        """
        Calculate simple linear regression using numpy.
        Returns slope, intercept, and R-squared.
        """
        n = len(x)
        if n < 2:
            return 0, 0, 0
        
        # Calculate means
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Calculate slope (beta)
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        
        if denominator == 0:
            return 0, mean_y, 0
            
        slope = numerator / denominator
        
        # Calculate intercept (alpha)
        intercept = mean_y - slope * mean_x
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_total = np.sum((y - mean_y) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        
        if ss_total == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_residual / ss_total)
            
        return slope, intercept, r_squared

    def linear_regression_predict(self, prices, forecast_horizon=5):
        """
        Use linear regression to predict future prices based on historical data.
        Returns predicted prices and the slope of the trend.
        """
        if len(prices) < 10:  # Need sufficient data for prediction
            return None, 0
            
        # Convert to numpy array for processing
        price_array = np.array(list(prices))
        x = np.arange(len(price_array))
        
        # Fit linear regression model
        slope, intercept, r_squared = self.simple_linear_regression(x, price_array)
        
        # Predict future prices
        future_predictions = []
        for i in range(len(price_array), len(price_array) + forecast_horizon):
            future_predictions.append(slope * i + intercept)
        
        return np.array(future_predictions), slope

    #
    # ========= VOLCANO OPTIONS METHODS =========
    #
    def update_volcano_price_history(self, product: str, price: float):
        self.volcano_price_history[product].append(price)

    def calculate_features_volcano(self, prices: np.ndarray) -> tuple:
        """Calculate technical features from price history for volcano strategy"""
        if len(prices) < 5:
            return 0, 0, 0, 0, 0
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        mean_reversion = (prices[-1] - np.mean(prices[-10:])) / np.mean(prices[-10:]) if len(prices) >= 10 else 0
        trend = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0
        recent_accuracy = 1 - np.mean(np.abs(np.diff(prices[-5:]) / prices[-5:-1])) if len(prices) >= 5 else 0
        return volatility, momentum, mean_reversion, trend, recent_accuracy

    def predict_future_price_volcano(self) -> tuple:
        """Predict future price for the underlying using volcano features and linear regression"""
        if len(self.volcano_price_history[self.volcano_underlying]) < 5:
            return None, 0.0
        
        # Get the underlying price history
        prices = np.array(list(self.volcano_price_history[self.volcano_underlying]))
        
        # Use linear regression for primary prediction
        future_prices, trend_slope = self.linear_regression_predict(self.volcano_price_history[self.volcano_underlying], 5)
        
        # If linear regression provides a prediction, use it as the base
        if future_prices is not None:
            base_prediction = future_prices[0]  # Use the first predicted price
            confidence = min(1.0, 0.6 + abs(trend_slope) * 5)  # Scale confidence based on trend strength
        else:
            # Fall back to the old method if linear regression doesn't have enough data
            volatility, momentum, mean_reversion, trend, recent_accuracy = self.calculate_features_volcano(prices)
            momentum_weight = 1.0 - min(1.0, abs(mean_reversion) * 2)
            base_prediction = prices[-1]
            base_prediction = base_prediction * (1 + 0.3 * momentum * momentum_weight - 0.2 * mean_reversion - 0.1 * trend)
            volatility_factor = 1 - min(1.0, volatility * 10)
            mean_reversion_factor = 1 - min(1.0, abs(mean_reversion))
            confidence = volatility_factor * mean_reversion_factor * recent_accuracy
        
        return base_prediction, confidence

    def calculate_m_t_v_t(self, spot_price: float, strike_price: int, TTE: float, option_price: float) -> tuple:
        """
        Calculate m_t and v_t using a simplified Black-Scholes approach.
        m_t = log(K/St) / sqrt(TTE)
        v_t is a placeholder for implied volatility.
        """
        m_t = math.log(strike_price / spot_price) / math.sqrt(TTE)
        v_t = self.calculate_implied_volatility(spot_price, strike_price, TTE, option_price)
        return m_t, v_t

    def calculate_implied_volatility(self, S: float, K: float, T: float, market_price: float) -> float:
        """Calculate implied volatility using the Black-Scholes formula"""
        sigma = 0.2  # Starting guess for volatility
        tolerance = 0.0001
        max_iterations = 100

        for _ in range(max_iterations):
            price = self.black_scholes(S, K, T, sigma)
            vega = self.calculate_vega(S, K, T, sigma)
            if vega == 0:
                break
            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma
            sigma = sigma + diff / vega
        return sigma

    def black_scholes(self, S: float, K: float, T: float, sigma: float) -> float:
        """Black-Scholes formula for option price"""
        r = 0.05  # Assume a constant risk-free rate
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        return price

    def norm_cdf(self, x: float) -> float:
        """Cumulative distribution function for the standard normal distribution"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def calculate_vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate Vega (the sensitivity of the option price to changes in volatility)"""
        r = 0.05  # Assume a constant risk-free rate
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * math.sqrt(T) * self.norm_pdf(d1)

    def norm_pdf(self, x: float) -> float:
        """Standard normal probability density function"""
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    def find_trading_opportunity_volcano(self, spot_price: float, option_price: float, strike_price: int,
                                           predicted_price: float, confidence: float) -> tuple:
        if confidence < self.volcano_confidence_threshold:
            return None, 0
        expected_intrinsic = max(0, predicted_price - strike_price)
        current_intrinsic = max(0, spot_price - strike_price)
        safety_margin_value = self.volcano_safety_margin * spot_price
        price_diff = option_price - (expected_intrinsic - safety_margin_value)
        if abs(price_diff) < self.volcano_min_profit:
            return None, 0
        volatility, _, _, _, _ = self.calculate_features_volcano(np.array(list(self.volcano_price_history[self.volcano_underlying])))
        volatility_factor = 1 - min(1.0, volatility * 5)
        price_ratio = min(1.0, abs(price_diff) / spot_price)
        scaled_quantity = int(self.volcano_base_quantity * confidence * price_ratio * volatility_factor)
        scaled_quantity = max(1, scaled_quantity)
        if price_diff > 0:
            return "SELL_OPTION", scaled_quantity
        else:
            return "BUY_OPTION", scaled_quantity

    def run_volcano(self, sub_state: TradingState):
        result = {}
        
        # Skip all the complex logic and just trade the two profitable options aggressively
        for option_name, strike_price in self.volcano_options.items():
            if option_name in sub_state.order_depths:
                option_depth = sub_state.order_depths[option_name]
                if option_depth.buy_orders and option_depth.sell_orders:
                    option_best_bid = max(option_depth.buy_orders.keys())
                    option_best_ask = min(option_depth.sell_orders.keys())
                    
                    # Get current position
                    option_position = sub_state.position.get(option_name, 0)
                    max_position = self.volcano_position_limits[option_name]
                    
                    # Simple strategy: if position is negative, buy; if position is positive, sell
                    if option_position < 0:
                        # We're short, so buy to reduce position
                        buy_quantity = min(max_position + option_position, 10)
                        if buy_quantity > 0:
                            result[option_name] = [Order(option_name, option_best_ask, buy_quantity)]
                    else:
                        # We're long or neutral, so sell
                        sell_quantity = min(max_position - option_position, 10)
                        if sell_quantity > 0:
                            result[option_name] = [Order(option_name, option_best_bid, -sell_quantity)]
        
        return result, 0, ""

    #
    # ========= MAIN run() METHOD =========
    #
    def run(self, global_state: TradingState):
        # Create sub-states for each strategy using global_state values
        state_resin = self.create_sub_state(global_state, self.resin_positions)
        orders_resin, conv_resin, data_resin = self.run_resin(state_resin)

        state_pattern = self.create_sub_state(global_state, self.pattern_positions)
        orders_pattern, conv_pattern, data_pattern = self.run_pattern(state_pattern)

        state_volcano = self.create_sub_state(global_state, self.volcano_positions)
        orders_volcano, conv_volcano, data_volcano = self.run_volcano(state_volcano)

        # Simplified combined orders - Don't filter any products to ensure trading happens
        combined_orders = {}
        
        # Add orders from each strategy
        for strategy_orders in [orders_resin, orders_pattern, orders_volcano]:
            for product, orders in strategy_orders.items():
                # Skip SQUID_INK as it was not profitable previously
                if product == "SQUID_INK":
                    continue
                    
                # For all other products, include the orders without filtering
                if product not in combined_orders:
                    combined_orders[product] = []
                combined_orders[product].extend(orders)
                
        conversions = conv_resin + conv_pattern + conv_volcano
        combined_data = f"{data_resin} | {data_pattern} | {data_volcano}"
        return combined_orders, conversions, combined_data
 
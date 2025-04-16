import numpy as np
import math
from collections import deque, defaultdict
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState
import jsonpickle  # if needed for further serialization

class Trader:
    def __init__(self):
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
        self.iteration_pattern = 0
        self.pattern_products = ["SQUID_INK"]
        self.pattern_price_history = {
            "SQUID_INK": deque(maxlen=100)
        }
        self.recent_highs = {"SQUID_INK": deque(maxlen=10)}
        self.recent_lows = {"SQUID_INK": deque(maxlen=10)}
        self.rsi_period = 14
        self.rsi_values = {"SQUID_INK": deque(maxlen=20)}
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volatility_window_pattern = 20
        self.volatility_values = {"SQUID_INK": deque(maxlen=50)}
        self.volatility_breakout_threshold = 1.5
        self.positions_pattern = {"SQUID_INK": 0}
        self.position_limit_pattern = int(15 * self.alloc_pattern)
        self.base_trade_size = int(3 * self.alloc_pattern)
        self.max_trade_size = int(7 * self.alloc_pattern)
        self.trade_duration = {"SQUID_INK": 0}
        self.max_trade_duration = 15
        self.market_state = {"SQUID_INK": "neutral"}
        self.pattern_detected = {"SQUID_INK": None}
        self.historical_pl = 0
        self.trade_started_at = {"SQUID_INK": 0}

        # ========= Volcano Options Parameters ==========
        self.volcano_underlying = "VOLCANIC_ROCK"
        self.volcano_options = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        self.volcano_position_limits = {self.volcano_underlying: 300}
        for option in self.volcano_options:
            self.volcano_position_limits[option] = 100

        self.volcano_price_history = {self.volcano_underlying: deque(maxlen=100)}
        for option in self.volcano_options:
            self.volcano_price_history[option] = deque(maxlen=100)

        self.volcano_confidence_threshold = 0.7
        self.volcano_min_profit = 10
        self.volcano_safety_margin = 0.05
        self.volcano_base_quantity = 25
        self.volcano_max_drawdown = 0.1
        self.volcano_stop_loss = 0.05
        self.volcano_initial_capital = 0
        self.volcano_current_capital = 0
        self.volcano_max_capital = 0
        self.volcano_last_trade_prices = {}

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
        for strat_orders in orders_list:
            for product, orders in strat_orders.items():
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

        pos_imbalance = 0  # (Could be computed based on your current position.)
        buy_size = max(1, int(self.base_position_size * (1 - pos_imbalance)))
        sell_size = max(1, int(self.base_position_size * (1 + pos_imbalance)))

        buy_price = fair_value - spread // 2
        if buy_price > max(od.buy_orders.keys()):
            orders.append(Order("RAINFOREST_RESIN", buy_price, buy_size))

        sell_price = fair_value + spread // 2
        if sell_price < min(od.sell_orders.keys()):
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
    def calculate_rsi(self, prices: deque, period=14) -> float:
        if len(prices) < period + 1:
            return 50
        arr = np.array(list(prices)[-period-1:])
        deltas = np.diff(arr)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def pattern_calculate_volatility(self, prices: deque, window=20) -> float:
        if len(prices) < window:
            return 0.01
        recent = list(prices)[-window:]
        return np.std(recent) / np.mean(recent)

    def detect_patterns(self, product: str, price: float):
        prices = self.pattern_price_history[product]
        if len(prices) < 5:
            return None
        pattern = None
        recent_prices = list(prices)[-5:]
        if len(prices) > 2:
            if len(recent_prices) >= 3 and recent_prices[-2] > recent_prices[-3] and recent_prices[-2] > recent_prices[-1]:
                self.recent_highs[product].append(recent_prices[-2])
            if len(recent_prices) >= 3 and recent_prices[-2] < recent_prices[-3] and recent_prices[-2] < recent_prices[-1]:
                self.recent_lows[product].append(recent_prices[-2])
            if len(self.recent_lows[product]) >= 2:
                lows = list(self.recent_lows[product])
                if abs(lows[-1] - lows[-2]) / lows[-1] < 0.03:
                    pattern = "double_bottom"
            if len(self.recent_highs[product]) >= 2:
                highs = list(self.recent_highs[product])
                if abs(highs[-1] - highs[-2]) / highs[-1] < 0.03:
                    pattern = "double_top"
        if len(prices) > 10:
            rng = max(list(prices)[-10:]) - min(list(prices)[-10:])
            if rng > 0:
                if price > max(list(prices)[-10:-1]) * 1.02:
                    pattern = "breakout_up"
                elif price < min(list(prices)[-10:-1]) * 0.98:
                    pattern = "breakout_down"
        rsi_val = self.calculate_rsi(prices, self.rsi_period)
        self.rsi_values[product].append(rsi_val)
        vol_val = self.pattern_calculate_volatility(prices, self.volatility_window_pattern)
        self.volatility_values[product].append(vol_val)
        if rsi_val < self.rsi_oversold:
            pattern = "oversold"
        if rsi_val > self.rsi_overbought:
            pattern = "overbought"
        if len(self.volatility_values[product]) > 5:
            avg_vol = np.mean(list(self.volatility_values[product])[-5:])
            if vol_val > avg_vol * self.volatility_breakout_threshold:
                pattern = "volatility_breakout_up" if recent_prices[-1] > recent_prices[-2] else "volatility_breakout_down"
        return pattern

    def determine_market_state(self, product: str, pattern: str) -> str:
        rsi_hist = self.rsi_values[product]
        rsi_val = rsi_hist[-1] if len(rsi_hist) else 50
        state = "neutral"
        if pattern in ["double_bottom", "oversold", "breakout_up", "volatility_breakout_up"]:
            state = "bullish"
        elif pattern in ["double_top", "overbought", "breakout_down", "volatility_breakout_down"]:
            state = "bearish"
        if rsi_val < 20:
            state = "strongly_bullish"
        elif rsi_val > 80:
            state = "strongly_bearish"
        return state

    def calculate_position_size(self, product: str, price: float, state: str) -> int:
        size = self.base_trade_size
        if state in ["strongly_bullish", "strongly_bearish"]:
            size = int(size * 1.5)
        current_pos = self.positions_pattern[product]
        factor = max(0.5, 1.0 - (abs(current_pos) / self.position_limit_pattern) * 0.7)
        size = max(1, int(size * factor))
        if len(self.volatility_values[product]) > 5:
            recent_vol = self.volatility_values[product][-1]
            avg_vol = np.mean(list(self.volatility_values[product])[-5:])
            if recent_vol > avg_vol * 1.5:
                size = max(1, int(size * 0.7))
            elif recent_vol < avg_vol * 0.5:
                size = min(self.max_trade_size, int(size * 1.2))
        size = min(size, self.max_trade_size)
        available_long = self.position_limit_pattern - current_pos
        available_short = self.position_limit_pattern + current_pos
        if state in ["bullish", "strongly_bullish"]:
            size = min(size, available_long)
        elif state in ["bearish", "strongly_bearish"]:
            size = min(size, available_short)
        return size

    def should_exit_trade(self, product: str) -> bool:
        if self.trade_duration[product] > self.max_trade_duration:
            return True
        curr_pos = self.positions_pattern[product]
        curr_state = self.market_state[product]
        if curr_pos > 0 and curr_state in ["bearish", "strongly_bearish"]:
            return True
        if curr_pos < 0 and curr_state in ["bullish", "strongly_bullish"]:
            return True
        return False

    def run_pattern(self, sub_state: TradingState):
        self.iteration_pattern += 1
        result = {}
        pl_diff = 0
        if hasattr(sub_state, 'observations') and sub_state.observations:
            try:
                pl_diff = float(sub_state.observations.PROFIT_AND_LOSS)
            except Exception:
                pl_diff = 0
            self.historical_pl = pl_diff
        for product in self.pattern_products:
            od = sub_state.order_depths.get(product, OrderDepth())
            if not od.buy_orders or not od.sell_orders:
                continue
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.pattern_price_history[product].append(mid_price)
            current_pos = sub_state.position.get(product, 0)
            self.positions_pattern[product] = current_pos
            self.trade_duration[product] = self.trade_duration.get(product, 0) + (1 if current_pos != 0 else 0)
            pattern = self.detect_patterns(product, mid_price)
            self.pattern_detected[product] = pattern
            mkt_state = self.determine_market_state(product, pattern)
            self.market_state[product] = mkt_state
            orders = []
            if current_pos != 0 and self.should_exit_trade(product):
                orders.append(Order(product, best_bid if current_pos > 0 else best_ask, -current_pos))
                self.trade_duration[product] = 0
            else:
                if current_pos == 0 or \
                   (current_pos > 0 and mkt_state in ["bullish", "strongly_bullish"]) or \
                   (current_pos < 0 and mkt_state in ["bearish", "strongly_bearish"]):
                    trade_size = self.calculate_position_size(product, mid_price, mkt_state)
                    if mkt_state in ["bullish", "strongly_bullish"]:
                        buy_size = min(trade_size, self.position_limit_pattern - current_pos)
                        if buy_size > 0:
                            orders.append(Order(product, best_ask, buy_size))
                        if current_pos == 0:
                            self.trade_started_at[product] = self.iteration_pattern
                    elif mkt_state in ["bearish", "strongly_bearish"]:
                        sell_size = min(trade_size, self.position_limit_pattern + current_pos)
                        if sell_size > 0:
                            orders.append(Order(product, best_bid, -sell_size))
                        if current_pos == 0:
                            self.trade_started_at[product] = self.iteration_pattern
            if orders:
                result[product] = orders
        debug_info = f"Iter: {self.iteration_pattern}, PL: {pl_diff:.2f}"
        return result, 0, debug_info

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
        """Predict future price for the underlying using volcano features"""
        if len(self.volcano_price_history[self.volcano_underlying]) < 5:
            return None, 0.0
        prices = np.array(list(self.volcano_price_history[self.volcano_underlying]))
        volatility, momentum, mean_reversion, trend, recent_accuracy = self.calculate_features_volcano(prices)
        momentum_weight = 1.0 - min(1.0, abs(mean_reversion) * 2)
        base_prediction = prices[-1]
        prediction = base_prediction * (1 + 0.3 * momentum * momentum_weight - 0.2 * mean_reversion - 0.1 * trend)
        volatility_factor = 1 - min(1.0, volatility * 10)
        mean_reversion_factor = 1 - min(1.0, abs(mean_reversion))
        confidence = volatility_factor * mean_reversion_factor * recent_accuracy
        return prediction, confidence

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
        if self.volcano_initial_capital == 0:
            try:
                self.volcano_initial_capital = float(sub_state.traderData) if sub_state.traderData else 0
            except (ValueError, TypeError):
                self.volcano_initial_capital = 0
        self.volcano_current_capital = self.volcano_initial_capital
        self.volcano_max_capital = max(self.volcano_max_capital, self.volcano_current_capital)
        drawdown = (self.volcano_max_capital - self.volcano_current_capital) / self.volcano_max_capital if self.volcano_max_capital > 0 else 0
        if drawdown > self.volcano_max_drawdown:
            return result, 0, ""
        if self.volcano_underlying in sub_state.order_depths:
            spot_depth = sub_state.order_depths[self.volcano_underlying]
            if spot_depth.buy_orders and spot_depth.sell_orders:
                spot_best_bid = max(spot_depth.buy_orders.keys())
                spot_best_ask = min(spot_depth.sell_orders.keys())
                spot_mid_price = (spot_best_bid + spot_best_ask) / 2
                self.update_volcano_price_history(self.volcano_underlying, spot_mid_price)
                predicted_price, confidence = self.predict_future_price_volcano()
                if predicted_price is not None:
                    for option_name, strike_price in self.volcano_options.items():
                        if option_name in sub_state.order_depths:
                            option_depth = sub_state.order_depths[option_name]
                            if option_depth.buy_orders and option_depth.sell_orders:
                                option_best_bid = max(option_depth.buy_orders.keys())
                                option_best_ask = min(option_depth.sell_orders.keys())
                                option_mid_price = (option_best_bid + option_best_ask) / 2
                                self.update_volcano_price_history(option_name, option_mid_price)
                                direction, quantity = self.find_trading_opportunity_volcano(
                                    spot_mid_price, option_mid_price, strike_price,
                                    predicted_price, confidence
                                )
                                if direction is not None:
                                    spot_position = sub_state.position.get(self.volcano_underlying, 0)
                                    option_position = sub_state.position.get(option_name, 0)
                                    if direction == "SELL_OPTION":
                                        max_option_quantity = min(
                                            self.volcano_position_limits[option_name] + option_position,
                                            quantity
                                        )
                                        max_spot_quantity = min(
                                            self.volcano_position_limits[self.volcano_underlying] - spot_position,
                                            quantity
                                        )
                                        final_quantity = min(max_option_quantity, max_spot_quantity)
                                        if final_quantity > 0:
                                            result[option_name] = [Order(option_name, option_best_bid, -final_quantity)]
                                            self.volcano_last_trade_prices[option_name] = option_mid_price
                                    elif direction == "BUY_OPTION":
                                        max_option_quantity = min(
                                            self.volcano_position_limits[option_name] - option_position,
                                            quantity
                                        )
                                        max_spot_quantity = min(
                                            self.volcano_position_limits[self.volcano_underlying] + spot_position,
                                            quantity
                                        )
                                        final_quantity = min(max_option_quantity, max_spot_quantity)
                                        if final_quantity > 0:
                                            result[option_name] = [Order(option_name, option_best_ask, final_quantity)]
                                            self.volcano_last_trade_prices[option_name] = option_mid_price
        return result, 0, str(self.volcano_current_capital)

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

        # Combine orders from each strategy without netting (no volcano underlying orders)
        combined_orders = self.combine_orders_no_netting([orders_resin, orders_pattern, orders_volcano])
        conversions = conv_resin + conv_pattern + conv_volcano
        combined_data = f"{data_resin} | {data_pattern} | {data_volcano}"
        return combined_orders, conversions, combined_data
 
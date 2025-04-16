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

    # ========= Option Volatility Methods =========
    def calculate_m_t(self, K: float, St: float, TTE: float) -> float:
        """
        Calculate m_t = log(K / St) / sqrt(TTE)
        """
        return math.log(K / St) / math.sqrt(TTE)

    def calculate_v_t(self, S: float, K: float, T: float, r: float, market_price: float) -> float:
        """
        Calculate implied volatility (v_t) using the Black-Scholes model.
        We will iterate to find the implied volatility that gives us the market price.
        """
        sigma = 0.2  # Start with an initial guess for volatility
        tolerance = 0.0001  # Acceptable tolerance for error in price
        max_iterations = 100  # Max iterations

        for _ in range(max_iterations):
            price = self.black_scholes(S, K, T, r, sigma)
            vega = self.calculate_vega(S, K, T, r, sigma)  # Vega is the derivative of price with respect to volatility

            if vega == 0:
                break

            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma  # Found the implied volatility

            sigma = sigma + diff / vega  # Update sigma using Newton-Raphson

        return sigma  # Return the found implied volatility

    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        S: Spot price of the underlying
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        sigma: Volatility (annualized)
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        return price

    def norm_cdf(self, x: float) -> float:
        """
        Normal cumulative distribution function (CDF).
        """
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def calculate_vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vega (the derivative of option price with respect to volatility).
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * math.sqrt(T) * self.norm_pdf(d1)
        return vega

    def norm_pdf(self, x: float) -> float:
        """
        Normal probability density function (PDF).
        """
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    def run_volcano(self, sub_state: TradingState):
        result = {}

        # For each option, calculate the implied volatility (v_t) and m_t
        for option_name, strike_price in self.volcano_options.items():
            if option_name in sub_state.order_depths:
                option_depth = sub_state.order_depths[option_name]
                if option_depth.buy_orders and option_depth.sell_orders:
                    # Calculate mid-price for the option
                    option_best_bid = max(option_depth.buy_orders.keys())
                    option_best_ask = min(option_depth.sell_orders.keys())
                    option_mid_price = (option_best_bid + option_best_ask) / 2

                    # Calculate the m_t (log(K/St) / sqrt(TTE))
                    TTE = 30 / 365  # Example Time to Expiry, 30 days remaining
                    spot_price = sub_state.order_depths[self.volcano_underlying].mid_price  # Example spot price of the underlying
                    m_t = self.calculate_m_t(strike_price, spot_price, TTE)

                    # Estimate implied volatility (v_t)
                    v_t = self.calculate_v_t(spot_price, strike_price, TTE, 0.05, option_mid_price)

                    # Store results
                    if option_name not in result:
                        result[option_name] = []
                    result[option_name].append((m_t, v_t))

        # Fit the parabolic curve and calculate base IV for each option
        for option_name, data in result.items():
            m_t_values, v_t_values = zip(*data)
            coefficients = np.polyfit(m_t_values, v_t_values, 2)
            base_iv = np.polyval(coefficients, 0)
            print(f"Base implied volatility for {option_name} at m_t=0: {base_iv}")
            # Store the base IV for future analysis
            result[option_name].append(('Base IV', base_iv))

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

        # Combine orders from each strategy without netting (no volcano underlying orders)
        combined_orders = self.combine_orders_no_netting([orders_resin, orders_pattern, orders_volcano])
        conversions = conv_resin + conv_pattern + conv_volcano
        combined_data = f"{data_resin} | {data_pattern} | {data_volcano}"
        return combined_orders, conversions, combined_data

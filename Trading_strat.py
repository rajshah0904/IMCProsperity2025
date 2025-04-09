from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import numpy as np
from collections import deque
import logging

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
        
        # Debug counter
        self.iteration = 0
        self.last_signal_iteration = 0
        self.last_trade_iteration = 0

        # Pairs Trading Parameters
        self.pairs_position_limit = 15
        self.pairs_position_size = 3
        self.short_window = 3
        self.long_window = 10
        self.target_ratio = 0.90  # SQUID_INK/KELP target ratio
        self.ratio_std = 0.02
        
        # Price history for pairs
        self.squid_prices = deque(maxlen=50)
        self.kelp_prices = deque(maxlen=50)
        self.ratio_history = deque(maxlen=50)
        
        # Position tracking
        self.positions = {"SQUID_INK": 0, "KELP": 0}
        self.position_values = {"SQUID_INK": 0, "KELP": 0}
        
        # Trade counting
        self.trades_made = 0
        self.profitable_trades = 0
        
        # Force initial trades
        self.first_iteration = True

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

    def detect_regime(self) -> str:
        """Detect market regime based on recent price action"""
        if len(self.ratio_history) < 10:
            return "neutral"
            
        # Get recent ratio movements
        recent_ratios = list(self.ratio_history)[-10:]
        if len(recent_ratios) < 2:
            return "neutral"
            
        # Calculate trend strength
        upward_moves = sum(1 for i in range(1, len(recent_ratios)) if recent_ratios[i] > recent_ratios[i-1])
        trend_strength = (upward_moves / (len(recent_ratios)-1)) - 0.5
        
        # Calculate volatility
        ratio_std = np.std(recent_ratios)
        
        # Update regime
        if abs(trend_strength) > 0.3:
            # Strong trend
            return "trending" if trend_strength > 0 else "counter_trend"
        elif ratio_std < 0.01:
            # Low volatility
            return "range_bound"
        else:
            return "neutral"
            
    def update_parameters(self, squid_price: float, kelp_price: float, pl_diff: float):
        """Update strategy parameters based on recent performance"""
        # Update recent performance
        pl_change = pl_diff - self.historical_pl
        self.historical_pl = pl_diff
        
        # Update regime
        self.regime = self.detect_regime()
        
        # Calculate current ratio
        ratio = squid_price / kelp_price
        
        # Adaptive adjustment based on current market regime
        if self.regime == "trending":
            # In trending markets, be more aggressive
            self.pairs_position_size = min(4, self.pairs_position_limit // 4)
            self.short_window = 2
            self.min_profit_threshold = 0.0002
        elif self.regime == "counter_trend":
            # In counter-trend markets, be more cautious
            self.pairs_position_size = max(1, self.pairs_position_limit // 10)
            self.short_window = 4
            self.min_profit_threshold = 0.001
        elif self.regime == "range_bound":
            # In range-bound markets, focus on mean reversion
            self.pairs_position_size = min(3, self.pairs_position_limit // 5)
            self.short_window = 3
            self.min_profit_threshold = 0.0005
        else:
            # Neutral regime
            self.pairs_position_size = 2
            self.short_window = 3
            self.min_profit_threshold = 0.0008
            
        # Update target ratio based on recent successful trades
        if pl_change > 0 and len(self.ratio_history) > 5:
            # If making money, gradually adjust target ratio
            recent_ratios = list(self.ratio_history)[-5:]
            new_target = np.mean(recent_ratios)
            # Gradually blend new target with old
            self.target_ratio = 0.8 * self.target_ratio + 0.2 * new_target
            
        # Adjust position sizing based on recent volatility
        if len(self.ratio_history) > self.volatility_window:
            recent_vols = np.std(list(self.ratio_history)[-self.volatility_window:])
            if recent_vols > 0:
                vol_adjust = min(2.0, max(0.5, self.ratio_std / recent_vols))
                self.pairs_position_size = max(1, int(self.pairs_position_size * vol_adjust))

    def calculate_pair_signals(self, squid_mid: float, kelp_mid: float, pl_diff: float) -> tuple:
        """Calculate trading signals for pairs trading with advanced signal generation"""
        # Add latest prices
        self.squid_prices.append(squid_mid)
        self.kelp_prices.append(kelp_mid)
        
        # Calculate current ratio
        if kelp_mid == 0:
            return 0, 0, float('inf'), 0
            
        ratio = squid_mid / kelp_mid
        self.ratio_history.append(ratio)
        
        # Track the last time we calculated a signal
        self.last_signal_iteration = self.iteration
        
        # Update strategy parameters
        self.update_parameters(squid_mid, kelp_mid, pl_diff)
        
        if len(self.squid_prices) < self.short_window:
            return 0, 0, float('inf'), 0
            
        # Calculate spread between SQUID_INK and KELP
        spread = squid_mid - (kelp_mid * self.target_ratio)
        self.spread_history.append(spread)
        
        # Calculate exponential moving averages
        short_ema_ratio = np.mean(list(self.ratio_history)[-self.short_window:])
        long_ema_ratio = np.mean(list(self.ratio_history)[-self.long_window:]) if len(self.ratio_history) >= self.long_window else short_ema_ratio
        
        # Calculate spread EMAs
        short_ema_spread = np.mean(list(self.spread_history)[-self.short_window:])
        long_ema_spread = np.mean(list(self.spread_history)[-self.long_window:]) if len(self.spread_history) >= self.long_window else short_ema_spread
        
        # Calculate ratio deviation and momentum
        ratio_dev = (ratio - self.target_ratio) / self.ratio_std
        momentum = short_ema_ratio - long_ema_ratio
        
        # Calculate spread momentum
        spread_momentum = short_ema_spread - long_ema_spread
        
        # Dynamic threshold based on regime
        if self.regime == "trending":
            base_threshold = 1.2
        elif self.regime == "counter_trend":
            base_threshold = 1.8
        elif self.regime == "range_bound":
            base_threshold = 1.5
        else:
            base_threshold = 1.6
            
        # Adjust threshold based on position
        squid_pos = self.positions["SQUID_INK"]
        position_factor = 1.0 + (abs(squid_pos) / self.pairs_position_limit) * 0.7
        
        # Final threshold calculation
        threshold = base_threshold * position_factor
        
        return ratio_dev, momentum, threshold, spread_momentum

    def pairs_trade(self, squid_depth: OrderDepth, kelp_depth: OrderDepth) -> List[Order]:
        """Simplified pairs trading strategy that guarantees execution"""
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
        
        # Force initial trade on first iteration to get started
        if self.first_iteration:
            orders.append(Order("SQUID_INK", squid_best_ask, 1))
            orders.append(Order("KELP", kelp_best_ask, 1))
            self.first_iteration = False
            return orders
        
        # Add latest prices
        self.squid_prices.append(squid_mid)
        self.kelp_prices.append(kelp_mid)
        
        # Calculate current ratio
        if kelp_mid == 0:
            return orders
            
        ratio = squid_mid / kelp_mid
        self.ratio_history.append(ratio)
        
        # Need enough history
        if len(self.ratio_history) < self.short_window:
            return orders
            
        # Calculate short and long term averages
        short_avg = np.mean(list(self.ratio_history)[-self.short_window:])
        long_avg = np.mean(list(self.ratio_history)[-self.long_window:]) if len(self.ratio_history) >= self.long_window else short_avg
        
        # Calculate z-score (how many standard deviations from target)
        dev = (ratio - self.target_ratio) / self.ratio_std
        
        # Current positions
        squid_pos = self.positions["SQUID_INK"]
        kelp_pos = self.positions["KELP"]
        
        # Trade size
        size = self.pairs_position_size
        
        # Simple trading logic with guaranteed execution
        if dev > 1.0:  # SQUID_INK overvalued
            # Ensure we don't exceed position limits
            if squid_pos > -self.pairs_position_limit and kelp_pos < self.pairs_position_limit:
                # Available position capacity
                available_squid = self.pairs_position_limit + squid_pos
                available_kelp = self.pairs_position_limit - kelp_pos
                trade_size = min(size, available_squid, available_kelp)
                
                if trade_size > 0:
                    # Market orders to ensure execution
                    orders.append(Order("SQUID_INK", squid_best_bid, -trade_size))
                    orders.append(Order("KELP", kelp_best_ask, trade_size))
        
        elif dev < -1.0:  # SQUID_INK undervalued
            # Ensure we don't exceed position limits
            if squid_pos < self.pairs_position_limit and kelp_pos > -self.pairs_position_limit:
                # Available position capacity
                available_squid = self.pairs_position_limit - squid_pos
                available_kelp = self.pairs_position_limit + kelp_pos
                trade_size = min(size, available_squid, available_kelp)
                
                if trade_size > 0:
                    # Market orders to ensure execution
                    orders.append(Order("SQUID_INK", squid_best_ask, trade_size))
                    orders.append(Order("KELP", kelp_best_bid, -trade_size))
        
        # Simple profit taking
        elif abs(dev) < 0.3:  # Close to target ratio
            if squid_pos > 0:
                orders.append(Order("SQUID_INK", squid_best_bid, -min(size, squid_pos)))
            elif squid_pos < 0:
                orders.append(Order("SQUID_INK", squid_best_ask, min(size, -squid_pos)))
                
            if kelp_pos > 0:
                orders.append(Order("KELP", kelp_best_bid, -min(size, kelp_pos)))
            elif kelp_pos < 0:
                orders.append(Order("KELP", kelp_best_ask, min(size, -kelp_pos)))
                
        return orders

    def run(self, state: TradingState):
        """
        Trading strategy for RAINFOREST_RESIN (market making) and SQUID_INK/KELP (pairs trading)
        Returns: Dict[str, List[Order]], int, str
        """
        self.iteration += 1
        result = {}
        
        # Get P&L value - Fixed for Observation object
        pl_diff = 0
        if hasattr(state, 'observations') and state.observations:
            # Safe access to observations
            try:
                pl_diff = float(state.observations.PROFIT_AND_LOSS) if hasattr(state.observations, 'PROFIT_AND_LOSS') else 0
            except (AttributeError, TypeError):
                pl_diff = 0
        
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

        # Pairs trade SQUID_INK and KELP - aggressive execution
        if "SQUID_INK" in state.order_depths and "KELP" in state.order_depths:
            pairs_orders = self.pairs_trade(
                state.order_depths["SQUID_INK"],
                state.order_depths["KELP"]
            )
            
            # Split orders by product and add to result
            squid_orders = [order for order in pairs_orders if order.symbol == "SQUID_INK"]
            kelp_orders = [order for order in pairs_orders if order.symbol == "KELP"]
            
            if squid_orders:
                result["SQUID_INK"] = squid_orders
            if kelp_orders:
                result["KELP"] = kelp_orders
        
        # Update positions and position values
        for product in state.position:
            if product in self.positions:
                self.positions[product] = state.position[product]
                if product in state.order_depths:
                    depth = state.order_depths[product]
                    if depth.buy_orders and depth.sell_orders:
                        mid_price = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
                        if self.positions[product] != 0:
                            self.position_values[product] = mid_price

        # Debug info about pairs trading activity
        debug_info = ""
        if "SQUID_INK" in state.order_depths and "KELP" in state.order_depths:
            squid_depth = state.order_depths["SQUID_INK"]
            kelp_depth = state.order_depths["KELP"]
            
            if squid_depth.buy_orders and squid_depth.sell_orders and kelp_depth.buy_orders and kelp_depth.sell_orders:
                squid_mid = (max(squid_depth.buy_orders.keys()) + min(squid_depth.sell_orders.keys())) / 2
                kelp_mid = (max(kelp_depth.buy_orders.keys()) + min(kelp_depth.sell_orders.keys())) / 2
                current_ratio = squid_mid / kelp_mid
                
                debug_info = f"Iter: {self.iteration}, Ratio: {current_ratio:.3f}, Target: {self.target_ratio:.3f}, "
                debug_info += f"Last Signal: {self.last_signal_iteration}, Last Trade: {self.last_trade_iteration}, "
                debug_info += f"SQUID pos: {self.positions.get('SQUID_INK', 0)}, KELP pos: {self.positions.get('KELP', 0)}, "
                debug_info += f"Orders: SQUID {len(squid_orders)}, KELP {len(kelp_orders)}"

        return result, 0, debug_info 
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

        # Pairs Trading Parameters - Optimized based on performance data
        self.pairs_position_limit = 15  # Smaller position limit to be safer
        self.pairs_position_size = 3    # Smaller trade size
        self.short_window = 3          
        self.long_window = 10           
        self.volatility_window = 5      
        self.min_profit_threshold = 0.0005  
        
        # Price history for pairs
        self.squid_prices = deque(maxlen=50)
        self.kelp_prices = deque(maxlen=50)
        self.ratio_history = deque(maxlen=50)
        self.spread_history = deque(maxlen=50)
        
        # Position tracking
        self.positions = {"SQUID_INK": 0, "KELP": 0}
        self.position_values = {"SQUID_INK": 0, "KELP": 0}
        
        # Trade counting
        self.trades_made = 0
        self.profitable_trades = 0
        
        # Initial ratio parameters - dynamically updated
        self.target_ratio = 0.90        # SQUID_INK/KELP target ratio (adjusted based on data)
        self.ratio_std = 0.02           
        
        # Adaptive position sizing
        self.max_position_held = 0
        self.min_profit_target = 5     # Lower profit target for more trades
        
        # Trend detection
        self.trend_strength = 0
        self.regime = "neutral"         
        
        # Execution optimization
        self.squid_avg_fill = 0
        self.kelp_avg_fill = 0
        self.historical_pl = 0
        
        # Force initial trades to build history
        self.force_initial_trades = True
        self.initial_trades_count = 0

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

    def pairs_trade(self, squid_depth: OrderDepth, kelp_depth: OrderDepth, pl_diff: float) -> List[Order]:
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
        
        # Force initial trades to build price history if needed
        if self.force_initial_trades and self.initial_trades_count < 2:
            # Small initial position to establish history
            orders.append(Order("SQUID_INK", squid_best_ask, 1))
            orders.append(Order("KELP", kelp_best_ask, 1))
            self.initial_trades_count += 1
            self.last_trade_iteration = self.iteration
            return orders
        
        # Calculate signals with P&L feedback
        ratio_dev, momentum, threshold, spread_momentum = self.calculate_pair_signals(squid_mid, kelp_mid, pl_diff)
        
        # Current positions
        squid_pos = self.positions["SQUID_INK"]
        kelp_pos = self.positions["KELP"]
        
        # Dynamic position sizing based on multiple factors
        base_size = self.pairs_position_size
        
        # Scale based on signal strength
        signal_strength = min(1.5, abs(ratio_dev) / threshold)
        dynamic_size = max(1, min(base_size, int(base_size * signal_strength)))
        
        # Increase position size when winning, decrease when losing
        if pl_diff > 0:
            dynamic_size = min(dynamic_size + 1, self.pairs_position_limit // 5)
        elif pl_diff < -50:  # If losing substantially
            dynamic_size = max(1, dynamic_size - 1)
            
        # Key trading signals
        squid_overvalued = ratio_dev > threshold and (momentum > 0 or spread_momentum > 0)
        squid_undervalued = ratio_dev < -threshold and (momentum < 0 or spread_momentum < 0)
        
        # Force smaller trade size for initial trades
        if abs(squid_pos) < 2 or abs(kelp_pos) < 2:
            dynamic_size = 1
        
        # Primary trading logic
        if squid_overvalued:
            # SQUID_INK overvalued - sell SQUID_INK, buy KELP
            if squid_pos > -self.pairs_position_limit and kelp_pos < self.pairs_position_limit:
                # Scale order size based on distance from position limit
                available_squid = self.pairs_position_limit + squid_pos
                available_kelp = self.pairs_position_limit - kelp_pos
                size = min(dynamic_size, available_squid, available_kelp)
                
                if size > 0:
                    # Track trade
                    self.last_trade_iteration = self.iteration
                    
                    # Use better execution prices - immediately cross spread if signal is strong
                    squid_price = squid_best_bid if ratio_dev < threshold * 1.5 else squid_best_bid - 1
                    kelp_price = kelp_best_ask if ratio_dev < threshold * 1.5 else kelp_best_ask + 1
                    
                    orders.append(Order("SQUID_INK", squid_price, -size))
                    orders.append(Order("KELP", kelp_price, size))
                
        elif squid_undervalued:
            # SQUID_INK undervalued - buy SQUID_INK, sell KELP
            if squid_pos < self.pairs_position_limit and kelp_pos > -self.pairs_position_limit:
                # Scale order size based on distance from position limit
                available_squid = self.pairs_position_limit - squid_pos
                available_kelp = self.pairs_position_limit + kelp_pos
                size = min(dynamic_size, available_squid, available_kelp)
                
                if size > 0:
                    # Track trade
                    self.last_trade_iteration = self.iteration
                    
                    # Use better execution prices
                    squid_price = squid_best_ask if abs(ratio_dev) < threshold * 1.5 else squid_best_ask + 1
                    kelp_price = kelp_best_bid if abs(ratio_dev) < threshold * 1.5 else kelp_best_bid - 1
                    
                    orders.append(Order("SQUID_INK", squid_price, size))
                    orders.append(Order("KELP", kelp_price, -size))
        
        # More aggressive profit taking when approaching extreme positions
        profit_threshold = self.min_profit_threshold
        position_pct = abs(squid_pos) / self.pairs_position_limit
        
        # Reduce profit threshold as position grows
        if position_pct > 0.5:
            profit_threshold = self.min_profit_threshold * 0.75
        
        # Profit taking with dynamic thresholds
        if abs(ratio_dev) < threshold * 0.4 or (position_pct > 0.7 and (ratio_dev * squid_pos) < 0):
            # Close SQUID_INK positions when profitable or when signal reverses with large position
            if squid_pos > 0 and squid_best_bid > self.position_values["SQUID_INK"] * (1 + profit_threshold):
                orders.append(Order("SQUID_INK", squid_best_bid, -min(dynamic_size, squid_pos)))
                self.last_trade_iteration = self.iteration
            elif squid_pos < 0 and squid_best_ask < self.position_values["SQUID_INK"] * (1 - profit_threshold):
                orders.append(Order("SQUID_INK", squid_best_ask, min(dynamic_size, -squid_pos)))
                self.last_trade_iteration = self.iteration
            
            # Close KELP positions
            if kelp_pos > 0 and kelp_best_bid > self.position_values["KELP"] * (1 + profit_threshold):
                orders.append(Order("KELP", kelp_best_bid, -min(dynamic_size, kelp_pos)))
                self.last_trade_iteration = self.iteration
            elif kelp_pos < 0 and kelp_best_ask < self.position_values["KELP"] * (1 - profit_threshold):
                orders.append(Order("KELP", kelp_best_ask, min(dynamic_size, -kelp_pos)))
                self.last_trade_iteration = self.iteration
        
        # Emergency risk management - reduce position when losing significantly
        if pl_diff < -100 and abs(squid_pos) > self.pairs_position_limit / 2:
            # Cut position by half when in significant drawdown
            if squid_pos > 0:
                orders.append(Order("SQUID_INK", squid_best_bid, -min(squid_pos // 2, 5)))
                self.last_trade_iteration = self.iteration
            elif squid_pos < 0:
                orders.append(Order("SQUID_INK", squid_best_ask, min(-squid_pos // 2, 5)))
                self.last_trade_iteration = self.iteration
                
            if kelp_pos > 0:
                orders.append(Order("KELP", kelp_best_bid, -min(kelp_pos // 2, 5)))
                self.last_trade_iteration = self.iteration
            elif kelp_pos < 0:
                orders.append(Order("KELP", kelp_best_ask, min(-kelp_pos // 2, 5)))
                self.last_trade_iteration = self.iteration
        
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

        # Pairs trade SQUID_INK and KELP
        squid_orders = []
        kelp_orders = []
        
        if "SQUID_INK" in state.order_depths and "KELP" in state.order_depths:
            pairs_orders = self.pairs_trade(
                state.order_depths["SQUID_INK"],
                state.order_depths["KELP"],
                pl_diff
            )
            
            # Split orders by product
            squid_orders = [order for order in pairs_orders if order.symbol == "SQUID_INK"]
            kelp_orders = [order for order in pairs_orders if order.symbol == "KELP"]
            
            # Ensure we're not exceeding position limits
            if "SQUID_INK" in state.position:
                squid_pos = state.position["SQUID_INK"]
                for order in squid_orders[:]:
                    if (squid_pos + order.quantity > self.pairs_position_limit or 
                        squid_pos + order.quantity < -self.pairs_position_limit):
                        squid_orders.remove(order)
            
            if "KELP" in state.position:
                kelp_pos = state.position["KELP"]
                for order in kelp_orders[:]:
                    if (kelp_pos + order.quantity > self.pairs_position_limit or 
                        kelp_pos + order.quantity < -self.pairs_position_limit):
                        kelp_orders.remove(order)
            
            # Make sure we have valid orders after filtering
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
                        # Only update position value if we have a position
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
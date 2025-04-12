from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import numpy as np
from collections import deque
import logging

class Trader:
    def __init__(self):
        # Debug counter
        self.iteration = 0
        
        # Products to trade
        self.products = ["SQUID_INK", "KELP"]
        
        # Price history 
        self.price_history = {
            "SQUID_INK": deque(maxlen=100),
            "KELP": deque(maxlen=100)
        }
        
        # Track highs and lows for pattern recognition
        self.recent_highs = {"SQUID_INK": deque(maxlen=10), "KELP": deque(maxlen=10)}
        self.recent_lows = {"SQUID_INK": deque(maxlen=10), "KELP": deque(maxlen=10)}
        
        # Oscillator parameters
        self.rsi_period = 14
        self.rsi_values = {"SQUID_INK": deque(maxlen=20), "KELP": deque(maxlen=20)}
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Volatility parameters
        self.volatility_window = 20
        self.volatility_values = {"SQUID_INK": deque(maxlen=50), "KELP": deque(maxlen=50)}
        self.volatility_breakout_threshold = 1.5  # Multiple of average volatility
        
        # Position tracking
        self.positions = {"SQUID_INK": 0, "KELP": 0}
        
        # Position limits
        self.position_limit = 15
        
        # Trade management
        self.base_trade_size = 3
        self.max_trade_size = 7
        self.trade_duration = {"SQUID_INK": 0, "KELP": 0}  # Track how long we've been in trades
        self.max_trade_duration = 15  # Exit trades after this many iterations
        
        # Pattern states
        self.market_state = {"SQUID_INK": "neutral", "KELP": "neutral"}
        self.pattern_detected = {"SQUID_INK": None, "KELP": None}
        
        # Performance tracking
        self.historical_pl = 0
        self.trade_started_at = {"SQUID_INK": 0, "KELP": 0}
        
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50  # Default neutral value if not enough data
            
        # Convert to numpy array
        prices_array = np.array(list(prices)[-period-1:])
        
        # Calculate price changes
        deltas = np.diff(prices_array)
        
        # Calculate gains and losses
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100  # No losses means RSI = 100
            
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_volatility(self, prices, window=20):
        """Calculate price volatility using standard deviation"""
        if len(prices) < window:
            return 0.01  # Default low value if not enough data
            
        # Get recent prices
        recent_prices = list(prices)[-window:]
        
        # Calculate normalized volatility (std deviation / mean)
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        return volatility
    
    def detect_patterns(self, product, price):
        """Detect common price patterns"""
        prices = self.price_history[product]
        if len(prices) < 5:
            return None
            
        # Recently detected patterns
        pattern = None
        
        # Get recent prices
        recent_prices = list(prices)[-5:]
        
        # Update recent highs and lows
        if len(prices) > 2:
            # A local high is formed when the middle point is higher than points on either side
            if len(recent_prices) >= 3 and recent_prices[-2] > recent_prices[-3] and recent_prices[-2] > recent_prices[-1]:
                self.recent_highs[product].append(recent_prices[-2])
                
            # A local low is formed when the middle point is lower than points on either side
            if len(recent_prices) >= 3 and recent_prices[-2] < recent_prices[-3] and recent_prices[-2] < recent_prices[-1]:
                self.recent_lows[product].append(recent_prices[-2])
        
        # Pattern 1: Double Bottom (W pattern) - bullish reversal
        if len(self.recent_lows[product]) >= 2:
            lows = list(self.recent_lows[product])
            # Two similar lows with a higher point between them
            if len(lows) >= 2 and abs(lows[-1] - lows[-2]) / lows[-1] < 0.03:  # Within 3%
                pattern = "double_bottom"
        
        # Pattern 2: Double Top (M pattern) - bearish reversal
        if len(self.recent_highs[product]) >= 2:
            highs = list(self.recent_highs[product])
            # Two similar highs with a lower point between them
            if len(highs) >= 2 and abs(highs[-1] - highs[-2]) / highs[-1] < 0.03:  # Within 3%
                pattern = "double_top"
        
        # Pattern 3: Breakout - price exceeds recent range significantly
        if len(prices) > 10:
            recent_range = max(recent_prices[-10:]) - min(recent_prices[-10:])
            if recent_range > 0:
                # Upward breakout
                if price > max(list(prices)[-10:-1]) * 1.02:  # 2% above recent max
                    pattern = "breakout_up"
                # Downward breakout
                elif price < min(list(prices)[-10:-1]) * 0.98:  # 2% below recent min
                    pattern = "breakout_down"
        
        # Calculate RSI and volatility for additional insights
        rsi = self.calculate_rsi(prices, self.rsi_period)
        self.rsi_values[product].append(rsi)
        
        volatility = self.calculate_volatility(prices, self.volatility_window)
        self.volatility_values[product].append(volatility)
        
        # Pattern 4: Oversold bounce
        if rsi < self.rsi_oversold:
            pattern = "oversold"
            
        # Pattern 5: Overbought reversal
        if rsi > self.rsi_overbought:
            pattern = "overbought"
            
        # Pattern 6: Volatility breakout
        if len(self.volatility_values[product]) > 5:
            avg_volatility = np.mean(list(self.volatility_values[product])[-5:])
            if volatility > avg_volatility * self.volatility_breakout_threshold:
                # Direction depends on recent price action
                if recent_prices[-1] > recent_prices[-2]:
                    pattern = "volatility_breakout_up"
                else:
                    pattern = "volatility_breakout_down"
        
        return pattern
        
    def determine_market_state(self, product, pattern):
        """Determine overall market state based on detected patterns and indicators"""
        # Get recent RSI values
        rsi_values = self.rsi_values[product]
        if not rsi_values:
            return "neutral"
            
        rsi = list(rsi_values)[-1] if rsi_values else 50
        
        # Default state
        state = "neutral"
        
        # CORRECT SIGNALS: Use patterns properly to generate signals
        if pattern == "double_bottom" or pattern == "oversold" or pattern == "breakout_up" or pattern == "volatility_breakout_up":
            state = "bullish"  # proper bullish signals
        elif pattern == "double_top" or pattern == "overbought" or pattern == "breakout_down" or pattern == "volatility_breakout_down":
            state = "bearish"  # proper bearish signals
            
        # Extreme RSI values with proper logic
        if rsi < 20:
            state = "strongly_bullish"  # extremely oversold, expect bounce
        elif rsi > 80:
            state = "strongly_bearish"  # extremely overbought, expect reversal
            
        return state
    
    def calculate_position_size(self, product, price, state):
        """Calculate position size based on market state and risk parameters"""
        # Base size - moderate to allow for scaling
        size = self.base_trade_size
        
        # Adjust based on conviction - increase size for high-conviction signals
        if state in ["strongly_bullish", "strongly_bearish"]:
            size = int(size * 1.5)  # More conviction = larger size
            
        # Adjust based on current position - reduce risk as position grows
        current_pos = abs(self.positions[product])
        position_factor = max(0.5, 1.0 - (current_pos / self.position_limit) * 0.7)
        size = max(1, int(size * position_factor))
        
        # Consider volatility - reduce size in high volatility environments
        if len(self.volatility_values[product]) > 5:
            recent_vol = list(self.volatility_values[product])[-1]
            avg_vol = np.mean(list(self.volatility_values[product])[-5:])
            
            if recent_vol > avg_vol * 1.5:
                # Higher volatility = smaller positions
                size = max(1, int(size * 0.7))
            elif recent_vol < avg_vol * 0.5:
                # Lower volatility = slightly larger positions
                size = min(self.max_trade_size, int(size * 1.2))
        
        # Cap at max size
        size = min(size, self.max_trade_size)
            
        # Ensure we don't exceed position limits
        available_long = self.position_limit - self.positions[product]
        available_short = self.position_limit + self.positions[product]
        
        if state in ["bullish", "strongly_bullish"]:
            size = min(size, available_long)
        elif state in ["bearish", "strongly_bearish"]:
            size = min(size, available_short)
            
        return size
    
    def should_exit_trade(self, product):
        """Determine if we should exit a current trade with proper risk management"""
        # Exit if we've been in the trade for the maximum duration
        if self.trade_duration[product] > self.max_trade_duration:
            return True
            
        # Exit if pattern suggests reversal against our position
        current_position = self.positions[product]
        current_state = self.market_state[product]
        
        # Exit when market state changes against our position
        if current_position > 0 and current_state in ["bearish", "strongly_bearish"]:
            return True
        elif current_position < 0 and current_state in ["bullish", "strongly_bullish"]:
            return True
        
        # Check for profit target or stop loss
        if len(self.price_history[product]) > 0:
            current_price = list(self.price_history[product])[-1]
            entry_price = 0
            
            # Calculate profit/loss percentage
            if product in self.positions and self.positions[product] != 0:
                entry_iteration = self.trade_started_at.get(product, 0)
                
                # If we have position but can't determine entry price, use conservative approach
                if entry_iteration > 0 and entry_iteration < self.iteration and entry_iteration < len(self.price_history[product]):
                    entry_idx = min(entry_iteration, len(self.price_history[product])-1)
                    # Approximate entry price from history
                    entry_price = list(self.price_history[product])[entry_idx]
                
                if entry_price > 0:
                    # For long positions
                    if current_position > 0:
                        pnl_pct = (current_price - entry_price) / entry_price
                        # Take profit at 5% gain
                        if pnl_pct > 0.05:
                            return True
                        # Stop loss at 2% loss
                        if pnl_pct < -0.02:
                            return True
                    
                    # For short positions
                    elif current_position < 0:
                        pnl_pct = (entry_price - current_price) / entry_price
                        # Take profit at 5% gain
                        if pnl_pct > 0.05:
                            return True
                        # Stop loss at 2% loss
                        if pnl_pct < -0.02:
                            return True
            
        return False
    
    def run(self, state: TradingState):
        """
        Pattern recognition trading strategy for SQUID_INK and KELP
        """
        self.iteration += 1
        result = {}
        
        # Get P&L value
        pl_diff = 0
        if hasattr(state, 'observations') and state.observations:
            try:
                pl_diff = float(state.observations.PROFIT_AND_LOSS) if hasattr(state.observations, 'PROFIT_AND_LOSS') else 0
            except (AttributeError, TypeError):
                pl_diff = 0
        
        # Track PL changes
        pl_change = pl_diff - self.historical_pl
        self.historical_pl = pl_diff
        
        # Trade each product individually
        for product in self.products:
            if product in state.order_depths:
                depth = state.order_depths[product]
                
                # Skip if no orders
                if not depth.buy_orders or not depth.sell_orders:
                    continue
                    
                # Calculate mid price
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                
                # Add to price history
                self.price_history[product].append(mid_price)
                
                # Current position
                current_position = state.position.get(product, 0)
                self.positions[product] = current_position
                
                # Update trade duration if we have a position
                if current_position != 0:
                    self.trade_duration[product] += 1
                else:
                    self.trade_duration[product] = 0
                
                # Detect patterns
                pattern = self.detect_patterns(product, mid_price)
                self.pattern_detected[product] = pattern
                
                # Determine market state
                market_state = self.determine_market_state(product, pattern)
                self.market_state[product] = market_state
                
                # Initialize orders
                orders = []
                
                # Check if we should exit existing position
                if current_position != 0 and self.should_exit_trade(product):
                    if current_position > 0:
                        # Exit long position
                        orders.append(Order(product, best_bid, -current_position))
                    else:
                        # Exit short position
                        orders.append(Order(product, best_ask, -current_position))
                    
                    # Reset trade duration
                    self.trade_duration[product] = 0
                else:
                    # Only enter new positions if no current position or adding to winning position
                    if current_position == 0 or (current_position > 0 and market_state in ["bullish", "strongly_bullish"]) or (current_position < 0 and market_state in ["bearish", "strongly_bearish"]):
                        # Calculate position size
                        trade_size = self.calculate_position_size(product, mid_price, market_state)
                        
                        if market_state in ["bullish", "strongly_bullish"]:
                            # Enter or add to long position if not already at limit
                            if current_position < self.position_limit:
                                buy_size = min(trade_size, self.position_limit - current_position)
                                if buy_size > 0:
                                    orders.append(Order(product, best_ask, buy_size))
                                    # Record trade start if new position
                                    if current_position == 0:
                                        self.trade_started_at[product] = self.iteration
                        
                        elif market_state in ["bearish", "strongly_bearish"]:
                            # Enter or add to short position if not already at limit
                            if current_position > -self.position_limit:
                                sell_size = min(trade_size, self.position_limit + current_position)
                                if sell_size > 0:
                                    orders.append(Order(product, best_bid, -sell_size))
                                    # Record trade start if new position
                                    if current_position == 0:
                                        self.trade_started_at[product] = self.iteration
                
                # Add orders to result
                if orders:
                    result[product] = orders
        
        # Generate debug info
        debug_info = f"Iter: {self.iteration}, PL: {pl_diff:.2f}, "
        
        for product in self.products:
            if product in state.order_depths:
                pattern = self.pattern_detected.get(product, "none")
                state_val = self.market_state.get(product, "neutral")
                pos = self.positions.get(product, 0)
                duration = self.trade_duration.get(product, 0)
                
                if len(self.rsi_values[product]) > 0:
                    rsi = list(self.rsi_values[product])[-1]
                else:
                    rsi = 50
                    
                orders_count = len(result.get(product, []))
                debug_info += f"{product}: Pattern={pattern}, State={state_val}, RSI={rsi:.1f}, Pos={pos}, Duration={duration}, Orders={orders_count}, "
        
        return result, 0, debug_info 
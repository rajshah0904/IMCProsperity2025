from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import string
import numpy as np
from collections import deque

# --- Helper Strategy 1: Arbitrage on PICNIC_BASKET1 & PICNIC_BASKET2 ---
class ArbitrageTrader:
    def __init__(self):
        # Define baskets with components and position limits
        self.products = {
            "PICNIC_BASKET1": {
                "components": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                "position_limit": 70
            },
            "PICNIC_BASKET2": {
                "components": {"CROISSANTS": 4, "DJEMBES": 2},
                "position_limit": 70
            }
        }
        self.position_limits = {"CROISSANTS": 300, "JAMS": 300, "DJEMBES": 100}
        self.min_profit = 10
        # Maintain fixed-size histories for prices and spreads
        self.price_history = {p: deque(maxlen=100) for p in 
                             ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]}
        self.spread_history = {
            "PICNIC_BASKET1": deque(maxlen=20),
            "PICNIC_BASKET2": deque(maxlen=20)
        }
        
    def get_volume_weighted_mid(self, order_depth: OrderDepth) -> float:
        """Calculate volume-weighted mid price for more accurate pricing"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])
        
        # Handle case when both volumes are zero
        if bid_vol + ask_vol == 0:
            return (best_bid + best_ask) / 2
            
        # Return volume-weighted mid price
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def calculate_basket_value(self, basket_name: str, order_depths: Dict[str, OrderDepth]) -> float:
        """Calculate the fair value of a basket using volume-weighted pricing"""
        basket = self.products[basket_name]
        total_value = 0.0
        
        for component, quantity in basket["components"].items():
            order_depth = order_depths.get(component, OrderDepth())
            vw_price = self.get_volume_weighted_mid(order_depth)
            
            if vw_price is None:
                return None
                
            total_value += vw_price * quantity
            
        return total_value

    def calculate_spread_stats(self, basket_name: str) -> Tuple[float, float]:
        """Calculate spread statistics for dynamic thresholds"""
        history = self.spread_history[basket_name]
        if len(history) < 2:
            return 0, 1
        return np.mean(history), np.std(history)

    def find_arbitrage_opportunity(self, basket_name: str, basket_price: float, 
                                 component_value: float) -> tuple:
        """Determine if there's an arbitrage opportunity with dynamic thresholds"""
        if component_value is None:
            return None, 0
            
        # Calculate price difference
        price_diff = basket_price - component_value
        
        # Store price difference in history
        self.spread_history[basket_name].append(price_diff)
        
        # Calculate spread statistics
        mean_spread, std_spread = self.calculate_spread_stats(basket_name)
        
        # Calculate z-score for dynamic threshold adjustment
        z_score = (price_diff - mean_spread) / std_spread if std_spread != 0 else 0
        
        # Use z-score to adjust profit threshold - higher for extreme values
        adjusted_profit = self.min_profit * (1 + abs(z_score) * 0.1)
        
        # Only trade if price difference exceeds dynamic threshold
        if abs(price_diff) < adjusted_profit:
            return None, 0
            
        if price_diff > 0:
            # Basket is overpriced, sell basket and buy components
            return "SELL", price_diff
        else:
            # Basket is underpriced, buy basket and sell components
            return "BUY", abs(price_diff)

    def execute_arbitrage(self, state: TradingState, basket_name: str, 
                         direction: str, profit: float) -> Dict[str, List[Order]]:
        """Execute arbitrage trades with improved risk management"""
        result = {}
        basket = self.products[basket_name]
        position = state.position.get(basket_name, 0)
        
        # Get basket order depth
        basket_depth = state.order_depths.get(basket_name, OrderDepth())
        if not basket_depth.buy_orders or not basket_depth.sell_orders:
            return result
            
        best_bid = max(basket_depth.buy_orders.keys())
        best_ask = min(basket_depth.sell_orders.keys())
        
        # Position-aware scaling factor - reduce size when adding to existing positions
        position_factor = 1.0
        if (direction == "BUY" and position > 0) or (direction == "SELL" and position < 0):
            position_factor = 0.7  # Reduce size when adding to existing position
        
        # Calculate maximum quantity we can trade
        if direction == "BUY":
            max_quantity = min(
                basket["position_limit"] - position,
                abs(basket_depth.sell_orders[best_ask])
            )
            order_price = best_ask
        else:
            max_quantity = min(
                basket["position_limit"] + position,
                basket_depth.buy_orders[best_bid]
            )
            order_price = best_bid
        
        # Apply position-based scaling
        max_quantity = int(max_quantity * position_factor)
        
        if max_quantity <= 0:
            return result
            
        # Plan component trades first to ensure we can execute all of them
        component_orders = {}
        all_components_executable = True
        
        for component, quantity in basket["components"].items():
            component_depth = state.order_depths.get(component, OrderDepth())
            if not component_depth.buy_orders or not component_depth.sell_orders:
                all_components_executable = False
                break
                
            component_bid = max(component_depth.buy_orders.keys())
            component_ask = min(component_depth.sell_orders.keys())
            
            if direction == "BUY":
                # Selling components when buying basket
                available = component_depth.buy_orders[component_bid]
                component_qty = min(max_quantity * quantity, available)
                if component_qty <= 0:
                    all_components_executable = False
                    break
                component_orders[component] = [Order(component, component_bid, -component_qty)]
            else:
                # Buying components when selling basket
                available = abs(component_depth.sell_orders[component_ask])
                component_qty = min(max_quantity * quantity, available)
                if component_qty <= 0:
                    all_components_executable = False
                    break
                component_orders[component] = [Order(component, component_ask, component_qty)]
        
        # Only execute if all components can be traded
        if all_components_executable:
            result[basket_name] = [Order(basket_name, order_price, 
                                        max_quantity if direction == "BUY" else -max_quantity)]
            result.update(component_orders)
            
        return result

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """Run arbitrage strategy with improved pricing and execution"""
        result = {}
        
        # Check arbitrage opportunities for each basket
        for basket_name in self.products:
            basket_depth = state.order_depths.get(basket_name, OrderDepth())
            if not basket_depth.buy_orders or not basket_depth.sell_orders:
                continue
                
            # Use volume-weighted pricing for basket
            basket_price = self.get_volume_weighted_mid(basket_depth)
            if basket_price is None:
                continue
            
            # Store price in history
            self.price_history[basket_name].append(basket_price)
            
            # Calculate synthetic basket value from components
            component_value = self.calculate_basket_value(basket_name, state.order_depths)
            if component_value is None:
                continue
                
            # Find arbitrage opportunity
            direction, profit = self.find_arbitrage_opportunity(
                basket_name, basket_price, component_value)
                
            if direction is not None:
                # Execute arbitrage with improved risk management
                arbitrage_orders = self.execute_arbitrage(state, basket_name, direction, profit)
                result.update(arbitrage_orders)
                
        return result, 0, "Arbitrage strategy complete"

# --- Helper Strategy 2: Pattern Recognition for SQUID_INK & KELP ---
class PatternTrader:
    def __init__(self):
        # Products for technical trading
        self.products = ["SQUID_INK", "KELP"]
        
        # Price history and indicators
        self.price_history = {product: deque(maxlen=100) for product in self.products}
        self.recent_highs = {product: deque(maxlen=10) for product in self.products}
        self.recent_lows = {product: deque(maxlen=10) for product in self.products}
        
        # RSI parameters
        self.rsi_period = 14
        self.rsi_values = {product: deque(maxlen=20) for product in self.products}
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Volatility tracking
        self.volatility_window = 20
        self.volatility_values = {product: deque(maxlen=50) for product in self.products}
        self.volatility_breakout_threshold = 1.5
        
        # Position tracking
        self.positions = {product: 0 for product in self.products}
        self.position_limit = 15
        
        # Trade management
        self.base_trade_size = 3
        self.max_trade_size = 7
        self.trade_duration = {product: 0 for product in self.products}
        self.max_trade_duration = 15
        
        # Pattern tracking
        self.market_state = {product: "neutral" for product in self.products}
        self.pattern_detected = {product: None for product in self.products}
        
        # Performance tracking
        self.trade_started_at = {product: 0 for product in self.products}
        self.iteration = 0
        
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
        
        # Calculate RSI for additional signals
        rsi = self.calculate_rsi(prices, self.rsi_period)
        self.rsi_values[product].append(rsi)
        
        # RSI-based signals take precedence over pattern signals
        if rsi < self.rsi_oversold:
            pattern = "oversold"
        elif rsi > self.rsi_overbought:
            pattern = "overbought"
            
        return pattern
        
    def determine_market_state(self, product, pattern):
        """Determine overall market state based on detected patterns"""
        # Default state
        state = "neutral"
        
        # Use patterns to generate signals
        if pattern == "double_bottom" or pattern == "oversold":
            state = "bullish"
        elif pattern == "double_top" or pattern == "overbought":
            state = "bearish"
            
        # Get RSI for extreme values
        rsi_values = self.rsi_values[product]
        if rsi_values:
            rsi = list(rsi_values)[-1]
            if rsi < 20:
                state = "strongly_bullish"  # extremely oversold
            elif rsi > 80:
                state = "strongly_bearish"  # extremely overbought
            
        return state
    
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """Run pattern recognition strategy for technical products"""
        self.iteration += 1
        result = {}
        
        # Process each technical product
        for product in self.products:
            if product not in state.order_depths:
                continue
                
            depth = state.order_depths[product]
            if not depth.buy_orders or not depth.sell_orders:
                continue
                
            # Calculate prices
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            
            # Update price history
            self.price_history[product].append(mid_price)
            
            # Current position
            current_position = state.position.get(product, 0)
            self.positions[product] = current_position
            
            # Update trade duration if we have a position
            if current_position != 0:
                self.trade_duration[product] += 1
            else:
                self.trade_duration[product] = 0
            
            # Detect patterns and determine market state
            pattern = self.detect_patterns(product, mid_price)
            self.pattern_detected[product] = pattern
            market_state = self.determine_market_state(product, pattern)
            self.market_state[product] = market_state
            
            # Initialize orders list
            orders = []
            
            # Simple RSI-based trading logic
            rsi = list(self.rsi_values[product])[-1] if self.rsi_values[product] else 50
            
            # Determine trading action based on RSI thresholds and position limits
            if rsi < 30 and current_position < self.position_limit:
                # Buy signal - oversold
                orders.append(Order(product, best_ask, min(3, self.position_limit - current_position)))
            elif rsi > 70 and current_position > -self.position_limit:
                # Sell signal - overbought
                orders.append(Order(product, best_bid, -min(3, self.position_limit + current_position)))
            
            # Exit trades that have been held too long
            if current_position != 0 and self.trade_duration[product] > self.max_trade_duration:
                if current_position > 0:
                    orders.append(Order(product, best_bid, -current_position))
                else:
                    orders.append(Order(product, best_ask, -current_position))
                self.trade_duration[product] = 0
            
            # Add orders to result if any were generated
            if orders:
                result[product] = orders
        
        # Generate debug message with RSI values
        debug_msg = "Pattern strategy: "
        for product in self.products:
            if product in state.order_depths and self.rsi_values[product]:
                rsi = list(self.rsi_values[product])[-1]
                pos = self.positions[product]
                debug_msg += f"{product}: RSI={rsi:.1f}, Pos={pos}, "
                
        return result, 0, debug_msg

# --- Combined Trader class in required format ---
class Trader:
    def __init__(self):
        self.arbitrage = ArbitrageTrader()
        self.pattern = PatternTrader()
    
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # Print required info - important for system integration
        print("traderData: " + (state.traderData if state.traderData else ""))
        print("Observations: " + str(state.observations))
        
        # Run each strategy
        orders_arb, conv_arb, debug_arb = self.arbitrage.run(state)
        orders_pattern, conv_pattern, debug_pattern = self.pattern.run(state)
        
        # Merge orders (assumes strategies trade on different products)
        result: Dict[str, List[Order]] = {}
        for product, orders in orders_arb.items():
            result[product] = orders
        for product, orders in orders_pattern.items():
            if product in result:
                result[product].extend(orders)
            else:
                result[product] = orders
        
        # Set trader state data
        traderData = "CombinedStrategy"  # Can be enhanced to track state between iterations
        conversions = conv_arb + conv_pattern
        
        # Generate combined debug info
        debug_info = debug_arb + " | " + debug_pattern
        print("Debug: " + debug_info)
        
        return result, conversions, traderData

from collections import deque
import numpy as np
import jsonpickle
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        # List of products to market make
        self.products = ["SQUID_INK", "KELP", "RAINFOREST_RESIN"]
        
        # Position limits
        self.position_limits = {
            "SQUID_INK": 15,
            "KELP": 15,
            "RAINFOREST_RESIN": 25
        }
        
        # Trading parameters - optimized based on research
        self.params = {
            "SQUID_INK": {
                "take_width": 4,         # Increased due to high volatility shown in charts
                "clear_width": 2,         # Wider clearing width for volatility
                "disregard_edge": 2,      # Increased due to price swings
                "join_edge": 4,           # Join orders within this distance
                "default_edge": 6,        # Wider spread for highly volatile product
                "soft_position_limit": 6, # More conservative limit due to high volatility
                "adverse_volume": 15,     # Keep research-based threshold
                "reversion_beta": -0.07,  # Reduced mean reversion - chart shows strong trends
                "timeframe": 3,           # Longer timeframe for mean reversion
                "trend_factor": 0.6,      # Add trend following - charts show strong trends
                "trend_lookback": 5,      # Lookback period for trend detection
            },
            "KELP": {
                "take_width": 3,          # Increased slightly based on observed volatility
                "clear_width": 2,         # Maintain clearing width
                "disregard_edge": 2,      # Maintain selectivity on pennying
                "join_edge": 3,           # Keep join range
                "default_edge": 5,        # Slightly wider spread based on volatility
                "soft_position_limit": 8,  # Slightly more conservative
                "adverse_volume": 15,     # Keep research-based volume threshold
                "reversion_beta": -0.13,  # Slightly reduced mean reversion based on chart patterns
                "timeframe": 4,           # Increased timeframe as chart shows longer cycles
                "trend_factor": 0.4,      # Moderate trend following component
                "trend_lookback": 4,      # Lookback period for trend detection
            },
            "RAINFOREST_RESIN": {
                "fair_value": 10000,      # Confirmed by chart - very stable around 10000
                "take_width": 3,          # Reduced - chart shows tighter range
                "clear_width": 2,         # Reduced clearing width for tighter trading
                "disregard_edge": 2,      # Reduced based on lower volatility
                "join_edge": 2,           # Tighter join edge for less volatile product
                "default_edge": 4,        # Tighter spread for less volatile product
                "soft_position_limit": 20, # Increased - low risk in mean-reverting pattern
                "adverse_volume": 20,     # Keep threshold for less liquid product
                "reversion_beta": -0.15,  # Stronger mean reversion - chart confirms tight range
                "timeframe": 3,           # Moderate timeframe
                "trend_factor": 0.0,      # No trend following - pure mean reversion
                "trend_lookback": 3,      # Not used but defined for consistency
            }
        }
        
        # Price history
        self.price_history = {product: deque(maxlen=100) for product in self.products}
        
        # Store last fair values
        self.last_fair_values = {product: None for product in self.products}
        
        # Store last mid prices for reversion calculation
        self.last_mid_prices = {product: deque(maxlen=20) for product in self.products}
        
        # Store returns history for each timeframe
        self.returns_history = {product: deque(maxlen=20) for product in self.products}
        
        # Store trend direction
        self.trend_direction = {product: 0 for product in self.products}
    
    def calculate_fair_value(self, product: str, order_depth: OrderDepth, trader_data: dict) -> float:
        """Calculate fair value with research-based mean reversion and trend following"""
        # Default to parameter fair value for RAINFOREST_RESIN
        if product == "RAINFOREST_RESIN" and not self.last_fair_values[product]:
            return self.params[product]["fair_value"]
            
        # Calculate mid price
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return self.last_fair_values[product]
            
        # Calculate normal mid price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Filter for large orders only to avoid adverse selection
        # This is based directly on the research methodology
        adverse_volume = self.params[product]["adverse_volume"]
        filtered_asks = [price for price in order_depth.sell_orders.keys() 
                        if abs(order_depth.sell_orders[price]) >= adverse_volume]
        filtered_bids = [price for price in order_depth.buy_orders.keys() 
                        if abs(order_depth.buy_orders[price]) >= adverse_volume]
        
        # Calculate filtered mid price if possible, otherwise use regular mid
        fair_value = mid_price
        
        if filtered_asks and filtered_bids:
            mm_ask = min(filtered_asks)
            mm_bid = max(filtered_bids)
            mm_mid = (mm_ask + mm_bid) / 2
            
            # Store mid price in history
            self.last_mid_prices[product].append(mm_mid)
            
            # Calculate mean reversion component if we have enough history
            mean_reversion_component = 0
            timeframe = self.params[product]["timeframe"]
            if len(self.last_mid_prices[product]) > timeframe:
                # Calculate return over the appropriate timeframe
                last_price = list(self.last_mid_prices[product])[-timeframe-1]
                current_price = mm_mid
                returns = (current_price - last_price) / last_price
                
                # Store return in history
                self.returns_history[product].append(returns)
                
                # Apply research-based mean reversion coefficient
                mean_reversion_component = returns * self.params[product]["reversion_beta"]
            
            # Calculate trend following component
            trend_component = 0
            trend_lookback = self.params[product]["trend_lookback"]
            trend_factor = self.params[product]["trend_factor"]
            
            if len(self.last_mid_prices[product]) > trend_lookback + 1:
                # Calculate short-term trend direction
                price_series = list(self.last_mid_prices[product])
                recent_prices = price_series[-trend_lookback:]
                
                # Simple trend detection - linear regression slope would be better
                # but this is efficient for performance
                short_term_trend = recent_prices[-1] - recent_prices[0]
                if abs(short_term_trend) > 0:
                    # Normalize trend to a percentage
                    normalized_trend = short_term_trend / recent_prices[0]
                    trend_component = normalized_trend * trend_factor
                    
                    # Store trend direction for market making
                    self.trend_direction[product] = 1 if normalized_trend > 0 else -1
                else:
                    self.trend_direction[product] = 0
            
            # Combine mean reversion and trend factors - SQUID_INK will have stronger trend component
            # RAINFOREST_RESIN will only use mean reversion (trend_factor=0)
            fair_value = mm_mid * (1 + mean_reversion_component + trend_component)
        
        # Update price history for volatility calculation
        self.price_history[product].append(fair_value)
        self.last_fair_values[product] = fair_value
            
        return fair_value
    
    def take_best_orders(self, product: str, fair_value: float, take_width: float, 
                        orders: List[Order], order_depth: OrderDepth, position: int,
                        buy_volume: int, sell_volume: int, prevent_adverse: bool = True,
                        adverse_volume: int = 0) -> (int, int):
        """Take profitable orders that are mispriced beyond take_width"""
        position_limit = self.position_limits[product]
        
        # Take underpriced asks
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -order_depth.sell_orders[best_ask]
            
            # Only take if not potentially adverse selection
            # Research shows small volumes can be adverse selection signals
            if not prevent_adverse or best_ask_volume <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    # Profitable to buy
                    quantity = min(best_ask_volume, position_limit - position - buy_volume)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_volume += quantity
                        # Update order_depth to reflect our order
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        
        # Take overpriced bids
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            
            # Only take if not potentially adverse selection
            if not prevent_adverse or best_bid_volume <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    # Profitable to sell
                    quantity = min(best_bid_volume, position_limit + position - sell_volume)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_volume += quantity
                        # Update order_depth to reflect our order
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        
        return buy_volume, sell_volume
    
    def clear_position(self, product: str, fair_value: float, clear_width: int,
                      orders: List[Order], order_depth: OrderDepth, position: int,
                      buy_volume: int, sell_volume: int) -> (int, int):
        """Clear unwanted position at acceptable prices"""
        position_after_take = position + buy_volume - sell_volume
        position_limit = self.position_limits[product]
        
        # Define prices for position clearing
        bid_price = round(fair_value - clear_width)
        ask_price = round(fair_value + clear_width)
        
        # If long, try to reduce position
        if position_after_take > 0:
            # Look for bids above our ask price
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items()
                               if price >= ask_price)
            clear_quantity = min(clear_quantity, position_after_take)
            sell_quantity = min(position_limit + position - sell_volume, clear_quantity)
            
            if sell_quantity > 0:
                orders.append(Order(product, ask_price, -sell_quantity))
                sell_volume += sell_quantity
        
        # If short, try to reduce position
        if position_after_take < 0:
            # Look for asks below our bid price
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items()
                               if price <= bid_price)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            buy_quantity = min(position_limit - position - buy_volume, clear_quantity)
            
            if buy_quantity > 0:
                orders.append(Order(product, bid_price, buy_quantity))
                buy_volume += buy_quantity
                
        return buy_volume, sell_volume
    
    def make_market(self, product: str, order_depth: OrderDepth, fair_value: float,
                   position: int, buy_volume: int, sell_volume: int) -> List[Order]:
        """Make market with trend-aware join/penny logic"""
        orders = []
        params = self.params[product]
        position_limit = self.position_limits[product]
        
        # Determine available order capacity
        buy_capacity = position_limit - position - buy_volume
        sell_capacity = position_limit + position - sell_volume
        
        # Skip if we can't make market
        if buy_capacity <= 0 and sell_capacity <= 0:
            return orders
            
        # Extract parameters
        disregard_edge = params["disregard_edge"]
        join_edge = params["join_edge"]
        default_edge = params["default_edge"]
        soft_limit = params["soft_position_limit"]
        trend_direction = self.trend_direction[product]
        
        # Find orders worth pennying or joining
        asks_above_fair = [price for price in order_depth.sell_orders.keys()
                          if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys()
                          if price < fair_value - disregard_edge]
        
        best_ask_above = min(asks_above_fair) if asks_above_fair else None
        best_bid_below = max(bids_below_fair) if bids_below_fair else None
        
        # Calculate our quote prices
        ask_price = round(fair_value + default_edge)
        if best_ask_above is not None:
            # Either join or penny the best ask
            if abs(best_ask_above - fair_value) <= join_edge:
                ask_price = best_ask_above  # join
            else:
                ask_price = best_ask_above - 1  # penny
        
        bid_price = round(fair_value - default_edge)
        if best_bid_below is not None:
            # Either join or penny the best bid
            if abs(fair_value - best_bid_below) <= join_edge:
                bid_price = best_bid_below  # join
            else:
                bid_price = best_bid_below + 1  # penny
        
        # Adjust prices based on position
        if position > soft_limit:
            # If position too long, lower ask to sell more
            ask_price -= 1
        elif position < -soft_limit:
            # If position too short, raise bid to buy more
            bid_price += 1
            
        # Adjust orders based on trend (from price pattern analysis)
        if trend_direction > 0:  # Uptrend
            # In uptrend be more aggressive buying, less aggressive selling
            bid_price += 1  # Bid higher
            
            # Calculate smaller sell size in uptrend (more conservative)
            if sell_capacity > 2:
                sell_capacity = int(sell_capacity * 0.7)  # Reduce sell size in uptrend
                
        elif trend_direction < 0:  # Downtrend
            # In downtrend be more aggressive selling, less aggressive buying
            ask_price -= 1  # Ask lower
            
            # Calculate smaller buy size in downtrend (more conservative)
            if buy_capacity > 2:
                buy_capacity = int(buy_capacity * 0.7)  # Reduce buy size in downtrend
        
        # Special case for RAINFOREST_RESIN - pure mean reversion
        if product == "RAINFOREST_RESIN":
            # If price is below 10000, be more aggressive buying
            if fair_value < params["fair_value"]:
                bid_price += 1
                if sell_capacity > 2:
                    sell_capacity = int(sell_capacity * 0.7)
            # If price is above 10000, be more aggressive selling
            elif fair_value > params["fair_value"]:
                ask_price -= 1
                if buy_capacity > 2:
                    buy_capacity = int(buy_capacity * 0.7)
        
        # Place orders if we have capacity
        if buy_capacity > 0 and bid_price < ask_price:
            orders.append(Order(product, bid_price, buy_capacity))
            
        if sell_capacity > 0 and ask_price > bid_price:
            orders.append(Order(product, ask_price, -sell_capacity))
            
        return orders
    
    def run(self, state: TradingState):
        """Three-phase market making with research-optimized parameters"""
        # Initialize result dictionary
        result = {}
        
        # Initialize trader data
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Update positions
        for product, position in state.position.items():
            if product in self.products:
                pass  # Position directly available from state
        
        # Process each product
        for product in self.products:
            # Skip if product not in the order book
            if product not in state.order_depths:
                continue
                
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            
            # Skip if no orders
            if not order_depth.buy_orders and not order_depth.sell_orders:
                continue
            
            # Calculate fair value with research-based mean reversion
            fair_value = self.calculate_fair_value(product, order_depth, trader_data)
            
            # Phase 1: Take mispriced orders (arbitrage)
            orders = []
            buy_volume = 0
            sell_volume = 0
            buy_volume, sell_volume = self.take_best_orders(
                product, fair_value, self.params[product]["take_width"],
                orders, order_depth, position, buy_volume, sell_volume,
                True, self.params[product]["adverse_volume"]
            )
            
            # Phase 2: Clear unwanted position
            buy_volume, sell_volume = self.clear_position(
                product, fair_value, self.params[product]["clear_width"],
                orders, order_depth, position, buy_volume, sell_volume
            )
            
            # Phase 3: Make market
            make_orders = self.make_market(
                product, order_depth, fair_value, position, 
                buy_volume, sell_volume
            )
            
            # Combine all orders
            orders.extend(make_orders)
            
            # Add to result if we have orders
            if orders:
                result[product] = orders
        
        # Encode trader data
        trader_data_encoded = jsonpickle.encode(trader_data)
        
        return result, 0, trader_data_encoded
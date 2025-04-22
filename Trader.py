from datamodel import Order, OrderDepth, TradingState, ConversionObservation
from typing import Dict, List
import math
import jsonpickle
import numpy as np

class Trader:
    def __init__(self):
        # Parameters specific to MAGNIFICENT_MACARONS
        self.product = "MAGNIFICENT_MACARONS"
        self.position_limit = 75  # Position limit as specified
        self.conversion_limit = 10  # Conversion limit as specified
        
        # Improved strategy parameters
        self.params = {
            "make_edge": 2,           # Initial edge for market making
            "make_min_edge": 0.5,     # Minimum edge for market making
            "make_probability": 0.6,  # Probability factor for edge calculation
            "init_make_edge": 1.5,    # Initial edge value
            "min_edge": 0.3,          # Minimum allowable edge
            "volume_avg_window": 10,  # Number of timestamps to average volume
            "volume_bar": 40,         # Volume threshold for edge adjustment
            "edge_adjust_rate": 0.15, # Continuous edge adjustment rate (smaller for smoother changes)
            "csi_threshold": 180,     # Critical Sunlight Index threshold
            "csi_weight": 0.15,       # Weight for CSI impact on pricing
            "sugar_weight": 0.2,      # Sugar price adjustment weight
            "sunlight_weight": -0.1,  # Sunlight index adjustment weight
            "position_scale_factor": 0.8, # Factor to scale down order sizes
            "conversion_threshold": 0.6  # Position threshold ratio for conversion
        }
        
        # Initialize trader state
        self.trader_state = {
            "curr_edge": self.params["init_make_edge"],
            "volume_history": [],
            "edge_history": [],
            "pnl_history": [],
            "sugar_history": [],
            "sunlight_history": [],
            "position_history": [],
            "csi_factor": 0,
            "last_conversion": 0,
            "last_timestamp": 0,
            "last_position": 0,
            "avg_trade_price": 0,
            "total_volume": 0
        }

    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        """
        Main trading function called by the exchange for each iteration.
        
        Args:
            state (TradingState): The current market state
            
        Returns:
            Dict[str, List[Order]]: Orders to be placed
            int: Conversion count
            str: Trader state data for the next round
        """
        # Initialize result dictionary for orders
        result = {}
        conversions = 0
        
        # Restore trader state if available
        if state.traderData and state.traderData != "":
            try:
                self.trader_state = jsonpickle.decode(state.traderData)
            except:
                # Keep the default state if deserialization fails
                pass
        
        # Store timestamp for time-based calculations
        self.trader_state["last_timestamp"] = state.timestamp
        
        # Print debug information
        print(f"Time: {state.timestamp}, Position: {state.position.get(self.product, 0)}")
        
        # Execute strategy if the product is available
        if self.product in state.order_depths:
            orders, conversions = self.run_macarons_strategy(state)
            result[self.product] = orders
            
            # Print what we're doing
            print(f"Placing {len(orders)} orders, conversion count: {conversions}")
            print(f"Trading parameters: {self.params}")
        
        # Serialize trader state for next round
        trader_data_str = jsonpickle.encode(self.trader_state)
        return result, conversions, trader_data_str
    
    def run_macarons_strategy(self, state: TradingState) -> (List[Order], int):
        """Execute trading strategy for MAGNIFICENT_MACARONS"""
        orders = []
        position = state.position.get(self.product, 0)
        
        # Calculate position change since last iteration for P&L tracking
        position_change = position - self.trader_state.get("last_position", 0)
        self.trader_state["last_position"] = position
        
        # Get order depths
        order_depth = state.order_depths[self.product]
        
        # Track own trades for more accurate P&L estimation
        if self.product in state.own_trades and state.own_trades[self.product]:
            for trade in state.own_trades[self.product]:
                volume = abs(trade.quantity)
                # Update average trade price weighted by volume
                if self.trader_state.get("total_volume", 0) == 0:
                    self.trader_state["avg_trade_price"] = trade.price
                    self.trader_state["total_volume"] = volume
                else:
                    total_volume = self.trader_state["total_volume"] + volume
                    self.trader_state["avg_trade_price"] = (
                        (self.trader_state["avg_trade_price"] * self.trader_state["total_volume"]) + 
                        (trade.price * volume)
                    ) / total_volume
                    self.trader_state["total_volume"] = total_volume
        
        # Print order depths for debugging
        print(f"Buy orders: {order_depth.buy_orders}, Sell orders: {order_depth.sell_orders}")
        
        # Get conversion observation if available
        observation = None
        if state.observations and hasattr(state.observations, 'conversionObservations'):
            if self.product in state.observations.conversionObservations:
                observation = state.observations.conversionObservations[self.product]
                
                # Update environmental data with continuous tracking
                self.update_environmental_data(observation, position)
                
                # Print observations for debugging
                print(f"Sugar: {observation.sugarPrice}, Sunlight: {observation.sunlightIndex}")
                print(f"CSI factor: {self.trader_state['csi_factor']}")
                print(f"Tariffs - Import: {observation.importTariff}, Export: {observation.exportTariff}")
                print(f"Transport fees: {observation.transportFees}")
                print(f"Conversion prices - Bid: {observation.bidPrice}, Ask: {observation.askPrice}")
        
        # Calculate conversion amount (now returns continuous values)
        conversions = self.smooth_position_conversion(position, state.timestamp, observation)
        
        # If clearing position, partially adjust the position for order placement
        adjusted_position = position
        if conversions != 0:
            adjusted_position = position - conversions
        
        # If we have observation data, proceed with trading strategy
        if observation:
            # Calculate adaptive edge with smoothed adjustments
            adap_edge = self.smooth_adaptive_edge(
                state.timestamp,
                observation,
                adjusted_position
            )
            
            # Take arbitrage opportunities first with improved logic
            take_orders, buy_order_volume, sell_order_volume = self.improved_arb_take(
                order_depth,
                observation,
                adap_edge,
                adjusted_position
            )
            orders.extend(take_orders)
            
            # Then place market making orders with dynamic sizing
            make_orders, _, _ = self.improved_arb_make(
                order_depth,
                observation,
                adjusted_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )
            orders.extend(make_orders)
            
            print(f"Adaptive edge: {adap_edge}")
            print(f"Orders placed: {[(o.price, o.quantity) for o in orders]}")
        else:
            # Simplified but still more gradual strategy when no observation data is available
            orders = self.fallback_strategy(order_depth, position)
        
        return orders, conversions
    
    def update_environmental_data(self, observation: ConversionObservation, position: int):
        """Update and maintain environmental data with continuous tracking"""
        # Update environmental histories
        self.trader_state["sugar_history"].append(observation.sugarPrice)
        self.trader_state["sunlight_history"].append(observation.sunlightIndex)
        self.trader_state["position_history"].append(position)
        
        # Maintain window sizes with consistent history length
        max_history = self.params["volume_avg_window"]
        if len(self.trader_state["sugar_history"]) > max_history:
            self.trader_state["sugar_history"].pop(0)
        if len(self.trader_state["sunlight_history"]) > max_history:
            self.trader_state["sunlight_history"].pop(0)
        if len(self.trader_state["position_history"]) > max_history:
            self.trader_state["position_history"].pop(0)
        
        # Calculate continuous CSI factor instead of binary counter
        distance_from_csi = self.params["csi_threshold"] - observation.sunlightIndex
        if distance_from_csi > 0:
            # Gradual increase when below threshold
            self.trader_state["csi_factor"] = min(1.0, 
                self.trader_state.get("csi_factor", 0) + 0.1)
        else:
            # Gradual decrease when above threshold
            self.trader_state["csi_factor"] = max(0.0, 
                self.trader_state.get("csi_factor", 0) - 0.1)
    
    def smooth_implied_bid_ask(self, observation: ConversionObservation) -> (float, float):
        """Calculate implied bid and ask prices with smooth adjustments for environmental factors"""
        # Base implied prices from conversion facility
        implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees
        implied_ask = observation.askPrice + observation.importTariff + observation.transportFees
        
        # Get continuous CSI factor
        csi_factor = self.trader_state.get("csi_factor", 0)
        
        # Apply sunlight effect using continuous factor
        if csi_factor > 0:
            sunlight_effect = csi_factor * self.params["csi_weight"]
            implied_bid -= sunlight_effect
            implied_ask += sunlight_effect
        
        # Apply sugar price effect if we have history
        if len(self.trader_state["sugar_history"]) > 3:
            sugar_prices = self.trader_state["sugar_history"]
            # Calculate normalized sugar trend
            sugar_avg = np.mean(sugar_prices)
            sugar_std = max(1, np.std(sugar_prices))
            latest_sugar = sugar_prices[-1]
            sugar_z_score = (latest_sugar - sugar_avg) / sugar_std
            
            # Apply continuous effect based on z-score
            sugar_effect = sugar_z_score * self.params["sugar_weight"]
            implied_bid += sugar_effect
            implied_ask += sugar_effect
        
        return implied_bid, implied_ask
    
    def smooth_adaptive_edge(self, timestamp: int, observation: ConversionObservation, position: int) -> float:
        """Calculate adaptive edge with continuous adjustments"""
        curr_edge = self.trader_state.get("curr_edge", self.params["init_make_edge"])
        
        # Initialize edge on first call
        if timestamp == 0:
            self.trader_state["curr_edge"] = self.params["init_make_edge"]
            return self.params["init_make_edge"]
        
        # Add position to history without resetting
        self.trader_state["volume_history"].append(abs(position))
        self.trader_state["edge_history"].append(curr_edge)
        
        # Maintain window size
        max_history = self.params["volume_avg_window"]
        if len(self.trader_state["volume_history"]) > max_history:
            self.trader_state["volume_history"].pop(0)
        if len(self.trader_state["edge_history"]) > max_history:
            self.trader_state["edge_history"].pop(0)
        
        # Need enough history to make adjustments
        if len(self.trader_state["volume_history"]) < 3:
            return curr_edge
        
        # Calculate recent volume trends
        volume_avg = np.mean(self.trader_state["volume_history"])
        volume_ratio = volume_avg / self.params["volume_bar"]
        
        # Calculate target edge based on market conditions
        target_edge = curr_edge
        
        # If getting filled at high volume, gradually increase edge
        if volume_ratio > 0.8:
            # More aggressive increase when very high volume
            increase_factor = min(1.0, volume_ratio - 0.7)
            target_edge = curr_edge * (1 + increase_factor * self.params["edge_adjust_rate"])
        # If low volume, try reducing edge to attract more trades
        elif volume_ratio < 0.3 and curr_edge > self.params["make_min_edge"]:
            # Gradually decrease
            decrease_factor = 0.3 - volume_ratio
            target_edge = curr_edge * (1 - decrease_factor * self.params["edge_adjust_rate"])
        
        # Adjust for market volatility
        if len(self.trader_state["sunlight_history"]) > 3:
            sunlight_std = np.std(self.trader_state["sunlight_history"])
            # Normalize volatility measure
            vol_factor = min(1.0, sunlight_std / 20)
            # Higher volatility = wider edge
            target_edge *= (1 + vol_factor * 0.1)
        
        # Enforce edge limits
        target_edge = max(self.params["make_min_edge"], 
                          min(self.params["make_edge"] * 2, target_edge))
        
        # Smooth the transition to target (don't change abruptly)
        new_edge = curr_edge * 0.85 + target_edge * 0.15
        
        # Store and return the updated edge
        self.trader_state["curr_edge"] = new_edge
        return new_edge
    
    def smooth_position_conversion(self, position: int, timestamp: int, observation: ConversionObservation) -> int:
        """Calculate position conversion amount with gradual approach"""
        # Don't convert if no position or no observation data
        if position == 0 or not observation:
            return 0
        
        # Calculate absolute position ratio (how close we are to limit)
        position_ratio = abs(position) / self.position_limit
        
        # Don't convert if position is small relative to limit
        if position_ratio < self.params["conversion_threshold"]:
            return 0
            
        # Time-based cooldown check with continuous factor
        last_conversion = self.trader_state.get("last_conversion", 0)
        if last_conversion > 0:
            time_since_last = timestamp - last_conversion
            # Continuous probability based on time elapsed
            if time_since_last < 15:  # Min cooldown
                return 0
            
            # Gradually increase probability of conversion after cooldown
            conversion_probability = min(1.0, (time_since_last - 15) / 30)
            if np.random.random() > conversion_probability:
                return 0  # Random chance to skip conversion
                
        # Determine direction
        direction = -1 if position > 0 else 1
        
        # Calculate conversion amount (more aggressive for larger positions)
        ratio_above_threshold = position_ratio - self.params["conversion_threshold"]
        conversion_ratio = min(0.8, ratio_above_threshold * 2)  # Convert up to 80% of position
        
        # Calculate desired conversion amount
        conversion_amount = int(position * conversion_ratio)
        
        # Ensure negative for positive positions (selling) and positive for negative positions (buying)
        conversion_amount = -abs(conversion_amount) if position > 0 else abs(conversion_amount)
        
        # Clamp to conversion limit
        conversion_amount = max(-self.conversion_limit, min(self.conversion_limit, conversion_amount))
        
        # Update last conversion time if we're actually converting
        if conversion_amount != 0:
            self.trader_state["last_conversion"] = timestamp
            
        return conversion_amount
    
    def improved_arb_take(self, order_depth: OrderDepth, observation: ConversionObservation, 
                adap_edge: float, position: int) -> (List[Order], int, int):
        """Take profitable arbitrage opportunities with improved continuous logic"""
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Calculate implied prices with smooth adjustments
        implied_bid, implied_ask = self.smooth_implied_bid_ask(observation)
        
        # Calculate available quantities with scaling for smoother transitions
        position_ratio = abs(position) / self.position_limit
        
        # Scale quantities based on position - smaller orders when close to limits
        scaling = max(0.1, 1 - position_ratio * self.params["position_scale_factor"])
        max_buy = int((self.position_limit - position) * scaling)
        max_sell = int((self.position_limit + position) * scaling)

        # Dynamic edge for taking based on position
        take_edge = adap_edge * self.params["make_probability"] * (1 + position_ratio * 0.2)
        
        # Take sell orders that are below our implied bid with probabilistic approach
        if order_depth.sell_orders and max_buy > 0:
            for price in sorted(list(order_depth.sell_orders.keys())):
                # Calculate continuous profitability factor
                profit_margin = (implied_bid - price) / implied_bid
                
                if profit_margin <= 0:
                    continue  # Skip unprofitable orders
                
                # Scale quantity based on profitability (take more of better deals)
                profitability_scale = min(1.0, profit_margin * 10)
                quantity = min(abs(order_depth.sell_orders[price]), 
                              int(max_buy * profitability_scale))
                
                if quantity > 0:
                    orders.append(Order(self.product, round(price), quantity))
                    buy_order_volume += quantity
                    max_buy -= quantity
                    if max_buy <= 0:
                        break

        # Take buy orders that are above our implied ask with probabilistic approach
        if order_depth.buy_orders and max_sell > 0:
            for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
                # Calculate continuous profitability factor
                profit_margin = (price - implied_ask) / implied_ask
                
                if profit_margin <= 0:
                    continue  # Skip unprofitable orders
                
                # Scale quantity based on profitability (take more of better deals)
                profitability_scale = min(1.0, profit_margin * 10)
                quantity = min(abs(order_depth.buy_orders[price]), 
                              int(max_sell * profitability_scale))
                
                if quantity > 0:
                    orders.append(Order(self.product, round(price), -quantity))
                    sell_order_volume += quantity
                    max_sell -= quantity
                    if max_sell <= 0:
                        break

        return orders, buy_order_volume, sell_order_volume
    
    def improved_arb_make(self, order_depth: OrderDepth, observation: ConversionObservation,
               position: int, edge: float, buy_order_volume: int, sell_order_volume: int) -> (List[Order], int, int):
        """Place market making orders with dynamic sizing and continuous price adjustments"""
        orders = []
        
        # Calculate implied bid/ask with smooth adjustments
        implied_bid, implied_ask = self.smooth_implied_bid_ask(observation)
        
        # Calculate position-adjusted edge (wider when closer to position limits)
        position_ratio = abs(position) / self.position_limit
        adjusted_edge = edge * (1 + position_ratio * 0.5)
        
        # Calculate bid and ask prices with continuous adjustments
        bid = implied_bid - adjusted_edge
        ask = implied_ask + adjusted_edge
        
        # Adjust for position imbalance (encourage mean reversion)
        if position > 0:
            # Long bias - tighten bid (less aggressive buying), widen ask (more aggressive selling)
            bid_skew = min(1.0, position / (self.position_limit * 0.5)) * adjusted_edge * 0.3
            bid -= bid_skew
            ask -= bid_skew * 0.3  # Slightly lower ask to encourage selling
        elif position < 0:
            # Short bias - tighten ask (less aggressive selling), widen bid (more aggressive buying)
            ask_skew = min(1.0, abs(position) / (self.position_limit * 0.5)) * adjusted_edge * 0.3
            ask += ask_skew
            bid += ask_skew * 0.3  # Slightly higher bid to encourage buying
        
        # Calculate foreign mid price and try modified aggressive ask
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.2  # Less aggressive than original
        
        # Use aggressive ask if it's still profitable, with continuous transition
        if aggressive_ask > implied_ask:
            # Blend between regular ask and aggressive ask based on profitability
            profit_ratio = min(1.0, (aggressive_ask - implied_ask) / edge)
            ask = ask * (1 - profit_ratio * 0.7) + aggressive_ask * (profit_ratio * 0.7)
        
        # Round prices to avoid fractional ticks
        bid_price = round(bid)
        ask_price = round(ask)
        
        # Avoid crossed or locked market
        if bid_price >= ask_price:
            mid_price = (bid + ask) / 2
            bid_price = math.floor(mid_price - 0.5)
            ask_price = math.ceil(mid_price + 0.5)
        
        # Check for large orders in the book to avoid getting stuck behind them
        filtered_asks = [price for price in order_depth.sell_orders.keys() 
                         if abs(order_depth.sell_orders[price]) >= 30]
        filtered_bids = [price for price in order_depth.buy_orders.keys() 
                         if abs(order_depth.buy_orders[price]) >= 20]
        
        # If there are large orders, try to place just in front with continuous approach
        if filtered_asks and min(filtered_asks) < ask_price:
            # Calculate best price relative to large order
            target_ask = min(filtered_asks) - 1
            if target_ask > implied_ask:
                # Blend between calculated ask and competitive ask
                competitiveness = min(1.0, (target_ask - implied_ask) / edge)
                ask_price = int(ask_price * (1 - competitiveness) + target_ask * competitiveness)
            
        if filtered_bids and max(filtered_bids) > bid_price:
            # Calculate best price relative to large order
            target_bid = max(filtered_bids) + 1
            if target_bid < implied_bid:
                # Blend between calculated bid and competitive bid
                competitiveness = min(1.0, (implied_bid - target_bid) / edge)
                bid_price = int(bid_price * (1 - competitiveness) + target_bid * competitiveness)
        
        # Dynamic order sizing based on position and edge
        position_ratio = abs(position) / self.position_limit
        size_scaling = max(0.2, 1 - position_ratio)
        
        # Calculate remaining quantities with dynamic scaling
        max_buy = self.position_limit - (position + buy_order_volume)
        max_sell = self.position_limit + (position - sell_order_volume)
        
        # Scale buy quantity based on position (smaller when close to limits)
        buy_quantity = int(max_buy * size_scaling)
        if position > 0:
            # Additional scaling when long to reduce risk
            buy_quantity = int(buy_quantity * max(0.1, 1 - (position / self.position_limit)))
            
        # Scale sell quantity based on position (smaller when close to limits)
        sell_quantity = int(max_sell * size_scaling)
        if position < 0:
            # Additional scaling when short to reduce risk
            sell_quantity = int(sell_quantity * max(0.1, 1 - (abs(position) / self.position_limit)))
        
        # Place market making orders if we have capacity
        if buy_quantity > 0 and bid_price > 0:
            orders.append(Order(self.product, bid_price, buy_quantity))
        
        if sell_quantity > 0 and ask_price > 0:
            orders.append(Order(self.product, ask_price, -sell_quantity))
        
        return orders, buy_order_volume, sell_order_volume
    
    def fallback_strategy(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Simplified fallback strategy when no observation data is available"""
        orders = []
        
        # Calculate available capacity with gradual scaling
        position_ratio = abs(position) / self.position_limit
        scaling = max(0.2, 1 - position_ratio * 0.7)
        
        # Calculate best prices and potential mid price
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        
        if best_bid > 0 and best_ask < float('inf'):
            mid_price = (best_bid + best_ask) / 2
            
            # Dynamic buy limit based on mid price and position
            if best_ask < mid_price * 1.05:  # Only buy if ask is reasonably close to mid
                buy_limit = mid_price * 0.97  # Don't buy above this price
                buy_capacity = int((self.position_limit - position) * scaling)
                
                if best_ask < buy_limit and buy_capacity > 0:
                    buy_quantity = min(abs(order_depth.sell_orders[best_ask]), buy_capacity)
                    if buy_quantity > 0:
                        orders.append(Order(self.product, best_ask, buy_quantity))
            
            # Dynamic sell limit based on mid price and position
            if best_bid > mid_price * 0.95:  # Only sell if bid is reasonably close to mid
                sell_limit = mid_price * 1.03  # Don't sell below this price
                sell_capacity = int((self.position_limit + position) * scaling)
                
                if best_bid > sell_limit and sell_capacity > 0:
                    sell_quantity = min(order_depth.buy_orders[best_bid], sell_capacity)
                    if sell_quantity > 0:
                        orders.append(Order(self.product, best_bid, -sell_quantity))
                        
        # Add market making orders if thin book
        if len(orders) == 0 and best_bid > 0 and best_ask < float('inf'):
            mid_price = (best_bid + best_ask) / 2
            
            # Gradual spread based on position
            spread = mid_price * 0.04 * (1 + position_ratio)
            
            # Calculate buy capacity with position scaling
            buy_capacity = int((self.position_limit - position) * scaling * 0.5)
            if buy_capacity > 0:
                bid_price = int(mid_price - spread/2)
                if bid_price > 0:
                    orders.append(Order(self.product, bid_price, buy_capacity))
            
            # Calculate sell capacity with position scaling
            sell_capacity = int((self.position_limit + position) * scaling * 0.5)
            if sell_capacity > 0:
                ask_price = int(mid_price + spread/2)
                if ask_price > 0:
                    orders.append(Order(self.product, ask_price, -sell_capacity))
        
        return orders 
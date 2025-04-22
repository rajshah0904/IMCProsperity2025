from datamodel import Order, OrderDepth, TradingState, UserId, ConversionObservation
from typing import Dict, List
import numpy as np
import jsonpickle
import math

class Trader:
    def __init__(self):
        # Initialize trader data structure
        self.macarons_trader = MacaronTrader()
        # Add more product traders here as needed
    
    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        """
        Main trading function called by the exchange for each trading iteration.
        
        Args:
            state (TradingState): The current market state
            
        Returns:
            Dict[str, List[Order]]: Orders to be placed
            int: Conversion count
            str: Trader state data for the next round
        """
        # Initialize trader data from previous round if available
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Initialize result dictionary for orders
        result = {}
        conversions = 0
        
        # Execute MAGNIFICENT_MACARONS trading strategy
        if "MAGNIFICENT_MACARONS" in state.order_depths:
            macarons_orders, macarons_conversions = self.macarons_trader.run_strategy(state, trader_data)
            result["MAGNIFICENT_MACARONS"] = macarons_orders
            conversions += macarons_conversions
        
        # Add more product strategies here
        
        # Serialize trader data for next round
        trader_data_str = jsonpickle.encode(trader_data)
        
        return result, conversions, trader_data_str

class MacaronTrader:
    def __init__(self):
        self.product = "MAGNIFICENT_MACARONS"
        self.position_limit = 100  # Position limit for macarons
        
        # Trading parameters based on correlation analysis
        self.params = {
            "make_edge": 2,           # Initial edge for market making
            "make_min_edge": 0.5,     # Minimum edge for market making
            "make_probability": 0.566, # Probability factor for market making
            "volume_avg_timestamp": 5, # Number of timestamps to average volume
            "volume_bar": 75,         # Volume threshold for edge adjustment
            "dec_edge_discount": 0.8, # Discount for decreasing edge
            "step_size": 0.5,         # Step size for edge adjustments
            "sugar_correlation": 0.38, # Correlation coefficient for sugar price
            "sunlight_correlation": -0.23, # Correlation for sunlight index
            "sugar_window": 10,       # Best window for sugar price
            "sunlight_window": 250,   # Best window for sunlight index
            "sugar_threshold": 3.0    # Threshold for significant sugar price deviation
        }
    
    def implied_bid_ask(self, observation: ConversionObservation) -> (float, float):
        """Calculate implied bid and ask prices adjusting for tariffs and fees"""
        implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        implied_ask = observation.askPrice + observation.importTariff + observation.transportFees
        return implied_bid, implied_ask
    
    def adaptive_edge(self, timestamp: int, curr_edge: float, position: int, trader_data: dict, observation: ConversionObservation) -> float:
        """Dynamically adjust trading edge based on market conditions"""
        # Initialize trader data if needed
        if "MAGNIFICENT_MACARONS" not in trader_data:
            trader_data["MAGNIFICENT_MACARONS"] = {
                "curr_edge": self.params["make_edge"],
                "volume_history": [],
                "optimized": False,
                "sugar_price_history": [],
                "sunlight_index_history": []
            }
        
        # Reset on first timestamp
        if timestamp == 0:
            trader_data["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params["make_edge"]
            trader_data["MAGNIFICENT_MACARONS"]["volume_history"] = []
            trader_data["MAGNIFICENT_MACARONS"]["sugar_price_history"] = []
            trader_data["MAGNIFICENT_MACARONS"]["sunlight_index_history"] = []
            trader_data["MAGNIFICENT_MACARONS"]["optimized"] = False
            return self.params["make_edge"]
        
        # Track position volume history
        trader_data["MAGNIFICENT_MACARONS"]["volume_history"].append(abs(position))
        if len(trader_data["MAGNIFICENT_MACARONS"]["volume_history"]) > self.params["volume_avg_timestamp"]:
            trader_data["MAGNIFICENT_MACARONS"]["volume_history"].pop(0)
        
        # Track environmental factors
        trader_data["MAGNIFICENT_MACARONS"]["sugar_price_history"].append(observation.sugarPrice)
        trader_data["MAGNIFICENT_MACARONS"]["sunlight_index_history"].append(observation.sunlightIndex)
        
        # Maintain window sizes
        if len(trader_data["MAGNIFICENT_MACARONS"]["sugar_price_history"]) > self.params["sugar_window"]:
            trader_data["MAGNIFICENT_MACARONS"]["sugar_price_history"].pop(0)
        if len(trader_data["MAGNIFICENT_MACARONS"]["sunlight_index_history"]) > self.params["sunlight_window"]:
            trader_data["MAGNIFICENT_MACARONS"]["sunlight_index_history"].pop(0)
        
        # Volume-based edge adjustment
        if len(trader_data["MAGNIFICENT_MACARONS"]["volume_history"]) >= self.params["volume_avg_timestamp"] and not trader_data["MAGNIFICENT_MACARONS"]["optimized"]:
            volume_avg = np.mean(trader_data["MAGNIFICENT_MACARONS"]["volume_history"])
            
            # Increase edge if consistently getting full size trades
            if volume_avg >= self.params["volume_bar"]:
                trader_data["MAGNIFICENT_MACARONS"]["volume_history"] = []
                trader_data["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge + self.params["step_size"]
                return curr_edge + self.params["step_size"]
            
            # Decrease edge if more profitable with reduced edge
            elif self.params["dec_edge_discount"] * self.params["volume_bar"] * (curr_edge - self.params["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params["step_size"] > self.params["make_min_edge"]:
                    trader_data["MAGNIFICENT_MACARONS"]["volume_history"] = []
                    trader_data["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge - self.params["step_size"]
                    trader_data["MAGNIFICENT_MACARONS"]["optimized"] = True
                    return curr_edge - self.params["step_size"]
                else:
                    trader_data["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params["make_min_edge"]
                    return self.params["make_min_edge"]
        
        # Environmental factor adjustments
        # 1. Sugar price movement analysis
        sugar_movement = 0
        if len(trader_data["MAGNIFICENT_MACARONS"]["sugar_price_history"]) >= self.params["sugar_window"]:
            earliest_sugar = trader_data["MAGNIFICENT_MACARONS"]["sugar_price_history"][0]
            latest_sugar = trader_data["MAGNIFICENT_MACARONS"]["sugar_price_history"][-1]
            sugar_movement = (latest_sugar - earliest_sugar) / earliest_sugar if earliest_sugar != 0 else 0
            
            # Calculate sugar price deviation from baseline (200)
            sugar_diff = abs(latest_sugar - 200)
            
            # Adjust edge based on sugar price correlation and deviation
            if sugar_diff > self.params["sugar_threshold"]:
                sugar_adjustment = self.params["step_size"] * sugar_movement * self.params["sugar_correlation"]
                edge_candidate = max(self.params["make_min_edge"], curr_edge + sugar_adjustment)
                # Only apply if adjustment is significant
                if abs(edge_candidate - curr_edge) > 0.1:
                    trader_data["MAGNIFICENT_MACARONS"]["curr_edge"] = edge_candidate
                    return edge_candidate
        
        # 2. Sunlight index movement analysis
        sunlight_movement = 0
        if len(trader_data["MAGNIFICENT_MACARONS"]["sunlight_index_history"]) >= self.params["sunlight_window"]:
            earliest_sunlight = trader_data["MAGNIFICENT_MACARONS"]["sunlight_index_history"][0]
            latest_sunlight = trader_data["MAGNIFICENT_MACARONS"]["sunlight_index_history"][-1]
            sunlight_movement = (latest_sunlight - earliest_sunlight) / earliest_sunlight if earliest_sunlight != 0 else 0
            
            # Adjust edge based on sunlight index correlation
            if abs(sunlight_movement) > 0.01:  # Threshold for significant movement
                sunlight_adjustment = self.params["step_size"] * sunlight_movement * self.params["sunlight_correlation"]
                edge_candidate = max(self.params["make_min_edge"], curr_edge + sunlight_adjustment)
                # Only apply if adjustment is significant
                if abs(edge_candidate - curr_edge) > 0.1:
                    trader_data["MAGNIFICENT_MACARONS"]["curr_edge"] = edge_candidate
                    return edge_candidate
        
        # No change if we get here
        trader_data["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge
        return curr_edge
    
    def arb_take(self, order_depth: OrderDepth, observation: ConversionObservation, edge: float, position: int) -> (List[Order], int, int):
        """Take arbitrage opportunities from the market"""
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0
        
        implied_bid, implied_ask = self.implied_bid_ask(observation)
        
        buy_quantity = self.position_limit - position
        sell_quantity = self.position_limit + position
        
        # Calculate sugar price deviation factor
        sugar_diff = abs(observation.sugarPrice - 200)
        edge_factor = 1.0
        if sugar_diff > self.params["sugar_threshold"]:
            edge_factor = 1.2  # More aggressive when sugar price deviates significantly
        
        # Calculate effective edge for taking
        take_edge = edge * edge_factor * self.params["make_probability"]
        
        # Take sell orders below our implied bid
        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - take_edge:
                break
                
            quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
            if quantity > 0:
                orders.append(Order(self.product, price, quantity))
                buy_order_volume += quantity
                buy_quantity -= quantity
                if buy_quantity <= 0:
                    break
        
        # Take buy orders above our implied ask
        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + take_edge:
                break
                
            quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
            if quantity > 0:
                orders.append(Order(self.product, price, -quantity))
                sell_order_volume += quantity
                sell_quantity -= quantity
                if sell_quantity <= 0:
                    break
        
        return orders, buy_order_volume, sell_order_volume
    
    def arb_make(self, order_depth: OrderDepth, observation: ConversionObservation, position: int, edge: float, buy_order_volume: int, sell_order_volume: int) -> (List[Order], int, int):
        """Make orders for market making"""
        orders = []
        
        implied_bid, implied_ask = self.implied_bid_ask(observation)
        
        # Calculate base bid/ask levels
        bid = implied_bid - edge
        ask = implied_ask + edge
        
        # Adjust ask price based on environmental factors
        sugar_diff = abs(observation.sugarPrice - 200)
        if sugar_diff > self.params["sugar_threshold"]:
            # Be more aggressive when sugar price deviates significantly
            foreign_mid = (observation.askPrice + observation.bidPrice) / 2
            aggressive_ask = foreign_mid - 1.6  # Similar to ORCHIDS
            
            # Use aggressive ask if it's profitable
            if aggressive_ask >= implied_ask + self.params["make_min_edge"]:
                ask = aggressive_ask
        
        # Adapt to market depth
        large_size_threshold = 40
        filtered_ask = [price for price in order_depth.sell_orders.keys() 
                       if abs(order_depth.sell_orders[price]) >= large_size_threshold]
        filtered_bid = [price for price in order_depth.buy_orders.keys() 
                       if abs(order_depth.buy_orders[price]) >= large_size_threshold]
        
        # Penny better levels if possible
        if filtered_ask and ask > min(filtered_ask):
            if min(filtered_ask) - 1 > implied_ask:
                ask = min(filtered_ask) - 1
            else:
                ask = implied_ask + edge
                
        if filtered_bid and bid < max(filtered_bid):
            if max(filtered_bid) + 1 < implied_bid:
                bid = max(filtered_bid) + 1
            else:
                bid = implied_bid - edge
        
        # Place orders
        buy_quantity = self.position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(self.product, int(bid), buy_quantity))
        
        sell_quantity = self.position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(self.product, int(ask), -sell_quantity))
        
        return orders, buy_order_volume, sell_order_volume
    
    def arb_clear(self, position: int) -> int:
        """Clear position if needed"""
        return -position  # Convert all position back to cash
    
    def run_strategy(self, state: TradingState, trader_data: dict) -> (List[Order], int):
        """Execute trading strategy for MAGNIFICENT_MACARONS"""
        orders = []
        conversions = 0
        
        # Check if our product is in the order depths
        if self.product in state.order_depths:
            # Get current position or default to 0
            position = state.position.get(self.product, 0)
            
            # Get conversion observation
            if self.product in state.observations.conversionObservations:
                observation = state.observations.conversionObservations[self.product]
                
                # Determine if we should convert position back to cash
                conversions = self.arb_clear(position)
                
                # Reset position after conversion
                adjusted_position = 0 if conversions != 0 else position
                
                # Calculate adaptive edge
                adaptive_edge = self.adaptive_edge(
                    state.timestamp,
                    trader_data.get("MAGNIFICENT_MACARONS", {}).get("curr_edge", self.params["make_edge"]),
                    adjusted_position,
                    trader_data,
                    observation
                )
                
                # Take arbitrage opportunities
                take_orders, buy_volume, sell_volume = self.arb_take(
                    state.order_depths[self.product],
                    observation,
                    adaptive_edge,
                    adjusted_position
                )
                
                # Market make
                make_orders, _, _ = self.arb_make(
                    state.order_depths[self.product],
                    observation,
                    adjusted_position,
                    adaptive_edge,
                    buy_volume,
                    sell_volume
                )
                
                # Combine orders
                orders = take_orders + make_orders
        
        return orders, conversions

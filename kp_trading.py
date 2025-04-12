from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import numpy as np
from collections import deque
import string
import jsonpickle
import math

class Product:
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"

BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 4,
        Product.JAMS: 6,
        Product.DJEMBES: 1,
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2,
    }
}

PARAMS = {
    Product.PICNIC_BASKET1: {
        "spread_mean": 0,  # Will be calculated from data
        "starting_its": 30000,
        "spread_std_window": 25,
        "zscore_threshold": 11.0,
        "target_position": 60,
        "min_width": 1,
        "max_width": 8,
        "mm_min_volume": 10,
    },
    Product.PICNIC_BASKET2: {
        "spread_mean": 0,  # Will be calculated from data
        "starting_its": 30000,
        "spread_std_window": 25,
        "zscore_threshold": 11.0,
        "target_position": 60,
        "min_width": 1,
        "max_width": 8,
        "mm_min_volume": 10,
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 250,
            Product.DJEMBES: 250
        }

        # Initialize price history for components and basket
        self.price_history = {
            "CROISSANTS": deque(maxlen=200),
            "JAMS": deque(maxlen=200),
            "PICNIC_BASKET2": deque(maxlen=200)  # Fixed product name
        }
        
        # Initialize spread history
        self.spread_history = deque(maxlen=200)
        
        # Position tracking
        self.positions = {
            "CROISSANTS": 0,
            "JAMS": 0,
            "PICNIC_BASKET2": 0  # Fixed product name
        }
        
        # Position limits (adjusted based on order book volumes)
        self.position_limits = {
            "CROISSANTS": 100,  # Typical volume ~100-150
            "JAMS": 150,        # Typical volume ~150-250
            "PICNIC_BASKET2": 30 # Typical volume ~20-40
        }
        
        # Trading parameters
        self.zscore_entry = 2.0  # More conservative entry
        self.zscore_exit = 0.5
        self.window_size = 100
        
        # Basket composition (4 CROISSANTS + 2 JAMS)
        self.basket_weights = {
            "CROISSANTS": 4,
            "JAMS": 2
        }

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid price from order depth"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def calculate_fair_value(self, croissant_price: float, jam_price: float) -> float:
        """Calculate fair value of Basket 2 based on component prices"""
        if croissant_price is None or jam_price is None:
            return None
        return 4 * croissant_price + 2 * jam_price

    def calculate_zscore(self, spread: float) -> float:
        """Calculate z-score of the spread"""
        if len(self.spread_history) < self.window_size:
            return 0
            
        spread_array = np.array(self.spread_history)
        mean = np.mean(spread_array[-self.window_size:])
        std = np.std(spread_array[-self.window_size:])
        
        if std == 0:
            return 0
            
        return (spread - mean) / std

    def get_market_orders(self, product: str, target_position: int, order_depth: OrderDepth) -> List[Order]:
        """Generate orders to reach target position"""
        current_position = self.positions[product]
        required_change = target_position - current_position
        orders = []

        if required_change > 0:  # Need to buy
            for price in sorted(order_depth.sell_orders.keys()):
                volume = -order_depth.sell_orders[price]  # Convert to positive volume
                volume_to_trade = min(volume, required_change)
                if volume_to_trade > 0:
                    orders.append(Order(product, price, volume_to_trade))
                    required_change -= volume_to_trade
                if required_change <= 0:
                    break

        elif required_change < 0:  # Need to sell
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                volume = order_depth.buy_orders[price]
                volume_to_trade = max(-volume, required_change)
                if volume_to_trade < 0:
                    orders.append(Order(product, price, volume_to_trade))
                    required_change -= volume_to_trade
                if required_change >= 0:
                    break

        return orders

    def get_swmid(self, order_depth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth], basket_type: str) -> OrderDepth:
        # Get basket weights
        weights = BASKET_WEIGHTS[basket_type]
        CROISSANTS_PER_BASKET = weights[Product.CROISSANTS]
        JAMS_PER_BASKET = weights[Product.JAMS]
        DJEMBES_PER_BASKET = weights.get(Product.DJEMBES, 0)  # Default to 0 if not present

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else 0
        croissants_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else float('inf')
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float('inf')
        
        # Only include djembes if they're part of the basket
        djembes_best_bid = 0
        djembes_best_ask = float('inf')
        if DJEMBES_PER_BASKET > 0:
            djembes_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys()) if order_depths[Product.DJEMBES].buy_orders else 0
            djembes_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys()) if order_depths[Product.DJEMBES].sell_orders else float('inf')

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = croissants_best_bid * CROISSANTS_PER_BASKET + jams_best_bid * JAMS_PER_BASKET + djembes_best_bid * DJEMBES_PER_BASKET
        implied_ask = croissants_best_ask * CROISSANTS_PER_BASKET + jams_best_ask * JAMS_PER_BASKET + djembes_best_ask * DJEMBES_PER_BASKET

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid] // CROISSANTS_PER_BASKET
            jams_bid_volume = order_depths[Product.JAMS].buy_orders[jams_best_bid] // JAMS_PER_BASKET
            implied_bid_volume = min(croissants_bid_volume, jams_bid_volume)
            
            if DJEMBES_PER_BASKET > 0:
                djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders[djembes_best_bid] // DJEMBES_PER_BASKET
                implied_bid_volume = min(implied_bid_volume, djembes_bid_volume)
                
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float('inf'):
            croissants_ask_volume = -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask] // CROISSANTS_PER_BASKET
            jams_ask_volume = -order_depths[Product.JAMS].sell_orders[jams_best_ask] // JAMS_PER_BASKET
            implied_ask_volume = min(croissants_ask_volume, jams_ask_volume)
            
            if DJEMBES_PER_BASKET > 0:
                djembes_ask_volume = -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask] // DJEMBES_PER_BASKET
                implied_ask_volume = min(implied_ask_volume, djembes_ask_volume)
                
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_history: List[float]):
        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, product)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        
        if basket_swmid is None or synthetic_swmid is None:
            return None
            
        spread = basket_swmid - synthetic_swmid
        spread_history.append(spread)
        
        if len(spread_history) < self.params[product]["spread_std_window"]:
            return None
            
        spread_std = np.std(spread_history[-self.params[product]["spread_std_window"]:])
        if spread_std == 0:
            return None
            
        spread_mean = (np.sum(spread_history) + (self.params[product]["spread_mean"] * self.params[product]["starting_its"])) / \
                     (self.params[product]["starting_its"] + len(spread_history))
        zscore = (spread - spread_mean) / spread_std
        
        if zscore >= self.params[product]["zscore_threshold"]:
            if basket_position == -self.params[product]["target_position"]:
                return None
            target_quantity = abs(-self.params[product]["target_position"] - basket_position)

            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(product, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)]
            
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths, product)
            aggregate_orders[product] = basket_orders
            return aggregate_orders
            
        if zscore <= -self.params[product]["zscore_threshold"]:
            if basket_position == self.params[product]["target_position"]:
                return None
            target_quantity = abs(self.params[product]["target_position"] - basket_position)

            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(product, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]

            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths, product)
            aggregate_orders[product] = basket_orders
            return aggregate_orders
    
        return None

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth], basket_type: str) -> Dict[str, List[Order]]:
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }
        
        # Only include djembes if they're part of the basket
        weights = BASKET_WEIGHTS[basket_type]
        if Product.DJEMBES in weights:
            component_orders[Product.DJEMBES] = []

        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_type)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float("inf")

        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity

            if quantity > 0 and price >= best_ask:
                croissants_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                if Product.DJEMBES in weights:
                    djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                if Product.DJEMBES in weights:
                    djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                continue

            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * weights[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * weights[Product.JAMS],
            )

            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            
            if Product.DJEMBES in weights:
                djembes_order = Order(
                    Product.DJEMBES, djembes_price, quantity * weights[Product.DJEMBES]
                )
                component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        # Initialize spread history if not exists
        if "spread_history" not in traderObject:
            traderObject["spread_history"] = []

        # Trade PICNIC_BASKET1
        if Product.PICNIC_BASKET1 in state.order_depths:
            basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
            spread_orders = self.spread_orders(
                state.order_depths, 
                Product.PICNIC_BASKET1, 
                basket_position, 
                traderObject["spread_history"]
            )
            if spread_orders is not None:
                result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
                result[Product.JAMS] = spread_orders[Product.JAMS]
                result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
                result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]

        # Trade PICNIC_BASKET2
        if Product.PICNIC_BASKET2 in state.order_depths:
            basket_position = state.position.get(Product.PICNIC_BASKET2, 0)
            spread_orders = self.spread_orders(
                state.order_depths, 
                Product.PICNIC_BASKET2, 
                basket_position, 
                traderObject["spread_history"]
            )
            if spread_orders is not None:
                result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
                result[Product.JAMS] = spread_orders[Product.JAMS]
                result[Product.PICNIC_BASKET2] = spread_orders[Product.PICNIC_BASKET2]

        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List

class Trader:
    def run(self, state: TradingState):
        result = {}
        acceptable_prices = {
            "KELP": 2019.05,
            "RAINFOREST_RESIN": 10000.05
        }

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            acceptable_price = acceptable_prices.get(product, 0)

            # Sort sell orders (lowest price first)
            for ask_price in sorted(order_depth.sell_orders):
                ask_volume = order_depth.sell_orders[ask_price]
                if ask_price < acceptable_price - 2:
                    # Buy if ask is significantly lower than fair value
                    orders.append(Order(product, ask_price, -ask_volume))

            # Sort buy orders (highest price first)
            for bid_price in sorted(order_depth.buy_orders, reverse=True):
                bid_volume = order_depth.buy_orders[bid_price]
                if bid_price > acceptable_price + 2:
                    # Sell if bid is significantly higher than fair value
                    orders.append(Order(product, bid_price, -bid_volume))

            result[product] = orders

        traderData = ""  # No state to track across rounds yet
        conversions = 0  # No conversion logic used for now

        return result, conversions, traderData

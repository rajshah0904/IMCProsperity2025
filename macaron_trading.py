from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import string
import jsonpickle
import numpy as np
import math


class Product:
    MAGNIFICIENT_MACARONS = "MAGNIFICIENT_MACARONS"


PARAMS = {
    Product.MAGNIFICIENT_MACARONS: {
        "make_edge": 2.5,
        "make_min_edge": 0.5,
        "make_probability": 0.5,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 10,
        "volume_bar": 50,
        "dec_edge_discount": 0.9,
        "step_size": 0.7,
        "sunlight_adjustment": 0.05,
        "sugar_adjustment": 0.03946426036817559,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.MAGNIFICIENT_MACARONS: 100}

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def macarons_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return (
            observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1,
            observation.askPrice + observation.importTariff + observation.transportFees,
        )

    def macarons_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        traderObject: dict,
    ) -> float:
        if timestamp == 0:
            traderObject["MAGNIFICIENT_MACARONS"] = {
                "curr_edge": self.params[Product.MAGNIFICIENT_MACARONS]["init_make_edge"],
                "volume_history": [],
                "optimized": False,
            }
            return self.params[Product.MAGNIFICIENT_MACARONS]["init_make_edge"]

        # Timestamp not 0
        traderObject["MAGNIFICIENT_MACARONS"]["volume_history"].append(abs(position))
        if len(traderObject["MAGNIFICIENT_MACARONS"]["volume_history"]) > self.params[Product.MAGNIFICIENT_MACARONS]["volume_avg_timestamp"]:
            traderObject["MAGNIFICIENT_MACARONS"]["volume_history"].pop(0)

        if len(traderObject["MAGNIFICIENT_MACARONS"]["volume_history"]) < self.params[Product.MAGNIFICIENT_MACARONS]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject["MAGNIFICIENT_MACARONS"]["optimized"]:
            volume_avg = np.mean(traderObject["MAGNIFICIENT_MACARONS"]["volume_history"])

            # Bump up edge if consistently getting lifted full size
            if volume_avg >= self.params[Product.MAGNIFICIENT_MACARONS]["volume_bar"]:
                traderObject["MAGNIFICIENT_MACARONS"]["volume_history"] = []  # clear volume history if edge changed
                traderObject["MAGNIFICIENT_MACARONS"]["curr_edge"] = curr_edge + self.params[Product.MAGNIFICIENT_MACARONS]["step_size"]
                return curr_edge + self.params[Product.MAGNIFICIENT_MACARONS]["step_size"]

            # Decrement edge if more cash with less edge, included discount
            elif self.params[Product.MAGNIFICIENT_MACARONS]["dec_edge_discount"] * self.params[Product.MAGNIFICIENT_MACARONS]["volume_bar"] * (curr_edge - self.params[Product.MAGNIFICIENT_MACARONS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.MAGNIFICIENT_MACARONS]["step_size"] > self.params[Product.MAGNIFICIENT_MACARONS]["min_edge"]:
                    traderObject["MAGNIFICIENT_MACARONS"]["volume_history"] = []  # clear volume history if edge changed
                    traderObject["MAGNIFICIENT_MACARONS"]["curr_edge"] = curr_edge - self.params[Product.MAGNIFICIENT_MACARONS]["step_size"]
                    traderObject["MAGNIFICIENT_MACARONS"]["optimized"] = True
                    return curr_edge - self.params[Product.MAGNIFICIENT_MACARONS]["step_size"]
                else:
                    traderObject["MAGNIFICIENT_MACARONS"]["curr_edge"] = self.params[Product.MAGNIFICIENT_MACARONS]["min_edge"]
                    return self.params[Product.MAGNIFICIENT_MACARONS]["min_edge"]

        traderObject["MAGNIFICIENT_MACARONS"]["curr_edge"] = curr_edge
        return curr_edge

    def macarons_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICIENT_MACARONS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = implied_ask + adap_edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[Product.MAGNIFICIENT_MACARONS]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)  # max amount to buy
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICIENT_MACARONS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)  # max amount to sell
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICIENT_MACARONS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def macarons_arb_clear(self, position: int) -> int:
        conversions = -position
        return conversions

    def macarons_arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICIENT_MACARONS]

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        bid = implied_bid - edge
        ask = implied_ask + edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask >= implied_ask + self.params[Product.MAGNIFICIENT_MACARONS]['min_edge']:
            ask = aggressive_ask

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 40]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 25]

        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MAGNIFICIENT_MACARONS, round(bid), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MAGNIFICIENT_MACARONS, round(ask), -sell_quantity))  # Sell order

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.MAGNIFICIENT_MACARONS in self.params and Product.MAGNIFICIENT_MACARONS in state.order_depths:
            macarons_position = (
                state.position[Product.MAGNIFICIENT_MACARONS]
                if Product.MAGNIFICIENT_MACARONS in state.position
                else 0
            )

            conversions = self.macarons_arb_clear(macarons_position)

            adap_edge = self.macarons_adap_edge(
                state.timestamp,
                traderObject["MAGNIFICIENT_MACARONS"]["curr_edge"],
                macarons_position,
                traderObject,
            )

            macarons_position = 0

            macarons_take_orders, buy_order_volume, sell_order_volume = self.macarons_arb_take(
                state.order_depths[Product.MAGNIFICIENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICIENT_MACARONS],
                adap_edge,
                macarons_position,
            )

            macarons_make_orders, _, _ = self.macarons_arb_make(
                state.order_depths[Product.MAGNIFICIENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICIENT_MACARONS],
                macarons_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            result[Product.MAGNIFICIENT_MACARONS] = (
                macarons_take_orders + macarons_make_orders
            )

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData

import pandas as pd
import numpy as np
import itertools
from skopt import gp_minimize
from skopt.space import Real, Integer
import jsonpickle
from typing import List

# Function to load CSVs
def load_data(file_path):
    data = pd.read_csv(file_path, sep=';')  # Specify semicolon as the separator
    print(f"Columns in {file_path}: {data.columns}")  # Print columns to inspect their names
    return data

# Define a basic P/L calculation using order prices
def calculate_profit_loss(buy_orders, sell_orders):
    profit_loss = 0
    # Assuming you buy at buy_orders and sell at sell_orders
    for buy_order, sell_order in zip(buy_orders, sell_orders):
        profit_loss += (sell_order['price'] - buy_order['price']) * min(buy_order['quantity'], sell_order['quantity'])
    return profit_loss

# Product class
class Product:
    MAGNIFICIENT_MACARONS = "MAGNIFICIENT_MACARONS"

# Default parameters
PARAMS = {
    Product.MAGNIFICIENT_MACARONS: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
        "make_edge": 2,
        "make_min_edge": 1,
        "make_probability": 0.566,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 5,
        "volume_bar": 75,
        "dec_edge_discount": 0.8,
        "step_size": 0.5,
        "sunlight_adjustment": 0.1,  # Sunlight influence factor
        "sugar_adjustment": 0.05  # Sugar influence factor
    }
}

# Trader class that handles order management and P/L calculation
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.MAGNIFICIENT_MACARONS: 100}  # Only Magnificent Macarons are traded

    # Fair value calculation for Magnificent Macarons using Sunlight and Sugar
    def macaron_fair_value(self, order_depth) -> float:
        if len(order_depth['sell_orders']) != 0 and len(order_depth['buy_orders']) != 0:
            best_ask = min(order_depth['sell_orders'].keys())
            best_bid = max(order_depth['buy_orders'].keys())

            # Calculate fair value using the average of the best bid and ask
            fair_value = (best_ask + best_bid) / 2

            # Adjust fair value based on Sunlight and Sugar factors
            sunlight_factor = self.params[Product.MAGNIFICIENT_MACARONS].get("sunlight_adjustment", 0.1)
            sugar_factor = self.params[Product.MAGNIFICIENT_MACARONS].get("sugar_adjustment", 0.05)

            # Adjust fair value based on sunlight and sugar
            adjusted_fair_value = fair_value + sunlight_factor * (best_ask - best_bid) - sugar_factor
            return adjusted_fair_value
        return None

    def macaron_arb_take(self, order_depth, observation, adap_edge, position) -> (List[dict], int, int):
        orders = []
        position_limit = self.LIMIT[Product.MAGNIFICIENT_MACARONS]
        buy_order_volume = 0
        sell_order_volume = 0

        fair_value = self.macaron_fair_value(order_depth)

        implied_bid = observation['bidPrice'] - observation['exportTariff'] - observation['transportFees']
        implied_ask = observation['askPrice'] + observation['importTariff'] + observation['transportFees']

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        bid = implied_bid - adap_edge
        ask = implied_ask + adap_edge

        for price in sorted(order_depth['sell_orders'].keys()):
            if price > implied_bid - adap_edge:
                break
            quantity = min(abs(order_depth['sell_orders'][price]), buy_quantity)
            if quantity > 0:
                orders.append({"product": Product.MAGNIFICIENT_MACARONS, "price": round(price), "quantity": quantity})
                buy_order_volume += quantity

        for price in sorted(order_depth['buy_orders'].keys(), reverse=True):
            if price < implied_ask + adap_edge:
                break
            quantity = min(abs(order_depth['buy_orders'][price]), sell_quantity)
            if quantity > 0:
                orders.append({"product": Product.MAGNIFICIENT_MACARONS, "price": round(price), "quantity": -quantity})
                sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def macaron_arb_make(self, order_depth, observation, position, edge, buy_order_volume, sell_order_volume) -> (List[dict], int, int):
        orders = []
        position_limit = self.LIMIT[Product.MAGNIFICIENT_MACARONS]

        fair_value = self.macaron_fair_value(order_depth)

        implied_bid = observation['bidPrice'] - observation['exportTariff'] - observation['transportFees']
        implied_ask = observation['askPrice'] + observation['importTariff'] + observation['transportFees']

        bid = implied_bid - edge
        ask = implied_ask + edge

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append({"product": Product.MAGNIFICIENT_MACARONS, "price": round(bid), "quantity": buy_quantity})

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append({"product": Product.MAGNIFICIENT_MACARONS, "price": round(ask), "quantity": -sell_quantity})

        return orders, buy_order_volume, sell_order_volume

    def run(self, state):
        traderObject = {}
        if state['traderData'] != None and state['traderData'] != "":
            traderObject = jsonpickle.decode(state['traderData'])

        result = {}
        conversions = 0

        if Product.MAGNIFICIENT_MACARONS in self.params and Product.MAGNIFICIENT_MACARONS in state['order_depths']:
            if "MAGNIFICIENT_MACARONS" not in traderObject:
                traderObject["MAGNIFICIENT_MACARONS"] = {"curr_edge": self.params[Product.MAGNIFICIENT_MACARONS]["init_make_edge"], "volume_history": [], "optimized": False}
            
            macaron_position = state['position'][Product.MAGNIFICIENT_MACARONS] if Product.MAGNIFICIENT_MACARONS in state['position'] else 0

            adap_edge = self.macaron_arb_take(state['order_depths'][Product.MAGNIFICIENT_MACARONS], state['observations'][Product.MAGNIFICIENT_MACARONS], traderObject["MAGNIFICIENT_MACARONS"]["curr_edge"], macaron_position)

            macaron_take_orders, buy_order_volume, sell_order_volume = self.macaron_arb_take(
                state['order_depths'][Product.MAGNIFICIENT_MACARONS],
                state['observations'][Product.MAGNIFICIENT_MACARONS],
                adap_edge,
                macaron_position,
            )

            macaron_make_orders, buy_order_volume, sell_order_volume = self.macaron_arb_make(
                state['order_depths'][Product.MAGNIFICIENT_MACARONS],
                state['observations'][Product.MAGNIFICIENT_MACARONS],
                macaron_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            result[Product.MAGNIFICIENT_MACARONS] = (
                macaron_take_orders + macaron_make_orders
            )

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData

# Function to load the data and use the algorithm with different parameters
def objective(params, prices_data, trades_data):
    param_dict = {
        'make_edge': params[0],
        'make_min_edge': params[1],
        'make_probability': params[2],
        'init_make_edge': params[3],
        'min_edge': params[4],
        'volume_avg_timestamp': params[5],
        'volume_bar': params[6],
        'dec_edge_discount': params[7],
        'step_size': params[8],
        'sunlight_adjustment': params[9],
        'sugar_adjustment': params[10]
    }

    trader = Trader(params=param_dict)

    total_profit_loss = 0
    for day in range(len(prices_data)):
        price_data = prices_data[day]
        trade_data = trades_data[day]

        # Fair value calculation using correct column names (ask_price_1 and bid_price_1)
        fair_value = (min(price_data['ask_price_1']) + max(price_data['bid_price_1'])) / 2  # Fair value calculation

        make_edge = param_dict['make_edge']
        buy_price = fair_value - make_edge
        sell_price = fair_value + make_edge

        quantity = 10

        buy_orders = [{"price": buy_price, "quantity": quantity}]
        sell_orders = [{"price": sell_price, "quantity": quantity}]

        total_profit_loss += calculate_profit_loss(buy_orders, sell_orders)

    return -total_profit_loss  # Minimize negative P/L to maximize profit

# Bayesian optimization setup
def optimize_params(prices_data, trades_data):
    param_space = [
        Real(1, 2.5, name='make_edge'),
        Real(0.5, 1.5, name='make_min_edge'),
        Real(0.4, 0.8, name='make_probability'),
        Integer(1, 2, name='init_make_edge'),
        Real(0.5, 1, name='min_edge'),
        Integer(5, 10, name='volume_avg_timestamp'),
        Integer(50, 100, name='volume_bar'),
        Real(0.6, 0.9, name='dec_edge_discount'),
        Real(0.1, 0.7, name='step_size'),
        Real(0.05, 0.1, name='sunlight_adjustment'),
        Real(0.03, 0.05, name='sugar_adjustment')
    ]

    result = gp_minimize(lambda params: objective(params, prices_data, trades_data), 
                         dimensions=param_space, 
                         acq_func="LCB",  # Expected Improvement acquisition function
                         n_calls=200,  # Number of evaluations (iterations)
                         random_state=42)

    best_params = dict(zip([dim.name for dim in param_space], result.x))
    best_p_l = -result.fun

    return best_params, best_p_l

# Load CSVs exactly as requested
prices_data = [
    load_data("prices_round_4_day_1.csv"),
    load_data("prices_round_4_day_2.csv"),
    load_data("prices_round_4_day_3.csv")
]

trades_data = [
    load_data("trades_round_4_day_1.csv"),
    load_data("trades_round_4_day_2.csv"),
    load_data("trades_round_4_day_3.csv")
]


# Run optimization
best_params, best_p_l = optimize_params(prices_data, trades_data)

print("Best Parameters:", best_params)
print("Best Profit/Loss:", best_p_l)

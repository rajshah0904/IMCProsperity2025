from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import List, Dict
import jsonpickle
import numpy as np

class Trader:
    def __init__(self):
        # Only MAGNIFICENT_MACARONS
        self.product = "MAGNIFICENT_MACARONS"
        self.position_limit = 100

        # Fine‑tuned parameters for macarons
        self.params = {
            "make_edge": 2.0,
            "make_min_edge": 0.5,
            "make_probability": 0.566,
            "volume_avg_timestamp": 5,
            "volume_bar": 75,
            "dec_edge_discount": 0.8,
            "step_size": 0.5,
            "sugar_correlation": 0.38,
            "sunlight_correlation": -0.23,
            "sugar_window": 10,
            "sunlight_window": 250,
            "sugar_threshold": 3.0,
            "aggressive_offset": 1.6,
            "large_order_size": 40,
        }

    def run(self, state: TradingState):
        # Print state for debugging
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result: Dict[str, List[Order]] = {}
        conversions = 0

        # Decode or initialize peristent state
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}

        # Process each product in the state
        for product, depth in state.order_depths.items():
            orders: List[Order] = []

            if product == self.product:
                obs: ConversionObservation = state.observations.conversionObservations.get(product)
                pos = state.position.get(product, 0)
                if depth.buy_orders and depth.sell_orders and obs:
                    # Run custom macarons logic
                    orders, conv = self._run_macarons(depth, obs, pos, trader_data)
                    conversions += conv

            # Attach orders for this product
            result[product] = orders

        # Persist state for next call
        traderData = jsonpickle.encode(trader_data)
        return result, conversions, traderData

    def _run_macarons(self, depth: OrderDepth, obs: ConversionObservation, pos: int, td: dict):
        p = self.params
        d = td.setdefault(self.product, {
            "curr_edge": p["make_edge"],
            "volume_history": [],
            "optimized": False,
            "sugar_price_history": [],
            "sunlight_index_history": []
        })

        # 1) adapt edge
        edge = self._adaptive_edge(depth, obs, pos, d)

        # 2) take
        take_orders, bought, sold = self._arb_take(depth, obs, edge, pos)

        # 3) make
        make_orders, _, _ = self._arb_make(depth, obs, pos, edge, bought, sold)

        # count conversions as fills
        conversions = bought + sold
        return take_orders + make_orders, conversions

    def _adaptive_edge(self, od, obs, pos, d) -> float:
        p = self.params
        curr = d["curr_edge"]

        # volume history
        vh = d["volume_history"]; vh.append(abs(pos))
        if len(vh) > p["volume_avg_timestamp"]: vh.pop(0)

        # sugar/sunlight history
        sp = d["sugar_price_history"]; sp.append(obs.sugarPrice)
        if len(sp) > p["sugar_window"]: sp.pop(0)
        si = d["sunlight_index_history"]; si.append(obs.sunlightIndex)
        if len(si) > p["sunlight_window"]: si.pop(0)

        # bump edge if volume high
        if len(vh) >= p["volume_avg_timestamp"] and not d["optimized"]:
            avg = np.mean(vh)
            if avg >= p["volume_bar"]:
                vh.clear()
                d["curr_edge"] = curr + p["step_size"]
                return d["curr_edge"]
            if p["dec_edge_discount"] * p["volume_bar"] * (curr - p["step_size"]) > avg * curr:
                new = max(p["make_min_edge"], curr - p["step_size"])
                vh.clear(); d["curr_edge"] = new; d["optimized"] = True
                return new

        # sugar adjustment
        if len(sp) >= p["sugar_window"]:
            diff = abs(sp[-1] - sp[0])
            if diff > p["sugar_threshold"]:
                move = (sp[-1] - sp[0]) / (sp[0] or 1)
                adj = p["step_size"] * move * p["sugar_correlation"]
                cand = max(p["make_min_edge"], curr + adj)
                if abs(cand - curr) > 0.1:
                    d["curr_edge"] = cand
                    return cand

        # sunlight adjustment
        if len(si) >= p["sunlight_window"]:
            move = (si[-1] - si[0]) / (si[0] or 1)
            if abs(move) > 0.01:
                adj = p["step_size"] * move * p["sunlight_correlation"]
                cand = max(p["make_min_edge"], curr + adj)
                if abs(cand - curr) > 0.1:
                    d["curr_edge"] = cand
                    return cand

        return curr

    def _arb_take(self, od: OrderDepth, obs: ConversionObservation, edge: float, pos: int):
        p = self.params
        orders = []
        ib, ia = self.implied_bid_ask(obs)
        buy_q  = self.position_limit - pos
        sell_q = self.position_limit + pos

        sugar_diff = abs(obs.sugarPrice - 200)
        ef = 1.2 if sugar_diff > p["sugar_threshold"] else 1.0
        take_edge = edge * ef * p["make_probability"]

        # take from sell side
        for price in sorted(od.sell_orders):
            if price > ib - take_edge: break
            qty = min(-od.sell_orders[price], buy_q)
            if qty > 0:
                orders.append(Order(self.product, price, qty))
                buy_q -= qty
            if buy_q <= 0: break

        # take from buy side
        for price in sorted(od.buy_orders, reverse=True):
            if price < ia + take_edge: break
            qty = min(od.buy_orders[price], sell_q)
            if qty > 0:
                orders.append(Order(self.product, price, -qty))
                sell_q -= qty
            if sell_q <= 0: break

        bought = (self.position_limit - pos) - buy_q
        sold   = (self.position_limit + pos) - sell_q
        return orders, bought, sold

    def _arb_make(self, od: OrderDepth, obs: ConversionObservation, pos: int, edge: float, bv: int, sv: int):
        p = self.params
        orders = []
        ib, ia = self.implied_bid_ask(obs)

        bid = ib - edge
        ask = ia + edge

        # aggressive ask adjustment
        if abs(obs.sugarPrice - 200) > p["sugar_threshold"]:
            mid = (obs.askPrice + obs.bidPrice) / 2
            ag  = mid - p["aggressive_offset"]
            if ag >= ia + p["make_min_edge"]:
                ask = ag

        # depth adaptation
        large = p["large_order_size"]
        f_asks = [pr for pr, v in od.sell_orders.items() if abs(v) >= large]
        f_bids = [pr for pr, v in od.buy_orders.items()  if abs(v) >= large]

        if f_asks and ask > min(f_asks):
            ask = min(f_asks) - 1 if min(f_asks)-1 > ia else ia + edge
        if f_bids and bid < max(f_bids):
            bid = max(f_bids) + 1 if max(f_bids)+1 < ib else ib - edge

        # place make orders
        bq = self.position_limit - (pos + bv)
        if bq > 0:
            orders.append(Order(self.product, int(bid),  bq))
        sq = self.position_limit + (pos - sv)
        if sq > 0:
            orders.append(Order(self.product, int(ask), -sq))

        return orders, bv, sv

    def implied_bid_ask(self, obs: ConversionObservation):
        ib = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        ia = obs.askPrice + obs.importTariff + obs.transportFees
        return ib, ia

# Submission identifier: 59f81e67-f6c6-4254-b61e-39661eac6141

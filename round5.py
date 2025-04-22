import numpy as np
import math
import jsonpickle
from collections import deque, defaultdict
from typing import Dict, List, Tuple
from datamodel import Order, OrderDepth, TradingState, ConversionObservation

class Trader:
    def __init__(self):
        # ========= Sub-accounts for each strategy ==========
        self.resin_positions: Dict[str, int] = defaultdict(int)
        self.pattern_positions: Dict[str, int] = defaultdict(int)
        self.volcano_positions: Dict[str, int] = defaultdict(int)
        self.macaron_positions: Dict[str, int] = defaultdict(int)
        self.basket_positions: Dict[str, int] = defaultdict(int)

        # ========= Resin Market Maker Parameters ==========
        self.alloc_resin = 0.20
        self.base_spread = 4
        self.market_position_limit = int(20 * self.alloc_resin)
        self.base_position_size = int(10 * self.alloc_resin)
        self.max_spread = 8
        self.min_spread = 2
        self.volatility_window = 20
        self.resin_price_history: List[float] = []

        # ========= Pattern Recognition Parameters ==========
        self.alloc_pattern = 0.20
        self.iteration_pattern = 0
        self.pattern_products = ["SQUID_INK", "KELP"]
        self.pattern_price_history = {p: deque(maxlen=100) for p in self.pattern_products}
        self.recent_highs = {p: deque(maxlen=10) for p in self.pattern_products}
        self.recent_lows = {p: deque(maxlen=10) for p in self.pattern_products}
        self.rsi_period = 14
        self.rsi_values = {p: deque(maxlen=20) for p in self.pattern_products}
        self.volatility_window_pattern = 20
        self.volatility_values = {p: deque(maxlen=50) for p in self.pattern_products}
        self.position_limit_pattern = int(15 * self.alloc_pattern)
        self.base_trade_size = int(3 * self.alloc_pattern)
        self.max_trade_size = int(7 * self.alloc_pattern)
        self.trade_duration = {p: 0 for p in self.pattern_products}
        self.market_state = {p: "neutral" for p in self.pattern_products}
        self.pattern_detected = {p: None for p in self.pattern_products}
        self.historical_pl = 0
        self.trade_started_at = {p: 0 for p in self.pattern_products}

        # ========= Volcano Options Parameters ==========
        self.volcano_underlying = "VOLCANIC_ROCK"
        self.volcano_options = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        self.volcano_position_limits = {self.volcano_underlying: 300}
        for opt in self.volcano_options:
            self.volcano_position_limits[opt] = 100
        self.volcano_price_history = {self.volcano_underlying: deque(maxlen=100)}
        for opt in self.volcano_options:
            self.volcano_price_history[opt] = deque(maxlen=100)
        self.volcano_confidence_threshold = 0.7
        self.volcano_min_profit = 10
        self.volcano_safety_margin = 0.05
        self.volcano_base_quantity = 25
        self.volcano_max_drawdown = 0.1
        self.volcano_initial_capital = 0
        self.volcano_current_capital = 0
        self.volcano_max_capital = 0
        self.volcano_last_trade_prices: Dict[str, float] = {}

        # ========= MAGNIFICENT MACARONS Parameters ==========
        self.macaron_product = "MAGNIFICENT_MACARONS"
        self.macaron_position_limit = 75
        self.macaron_conversion_limit = 10
        self.macaron_params = {
            "make_edge": 2,
            "make_min_edge": 0.5,
            "make_probability": 0.6,
            "init_make_edge": 1.5,
            "min_edge": 0.3,
            "volume_avg_window": 10,
            "volume_bar": 40,
            "edge_adjust_rate": 0.15,
            "csi_threshold": 180,
            "csi_weight": 0.15,
            "sugar_weight": 0.2,
            "sunlight_weight": -0.1,
            "position_scale_factor": 0.8,
            "conversion_threshold": 0.6
        }
        self.macaron_state = {
            "curr_edge": self.macaron_params["init_make_edge"],
            "volume_history": [],
            "edge_history": [],
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

        # ========= PICNIC BASKET Parameters ==========
        self.alloc_basket = 0.20
        self.basket_products = {
            "PICNIC_BASKET1": {
                "components": {
                    "CROISSANTS": 6,
                    "JAMS": 3,
                    "DJEMBES": 1
                },
                "position_limit": int(70 * self.alloc_basket)
            },
            "PICNIC_BASKET2": {
                "components": {
                    "CROISSANTS": 4,
                    "DJEMBES": 2
                },
                "position_limit": int(70 * self.alloc_basket)
            }
        }
        self.basket_position_limits = {
            "CROISSANTS": int(300 * self.alloc_basket),
            "JAMS": int(300 * self.alloc_basket),
            "DJEMBES": int(100 * self.alloc_basket)
        }
        self.basket_min_profit = 10
        self.basket_price_history = {
            product: deque(maxlen=100)
            for product in ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]
        }
        self.basket_coefficients = {
            "CROISSANTS": {
                'ar_coef': [-0.13778924473364898, -0.017001430663861707,
                            -0.007126079130103736, 0.003956475454297141,
                            0.0005939005629995958],
                'ma_coef': [0.0] * 5,
                'intercept': -0.13778924473364898,
                'sigma2': 1.0
            },
            "JAMS": {
                'ar_coef': [-0.07681975802925964, 0.00855102932364836,
                            0.004110743183574504, -0.001057105898536105,
                            -0.0070153095407396475],
                'ma_coef': [0.0] * 5,
                'intercept': -0.07681975802925964,
                'sigma2': 1.0
            },
            "DJEMBES": {
                'ar_coef': [0.014983408336735213, 0.009723454198626398,
                            -0.0072454381893885264, 0.0010821369386140355,
                            -0.002663542650568304],
                'ma_coef': [0.0] * 5,
                'intercept': 0.014983408336735213,
                'sigma2': 1.0
            }
        }
        self.basket_std_threshold = 1.5

    def create_sub_state(self, global_state: TradingState, local_positions: Dict[str, int]) -> TradingState:
        return TradingState(
            traderData=global_state.traderData,
            timestamp=global_state.timestamp,
            listings=global_state.listings,
            order_depths=global_state.order_depths,
            own_trades=global_state.own_trades,
            market_trades=global_state.market_trades,
            position=local_positions,
            observations=global_state.observations
        )

    def combine_orders_no_netting(self, orders_list: List[Dict[str, List[Order]]]) -> Dict[str, List[Order]]:
        combined: Dict[str, List[Order]] = {}
        for strat_orders in orders_list:
            for product, orders in strat_orders.items():
                if product != self.volcano_underlying:
                    combined.setdefault(product, []).extend(orders)
        return combined

    # ========= RESIN MARKET MAKER METHODS =========
    def resin_calculate_volatility(self) -> float:
        if len(self.resin_price_history) < 2:
            return 0
        rets = np.diff(self.resin_price_history)
        return np.std(rets) if len(rets) > 0 else 0

    def calculate_dynamic_spread(self) -> int:
        vol = self.resin_calculate_volatility()
        spread = self.base_spread + int(vol * 10)
        return max(self.min_spread, min(self.max_spread, spread))

    def calculate_fair_value(self, od: OrderDepth) -> int:
        if not od.buy_orders or not od.sell_orders:
            return 0
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = od.sell_orders[best_ask]
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return (best_bid + best_ask) // 2
        return int(round((best_bid * ask_vol + best_ask * bid_vol) / total_vol))

    def generate_resin_orders(self, od: OrderDepth) -> List[Order]:
        orders: List[Order] = []
        if not od.buy_orders or not od.sell_orders:
            return orders
        fair = self.calculate_fair_value(od)
        spread = self.calculate_dynamic_spread()
        self.resin_price_history.append(fair)
        if len(self.resin_price_history) > self.volatility_window:
            self.resin_price_history.pop(0)
        buy_price = fair - spread//2
        sell_price = fair + spread//2
        if buy_price > max(od.buy_orders.keys()):
            orders.append(Order("RAINFOREST_RESIN", buy_price, self.base_position_size))
        if sell_price < min(od.sell_orders.keys()):
            orders.append(Order("RAINFOREST_RESIN", sell_price, -self.base_position_size))
        return orders

    def run_resin(self, sub_state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        res = {}
        if "RAINFOREST_RESIN" in sub_state.order_depths:
            res["RAINFOREST_RESIN"] = self.generate_resin_orders(sub_state.order_depths["RAINFOREST_RESIN"])
        return res, 0, ""

    # ========= PATTERN RECOGNITION METHODS =========
    def calculate_rsi(self, prices: deque, period=14) -> float:
        if len(prices) < period+1:
            return 50
        arr = np.array(list(prices)[-period-1:])
        deltas = np.diff(arr)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100
        rs = avg_gain/avg_loss
        return 100 - (100/(1+rs))

    def detect_patterns(self, product: str, price: float):
        prices = self.pattern_price_history[product]
        if len(prices) < 5:
            return None
        pattern = None
        recent = list(prices)[-5:]
        if recent[-2] > recent[-3] and recent[-2] > recent[-1]:
            self.recent_highs[product].append(recent[-2])
        if recent[-2] < recent[-3] and recent[-2] < recent[-1]:
            self.recent_lows[product].append(recent[-2])
        if len(self.recent_lows[product])>=2:
            lows = list(self.recent_lows[product])
            if abs(lows[-1]-lows[-2])/lows[-1]<0.03:
                pattern="double_bottom"
        if len(self.recent_highs[product])>=2:
            highs=list(self.recent_highs[product])
            if abs(highs[-1]-highs[-2])/highs[-1]<0.03:
                pattern="double_top"
        rsi=self.calculate_rsi(prices)
        self.rsi_values[product].append(rsi)
        if rsi<30:
            pattern="oversold"
        if rsi>70:
            pattern="overbought"
        return pattern

    def run_pattern(self, sub_state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        res, debug = {}, ""
        for p in self.pattern_products:
            if p not in sub_state.order_depths: continue
            od=sub_state.order_depths[p]
            if not od.buy_orders or not od.sell_orders: continue
            mid=(max(od.buy_orders)+min(od.sell_orders))/2
            self.pattern_price_history[p].append(mid)
            pattern=self.detect_patterns(p, mid)
            orders=[]
            pos=sub_state.position.get(p,0)
            if pos==0 and pattern in ("double_bottom","oversold"):
                orders.append(Order(p, min(od.sell_orders), 1))
            if pos>0 and pattern in ("double_top","overbought"):
                orders.append(Order(p, max(od.buy_orders), -pos))
            if orders:
                res[p]=orders
                debug=f"PATTERN {p}:{pattern}"
        return res, 0, debug

    # ========= VOLCANO OPTIONS METHODS =========
    def update_volcano_price_history(self, prod: str, price: float):
        self.volcano_price_history[prod].append(price)

    def calculate_features_volcano(self, prices: np.ndarray) -> Tuple[float,float,float,float,float]:
        if len(prices)<5: return 0,0,0,0,0
        rets=np.diff(prices)/prices[:-1]
        vol=np.std(rets[-5:])
        mom=(prices[-1]-prices[-5])/prices[-5]
        rev=(prices[-1]-np.mean(prices[-10:]))/np.mean(prices[-10:]) if len(prices)>=10 else 0
        trend=(prices[-1]-prices[-3])/prices[-3] if len(prices)>=3 else 0
        acc=1-np.mean(np.abs(np.diff(prices[-5:])/prices[-5:-1]))
        return vol,mom,rev,trend,acc

    def predict_future_price_volcano(self) -> Tuple[float,float]:
        hist=self.volcano_price_history[self.volcano_underlying]
        if len(hist)<5: return None,0
        arr=np.array(hist)
        vol,mom,rev,trend,acc=self.calculate_features_volcano(arr)
        base=arr[-1]
        pred=base*(1+0.3*mom*(1-min(1,abs(rev)*2))-0.2*rev-0.1*trend)
        conf=(1-min(1,vol*10))*(1-min(1,abs(rev)))*acc
        return pred,conf

    def find_trading_opportunity_volcano(self,spot,opt_price,strike,pred,conf)->Tuple[str,int]:
        if conf<self.volcano_confidence_threshold: return None,0
        intrinsic=pred-strike
        cur_intr=max(0,spot-strike)
        diff=opt_price-(intrinsic-self.volcano_safety_margin*spot)
        if abs(diff)<self.volcano_min_profit: return None,0
        qty=int(self.volcano_base_quantity*conf*min(1,abs(diff)/spot)*(1-min(1,np.std(np.diff(np.array(self.volcano_price_history[self.volcano_underlying]))))*5))
        qty=max(1,qty)
        return ("SELL_OPTION",qty) if diff>0 else ("BUY_OPTION",qty)

    def run_volcano(self, sub: TradingState) -> Tuple[Dict[str,List[Order]],int,str]:
        res={}
        if self.volcano_initial_capital==0:
            try: self.volcano_initial_capital=float(sub.traderData)
            except: self.volcano_initial_capital=0
        self.volcano_current_capital=self.volcano_initial_capital
        self.volcano_max_capital=max(self.volcano_max_capital,self.volcano_current_capital)
        if self.volcano_max_capital>0 and (self.volcano_max_capital-self.volcano_current_capital)/self.volcano_max_capital>self.volcano_max_drawdown:
            return res,0,""
        if self.volcano_underlying in sub.order_depths:
            od=sub.order_depths[self.volcano_underlying]
            if od.buy_orders and od.sell_orders:
                mid=(max(od.buy_orders)+min(od.sell_orders))/2
                self.update_volcano_price_history(self.volcano_underlying,mid)
                pred,conf=self.predict_future_price_volcano()
                if pred is not None:
                    for opt,strike in self.volcano_options.items():
                        od_opt=sub.order_depths.get(opt)
                        if not od_opt: continue
                        if not od_opt.buy_orders or not od_opt.sell_orders: continue
                        mid_opt=(max(od_opt.buy_orders)+min(od_opt.sell_orders))/2
                        self.update_volcano_price_history(opt,mid_opt)
                        dir,qty=self.find_trading_opportunity_volcano(mid,mid_opt,strike,pred,conf)
                        if dir and qty>0:
                            pos_spot=sub.position.get(self.volcano_underlying,0)
                            pos_opt=sub.position.get(opt,0)
                            if dir=="SELL_OPTION":
                                qty=min(qty,pos_opt+self.volcano_position_limits[opt],self.volcano_position_limits[self.volcano_underlying]-pos_spot)
                                if qty>0: res[opt]=[Order(opt,max(od_opt.buy_orders),-qty)]
                            else:
                                qty=min(qty,self.volcano_position_limits[opt]-pos_opt,self.volcano_position_limits[self.volcano_underlying]+pos_spot)
                                if qty>0: res[opt]=[Order(opt,min(od_opt.sell_orders),qty)]
        return res,0,str(self.volcano_current_capital)

    # ========= MACARON MARKET MAKER METHODS =========
    def update_environmental_data(self, obs: ConversionObservation, pos: int):
        st=self.macaron_state
        st['sugar_history'].append(obs.sugarPrice)
        st['sunlight_history'].append(obs.sunlightIndex)
        st['position_history'].append(pos)
        dist=self.macaron_params['csi_threshold']-obs.sunlightIndex
        st['csi_factor']=min(1,st.get('csi_factor',0)+0.1) if dist>0 else max(0,st.get('csi_factor',0)-0.1)
        for k in ('sugar_history','sunlight_history','position_history'):
            if len(st[k])>self.macaron_params['volume_avg_window']: st[k].pop(0)

    def smooth_implied_bid_ask(self, obs: ConversionObservation) -> Tuple[float,float]:
        st=self.macaron_state
        bid=obs.bidPrice-obs.exportTariff-obs.transportFees
        ask=obs.askPrice+obs.importTariff+obs.transportFees
        cf=st['csi_factor']
        bid-=cf*self.macaron_params['csi_weight']
        ask+=cf*self.macaron_params['csi_weight']
        if len(st['sugar_history'])>3:
            arr=np.array(st['sugar_history'])
            zs=(arr[-1]-arr.mean())/max(1,arr.std())
            bid+=zs*self.macaron_params['sugar_weight']
            ask+=zs*self.macaron_params['sugar_weight']
        return bid,ask

    def smooth_adaptive_edge(self, ts:int, obs:ConversionObservation, pos:int)->float:
        st=self.macaron_state; p=self.macaron_params
        ce=st['curr_edge']
        if st['last_timestamp']==0: st['curr_edge']=p['init_make_edge']; return p['init_make_edge']
        st['volume_history'].append(abs(pos)); st['edge_history'].append(ce)
        if len(st['volume_history'])>p['volume_avg_window']: st['volume_history'].pop(0); st['edge_history'].pop(0)
        if len(st['volume_history'])<3: return ce
        vol_avg=np.mean(st['volume_history'])/p['volume_bar']
        target=ce
        if vol_avg>0.8: target*=1+(vol_avg-0.7)*p['edge_adjust_rate']
        elif vol_avg<0.3 and ce>p['make_min_edge']: target*=1-(0.3-vol_avg)*p['edge_adjust_rate']
        if len(st['sunlight_history'])>3:
            sf=np.std(st['sunlight_history'])/20
            target*=1+sf*0.1
        target=max(p['make_min_edge'],min(p['make_edge']*2,target))
        ne=ce*0.85+target*0.15; st['curr_edge']=ne
        return ne

    def improved_arb_take(self, od:OrderDepth, obs:ConversionObservation, edge:float, pos:int) -> Tuple[List[Order],int,int]:
        orders=[]; buy_vol=sell_vol=0; bid_i,ask_i=self.smooth_implied_bid_ask(obs)
        pmult=min(1,abs(pos)/self.macaron_position_limit*self.macaron_params['position_scale_factor'])
        max_buy=int((self.macaron_position_limit-pos)*(1-pmult))
        max_sell=int((self.macaron_position_limit+pos)*(1-pmult))
        for price in sorted(od.sell_orders):
            if max_buy<=0: break
            m=(bid_i-price)/bid_i
            if m<=0: continue
            q=min(od.sell_orders[price],int(max_buy*min(1,m*10)))
            if q>0: orders.append(Order(self.macaron_product,round(price),q)); buy_vol+=q; max_buy-=q
        for price in sorted(od.buy_orders, reverse=True):
            if max_sell<=0: break
            m=(price-ask_i)/ask_i
            if m<=0: continue
            q=min(od.buy_orders[price],int(max_sell*min(1,m*10)))
            if q>0: orders.append(Order(self.macaron_product,round(price),-q)); sell_vol+=q; max_sell-=q
        return orders,buy_vol,sell_vol

    def improved_arb_make(self, od:OrderDepth, obs:ConversionObservation, pos:int, edge:float, bv:int, sv:int) -> Tuple[List[Order],int,int]:
        orders=[]; bid_i,ask_i=self.smooth_implied_bid_ask(obs)
        pr=abs(pos)/self.macaron_position_limit; ae=edge*(1+pr*0.5)
        bid,ask=bid_i-ae,ask_i+ae
        if pos>0: s=pos/(self.macaron_position_limit*0.5)*ae*0.3; bid-=s; ask-=s*0.3
        if pos<0: s=abs(pos)/(self.macaron_position_limit*0.5)*ae*0.3; ask+=s; bid+=s*0.3
        large_asks=[p for p,q in od.sell_orders.items() if q>=30]
        large_bids=[p for p,q in od.buy_orders.items() if q>=20]
        if large_asks and min(large_asks)<ask: ask=int(ask*(1-0.7)+(min(large_asks)-1)*0.7)
        if large_bids and max(large_bids)>bid: bid=int(bid*(1-0.7)+(max(large_bids)+1)*0.7)
        cap_buy=self.macaron_position_limit-(pos+bv)
        cap_sell=self.macaron_position_limit+(pos-sv)
        size_scale=max(0.2,1-pr)
        qb=int(cap_buy*size_scale); qs=int(cap_sell*size_scale)
        if qb>0 and bid>0: orders.append(Order(self.macaron_product,round(bid),qb))
        if qs>0 and ask>0: orders.append(Order(self.macaron_product,round(ask),-qs))
        return orders,bv,sv

    def smooth_position_conversion(self, pos:int, ts:int, obs:ConversionObservation) -> int:
        if pos==0 or not obs: return 0
        r=abs(pos)/self.macaron_position_limit
        if r<self.macaron_params['conversion_threshold']: return 0
        last=self.macaron_state['last_conversion']
        if last>0 and ts-last<15: return 0
        if last>0:
            prob=min(1,(ts-last-15)/30)
            if np.random.random()>prob: return 0
        amt=int(abs(pos)*(min(0.8,(r-self.macaron_params['conversion_threshold'])*2)))
        amt=min(amt,self.macaron_conversion_limit)
        conv=-amt if pos>0 else amt
        if conv!=0: self.macaron_state['last_conversion']=ts
        return conv

    def run_macarons(self, state: TradingState) -> Tuple[Dict[str,List[Order]],int,str]:
        res = {}
        obs = None
        if state.observations and hasattr(state.observations,'conversionObservations'):
            obs = state.observations.conversionObservations.get(self.macaron_product)
        pos = state.position.get(self.macaron_product,0)
        conv = self.smooth_position_conversion(pos,state.timestamp,obs)
        adj_pos = pos-conv
        orders = []
        if obs:
            self.update_environmental_data(obs,pos)
            edge = self.smooth_adaptive_edge(state.timestamp,obs,adj_pos)
            take,bv,sv = self.improved_arb_take(state.order_depths[self.macaron_product],obs,edge,adj_pos)
            make,_,_ = self.improved_arb_make(state.order_depths[self.macaron_product],obs,adj_pos,edge,bv,sv)
            orders = take+make
        res = {self.macaron_product:orders}
        return res,conv,jsonpickle.encode(self.macaron_state)

    # ========= PICNIC BASKET METHODS =========
    def update_basket_price_history(self, product: str, price: float):
        self.basket_price_history[product].append(price)

    def predict_price(self, product: str) -> float:
        if len(self.basket_price_history[product]) < 5:
            return None
        prices = list(self.basket_price_history[product])
        coeffs = self.basket_coefficients[product]
        prediction = coeffs['intercept']
        for i in range(5):
            if i < len(prices):
                prediction += coeffs['ar_coef'][i] * prices[-(i+1)]
        return prediction

    def find_prediction_opportunity(self, product: str, current_price: float) -> (str, float):
        prediction = self.predict_price(product)
        if prediction is None:
            return None, 0
        recent_prices = list(self.basket_price_history[product])[-20:]
        if len(recent_prices) < 2:
            return None, 0
        std = np.std(recent_prices)
        if std == 0:
            return None, 0
        z_score = (current_price - prediction) / std
        if abs(z_score) > self.basket_std_threshold:
            return ("SELL", abs(z_score)) if z_score > 0 else ("BUY", abs(z_score))
        return None, 0

    def calculate_basket_value(self, basket_name: str, order_depths: Dict[str, OrderDepth]) -> float:
        basket = self.basket_products[basket_name]
        total_value = 0.0
        for component, quantity in basket["components"].items():
            od = order_depths.get(component, OrderDepth())
            if not od.buy_orders or not od.sell_orders:
                return None
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.update_basket_price_history(component, mid_price)
            total_value += mid_price * quantity
        return total_value

    def find_arbitrage_opportunity(self, basket_name: str, basket_price: float, component_value: float) -> (str, float):
        if component_value is None:
            return None, 0
        diff = basket_price - component_value
        if abs(diff) < self.basket_min_profit:
            return None, 0
        return ("SELL", diff) if diff > 0 else ("BUY", abs(diff))

    def execute_arbitrage(self, sub_state: TradingState, basket_name: str, direction: str, profit: float) -> Dict[str, List[Order]]:
        result = {}
        basket = self.basket_products[basket_name]
        position_here = sub_state.position.get(basket_name, 0)
        basket_depth = sub_state.order_depths.get(basket_name, OrderDepth())
        if not basket_depth.buy_orders or not basket_depth.sell_orders:
            return result

        best_bid = max(basket_depth.buy_orders.keys())
        best_ask = min(basket_depth.sell_orders.keys())
        max_qty = min(
            basket["position_limit"] - position_here if direction == "BUY" else basket["position_limit"] + position_here,
            basket_depth.buy_orders[best_bid] if direction == "SELL" else basket_depth.sell_orders[best_ask]
        )
        if max_qty <= 0:
            return result

        if direction == "BUY":
            result[basket_name] = [Order(basket_name, best_ask, max_qty)]
        else:
            result[basket_name] = [Order(basket_name, best_bid, -max_qty)]

        for component, needed in basket["components"].items():
            comp_pos = sub_state.position.get(component, 0)
            comp_depth = sub_state.order_depths.get(component, OrderDepth())
            if not comp_depth.buy_orders or not comp_depth.sell_orders:
                continue
            best_comp_bid = max(comp_depth.buy_orders.keys())
            best_comp_ask = min(comp_depth.sell_orders.keys())
            comp_limit = self.basket_position_limits[component]
            comp_max_qty = min(
                comp_limit - comp_pos if direction == "SELL" else comp_limit + comp_pos,
                comp_depth.buy_orders[best_comp_bid] if direction == "SELL" else comp_depth.sell_orders[best_comp_ask]
            )
            total_comp_qty = min(max_qty * needed, comp_max_qty)
            if total_comp_qty <= 0:
                continue

            if direction == "SELL":
                result[component] = [Order(component, best_comp_bid, total_comp_qty)]
            else:
                result[component] = [Order(component, best_comp_ask, -total_comp_qty)]
        return result

    def execute_prediction_trade(self, sub_state: TradingState, product: str, direction: str, confidence: float) -> Dict[str, List[Order]]:
        result = {}
        pos = sub_state.position.get(product, 0)
        od = sub_state.order_depths.get(product, OrderDepth())
        if not od.buy_orders or not od.sell_orders:
            return result

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        plimit = self.basket_position_limits[product]
        max_qty = min(
            plimit - pos if direction == "BUY" else plimit + pos,
            od.sell_orders[best_ask] if direction == "BUY" else od.buy_orders[best_bid]
        )
        quantity = int(max_qty * min(1.0, confidence / self.basket_std_threshold))
        if quantity <= 0:
            return result

        if direction == "BUY":
            result[product] = [Order(product, best_ask, quantity)]
        else:
            result[product] = [Order(product, best_bid, -quantity)]
        return result

    def run_basket(self, sub_state: TradingState):
        result = {}
        for product in ["CROISSANTS", "JAMS", "DJEMBES"]:
            od = sub_state.order_depths.get(product, OrderDepth())
            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders.keys())
                best_ask = min(od.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                self.update_basket_price_history(product, mid_price)
        for product in ["CROISSANTS", "JAMS", "DJEMBES"]:
            od = sub_state.order_depths.get(product, OrderDepth())
            if not od.buy_orders or not od.sell_orders:
                continue
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            direction, confidence = self.find_prediction_opportunity(product, mid_price)
            if direction is not None:
                orders = self.execute_prediction_trade(sub_state, product, direction, confidence)
                result.update(orders)
        for basket_name in self.basket_products:
            od_b = sub_state.order_depths.get(basket_name, OrderDepth())
            if not od_b.buy_orders or not od_b.sell_orders:
                continue
            best_bid = max(od_b.buy_orders.keys())
            best_ask = min(od_b.sell_orders.keys())
            basket_mid = (best_bid + best_ask) / 2
            c_val = self.calculate_basket_value(basket_name, sub_state.order_depths)
            if c_val is None:
                continue
            direction, profit = self.find_arbitrage_opportunity(basket_name, basket_mid, c_val)
            if direction is not None:
                arb_orders = self.execute_arbitrage(sub_state, basket_name, direction, profit)
                result.update(arb_orders)
        return result, 0, ""

    # ========= MAIN RUN =========
    def run(self, global_state: TradingState) -> Tuple[Dict[str,List[Order]],int,str]:
        sr = self.create_sub_state(global_state, self.resin_positions)
        or_,cr,dr = self.run_resin(sr)
        
        sp = self.create_sub_state(global_state, self.pattern_positions)
        op,cp,dp = self.run_pattern(sp)
        
        sv = self.create_sub_state(global_state, self.volcano_positions)
        ov,cv,dv = self.run_volcano(sv)
        
        sm = self.create_sub_state(global_state, self.macaron_positions)
        om,cm,dm = self.run_macarons(sm)
        
        sb = self.create_sub_state(global_state, self.basket_positions)
        ob,cb,db = self.run_basket(sb)
        
        combined = self.combine_orders_no_netting([or_,op,ov,om,ob])
        conv = cr+cp+cv+cm+cb
        data = "|".join([dr,dp,dv,dm,db])
        return combined,conv,data

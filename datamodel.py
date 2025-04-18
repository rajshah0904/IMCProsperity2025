from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Order:
    symbol: str
    price: int
    quantity: int

@dataclass
class Listing:
    symbol: str
    product: str
    denomination: str

class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class TradingState:
    def __init__(self, timestamp: int, position: Dict[str, int], order_depths: Dict[str, OrderDepth],
                 own_trades: Dict[str, List] = None, market_trades: Dict[str, List] = None,
                 listings: Dict[str, Listing] = None, observations: Dict[str, str] = None,
                 traderData: str = None):
        self.timestamp = timestamp
        self.position = position
        self.order_depths = order_depths
        self.own_trades = own_trades if own_trades is not None else {}
        self.market_trades = market_trades if market_trades is not None else {}
        self.listings = listings if listings is not None else {}
        self.observations = observations if observations is not None else {}
        self.traderData = traderData 
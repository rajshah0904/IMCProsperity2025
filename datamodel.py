from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Order:
    symbol: str
    price: int
    quantity: int

class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class TradingState:
    def __init__(self, timestamp: int, position: Dict[str, int], order_depths: Dict[str, OrderDepth]):
        self.timestamp = timestamp
        self.position = position
        self.order_depths = order_depths 
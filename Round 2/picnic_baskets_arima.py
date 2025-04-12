from collections import deque
import numpy as np
import pandas as pd
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class Trader:
    def __init__(self):
        # Product definitions with correct names
        self.products = {
            "PICNIC_BASKET1": {
                "components": {
                    "CROISSANTS": 6,
                    "JAMS": 3,
                    "DJEMBES": 1
                },
                "position_limit": 70
            },
            "PICNIC_BASKET2": {
                "components": {
                    "CROISSANTS": 4,
                    "DJEMBES": 2
                },
                "position_limit": 70
            }
        }
        
        # Position limits for individual products
        self.position_limits = {
            "CROISSANTS": 300,
            "JAMS": 300,
            "DJEMBES": 100
        }
        
        # Minimum profit threshold for arbitrage
        self.min_profit = 10
        
        # Price history for each product
        self.price_history = {
            product: deque(maxlen=100) for product in 
            ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]
        }
        
        # ARIMA models for each component
        self.arima_models = {
            "CROISSANTS": None,
            "JAMS": None,
            "DJEMBES": None
        }
        
        # Standard deviation thresholds for trading
        self.std_threshold = 1.5
        
        # Track last predictions
        self.last_predictions = {
            "CROISSANTS": None,
            "JAMS": None,
            "DJEMBES": None
        }
        
        # Store coefficients for each product
        self.coefficients = {
            "CROISSANTS": None,
            "JAMS": None,
            "DJEMBES": None
        }

    def update_price_history(self, product: str, price: float):
        """Update price history for a product"""
        self.price_history[product].append(price)

    def train_arima_model(self, product: str):
        """Train ARIMA model for a product and return coefficients"""
        if len(self.price_history[product]) < 30:  # Need minimum data points
            return None
            
        try:
            # Convert price history to numpy array
            prices = np.array(list(self.price_history[product]))
            
            # Fit ARIMA model
            model = ARIMA(prices, order=(5,1,0))
            model_fit = model.fit()
            
            # Get coefficients
            coefficients = {
                'ar_coef': model_fit.arparams.tolist(),
                'ma_coef': [0.0] * 5,  # No MA terms in our model
                'intercept': float(model_fit.params[0]),
                'sigma2': float(model_fit.scale)  # Using scale instead of sigma2
            }
            
            return coefficients
        except Exception as e:
            print(f"Error training ARIMA for {product}: {str(e)}")
            return None

    def save_coefficients(self):
        """Save coefficients to a CSV file"""
        # Create a DataFrame with the coefficients
        data = []
        for product in ["CROISSANTS", "JAMS", "DJEMBES"]:
            if self.coefficients[product] is not None:
                row = {
                    'product': product,
                    'ar_coef_1': self.coefficients[product]['ar_coef'][0],
                    'ar_coef_2': self.coefficients[product]['ar_coef'][1],
                    'ar_coef_3': self.coefficients[product]['ar_coef'][2],
                    'ar_coef_4': self.coefficients[product]['ar_coef'][3],
                    'ar_coef_5': self.coefficients[product]['ar_coef'][4],
                    'ma_coef_1': self.coefficients[product]['ma_coef'][0],
                    'ma_coef_2': self.coefficients[product]['ma_coef'][1],
                    'ma_coef_3': self.coefficients[product]['ma_coef'][2],
                    'ma_coef_4': self.coefficients[product]['ma_coef'][3],
                    'ma_coef_5': self.coefficients[product]['ma_coef'][4],
                    'intercept': self.coefficients[product]['intercept'],
                    'sigma2': self.coefficients[product]['sigma2']
                }
                data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv('arima_coefficients.csv', index=False)
            print("Coefficients saved to arima_coefficients.csv")

    def predict_price(self, product: str):
        """Predict next price for a product"""
        if self.arima_models[product] is None:
            self.arima_models[product] = self.train_arima_model(product)
            
        if self.arima_models[product] is None:
            return None
            
        try:
            prediction = self.arima_models[product].forecast(steps=1)[0]
            return prediction
        except:
            return None

    def calculate_basket_value(self, basket_name: str, order_depths: Dict[str, OrderDepth]) -> float:
        """Calculate the fair value of a basket based on its components"""
        basket = self.products[basket_name]
        total_value = 0.0
        
        for component, quantity in basket["components"].items():
            order_depth = order_depths.get(component, OrderDepth())
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return None
                
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.update_price_history(component, mid_price)
            total_value += mid_price * quantity
            
        return total_value

    def find_arbitrage_opportunity(self, basket_name: str, basket_price: float, 
                                 component_value: float) -> tuple:
        """Determine if there's an arbitrage opportunity and return the direction"""
        if component_value is None:
            return None, 0
            
        price_diff = basket_price - component_value
        if abs(price_diff) < self.min_profit:
            return None, 0
            
        if price_diff > 0:
            # Basket is overpriced, sell basket and buy components
            return "SELL", price_diff
        else:
            # Basket is underpriced, buy basket and sell components
            return "BUY", abs(price_diff)

    def find_prediction_opportunity(self, product: str, current_price: float) -> tuple:
        """Find trading opportunity based on ARIMA predictions"""
        prediction = self.predict_price(product)
        if prediction is None:
            return None, 0
            
        # Calculate standard deviation of recent prices
        recent_prices = list(self.price_history[product])[-20:]
        if len(recent_prices) < 2:
            return None, 0
            
        std = np.std(recent_prices)
        mean = np.mean(recent_prices)
        
        # Calculate z-score
        z_score = (current_price - prediction) / std
        
        if abs(z_score) > self.std_threshold:
            if z_score > 0:
                # Price is above prediction, sell
                return "SELL", abs(z_score)
            else:
                # Price is below prediction, buy
                return "BUY", abs(z_score)
                
        return None, 0

    def execute_arbitrage(self, state: TradingState, basket_name: str, 
                         direction: str, profit: float) -> Dict[str, List[Order]]:
        """Execute arbitrage trades for the given basket"""
        result = {}
        basket = self.products[basket_name]
        position = state.position.get(basket_name, 0)
        
        # Get basket order depth
        basket_depth = state.order_depths.get(basket_name, OrderDepth())
        if not basket_depth.buy_orders or not basket_depth.sell_orders:
            return result
            
        best_bid = max(basket_depth.buy_orders.keys())
        best_ask = min(basket_depth.sell_orders.keys())
        
        # Calculate maximum quantity we can trade
        max_quantity = min(
            basket["position_limit"] - position if direction == "BUY" else basket["position_limit"] + position,
            basket_depth.buy_orders[best_bid] if direction == "SELL" else basket_depth.sell_orders[best_ask]
        )
        
        if max_quantity <= 0:
            return result
            
        # Execute basket trade
        orders = []
        if direction == "BUY":
            orders.append(Order(basket_name, best_ask, max_quantity))
        else:
            orders.append(Order(basket_name, best_bid, -max_quantity))
        result[basket_name] = orders
        
        # Execute component trades
        for component, quantity in basket["components"].items():
            component_position = state.position.get(component, 0)
            component_depth = state.order_depths.get(component, OrderDepth())
            
            if not component_depth.buy_orders or not component_depth.sell_orders:
                continue
                
            component_best_bid = max(component_depth.buy_orders.keys())
            component_best_ask = min(component_depth.sell_orders.keys())
            
            component_max_quantity = min(
                self.position_limits[component] - component_position if direction == "SELL" 
                else self.position_limits[component] + component_position,
                component_depth.buy_orders[component_best_bid] if direction == "SELL" 
                else component_depth.sell_orders[component_best_ask]
            )
            
            component_quantity = min(max_quantity * quantity, component_max_quantity)
            if component_quantity <= 0:
                continue
                
            if direction == "SELL":
                result[component] = [Order(component, component_best_bid, component_quantity)]
            else:
                result[component] = [Order(component, component_best_ask, -component_quantity)]
                
        return result

    def execute_prediction_trade(self, state: TradingState, product: str, 
                               direction: str, confidence: float) -> Dict[str, List[Order]]:
        """Execute trades based on ARIMA predictions"""
        result = {}
        position = state.position.get(product, 0)
        order_depth = state.order_depths.get(product, OrderDepth())
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Calculate trade quantity based on confidence
        max_quantity = min(
            self.position_limits[product] - position if direction == "BUY" 
            else self.position_limits[product] + position,
            order_depth.sell_orders[best_ask] if direction == "BUY" 
            else order_depth.buy_orders[best_bid]
        )
        
        # Scale quantity based on confidence
        quantity = int(max_quantity * min(1.0, confidence / self.std_threshold))
        
        if quantity <= 0:
            return result
            
        if direction == "BUY":
            result[product] = [Order(product, best_ask, quantity)]
        else:
            result[product] = [Order(product, best_bid, -quantity)]
            
        return result

    def run(self, state: TradingState):
        """
        Analyze price data and output ARIMA coefficients
        Returns: Dict[str, List[Order]], int, str
        """
        result = {}
        
        # Update price histories
        for product in ["CROISSANTS", "JAMS", "DJEMBES"]:
            order_depth = state.order_depths.get(product, OrderDepth())
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                self.update_price_history(product, mid_price)
                
                # Train model and get coefficients
                if len(self.price_history[product]) >= 30:
                    self.coefficients[product] = self.train_arima_model(product)
                    if self.coefficients[product] is not None:
                        print(f"{product} coefficients: {self.coefficients[product]}")
                        self.save_coefficients()
        
        return result, 0, ""

def train_arima_models():
    # Read price data with semicolon separator
    prices_day_neg1 = pd.read_csv('Round 2/round-2-island-data-bottle/prices_round_2_day_-1.csv', sep=';')
    prices_day_0 = pd.read_csv('Round 2/round-2-island-data-bottle/prices_round_2_day_0.csv', sep=';')
    prices_day_1 = pd.read_csv('Round 2/round-2-island-data-bottle/prices_round_2_day_1.csv', sep=';')
    
    # Combine all data
    all_prices = pd.concat([prices_day_neg1, prices_day_0, prices_day_1])
    
    # Get first half of data for training
    train_data = all_prices.iloc[:len(all_prices)//2]
    
    # Products to train on
    products = ['CROISSANTS', 'JAMS', 'DJEMBES']
    
    # Store coefficients
    coefficients = {}
    
    for product in products:
        # Get mid prices for the product
        product_data = train_data[train_data['product'] == product]
        if len(product_data) < 30:  # Need minimum data points
            print(f"Not enough data for {product}")
            continue
            
        # Calculate mid prices
        mid_prices = (product_data['bid_price_1'] + product_data['ask_price_1']) / 2
        
        try:
            # Fit ARIMA model
            model = ARIMA(mid_prices, order=(5,1,0))
            model_fit = model.fit()
            
            # Get coefficients
            coefficients[product] = {
                'ar_coef': model_fit.arparams.tolist(),
                'ma_coef': [0.0] * 5,  # No MA terms in our model
                'intercept': float(model_fit.params[0]),
                'sigma2': float(model_fit.scale)  # Using scale instead of sigma2
            }
            
            print(f"Trained model for {product}")
            print(f"AR coefficients: {coefficients[product]['ar_coef']}")
            print(f"Intercept: {coefficients[product]['intercept']}")
            print(f"Sigma squared: {coefficients[product]['sigma2']}")
            print("---")
            
        except Exception as e:
            print(f"Error training ARIMA for {product}: {str(e)}")
    
    # Save coefficients to CSV
    data = []
    for product in products:
        if product in coefficients:
            row = {
                'product': product,
                'ar_coef_1': coefficients[product]['ar_coef'][0],
                'ar_coef_2': coefficients[product]['ar_coef'][1],
                'ar_coef_3': coefficients[product]['ar_coef'][2],
                'ar_coef_4': coefficients[product]['ar_coef'][3],
                'ar_coef_5': coefficients[product]['ar_coef'][4],
                'ma_coef_1': coefficients[product]['ma_coef'][0],
                'ma_coef_2': coefficients[product]['ma_coef'][1],
                'ma_coef_3': coefficients[product]['ma_coef'][2],
                'ma_coef_4': coefficients[product]['ma_coef'][3],
                'ma_coef_5': coefficients[product]['ma_coef'][4],
                'intercept': coefficients[product]['intercept'],
                'sigma2': coefficients[product]['sigma2']
            }
            data.append(row)
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv('arima_coefficients.csv', index=False)
        print("Coefficients saved to arima_coefficients.csv")

if __name__ == "__main__":
    train_arima_models() 
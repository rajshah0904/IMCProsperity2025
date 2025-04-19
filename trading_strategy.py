import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data():
    # Load the CSV files into DataFrames
    file_paths = [
        '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_1.csv',
        '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_2.csv',
        '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_3.csv'
    ]
    
    dataframes = [pd.read_csv(file_path, delimiter=';') for file_path in file_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def analyze_product_data(df, product_name):
    # Filter data for the specific product
    product_df = df[df['product'] == product_name].copy()
    if product_df.empty:
        print(f"No data found for product: {product_name}")
        return None
    
    # Sort by day and timestamp
    product_df = product_df.sort_values(by=['day', 'timestamp'])
    
    # Calculate price features
    product_df['price_change'] = product_df['mid_price'].diff()
    product_df['pct_change'] = product_df['mid_price'].pct_change()
    product_df['rolling_mean'] = product_df['mid_price'].rolling(window=10).mean()
    product_df['rolling_std'] = product_df['mid_price'].rolling(window=10).std()
    
    # Calculate bid-ask spread
    product_df['spread'] = product_df['ask_price_1'] - product_df['bid_price_1']
    product_df['relative_spread'] = product_df['spread'] / product_df['mid_price']
    
    # Calculate volume imbalance
    product_df['volume_imbalance'] = (product_df['bid_volume_1'] - product_df['ask_volume_1']) / (product_df['bid_volume_1'] + product_df['ask_volume_1'])
    
    return product_df

def linear_regression_forecast(df, forecast_horizon=10):
    # Prepare data for linear regression
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['mid_price'].values
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future prices
    future_X = np.arange(len(df), len(df) + forecast_horizon).reshape(-1, 1)
    future_predictions = model.predict(future_X)
    
    return future_predictions, model.coef_[0], model.intercept_

def compute_optimal_position(current_price, predicted_prices, volatility, max_position=100):
    # Calculate trend strength
    trend = predicted_prices[-1] - current_price
    trend_strength = abs(trend) / (volatility + 1e-6)  # Adding small value to avoid division by zero
    
    # Determine position direction
    direction = 1 if trend > 0 else -1
    
    # Scale position size based on trend strength 
    confidence = min(1.0, trend_strength / 3)  # Scale confidence, cap at 1.0
    position_size = int(max_position * confidence)
    
    return direction * position_size

def identify_trading_opportunities(df):
    # Get unique products
    products = df['product'].unique()
    
    # Store trading recommendations
    recommendations = {}
    
    for product in products:
        product_df = analyze_product_data(df, product)
        if product_df is None or len(product_df) < 20:  # Need sufficient data
            continue
            
        # Get latest data
        latest_data = product_df.iloc[-1]
        current_price = latest_data['mid_price']
        
        # Calculate volatility
        volatility = product_df['mid_price'].rolling(window=20).std().iloc[-1]
        
        # Forecast prices
        future_prices, slope, intercept = linear_regression_forecast(product_df)
        
        # Determine optimal position
        optimal_position = compute_optimal_position(current_price, future_prices, volatility)
        
        # Calculate expected profit
        expected_price_move = future_prices[-1] - current_price
        expected_profit = abs(expected_price_move) * abs(optimal_position)
        
        # Store recommendation
        trend_direction = "BULLISH" if slope > 0 else "BEARISH"
        strength = "STRONG" if abs(slope) > 0.01 else "MODERATE"
        
        recommendations[product] = {
            "current_price": current_price,
            "forecast_price": future_prices[-1],
            "trend": f"{strength} {trend_direction}",
            "optimal_position": optimal_position,
            "expected_profit": expected_profit,
            "volatility": volatility,
            "confidence": abs(optimal_position) / 100  # Normalized confidence
        }
    
    return recommendations

def rank_opportunities(recommendations):
    # Rank opportunities by expected profit
    ranked_opportunities = sorted(
        recommendations.items(), 
        key=lambda x: x[1]['expected_profit'], 
        reverse=True
    )
    return ranked_opportunities

def main():
    # Load and prepare data
    print("Loading data...")
    df = load_data()
    
    # Identify trading opportunities
    print("Identifying trading opportunities...")
    opportunities = identify_trading_opportunities(df)
    
    # Rank opportunities
    print("Ranking opportunities by expected profit...")
    ranked_opportunities = rank_opportunities(opportunities)
    
    # Print recommendations
    print("\n=== TRADING RECOMMENDATIONS ===")
    for i, (product, details) in enumerate(ranked_opportunities[:10], 1):
        print(f"\n{i}. {product}")
        print(f"   Current Price: {details['current_price']:.2f}")
        print(f"   Forecast Price: {details['forecast_price']:.2f}")
        print(f"   Trend: {details['trend']}")
        position_type = "BUY" if details['optimal_position'] > 0 else "SELL"
        print(f"   Recommendation: {position_type} {abs(details['optimal_position'])} units")
        print(f"   Expected Profit: {details['expected_profit']:.2f}")
        print(f"   Confidence: {details['confidence']:.2f}")
        print(f"   Volatility: {details['volatility']:.2f}")

if __name__ == "__main__":
    main() 
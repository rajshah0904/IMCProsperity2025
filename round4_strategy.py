import pandas as pd
import numpy as np
from collections import defaultdict
from round3_final import Trader
from datamodel import Order, OrderDepth, TradingState, Listing

def create_order_depth_from_data(row):
    """Create OrderDepth object from a row of CSV data"""
    order_depth = OrderDepth()
    
    # Add buy orders if available
    if not pd.isna(row['bid_price_1']):
        order_depth.buy_orders[row['bid_price_1']] = row['bid_volume_1']
    if not pd.isna(row['bid_price_2']):
        order_depth.buy_orders[row['bid_price_2']] = row['bid_volume_2']
    if not pd.isna(row['bid_price_3']):
        order_depth.buy_orders[row['bid_price_3']] = row['bid_volume_3']
    
    # Add sell orders if available
    if not pd.isna(row['ask_price_1']):
        order_depth.sell_orders[row['ask_price_1']] = row['ask_volume_1']
    if not pd.isna(row['ask_price_2']):
        order_depth.sell_orders[row['ask_price_2']] = row['ask_volume_2']
    if not pd.isna(row['ask_price_3']):
        order_depth.sell_orders[row['ask_price_3']] = row['ask_volume_3']
    
    return order_depth

def simulate_trading():
    # Initialize the Trader
    trader = Trader()
    
    # Load the CSV files into DataFrames
    file_paths = [
        '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_1.csv',
        '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_2.csv',
        '/Users/amrmotair/IMCProsperity2025/prices_round_4_day_3.csv'
    ]
    
    dataframes = [pd.read_csv(file_path, delimiter=';') for file_path in file_paths]
    
    # Keep track of positions and profit
    positions = defaultdict(int)
    total_profit = 0
    trades_executed = []
    
    # Process each day's data
    for day_idx, df in enumerate(dataframes, start=1):
        print(f"\nProcessing Day {day_idx}...")
        
        # Group data by timestamp to simulate real-time trading
        timestamps = df['timestamp'].unique()
        
        for timestamp in timestamps:
            # Get all products at this timestamp
            timestamp_data = df[df['timestamp'] == timestamp]
            
            # Create order_depths for each product
            order_depths = {}
            for _, row in timestamp_data.iterrows():
                product = row['product']
                order_depths[product] = create_order_depth_from_data(row)
            
            # Create a TradingState object
            state = TradingState(
                timestamp=timestamp,
                listings={product: Listing(product, product, "") for product in order_depths.keys()},
                order_depths=order_depths,
                own_trades={},
                market_trades={},
                position=positions,
                observations={"PROFIT_AND_LOSS": str(total_profit)},
                traderData=str(total_profit)
            )
            
            # Execute trader's strategy
            result_orders, conversions, _ = trader.run(state)
            
            # Process the orders (simulate execution)
            for product, orders in result_orders.items():
                for order in orders:
                    # Find the current price data
                    product_data = timestamp_data[timestamp_data['product'] == product]
                    if product_data.empty:
                        continue
                    
                    # Determine execution price based on order direction
                    if order.quantity > 0:  # BUY
                        execution_price = product_data['ask_price_1'].values[0]
                    else:  # SELL
                        execution_price = product_data['bid_price_1'].values[0]
                    
                    # Update position
                    positions[product] += order.quantity
                    
                    # Calculate profit impact
                    trade_profit = -order.quantity * execution_price
                    total_profit += trade_profit
                    
                    # Record the trade
                    trades_executed.append({
                        'day': day_idx,
                        'timestamp': timestamp,
                        'product': product,
                        'quantity': order.quantity,
                        'price': execution_price,
                        'profit': trade_profit,
                        'cumulative_profit': total_profit
                    })
                    
                    print(f"Executed: {order.quantity} {product} @ {execution_price} | Profit: {trade_profit:.2f} | Total: {total_profit:.2f}")
    
    # Calculate final P&L including marking positions to market
    final_positions_value = 0
    for product, quantity in positions.items():
        # Get the last price for each product
        last_prices = dataframes[-1][dataframes[-1]['product'] == product]['mid_price']
        if not last_prices.empty:
            last_price = last_prices.iloc[-1]
            position_value = quantity * last_price
            final_positions_value += position_value
    
    # Final profit calculation
    final_profit = total_profit + final_positions_value
    
    # Print results
    print("\n=== TRADING RESULTS ===")
    print(f"Total executed trades: {len(trades_executed)}")
    print(f"Final positions: {dict(positions)}")
    print(f"Final positions value: {final_positions_value:.2f}")
    print(f"Total profit (excluding positions): {total_profit:.2f}")
    print(f"Total profit (including positions): {final_profit:.2f}")
    
    # Analyze the most profitable products
    trade_df = pd.DataFrame(trades_executed)
    if not trade_df.empty:
        product_summary = trade_df.groupby('product').agg({
            'profit': 'sum',
            'quantity': lambda x: abs(x).sum()  # Total volume traded
        }).reset_index()
        
        product_summary['profit_per_trade'] = product_summary['profit'] / product_summary['quantity']
        product_summary = product_summary.sort_values('profit', ascending=False)
        
        print("\n=== PRODUCT PERFORMANCE ===")
        print(product_summary)
    
    return {
        'trades': trades_executed,
        'positions': positions,
        'total_profit': total_profit,
        'final_profit': final_profit
    }

if __name__ == "__main__":
    results = simulate_trading() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_price_data():
    # Load all three price files
    files = [
        'prices_round_1_day_-1.csv',
        'prices_round_1_day_0.csv',
        'prices_round_1_day_1.csv'
    ]
    
    dfs = []
    for file in files:
        df = pd.read_csv(file, sep=';')
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def analyze_product(df, product):
    # Filter data for specific product
    product_df = df[df['product'] == product]
    
    # Calculate basic statistics
    stats = {
        'mean_price': product_df['mid_price'].mean(),
        'std_price': product_df['mid_price'].std(),
        'min_price': product_df['mid_price'].min(),
        'max_price': product_df['mid_price'].max(),
        'price_range': product_df['mid_price'].max() - product_df['mid_price'].min(),
        'mean_spread': (product_df['ask_price_1'] - product_df['bid_price_1']).mean(),
        'mean_volume': product_df['bid_volume_1'].mean() + product_df['ask_volume_1'].mean()
    }
    
    # Calculate price changes
    product_df['price_change'] = product_df['mid_price'].diff()
    stats['mean_daily_change'] = product_df['price_change'].mean()
    stats['std_daily_change'] = product_df['price_change'].std()
    
    return stats

def analyze_correlations(df):
    # Calculate correlations between products
    products = df['product'].unique()
    correlations = {}
    
    for i, prod1 in enumerate(products):
        for prod2 in products[i+1:]:
            price1 = df[df['product'] == prod1]['mid_price']
            price2 = df[df['product'] == prod2]['mid_price']
            corr = price1.corr(price2)
            correlations[f"{prod1}-{prod2}"] = corr
    
    return correlations

def main():
    # Load data
    df = load_price_data()
    
    # Analyze each product
    products = df['product'].unique()
    print("\nProduct Analysis:")
    print("-" * 50)
    
    for product in products:
        stats = analyze_product(df, product)
        print(f"\n{product} Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
    
    # Analyze correlations
    correlations = analyze_correlations(df)
    print("\nProduct Correlations:")
    print("-" * 50)
    for pair, corr in correlations.items():
        print(f"{pair}: {corr:.4f}")
    
    # Plot price movements
    plt.figure(figsize=(15, 10))
    for product in products:
        product_df = df[df['product'] == product]
        plt.plot(product_df['timestamp'], product_df['mid_price'], label=product)
    
    plt.title('Price Movements Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Mid Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('price_movements.png')
    plt.close()

if __name__ == "__main__":
    main() 
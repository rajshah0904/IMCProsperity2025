import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to get previous returns
def get_prev_returns(df, col, its):
    prev_col = f"{col}_prev_{its}_its"
    df[prev_col] = df[col].shift(its)
    
    if col == 'SUGAR_DIFF':
        df[f"{col}_returns_from_{its}_its_ago"] = ((df[col] - df[prev_col]) / df[prev_col]).where(df['SUGAR_DIFF'] >= 3.0, 0)
    else:
        df[f"{col}_returns_from_{its}_its_ago"] = (df[col] - df[prev_col]) / df[prev_col]
    
    df.drop(columns=[prev_col], inplace=True)
    return df

# Function to get future returns
def get_future_returns(df, col, its):
    future_col = f"{col}_future_{its}_its"
    df[future_col] = df[col].shift(-its)
    
    if col == 'SUGAR_DIFF':
        df[f"{col}_returns_in_{its}_its"] = ((df[future_col] - df[col]) / df[col]).where(df['SUGAR_DIFF'] >= 3.0, 0)
    else:
        df[f"{col}_returns_in_{its}_its"] = (df[future_col] - df[col]) / df[col]
    
    df.drop(columns=[future_col], inplace=True)
    return df

# Function to generate returns dataframe
def generate_returns_dataframe(df, iteration):
    columns_to_process = ['MAGNIFICENT_MACARONS', 'TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'SUGAR_PRICE', 'SUNLIGHT_INDEX', 'SUGAR_DIFF']
    new_df = df.copy()

    for col in columns_to_process:
        new_df = get_prev_returns(new_df, col, iteration)
        if col == 'MAGNIFICENT_MACARONS':
            new_df = get_future_returns(new_df, col, iteration)
    
    return new_df

# Function to calculate correlation
def calculate_correlation(df, iteration):
    columns_to_process = ['TRANSPORT_FEES', 'EXPORT_TARIFF', 'IMPORT_TARIFF', 'SUGAR_PRICE', 'SUNLIGHT_INDEX', 'SUGAR_DIFF']
    correlations = {}

    for col in columns_to_process:
        corr = df[f"{col}_returns_from_{iteration}_its_ago"].corr(df[f"MAGNIFICENT_MACARONS_returns_in_{iteration}_its"], method='pearson')
        correlations[col] = corr

    return correlations

# Function to process data for a given day
def process_day_data(day):
    print(f"\nProcessing data for Day {day}...")
    observation_df = pd.read_csv(f"observations_round_4_day_{day}.csv", header=0)
    
    # Process the observation data
    processed_df = pd.DataFrame()
    processed_df['timestamp'] = observation_df['timestamp']
    processed_df['day'] = day
    
    # Find the price data for MAGNIFICENT_MACARONS
    # For now, we'll use the bid/ask from observations as a proxy
    processed_df['MAGNIFICENT_MACARONS'] = (observation_df['bidPrice'] + observation_df['askPrice']) / 2
    processed_df['TRANSPORT_FEES'] = observation_df['transportFees']
    processed_df['EXPORT_TARIFF'] = observation_df['exportTariff']
    processed_df['IMPORT_TARIFF'] = observation_df['importTariff']
    processed_df['SUGAR_PRICE'] = observation_df['sugarPrice']
    processed_df['SUNLIGHT_INDEX'] = observation_df['sunlightIndex']
    processed_df['SUGAR_DIFF'] = np.abs(processed_df['SUGAR_PRICE'] - 200)  # Difference from baseline
    
    # Smooth the MAGNIFICENT_MACARONS prices with exponential weighted moving average
    processed_df.loc[:, 'MAGNIFICENT_MACARONS'] = processed_df['MAGNIFICENT_MACARONS'].ewm(alpha=0.05).mean().reset_index(drop=True)
    
    return processed_df

# Process data for all three days
day1_df = process_day_data(1)
day2_df = process_day_data(2)
day3_df = process_day_data(3)

# Combine data from all days, adding a day marker
all_days_df = pd.concat([day1_df, day2_df, day3_df], ignore_index=True)

# Plot time series for each day
days = [1, 2, 3]
for day in days:
    day_df = all_days_df[all_days_df['day'] == day].copy()
    
    # Create a subplot figure with separate y-axes for each feature
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for each feature
    fig.add_trace(go.Scatter(x=day_df['timestamp'], y=day_df['MAGNIFICENT_MACARONS'], name='MAGNIFICENT_MACARONS'), secondary_y=False)
    fig.add_trace(go.Scatter(x=day_df['timestamp'], y=day_df['TRANSPORT_FEES'], name='TRANSPORT_FEES'), secondary_y=True)
    fig.add_trace(go.Scatter(x=day_df['timestamp'], y=day_df['EXPORT_TARIFF'], name='EXPORT_TARIFF'), secondary_y=True)
    fig.add_trace(go.Scatter(x=day_df['timestamp'], y=day_df['IMPORT_TARIFF'], name='IMPORT_TARIFF'), secondary_y=True)
    fig.add_trace(go.Scatter(x=day_df['timestamp'], y=day_df['SUGAR_PRICE'], name='SUGAR_PRICE'), secondary_y=True)
    fig.add_trace(go.Scatter(x=day_df['timestamp'], y=day_df['SUNLIGHT_INDEX'], name='SUNLIGHT_INDEX'), secondary_y=True)
    fig.add_trace(go.Scatter(x=day_df['timestamp'], y=day_df['SUGAR_DIFF'], name='SUGAR_DIFF'), secondary_y=True)
    
    # Set the layout and axis properties
    fig.update_layout(
        title=f'MAGNIFICENT_MACARONS and Feature Values over Time (Day {day})',
        xaxis_title='Timestamp',
        yaxis_title='MAGNIFICENT_MACARONS Price',
        legend=dict(x=0, y=1.15, orientation='h')
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="MAGNIFICENT_MACARONS", secondary_y=False)
    fig.update_yaxes(title_text="Other Features", secondary_y=True)
    
    # Save the plot
    fig.write_html(f"macaron_features_time_series_day{day}.html")
    print(f"Time series plot saved as 'macaron_features_time_series_day{day}.html'")

# Create a comparative plot to see how prices evolve across days
plt.figure(figsize=(12, 6))
for day in days:
    day_df = all_days_df[all_days_df['day'] == day]
    plt.plot(day_df['timestamp'], day_df['MAGNIFICENT_MACARONS'], label=f'Day {day}')

plt.xlabel('Timestamp')
plt.ylabel('MAGNIFICENT_MACARONS Price')
plt.title('MAGNIFICENT_MACARONS Price Evolution Across Days')
plt.legend()
plt.grid(True)
plt.savefig("macaron_price_evolution.png")
print("Price evolution plot saved as 'macaron_price_evolution.png'")

# Calculate correlations for different time windows using data from all days
iteration_candidates = [1, 5, 10, 50, 100, 250, 500, 700, 1000]
correlation_data = {}

# Create separate figures for each day and one for all days combined
fig_all, axes_all = plt.subplots(3, 3, figsize=(15, 10))
axes_all = axes_all.flatten()

# Analyze each day separately and then all days combined
day_dfs = {1: day1_df, 2: day2_df, 3: day3_df, 'all': all_days_df}

for day_label, df in day_dfs.items():
    correlation_data[day_label] = {}
    
    # Create a figure for this day
    if day_label != 'all':  # Skip individual day plots for brevity
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.flatten()
    
    for i, iteration in enumerate(iteration_candidates):
        print(f"Analyzing correlation for {day_label}, iteration: {iteration}")
        new_df = generate_returns_dataframe(df, iteration)
        correlations = calculate_correlation(new_df, iteration)
        correlation_data[day_label][iteration] = correlations
        
        # Print correlation values
        for col, corr in correlations.items():
            print(f"Correlation between {col}_returns_from_{iteration}_its_ago and MAGNIFICENT_MACARONS_returns_in_{iteration}_its: {corr}")
        
        # Create a subplot for this iteration in the day-specific figure
        if day_label != 'all':  # Skip individual day plots for brevity
            ax = axes[i]
            ax.bar(correlations.keys(), correlations.values())
            ax.set_title(f"Day {day_label}, Iter {iteration}")
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add to the all-days figure
        if day_label == 'all':
            ax = axes_all[i]
            ax.bar(correlations.keys(), correlations.values())
            ax.set_title(f"All Days, Iter {iteration}")
            ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Save the day-specific figure
    if day_label != 'all':  # Skip individual day plots for brevity
        plt.tight_layout()
        plt.savefig(f"macaron_correlation_day{day_label}.png")
        print(f"Correlation plots for Day {day_label} saved as 'macaron_correlation_day{day_label}.png'")

# Save the all-days figure
plt.figure(fig_all.number)
plt.tight_layout()
plt.savefig("macaron_correlation_all_days.png")
print("Correlation plots for all days combined saved as 'macaron_correlation_all_days.png'")

# Create heatmaps for each day and all days combined
for day_label, day_corr in correlation_data.items():
    heatmap_data = pd.DataFrame({k: v for k, v in day_corr.items()}).T
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f')
    plt.xlabel('Features')
    plt.ylabel('Timeframes (iterations)')
    plt.title(f"Correlation Heatmap for MAGNIFICENT_MACARONS - {'All Days' if day_label == 'all' else f'Day {day_label}'}")
    plt.tight_layout()
    plt.savefig(f"macaron_correlation_heatmap_{'all_days' if day_label == 'all' else f'day{day_label}'}.png")
    print(f"Correlation heatmap saved as 'macaron_correlation_heatmap_{'all_days' if day_label == 'all' else f'day{day_label}'}.png'")

# Analyze the effect of SUGAR_DIFF more closely across all days
sugar_threshold = 3.0  # Threshold for significant sugar price deviation
iteration = 100  # Focus on this time window

# Generate returns dataframe focused on SUGAR_DIFF for all days
df_sugar_diff = generate_returns_dataframe(all_days_df, iteration).dropna()
df_sugar_diff_filtered = df_sugar_diff[df_sugar_diff['SUGAR_DIFF'] >= sugar_threshold]

if not df_sugar_diff_filtered.empty:
    # Create a linear regression model
    X = df_sugar_diff_filtered[f"SUGAR_DIFF_returns_from_{iteration}_its_ago"].values.reshape(-1, 1)
    y = df_sugar_diff_filtered[f"MAGNIFICENT_MACARONS_returns_in_{iteration}_its"].values.reshape(-1, 1)
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    coefficient = model.coef_[0][0]
    
    # Equation
    equation = f"MAGNIFICENT_MACARONS_returns_in_{iteration}_its = {coefficient:.4f} * SUGAR_DIFF_returns_from_{iteration}_its_ago"
    print("\nPrediction equation for all days combined:")
    print(equation)
    
    # Make predictions
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R^2) Value: {r2:.4f}")
    
    # Plot the linear regression
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5)
    plt.plot(X, y_pred, color='red', linewidth=2)
    plt.xlabel(f"SUGAR_DIFF_returns_from_{iteration}_its_ago")
    plt.ylabel(f"MAGNIFICENT_MACARONS_returns_in_{iteration}_its")
    plt.title(f"Linear Regression (SUGAR_DIFF ≥ {sugar_threshold}) - All Days")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("macaron_sugar_diff_regression_all_days.png")
    print("Sugar difference regression plot saved as 'macaron_sugar_diff_regression_all_days.png'")
else:
    print(f"\nNo data points with SUGAR_DIFF >= {sugar_threshold} across all days")

# Calculate trading signals for all days combined
# Use the "all" day correlations
heatmap_data = pd.DataFrame({k: v for k, v in correlation_data['all'].items()}).T

# Identify iterations with strongest correlations for each feature
strongest_corr = {}
for col in heatmap_data.columns:
    strongest_iter = heatmap_data[col].abs().idxmax()
    strongest_corr[col] = {
        'iteration': strongest_iter,
        'correlation': heatmap_data.loc[strongest_iter, col]
    }

print("\nStrongest correlations for each feature:")
for feature, data in strongest_corr.items():
    print(f"{feature}: Iteration {data['iteration']}, Correlation: {data['correlation']:.4f}")

# Focus on the features with the strongest correlations
# Typically these would be SUGAR_PRICE and SUNLIGHT_INDEX based on our hypothesis
key_features = ['SUGAR_PRICE', 'SUNLIGHT_INDEX', 'SUGAR_DIFF']
selected_features = {f: strongest_corr[f] for f in key_features}

# Determine the max iteration needed for our signals
max_iter = max([data['iteration'] for data in selected_features.values()])

# Generate a dataframe with all necessary return calculations
signal_df = all_days_df.copy()
for feature, data in selected_features.items():
    iteration = data['iteration']
    signal_df = get_prev_returns(signal_df, feature, iteration)

# Get future MAGNIFICENT_MACARONS returns for the max iteration 
signal_df = get_future_returns(signal_df, 'MAGNIFICENT_MACARONS', max_iter)

# Drop NA values from calculations
signal_df = signal_df.dropna()

# Create a weighted signal using the correlation as weights
signal_df['WEIGHTED_SIGNAL'] = 0
for feature, data in selected_features.items():
    iteration = data['iteration']
    weight = data['correlation']
    signal_df['WEIGHTED_SIGNAL'] += signal_df[f"{feature}_returns_from_{iteration}_its_ago"] * weight

# Check the correlation of our weighted signal
signal_corr = signal_df['WEIGHTED_SIGNAL'].corr(signal_df[f'MAGNIFICENT_MACARONS_returns_in_{max_iter}_its'])
print(f"\nCorrelation of weighted signal with future returns: {signal_corr:.4f}")

# Plot the weighted signal vs. actual returns
plt.figure(figsize=(12, 6))
plt.scatter(signal_df['WEIGHTED_SIGNAL'], 
           signal_df[f'MAGNIFICENT_MACARONS_returns_in_{max_iter}_its'], 
           alpha=0.5)
plt.xlabel('Weighted Signal')
plt.ylabel(f'MAGNIFICENT_MACARONS Returns in {max_iter} iterations')
plt.title('Weighted Signal vs. Actual Returns')
plt.grid(True)
plt.tight_layout()
plt.savefig("macaron_weighted_signal.png")
print("Weighted signal plot saved as 'macaron_weighted_signal.png'")

# Calculate arbitrage spread (similar to what was done for ORCHIDS)
signal_df['ARB_SPREAD'] = signal_df['IMPORT_TARIFF'] + signal_df['TRANSPORT_FEES']

# Calculate daily averages of arbitrage spread
arb_spread_by_day = signal_df.groupby('day')['ARB_SPREAD'].agg(['mean', 'min', 'max'])
print("\nArbitrage spread by day:")
print(arb_spread_by_day)

# Calculate overall average
overall_arb_spread_avg = signal_df['ARB_SPREAD'].mean()
print(f"Overall average arbitrage spread: {overall_arb_spread_avg:.4f}")

# Analyze if MAGNIFICENT_MACARONS price follows a pattern when SUGAR_DIFF crosses thresholds
# Similar to what was done for HUMIDITY_DIFF with ORCHIDS
sugar_thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]

plt.figure(figsize=(15, 10))
for i, threshold in enumerate(sugar_thresholds):
    subset = signal_df[signal_df['SUGAR_DIFF'] >= threshold]
    if len(subset) > 0:
        plt.subplot(len(sugar_thresholds), 1, i+1)
        plt.scatter(subset['SUGAR_DIFF'], subset[f'MAGNIFICENT_MACARONS_returns_in_{max_iter}_its'], alpha=0.5)
        plt.xlabel(f'SUGAR_DIFF (≥ {threshold})')
        plt.ylabel(f'Returns in {max_iter} its')
        plt.title(f'Returns vs. SUGAR_DIFF (≥ {threshold}), n={len(subset)}')
        plt.grid(True)

plt.tight_layout()
plt.savefig("macaron_sugar_threshold_analysis.png")
print("Sugar threshold analysis plot saved as 'macaron_sugar_threshold_analysis.png'")

# Develop a simple trading strategy
# 1. When the weighted signal exceeds a certain threshold, go long
# 2. When the weighted signal falls below a negative threshold, go short
# 3. Otherwise, stay neutral

# Define signal thresholds (these can be optimized)
long_threshold = signal_df['WEIGHTED_SIGNAL'].quantile(0.75)
short_threshold = signal_df['WEIGHTED_SIGNAL'].quantile(0.25)

# Create position signals
signal_df['POSITION'] = 0  # Neutral by default
signal_df.loc[signal_df['WEIGHTED_SIGNAL'] >= long_threshold, 'POSITION'] = 1  # Long
signal_df.loc[signal_df['WEIGHTED_SIGNAL'] <= short_threshold, 'POSITION'] = -1  # Short

# Calculate strategy returns
signal_df['STRATEGY_RETURN'] = signal_df['POSITION'] * signal_df[f'MAGNIFICENT_MACARONS_returns_in_{max_iter}_its']

# Calculate cumulative returns of the strategy
signal_df['CUM_STRATEGY_RETURN'] = (1 + signal_df['STRATEGY_RETURN']).cumprod() - 1

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
signal_df['CUM_STRATEGY_RETURN'].plot()
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xlabel('Observation Index')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns of MAGNIFICENT_MACARONS Trading Strategy')
plt.grid(True)
plt.tight_layout()
plt.savefig("macaron_strategy_returns.png")
print("Strategy returns plot saved as 'macaron_strategy_returns.png'")

# Print strategy performance metrics
strategy_mean_return = signal_df['STRATEGY_RETURN'].mean()
strategy_std_return = signal_df['STRATEGY_RETURN'].std()
strategy_sharpe = strategy_mean_return / strategy_std_return if strategy_std_return > 0 else 0
strategy_final_return = signal_df['CUM_STRATEGY_RETURN'].iloc[-1]

print("\nStrategy Performance Summary:")
print(f"Mean Return: {strategy_mean_return:.6f}")
print(f"Standard Deviation: {strategy_std_return:.6f}")
print(f"Sharpe Ratio: {strategy_sharpe:.4f}")
print(f"Final Cumulative Return: {strategy_final_return:.4f} ({strategy_final_return*100:.2f}%)")

print("\nAnalysis complete!") 
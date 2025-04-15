import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

def calculate_features(prices: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Calculate features from price history"""
    if len(prices) < 20:
        # Not enough data for features
        return np.zeros(8)
        
    # Calculate returns at different timeframes
    returns_1 = (prices[-1] - prices[-2])/prices[-2] if len(prices) >= 2 else 0
    returns_5 = (prices[-1] - prices[-5])/prices[-5] if len(prices) >= 5 else returns_1
    returns_10 = (prices[-1] - prices[-10])/prices[-10] if len(prices) >= 10 else returns_5
    
    # Calculate moving averages of returns
    returns = np.diff(prices) / prices[:-1]
    ma_returns_5 = np.mean(returns[-5:]) if len(returns) >= 5 else returns[-1]
    ma_returns_10 = np.mean(returns[-10:]) if len(returns) >= 10 else ma_returns_5
    
    # Volatility of returns
    vol_returns_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
    vol_returns_10 = np.std(returns[-10:]) if len(returns) >= 10 else vol_returns_5
    
    # Mean reversion indicator
    current_price = prices[-1]
    ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
    mean_rev = (current_price - ma_10) / ma_10
    
    features = [
        returns_1,      # Most recent return
        returns_5,      # 5-period return
        returns_10,     # 10-period return
        ma_returns_5,   # Average of recent returns (5)
        ma_returns_10,  # Average of recent returns (10)
        vol_returns_5,  # Volatility of returns (5)
        vol_returns_10, # Volatility of returns (10)
        mean_rev       # Mean reversion indicator
    ]
    
    return np.array(features)

class VolcanicRockModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=2,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        self.scaler = RobustScaler()
        
    def prepare_data(self, data_dir: str, window_size: int = 20):
        """Prepare training data from price history"""
        all_features = []
        all_targets = []
        
        for day in [0, 1, 2]:
            trades_file = os.path.join(data_dir, f'trades_round_3_day_{day}.csv')
            
            print(f"Processing day {day}...")
            
            # Read trades data
            trades_df = pd.read_csv(trades_file, sep=';')
            volcanic_trades = trades_df[trades_df['symbol'] == 'VOLCANIC_ROCK'].sort_values('timestamp')
            
            if len(volcanic_trades) == 0:
                continue
                
            print(f"Found {len(volcanic_trades)} volcanic rock trades")
            
            # Convert to numpy arrays
            prices = volcanic_trades['price'].values
            timestamps = volcanic_trades['timestamp'].values
            
            # Generate features and targets
            for i in range(window_size, len(prices)-1):  # -1 to ensure we have next price for target
                price_window = prices[max(0, i-window_size):i+1]  # Include current price
                time_window = timestamps[max(0, i-window_size):i+1]
                
                features = calculate_features(price_window, time_window)
                
                # Target is the next return instead of next price
                current_price = prices[i]
                next_price = prices[i+1]
                target_return = (next_price - current_price) / current_price
                
                all_features.append(features)
                all_targets.append(target_return)
        
        # Convert to numpy arrays
        features_array = np.array(all_features)
        targets_array = np.array(all_targets)
        
        # Remove any invalid data
        valid_mask = ~np.any(np.isnan(features_array) | np.isinf(features_array), axis=1)
        features_array = features_array[valid_mask]
        targets_array = targets_array[valid_mask]
        
        # Scale features
        features_array = self.scaler.fit_transform(features_array)
        
        print(f"Final dataset shape: {features_array.shape}")
        return features_array, targets_array
    
    def train_test_split(self, features: np.ndarray, targets: np.ndarray, 
                        test_size: float = 0.4) -> tuple:
        """Split data chronologically into 60-40 train-test"""
        split_idx = int(len(features) * 0.6)
        return (features[:split_idx], features[split_idx:],
                targets[:split_idx], targets[split_idx:])
    
    def train(self, data_dir: str):
        """Train the model"""
        # Prepare data
        features, targets = self.prepare_data(data_dir)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = self.train_test_split(features, targets)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nTraining MSE: {train_mse:.6f}")
        print(f"Testing MSE: {test_mse:.6f}")
        print(f"Training R²: {train_r2:.2f}")
        print(f"Testing R²: {test_r2:.2f}")
        
        # Print feature importances
        feature_names = [
            'returns_1',
            'returns_5',
            'returns_10',
            'ma_returns_5',
            'ma_returns_10',
            'vol_returns_5',
            'vol_returns_10',
            'mean_rev'
        ]
        importances = self.model.feature_importances_
        print("\nFeature Importances:")
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
        
        return train_mse, test_mse, train_r2, test_r2
    
    def predict(self, price_history: np.ndarray, time_history: np.ndarray) -> float:
        """Predict the next return"""
        features = calculate_features(price_history, time_history)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        predicted_return = self.model.predict(features_scaled)[0]
        
        # Convert predicted return to price
        current_price = price_history[-1]
        predicted_price = current_price * (1 + predicted_return)
        return predicted_price

def main():
    model = VolcanicRockModel()
    data_dir = "round-3-island-data-bottle"
    train_mse, test_mse, train_r2, test_r2 = model.train(data_dir)
    
    print("\nModel Performance Summary:")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Testing MSE: {test_mse:.6f}")
    print(f"Training R²: {train_r2:.2f}")
    print(f"Testing R²: {test_r2:.2f}")

if __name__ == "__main__":
    main() 
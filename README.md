# MAGNIFICENT_MACARONS Trading Strategy

This repository contains a trading algorithm for the MAGNIFICENT_MACARONS product in the IMC Prosperity trading challenge.

## Strategy Overview

The MacaronTrader implements a sophisticated market-making strategy with several key components:

1. **Adaptive Edge Market-Making**: Dynamically adjusts the trading edge (spread) based on market conditions, trading volume, and environmental factors.

2. **Environmental Factor Integration**: 
   - Sugar Price Correlation (0.38): Utilizes a 10-iteration window to track and react to sugar price movements
   - Sunlight Index Correlation (-0.23): Uses a 250-iteration window to monitor sunlight trends and adjust trading parameters accordingly

3. **Smart Order Execution**:
   - Takes arbitrage opportunities when available
   - Places competitive market-making orders
   - Dynamically sizes orders based on position limits

4. **Position Management**:
   - Maintains a position limit of 100 units
   - Converts positions back to cash as needed

## Implementation

The code is organized into two main classes:

1. **Trader**: The main entry point that handles all trading decisions across multiple products
2. **MacaronTrader**: The specific implementation for MAGNIFICENT_MACARONS

### Key Parameters

```
"make_edge": 2            # Initial edge for market making
"make_min_edge": 0.5      # Minimum edge for market making
"sugar_correlation": 0.38 # Correlation coefficient for sugar price
"sunlight_correlation": -0.23 # Correlation for sunlight index
"sugar_window": 10        # Best window for sugar price
"sunlight_window": 250    # Best window for sunlight index
```

## Usage

To use this trader in the competition:

1. Import the Trader class from trader.py
2. The trader automatically handles MAGNIFICENT_MACARONS trading
3. The strategy can be extended to handle additional products

## Performance Considerations

- The adaptive edge mechanism continuously optimizes trading parameters
- Environmental factors (sugar price and sunlight index) provide an edge in market prediction
- Position management ensures compliance with exchange limits while maximizing profit opportunities 
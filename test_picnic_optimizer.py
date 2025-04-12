import numpy as np
from datamodel import OrderDepth, TradingState, Order
from picnic_basket_optimizer import Product, run_optimization

def generate_sample_order_depth(price: float, spread: float = 1.0, volume: int = 10) -> OrderDepth:
    """Generate sample order depth with given price and spread"""
    depth = OrderDepth()
    depth.buy_orders = {int(price - spread/2): volume}
    depth.sell_orders = {int(price + spread/2): volume}
    return depth

def generate_sample_state() -> TradingState:
    """Generate sample trading state with realistic prices"""
    state = TradingState()
    state.order_depths = {}
    
    # Generate component prices with some correlation
    base_price = np.random.normal(100, 10)
    croissant_price = base_price + np.random.normal(0, 2)
    jam_price = base_price + np.random.normal(0, 2)
    djembe_price = base_price + np.random.normal(0, 2)
    
    # Generate basket prices with some spread
    basket_1_price = (croissant_price * 6 + jam_price * 3 + djembe_price * 1) + np.random.normal(0, 5)
    basket_2_price = (croissant_price * 4 + jam_price * 2) + np.random.normal(0, 5)
    
    # Create order depths
    state.order_depths[Product.PICNIC_BASKET_1] = generate_sample_order_depth(basket_1_price)
    state.order_depths[Product.PICNIC_BASKET_2] = generate_sample_order_depth(basket_2_price)
    state.order_depths[Product.CROISSANT] = generate_sample_order_depth(croissant_price)
    state.order_depths[Product.JAM] = generate_sample_order_depth(jam_price)
    state.order_depths[Product.DJEMBE] = generate_sample_order_depth(djembe_price)
    
    # Initialize positions
    state.position = {
        Product.PICNIC_BASKET_1: 0,
        Product.PICNIC_BASKET_2: 0,
        Product.CROISSANT: 0,
        Product.JAM: 0,
        Product.DJEMBE: 0
    }
    
    return state

def run_optimization_test(num_steps: int = 100):
    """Run optimization test with generated data"""
    print("Running optimization test...")
    print(f"Generating {num_steps} steps of market data...")
    
    # Run optimization for multiple steps
    for i in range(num_steps):
        state = generate_sample_state()
        results = run_optimization(state)
        
        # Print results every 20 steps
        if (i + 1) % 20 == 0:
            print(f"\nStep {i + 1} Results:")
            if results:
                print(f"Best std_window: {results['best_std_window']}")
                print(f"Best zscore_threshold: {results['best_zscore_threshold']}")
                print(f"Best target_position: {results['best_target_position']}")
                print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            else:
                print("Not enough data yet for optimization")

if __name__ == "__main__":
    run_optimization_test() 
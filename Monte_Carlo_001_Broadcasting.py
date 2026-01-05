import numpy as np
import matplotlib.pyplot as plt

# Layman setup: Stock starts at $100, grows 10% avg/year but wiggles 20%
initial_price = 100
expected_return = 0.2  # 10% growth
volatility = 0.2       # 20% wiggles
days = 252             # Trading days in year
simulations = 100000   # Run 10,000 "what if" games

# Random daily changes: Big 2D array (sims rows, days columns)
random_changes = np.random.normal(0, volatility/np.sqrt(days), (simulations, days))

# Broadcasting magic: Add daily growth (scalar stretches to whole array!)
daily_returns = expected_return/days + random_changes

# Grow prices over time (cumulative product — vectorized!)
price_paths = initial_price * np.cumprod(1 + daily_returns, axis=1)

# Ending prices from all simulations
final_prices = price_paths[:, -1]

# Layman results
print(f"Average ending price: ${final_prices.mean():.2f}")
print(f"Worst 5% case: ${np.percentile(final_prices, 5):.2f} (risky!)")
print(f"Best 5% case: ${np.percentile(final_prices, 95):.2f} (lucky!)")

# Quick plot to see the "cloud" of possibilities
plt.figure(figsize=(10,6))
plt.plot(price_paths[:100].T, alpha=0.1)  # Show 100 paths
plt.title("Monte Carlo: 10,000 Possible Stock Price Futures")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

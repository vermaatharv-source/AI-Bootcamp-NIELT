import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 100 days of random daily returns for 3 assets (mean=0.001, std=0.02)
days = 100
returns = np.random.normal(loc=0.001, scale=0.02, size=(days, 3))

# Calculate cumulative returns for each asset
cumulative_returns = np.cumprod(1 + returns, axis=0) - 1

# Statistical analysis across assets
mean_returns = np.mean(returns, axis=0)
volatility = np.std(returns, axis=0)
correlation_matrix = np.corrcoef(returns, rowvar=False)

print("--- NumPy Financial Simulation ---")
print(f"Final Cumulative Returns: {cumulative_returns[-1]}")
print(f"Mean Daily Returns: {mean_returns}")
print(f"Asset Volatility (Std Dev): {volatility}")
print("Correlation Matrix:\n", correlation_matrix)
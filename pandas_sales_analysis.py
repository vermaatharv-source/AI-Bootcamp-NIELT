import numpy as np
import pandas as pd

# Create a synthetic dataset
np.random.seed(42)
dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
categories = ["Electronics", "Clothing", "Home & Kitchen"]

data = {
    "Date": np.random.choice(dates, size=100),
    "Category": np.random.choice(categories, size=100),
    "Units_Sold": np.random.randint(1, 15, size=100),
    "Unit_Price": np.random.uniform(10.0, 250.0, size=100),
}

df = pd.DataFrame(data)

# Data Cleaning & Feature Engineering
df["Total_Sales"] = df["Units_Sold"] * df["Unit_Price"]
df = df.sort_values("Date").reset_index(drop=True)

# Groupby Aggregation (Total sales and average units sold per category)
summary_table = (
    df.groupby("Category")
    .agg(
        Total_Revenue=("Total_Sales", "sum"),
        Average_Units=("Units_Sold", "mean"),
        Transaction_Count=("Total_Sales", "count"),
    )
    .reset_index()
)

print("--- Pandas Sales Summary ---")
print(summary_table.to_string(index=False))
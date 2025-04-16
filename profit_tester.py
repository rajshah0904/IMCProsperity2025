import pandas as pd

# Read CSV with semicolon delimiter.
df = pd.read_csv("8b2d13cb-44ab-40fa-96b9-a97b7b81a3a7.csv", delimiter=";")

# Convert the 'profit_and_loss' column to numeric (if not already) and coerce errors.
df["profit_and_loss"] = pd.to_numeric(df["profit_and_loss"], errors="coerce")

# Group by product and sum the profit_and_loss.
pl_summary = df.groupby("product")["profit_and_loss"].sum().reset_index()

# Identify products that are losing money (negative sum).
losing_products = pl_summary[pl_summary["profit_and_loss"] < 0]

print("Profit and Loss Summary by Product:")
print(pl_summary)
print("\nProducts losing money:")
print(losing_products)

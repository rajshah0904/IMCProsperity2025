import pandas as pd
# Read CSV with semicolon delimiter.
df = pd.read_csv("3398a6c4-4906-4ea6-8f5f-73c578cbc40c.csv", delimiter=";")

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

print(sum(pl_summary["profit_and_loss"]))


import pandas as pd

df = pd.read_csv("data/processed/test.csv")
half = len(df) // 2

df.iloc[:half].to_csv("data/processed/test_part1.csv", index=False)
df.iloc[half:].to_csv("data/processed/test_part2.csv", index=False)

print("Split complete:")
print(f"Part 1 rows: {half}")
print(f"Part 2 rows: {len(df) - half}")


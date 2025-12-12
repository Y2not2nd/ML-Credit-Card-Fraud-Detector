import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw dataset
df = pd.read_csv("data/raw/creditcard.csv")

# Basic validation
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Class"].value_counts())

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Stratified split due to heavy class imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Combine back for saving
train_df = X_train.copy()
train_df["Class"] = y_train.values

test_df = X_test.copy()
test_df["Class"] = y_test.values

# Save processed datasets
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nTrain set distribution:")
print(train_df["Class"].value_counts())

print("\nTest set distribution:")
print(test_df["Class"].value_counts())

print("\nData preparation complete.")

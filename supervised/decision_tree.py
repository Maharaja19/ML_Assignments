import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.datasets import load_wine

# ---------------- LOAD BUILT-IN DATASET ----------------
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Convert target to Yes/No style (binary classification for simplicity)
df['Placed'] = df['target'].apply(lambda x: "Yes" if x == 0 else "No")
df.drop(['target'], axis=1, inplace=True)

# ---------------- DISCRETIZE NUMERICAL FEATURES ----------------
# Convert continuous values into Low / Medium / High
for col in df.columns[:-1]:
    df[col] = pd.qcut(df[col], 3, labels=["Low", "Medium", "High"])

X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])
columns = df.columns

# ---------------- FUNCTIONS ----------------
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    ent = 0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def information_gain(X, y, index):
    base_entropy = entropy(y)
    print(f"\nEntropy before split = {base_entropy:.4f}")

    values = set(X[:, index])
    weighted_entropy = 0

    for v in values:
        subset_y = y[X[:, index] == v]
        e = entropy(subset_y)
        weight = len(subset_y) / len(y)
        weighted_entropy += weight * e
        print(f"  Value = {v}, Entropy = {e:.4f}")

    gain = base_entropy - weighted_entropy
    return gain

# ---------------- TRAINING ----------------
print("STEP 1: Training Decision Tree (ID3)\n")

gains = []
for i, col in enumerate(columns[:-1]):
    print(f"\nCalculating Information Gain for attribute: {col}")
    gain = information_gain(X, y, i)
    gains.append(gain)
    print(f"Information Gain = {gain:.4f}")

best_index = np.argmax(gains)

print("\nSTEP 2: Best Attribute Selected as Root")
print("Root Attribute:", columns[best_index])

# ---------------- USER INPUT ----------------
print("\nSTEP 3: Prediction using Root Attribute")

user_value = input(f"Enter value for {columns[best_index]} (Low/Medium/High): ")

subset = df[df[columns[best_index]] == user_value]

if subset.empty:
    print("No matching data found. Prediction not possible.")
else:
    prediction = subset['Placed'].mode()[0]
    print("Predicted Class:", prediction)
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import load_wine

# ---------------- LOAD BUILT-IN DATASET ----------------
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Convert numerical features to categorical (Low/Medium/High)
for col in df.columns[:-1]:
    df[col] = pd.qcut(df[col], 3, labels=["Low", "Medium", "High"])

X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

# ---------------- TRAIN NAIVE BAYES ----------------
def train_nb(X, y):
    model = {}
    classes = set(y)

    for cls in classes:
        model[cls] = {}
        X_cls = X[y == cls]

        for col in range(X.shape[1]):
            model[cls][col] = Counter(X_cls[:, col])

        model[cls]['total'] = len(X_cls)

    return model

# ---------------- PREDICTION ----------------
def predict_nb(model, x):
    probs = {}

    for cls in model:
        total_cls = model[cls]['total']
        probs[cls] = total_cls / len(y)   # Prior probability

        for i in range(len(x)):
            count = model[cls][i].get(x[i], 0)

            # Laplace smoothing
            probs[cls] *= (count + 1) / (total_cls + 3)

    return max(probs, key=probs.get)

# ---------------- TRAIN ----------------
model = train_nb(X, y)

# Test prediction
print("Prediction for first sample:", predict_nb(model, X[0]))
print("Actual class:", y[0])
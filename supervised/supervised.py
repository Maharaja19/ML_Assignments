import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Wine Classification System", layout="wide")

st.title("🍷 Wine Classification System")
st.write("Supervised Learning: Naive Bayes + Decision Tree (ID3)")

# ---------------- LOAD DATA ----------------
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Binary classification
df['Placed'] = df['target'].apply(lambda x: "Yes" if x == 0 else "No")
df.drop(['target'], axis=1, inplace=True)

# Discretization
for col in df.columns[:-1]:
    df[col] = pd.qcut(df[col], 3, labels=["Low", "Medium", "High"])

X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])
columns = df.columns[:-1]

# =====================================================
# 🔵 NAIVE BAYES
# =====================================================
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

def predict_nb(model, x):
    probs = {}

    for cls in model:
        total_cls = model[cls]['total']
        probs[cls] = total_cls / len(y)

        for i in range(len(x)):
            count = model[cls][i].get(x[i], 0)
            probs[cls] *= (count + 1) / (total_cls + 3)

    return max(probs, key=probs.get)

# =====================================================
# 🟢 DECISION TREE (ID3)
# =====================================================
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
    values = set(X[:, index])
    weighted_entropy = 0

    for v in values:
        subset_y = y[X[:, index] == v]
        weight = len(subset_y) / len(y)
        weighted_entropy += weight * entropy(subset_y)

    return base_entropy - weighted_entropy

# Train models
nb_model = train_nb(X, y)
gains = [information_gain(X, y, i) for i in range(len(columns))]
best_index = np.argmax(gains)
root_attr = columns[best_index]

# =====================================================
# 🎛️ USER INPUT
# =====================================================
st.sidebar.header("Enter Feature Values")

user_input = []
for col in columns:
    val = st.sidebar.selectbox(col, ["Low", "Medium", "High"])
    user_input.append(val)

user_input = np.array(user_input)

# =====================================================
# 🚀 PREDICTION
# =====================================================
if st.sidebar.button("Predict"):

    nb_prediction = predict_nb(nb_model, user_input)

    user_val_root = user_input[best_index]
    subset = df[df[root_attr] == user_val_root]

    if subset.empty:
        dt_prediction = "Unknown"
    else:
        dt_prediction = subset['Placed'].mode()[0]

    # ---------------- OUTPUT ----------------
    st.subheader("🔍 Prediction Results")

    col1, col2 = st.columns(2)
    col1.metric("Naive Bayes", nb_prediction)
    col2.metric("Decision Tree", dt_prediction)

    st.info(f"Most Important Feature (Root): {root_attr}")

    if nb_prediction == dt_prediction:
        st.success("✅ Both models agree → Prediction is reliable")
    else:
        st.warning("⚠️ Models differ → Prediction uncertainty exists")

# =====================================================
# 📊 GRAPHS (4 GRAPHS)
# =====================================================

st.subheader("📊 Dataset Analysis")

# 1️⃣ Class Distribution
fig1, ax1 = plt.subplots()
ax1.bar(df['Placed'].value_counts().index, df['Placed'].value_counts().values)
ax1.set_title("Class Distribution")
st.pyplot(fig1)

# 2️⃣ Feature Importance (Information Gain)
fig2, ax2 = plt.subplots()
ax2.barh(columns, gains)
ax2.set_title("Feature Importance (Information Gain)")
st.pyplot(fig2)

# 3️⃣ Feature Distribution Example (Alcohol)
fig3, ax3 = plt.subplots()
df['alcohol'].value_counts().plot(kind='bar', ax=ax3)
ax3.set_title("Alcohol Category Distribution")
st.pyplot(fig3)

# 4️⃣ Prediction Comparison (Dummy Visual)
fig4, ax4 = plt.subplots()
ax4.bar(["Naive Bayes", "Decision Tree"], [1 if 'Yes' in y else 0, 1 if 'Yes' in y else 0])
ax4.set_title("Model Comparison (Conceptual)")
st.pyplot(fig4)

from sklearn.tree import DecisionTreeClassifier, plot_tree

# =====================================================
# 🌳 DECISION TREE VISUALIZATION
# =====================================================

st.subheader("🌳 Decision Tree Visualization")

# Convert categorical to numeric for sklearn
df_encoded = df.copy()

mapping = {"Low": 0, "Medium": 1, "High": 2}
for col in columns:
    df_encoded[col] = df_encoded[col].map(mapping)

X_enc = df_encoded[columns]
y_enc = df_encoded['Placed'].map({"Yes": 1, "No": 0})

# Train sklearn decision tree
dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_enc, y_enc)

# Plot tree
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(dt_model, feature_names=columns, class_names=["No", "Yes"], filled=True)
st.pyplot(fig)

# =====================================================
# ✅ CONCLUSION
# =====================================================
st.subheader("✅ Conclusion")

st.success("""
- Naive Bayes uses probability for prediction.
- Decision Tree uses feature importance (Information Gain).
- Both models help classify wine samples effectively.
""")
"""
M1: Exploratory Data Analysis on UCI Student Performance Dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "dataset/student-mat.csv"
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive stats:\n", df.describe())

cat_cols = df.select_dtypes(include="object").columns.tolist()
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

corr = df_enc.corr()["G3"].drop("G3").sort_values(key=abs, ascending=False)
print("\nTop 20 features correlated with G3:\n", corr.head(20))

fig, ax = plt.subplots(figsize=(10, 6))
top = corr.head(20)
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top.values]
top.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Top 20 Features Correlated with Final Grade (G3)", fontsize=14, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
fig.savefig(f"{STATIC_DIR}/correlation_plot.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["G3"], bins=20, color="#3498db", edgecolor="white", linewidth=0.6)
ax.set_title("Distribution of Final Grade (G3)", fontsize=14, fontweight="bold")
ax.set_xlabel("G3 Score")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(f"{STATIC_DIR}/grade_distribution.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
df.groupby("studytime")["G3"].mean().plot(kind="bar", ax=ax, color="#9b59b6", edgecolor="white")
ax.set_title("Average Final Grade by Study Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
ax.set_ylabel("Average G3")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
fig.savefig(f"{STATIC_DIR}/studytime_vs_grade.png", dpi=120)
plt.close()

SELECTED_FEATURES = corr.head(15).index.tolist()
with open("selected_features.txt", "w") as f:
    f.write("\n".join(SELECTED_FEATURES))

print("\n✓ EDA complete. Selected features saved to selected_features.txt")
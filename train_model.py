import pandas as pd
import numpy as np
import pickle
import os

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset (TAB separated)
df = pd.read_csv("dataset/student-mat.csv", sep="\t")

print("Columns Found:")
print(df.columns.tolist())

# Check G3 exists
if "G3" not in df.columns:
    print("\nERROR: G3 column not found in dataset.")
    print("Check the dataset format.")
    exit()

# Convert categorical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()

df_enc = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=True
)

# Features and Target
features = [c for c in df_enc.columns if c not in ["G1", "G2", "G3"]]

X = df_enc[features]
y = df_enc["G3"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale Features
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_sc, y_train)

lr_pred = lr.predict(X_test_sc)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# Ridge Regression
ridge = Ridge(alpha=1.0)

ridge.fit(X_train_sc, y_train)

ridge_pred = ridge.predict(X_test_sc)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

# Cross Validation
lr_cv = -cross_val_score(
    lr,
    X_train_sc,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=5
).mean()

ridge_cv = -cross_val_score(
    ridge,
    X_train_sc,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=5
).mean()

print("\nModel Comparison")
print("-" * 40)
print(f"Linear Regression RMSE : {lr_rmse:.4f}")
print(f"Ridge Regression RMSE  : {ridge_rmse:.4f}")

# Select Best Model
if ridge_rmse <= lr_rmse:
    best_model = ridge
    best_model_name = "Ridge Regression"
else:
    best_model = lr
    best_model_name = "Linear Regression"

# Save Best Model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save Scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save Features
with open("models/features.pkl", "wb") as f:
    pickle.dump(features, f)

# Save Metrics
metrics = {
    "lr_rmse": round(lr_rmse, 4),
    "ridge_rmse": round(ridge_rmse, 4),
    "best": best_model_name,
    "features_count": len(features)
}

with open("models/metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("\nFiles Saved Successfully")
print("✓ best_model.pkl")
print("✓ models/scaler.pkl")
print("✓ models/features.pkl")
print("✓ models/metrics.pkl")

print(f"\nBest Model : {best_model_name}")
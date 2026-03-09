import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def main(args):
    # Simple demo: create synthetic data if no CSV provided
    if args.csv is None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "feat_num1": rng.normal(0, 1, size=500),
            "feat_num2": rng.normal(5, 2, size=500),
            "feat_cat": rng.choice(["A","B","C"], size=500)
        })
        y = 3.0*X["feat_num1"] - 0.8*X["feat_num2"] + (X["feat_cat"]=="B").astype(int)*2 + rng.normal(0, 0.5, size=500)
    else:
        df = pd.read_csv(args.csv)
        y = df[args.target]
        X = df.drop(columns=[args.target])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1
        ))
    ])

    # Split only for demo
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    model.fit(Xtr, ytr)
    preds = model.predict(Xva)
    mae = mean_absolute_error(yva, preds)
    print(f"[INFO] Validation MAE: {mae:.4f}")

    os.makedirs(args.models_dir, exist_ok=True)
    out_path = os.path.join(args.models_dir, "model.joblib")
    joblib.dump(model, out_path)
    print(f"[INFO] Saved model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV with target column")
    parser.add_argument("--target", type=str, default="SalePrice", help="Target column name")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()
    main(args)

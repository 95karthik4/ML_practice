"""
Feature Selector Tool
---------------------

This module implements multiple feature selection methods:
- Pearson correlation filter
- Chi-Square test
- Recursive Feature Elimination (RFE)
- Lasso regression (embedded)
- RandomForest importance (tree-based)
- LightGBM importance (tree-based)

It then ensembles (votes) across all methods to produce a ranked list of top features.

✅ How to use:
1. Import this file in your project:
   >>> from feature_selector import run_feature_selector

2. Load your dataset:
   >>> import pandas as pd
   >>> df = pd.read_csv("your_dataset.csv")

3. Run feature selection:
   >>> ranked, votes = run_feature_selector(df, target="YourTargetColumn", top_n=20)

4. Save results (optional):
   >>> from feature_selector import save_results
   >>> save_results(ranked, "selected_features.csv")
"""

# =========================
# Imports
# =========================
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor

# LightGBM is optional
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False
    print("⚠️ LightGBM not available; skipping LightGBM selector.")


# =========================
# Individual Feature Selectors
# =========================

def pearson_selector(X, y, threshold=0.15):
    """
    Select features based on Pearson correlation with the target.
    """
    df_temp = pd.concat([X, y.rename("target")], axis=1)
    corr = df_temp.corr()["target"].abs().drop("target")
    selected = corr[corr > threshold].sort_values(ascending=False)
    return selected.index.tolist(), selected.to_dict()


def chi2_selector(X, y, k=10, bins=10):
    """
    Select top features using Chi-Square test.
    """
    X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    y_binned = pd.qcut(y, q=bins, duplicates="drop").astype("category").cat.codes
    selector = SelectKBest(chi2, k=min(k, X.shape[1]))
    selector.fit(X_scaled, y_binned)
    return X.columns[selector.get_support()].tolist()


def rfe_selector(X, y, n_features=10):
    """
    Select features using Recursive Feature Elimination (RFE).
    """
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=min(n_features, X.shape[1]), step=1)
    selector.fit(X, y)
    return X.columns[selector.get_support()].tolist()


def lasso_selector(X, y):
    """
    Select features using LassoCV.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LassoCV(cv=5, random_state=42, n_alphas=100, max_iter=5000)
    model.fit(Xs, y)
    sfm = SelectFromModel(model, prefit=True, threshold="mean")
    mask = sfm.get_support()
    return X.columns[mask].tolist(), model


def rf_selector(X, y, n_features=10):
    """
    Select features using RandomForest importance.
    """
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importances.index[:min(n_features, len(importances))].tolist(), importances


def lgbm_selector(X, y, n_features=10):
    """
    Select features using LightGBM importance.
    """
    if not _HAS_LGB:
        raise RuntimeError("LightGBM not available in this environment")
    model = lgb.LGBMRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importances.index[:min(n_features, len(importances))].tolist(), importances


# =========================
# Ensemble Voting
# =========================

def auto_feature_selector(method_outputs, top_n=20):
    """
    Ensemble method: count votes across different selectors.
    """
    votes = {}
    for name, out in method_outputs.items():
        feats = out[0] if isinstance(out, tuple) else out
        for f in feats:
            votes[f] = votes.get(f, 0) + 1
    ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], votes


# =========================
# Wrapper for Any Dataset
# =========================

def run_feature_selector(df, target, top_n=20):
    """
    Run the full AutoFeatureSelector pipeline on any dataset.
    """
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric.columns and target in df.columns:
        numeric[target] = pd.to_numeric(df[target], errors="coerce")

    X = numeric.drop(columns=[target]).fillna(numeric.median())
    y = numeric[target].fillna(numeric[target].median())

    nunique = X.nunique()
    X = X.drop(columns=nunique[nunique <= 1].index.tolist(), errors="ignore")

    pearson_feats, _ = pearson_selector(X, y, threshold=0.15)
    chi2_feats = chi2_selector(X, y, k=10)
    rfe_feats = rfe_selector(X, y, n_features=10)
    lasso_feats, _ = lasso_selector(X, y)
    rf_feats, _ = rf_selector(X, y, n_features=10)
    if _HAS_LGB:
        lgbm_feats, _ = lgbm_selector(X, y, n_features=10)
    else:
        lgbm_feats = []

    methods = {
        "Pearson": pearson_feats,
        "Chi2": chi2_feats,
        "RFE": rfe_feats,
        "Lasso": lasso_feats,
        "RandomForest": rf_feats,
        "LightGBM": lgbm_feats
    }

    ranked, votes = auto_feature_selector(methods, top_n=top_n)
    return ranked, votes


# =========================
# Utility: Save Results
# =========================

def save_results(ranked, filename="auto_selected_features.csv"):
    """
    Save ranked features to CSV.
    """
    pd.DataFrame(ranked, columns=["feature","votes"]).to_csv(filename, index=False)
    print(f"✅ Saved feature ranking to {filename}")

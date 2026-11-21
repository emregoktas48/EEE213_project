"""EEE213 Week 10 method presentation: Random forest regression on the power-plant data.

Run from the repository folder:  python code/rf.py
Split (course standard): first 7654 rows training, last 1914 rows test.
Hyperparameters chosen by 5-fold cross-validation on the training rows only;
the test set is used once at the end.
"""
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

DATA = Path(__file__).with_name("ccpp.csv")
SEED = 0
FEATURES = ["AT", "V", "AP", "RH"]

d = np.genfromtxt(DATA, delimiter=",", names=True)
X = np.column_stack([d[c] for c in FEATURES])
y = d["PE"]
X_tr, X_te, y_tr, y_te = X[:7654], X[7654:], y[:7654], y[7654:]
mae = lambda a, b: np.mean(np.abs(a - b))
cv = KFold(5, shuffle=True, random_state=SEED)

print(f"[test] baseline (training mean {y_tr.mean():.2f} MW): MAE {mae(y_te, y_tr.mean()):.2f} MW")

# Effect of the number of trees (validation = 5-fold CV inside training rows)
print("\n[validation] number of trees -> CV MAE (MW)")
for n in (1, 10, 50, 100, 300):
    m = RandomForestRegressor(n_estimators=n, random_state=SEED, n_jobs=-1)
    s = -cross_val_score(m, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()
    print(f"  n_estimators={n:4d}: {s:.2f}")

# Effect of tree depth
print("\n[validation] max_depth -> CV MAE (MW)")
for depth in (2, 5, 10, None):
    m = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=SEED, n_jobs=-1)
    s = -cross_val_score(m, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()
    print(f"  max_depth={str(depth):>4}: {s:.2f}")

# Final choice by CV, then one test evaluation
grid = GridSearchCV(
    RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    {"max_depth": [10, 20, None], "min_samples_leaf": [1, 3, 5]},
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_tr, y_tr)
best = grid.best_estimator_
print(f"\n[validation] chosen: n_estimators=300, {grid.best_params_}, CV MAE {-grid.best_score_:.2f} MW")
print(f"[training]   MAE {mae(y_tr, best.predict(X_tr)):.2f} MW")
print(f"[test]       MAE {mae(y_te, best.predict(X_te)):.2f} MW")

print("\nFeature importance (mean decrease in impurity):")
for f, imp in sorted(zip(FEATURES, best.feature_importances_), key=lambda t: -t[1]):
    print(f"  {f}: {imp:.3f}")

# Out-of-range check: a forest cannot predict outside the training PE range
x_hot = np.array([[45.0, 75.0, 1005.0, 40.0]])   # 45 °C, hotter than any recorded hour
print(f"\nExtrapolation check: AT = 45 °C -> predicted {best.predict(x_hot)[0]:.2f} MW "
      f"(training PE minimum {y_tr.min():.2f} MW)")

"""EEE213 Week 09 lab: power-plant data checks, baseline and straight-line models.

Run from the repository folder:  python lab/w09.py
Data: Tüfekci and Kaya, Combined Cycle Power Plant, UCI ML Repository,
doi:10.24432/C5002N, CC BY 4.0. File: ccpp.csv (header line + 9568 rows).
"""
from pathlib import Path
import numpy as np

DATA = Path(__file__).with_name("ccpp.csv")
N_TRAIN = 7654          # first 7654 rows training, last 1914 test (course split)

data = np.genfromtxt(DATA, delimiter=",", names=True)
cols = data.dtype.names                      # ('AT', 'V', 'AP', 'RH', 'PE')
X = np.column_stack([data[c] for c in cols])
units = {"AT": "°C", "V": "cm Hg", "AP": "mbar", "RH": "%", "PE": "MW"}

# ---- 3.1 size and completeness ----
print("== 3.1 Size and completeness ==")
print(f"rows: {X.shape[0]}, columns: {X.shape[1]}")
print("columns:", ", ".join(f"{c} ({units[c]})" for c in cols))
print("missing values:", int(np.isnan(X).sum()))
_, counts = np.unique(X, axis=0, return_counts=True)
print("exact duplicate rows (extra copies):", int((counts - 1).sum()))

# ---- 3.2 ranges ----
print("\n== 3.2 Minimum and maximum ==")
for i, c in enumerate(cols):
    print(f"{c}: {X[:, i].min():.2f} - {X[:, i].max():.2f} {units[c]}")

# ---- 3.3 split ----
train, test = X[:N_TRAIN], X[N_TRAIN:]
print(f"\n== 3.3 Split (all rows kept) ==\ntraining rows: {len(train)}, test rows: {len(test)}")
uniq = np.unique(X, axis=0)
print(f"(separate experiment) rows after removing duplicates: {len(uniq)}")

# ---- 4.1 summary statistics (all rows) ----
print("\n== 4.1 Summary statistics (all 9568 rows) ==")
for c in ("AT", "PE"):
    v = data[c]
    print(f"{c}: mean {v.mean():.2f}, median {np.median(v):.2f}, std {v.std(ddof=1):.2f} {units[c]}")

# ---- 4.2 correlation with PE ----
print("\n== 4.2 Correlation with PE (all rows) ==")
r = {c: np.corrcoef(data[c], data["PE"])[0, 1] for c in cols[:-1]}
for c in r:
    print(f"r({c}, PE) = {r[c]:.3f}")
best = max(r, key=lambda c: abs(r[c]))
print(f"strongest: {best} (r = {r[best]:.3f})")

# ---- 5.1 baseline ----
y_tr, y_te = train[:, 4], test[:, 4]
mean_tr = y_tr.mean()
mae = lambda y, p: np.mean(np.abs(y - p))
print("\n== 5.1 Baseline ==")
print(f"training mean of PE: {mean_tr:.2f} MW")
print(f"baseline test MAE: {mae(y_te, mean_tr):.2f} MW")

# ---- 5.2 straight lines ----
print("\n== 5.2 Straight lines PE = a*x + b (fit on training rows) ==")
lines = {}
for i, c in enumerate(cols[:-1]):
    a, b = np.polyfit(train[:, i], y_tr, 1)
    lines[c] = (a, b)
    print(f"{c}: a = {a:.4f} MW/{units[c]}, b = {b:.2f} MW, "
          f"test MAE = {mae(y_te, a * test[:, i] + b):.2f} MW")

# ---- Separate experiments: cleaned data (same position-based split) ----
def experiment(keep, label):
    idx = np.arange(len(X))
    tr, te = X[keep & (idx < N_TRAIN)], X[keep & (idx >= N_TRAIN)]
    m = tr[:, 4].mean()
    print(f"\n== Separate experiment: {label} ==")
    print(f"training rows: {len(tr)}, test rows: {len(te)}")
    print(f"baseline: training mean {m:.2f} MW, test MAE {mae(te[:, 4], m):.2f} MW")
    for c in ("AT", "V"):
        j = cols.index(c)
        a_, b_ = np.polyfit(tr[:, j], tr[:, 4], 1)
        print(f"line from {c}: a = {a_:.4f}, b = {b_:.2f}, test MAE = {mae(te[:, 4], a_ * te[:, j] + b_):.2f} MW")

rh_ok = X[:, cols.index("RH")] <= 100.0
print(f"\nrows with RH > 100 % (physically impossible): {int((~rh_ok).sum())} "
      f"({int((~rh_ok[:N_TRAIN]).sum())} in training part, {int((~rh_ok[N_TRAIN:]).sum())} in test part)")
experiment(rh_ok, "rows with RH > 100 % removed")
_, first = np.unique(X, axis=0, return_index=True)
not_dup = np.zeros(len(X), bool); not_dup[first] = True
experiment(not_dup, "exact duplicates removed (first copy kept)")
experiment(rh_ok & not_dup, "RH > 100 % and duplicates removed")

# ---- 5.5 one prediction by hand (first test row) ----
FEATURE = "V"
i = cols.index(FEATURE)
a, b = lines[FEATURE]
row = N_TRAIN            # index of first test row
x, y = X[row, i], X[row, 4]
p = a * x + b
print("\n== 5.5 One prediction checked by hand ==")
print(f"file line: {row + 2} (header is line 1)")
print(f"{FEATURE} = {x} {units[FEATURE]}; predicted = {a:.4f}*{x} + {b:.2f} = {p:.2f} MW; "
      f"true = {y} MW; absolute error = {abs(y - p):.2f} MW")

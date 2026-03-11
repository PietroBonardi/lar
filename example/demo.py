import numpy as np
from lar import LiftAreaRatio

np.random.seed(42)
N = 2000
y = np.random.binomial(1, 0.10, N)

# good model: meaningful overlap between bads and goods
scores_good = np.clip(
    y * np.random.uniform(50, 100, N) +
    (1 - y) * np.random.uniform(0, 70, N), 0, 100)

# weak model: heavy overlap, low discrimination
scores_weak = np.clip(
    y * np.random.uniform(40, 90, N) +
    (1 - y) * np.random.uniform(10, 85, N), 0, 100)

lar = LiftAreaRatio(n_percentiles=100)

print("\n── Good Model ──")
result_good = lar.fit(scores_good, y)
lar.summary(result_good)
lar.plot(result_good, title="Good Model — Lift Area Ratio")

print("\n── Weak Model ──")
result_weak = lar.fit(scores_weak, y)
lar.summary(result_weak)
lar.plot(result_weak, title="Weak Model — Lift Area Ratio")
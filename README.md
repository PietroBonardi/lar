# LAR — Lift Area Ratio

A population-independent metric for binary classification model discrimination, designed as a more interpretable alternative to ROC-AUC.

---

## Why not just use AUC?

AUC averages performance across all thresholds equally — including thresholds you'll never use in production. Two models can have the same AUC but behave very differently at your actual operating point.

LAR answers a more practical question:

> *"How much better than random is my model at concentrating positives at the top of the ranking?"*

And unlike AUC, LAR is **population independent** — it gives comparable scores across datasets with different event rates.

---

## How it works

LAR is based on the **Event Rate Ratio (ERR)** — at a given percentile, how much more concentrated are positives compared to the overall population:

$$ERR = \frac{b/n}{B/N}$$

Where:
- `N` = total observations, `B` = total positives
- `n` = observations up to percentile, `b` = positives found up to percentile

Plotting ERR across all percentiles gives three reference curves:

- 🔵 **Oracle** — perfect model, all positives ranked first. ERR starts at `N/B` and drops sharply once all positives are found
- ⚫ **Your model** — sits between oracle and random
- 🔴 **Random** — flat line at ERR = 1, no discrimination

LAR is the ratio of the areas above the random line:

$$LAR = \frac{A}{O} = \frac{\text{area under model curve}}{\text{area under oracle curve}}$$

| LAR value | Meaning |
|-----------|---------|
| 0 | model has no skill (= random) |
| 1 | perfect model (= oracle) |
| 0.83 | model captures 83% of perfect discrimination |

Since both areas are scaled by the same oracle, the population event rate cancels out — making LAR directly comparable across different datasets and models.

---

## Installation

```bash
pip install lar
```

Or install from source:

```bash
git clone https://github.com/yourname/lar.git
cd lar
pip install -e .
```

---

## Quick start

```python
import numpy as np
from lar import LiftAreaRatio

# your model scores and true labels
scores = model.predict_proba(X)[:, 1]
y = df["default_flag"]

# compute LAR
lar = LiftAreaRatio(n_percentiles=100)
result = lar.fit(scores, y)

# print summary
lar.summary(result)
# =============================================
#   Lift Area Ratio — Summary
# =============================================
#   N (observations) : 2000
#   B (positives)    : 197
#   Event rate (ER)  : 9.85%
#   Max ERR (oracle) : 10.15
# ---------------------------------------------
#   Area model       : 192.66
#   Area oracle      : 231.72
#   LAR  = A / O     : 0.8314
# ---------------------------------------------
#   Interpretation   : model achieves 83.1% of perfect discrimination
# =============================================

# plot lift curves
lar.plot(result, title="My Model")
```

---

## API

### `LiftAreaRatio(n_percentiles=100)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_percentiles` | int | 100 | number of quantile bins |

### `.fit(scores, y) → LARResult`

| Parameter | Type | Description |
|-----------|------|-------------|
| `scores` | array-like | model risk scores (higher = more likely positive) |
| `y` | array-like | binary labels (1 = positive, 0 = negative) |

### `.summary(result)`
Prints a formatted summary of the LAR result.

### `.plot(result, title, figsize)`
Plots oracle, model and random lift curves with shaded areas A and O.

### `LARResult` fields

| Field | Description |
|-------|-------------|
| `lar` | final LAR score |
| `area_model` | area under model lift curve |
| `area_oracle` | area under oracle lift curve |
| `N` | total observations |
| `B` | total positives |
| `event_rate` | B / N |
| `lift_model` | DataFrame with percentile and ERR per bin |
| `lift_oracle` | DataFrame with percentile and ERR for oracle |

---

## Requirements

- Python >= 3.8
- numpy
- pandas
- matplotlib
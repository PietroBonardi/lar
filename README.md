# LAR // Lift Area Ratio

A population-independent discrimination metric — interpretable alternative to ROC-AUC



## [01] - why not AUC?

AUC averages performance across all thresholds equally. Two models can have the same AUC but behave very differently at your actual operating point.

```
// **LAR asks**
// how much better than random is my model at concentrating positives at the top of the ranking?

```

---

## [02] - how it works

LAR is based on **ERR** (Event Rate Ratio) — at a given percentile, how much more concentrated are positives compared to the overall population:

```
ERR = (b / n) / (B / N)

  N := total observations
  B := total positives
  n := observations up to percentile
  b := positives found up to percentile
```

Three reference curves:

```
[oracle]  ERR starts at N/B -> drops once all positives are found  /* upper bound */
[model]   sits between oracle and random
[random]  ERR = 1 flat                                             /* no skill    */
```

LAR := ratio of areas above the random line:

```
LAR = A / O
    = area(model) / area(oracle)
    ∈ [0, 1]
```

| `LAR` | meaning |
|-------|---------|
| `0.00` | no skill |
| `0.83` | 83% of perfect discrimination |
| `1.00` | oracle |

// NOTE: event rate cancels out - directly comparable across datasets

---

## [03] - install

```bash
$ pip install lift-area-ratio
```

---

## [04] - usage

```python
from lar import LiftAreaRatio

lar    = LiftAreaRatio(n_percentiles=100)
result = lar.fit(scores, y)

lar.summary(result)
# ================================================
#   N (observations) : 2000
#   B (positives)    : 197   | ER : 9.85%
#   max ERR (oracle) : 10.15
# ------------------------------------------------
#   area(model)      : 192.66
#   area(oracle)     : 231.72
#   LAR = A/O        : 0.8314  => 83.1% of perfect
# ================================================

lar.plot(result, title="My Model")
```

---

## [05] - API

```
LiftAreaRatio(n_percentiles=100)

  .fit(scores, y)  ->  LARResult
      scores  :: array-like  # higher = more likely positive
      y       :: array-like  # binary labels {0, 1}

  .summary(result)
  .plot(result, title, figsize, save=False, file_name="")
```

**`LARResult`**

```
.lar          # final score
.area_model   # area under model curve
.area_oracle  # area under oracle curve
.N            # total observations
.B            # total positives
.event_rate   # B / N
.lift_model   # DataFrame :: percentile, ERR per bin
.lift_oracle  # DataFrame :: percentile, ERR oracle
```

---

## [06] - requirements

```
python     >= 3.10
numpy      >= 2.2.0
pandas     >= 2.3.0
matplotlib >= 3.10.0
```
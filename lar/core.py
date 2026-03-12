import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class LARResult:
    """Holds the results of a LAR computation."""

    lar: float
    area_model: float
    area_oracle: float
    N: int
    B: int
    event_rate: float
    lift_model: pd.DataFrame
    lift_oracle: pd.DataFrame


class LiftAreaRatio:
    """
    Lift Area Ratio (LAR) — a population-independent measure of model
    discrimination power, defined as:

        LAR = Area under model lift curve / Area under oracle lift curve

    Where:
        - Oracle  = perfect model (all bads ranked first)
        - Random  = flat line at err = 1
        - LAR = 0 → model has no skill (= random)
        - LAR = 1 → model is perfect (= oracle)

    Parameters
    ----------
    n_percentiles : int
        Number of quantile bins (default: 100).

    Example
    -------
    >>> lar = LiftAreaRatio()
    >>> result = lar.fit(scores, labels)
    >>> print(result.lar)
    >>> lar.plot(result)
    """

    def __init__(self, n_percentiles: int = 100):
        self.n_percentiles = n_percentiles

    def fit(self, scores, labels) -> LARResult:
        """
        Compute LAR given scores and binary labels.

        Parameters
        ----------
        scores : array-like  — model risk scores (higher = riskier)
        labels   : array-like  — binary labels (1 = bad, 0 = good)

        Returns
        -------
        LARResult dataclass
        """
        scores, labels = self._to_series(scores, labels)
        N = len(scores)
        B = int(labels.sum())
        ER = B / N

        df = pd.DataFrame(
            {"scores": scores.astype(float), "labels": labels.astype(int), "obs": 1}
        )

        # --- bin into percentiles ---
        cuts, labels = pd.qcut(
            df.scores, self.n_percentiles, retbins=True, duplicates="drop"
        )
        actual_bins = len(labels) - 1
        if actual_bins != self.n_percentiles:
            warnings.warn(
                f"Duplicate edges detected — using {actual_bins} bins "
                f"instead of {self.n_percentiles}."
            )
        df["bin"] = cuts

        # --- model lift (real scores) ---
        lift_model = self._compute_lift(df, N, ER)

        # --- oracle lift (all labels ranked first, independent of model scores) ---
        # Assign artificial scores: top B positions = labels, rest = goods
        oracle_scores = pd.Series([float(N - i) for i in range(N)])
        oracle_target = pd.Series([1] * B + [0] * (N - B))  # 1 * num_bad + 0 * num_good
        df_oracle = pd.DataFrame(
            {"scores": oracle_scores, "labels": oracle_target, "obs": 1}
        )
        df_oracle["bin"] = pd.qcut(
            df_oracle.scores, self.n_percentiles, duplicates="drop"
        )
        lift_oracle = self._compute_lift(df_oracle, N, ER)

        # --- compute the area under ERR curve ---
        # Compute the lift
        area_model = self._trapezoid_area(lift_model)
        area_oracle = self._trapezoid_area(lift_oracle)
        lar = round(area_model / area_oracle, 4)

        return LARResult(
            lar=lar,
            area_model=round(area_model, 4),
            area_oracle=round(area_oracle, 4),
            N=N,
            B=B,
            event_rate=round(ER, 4),
            lift_model=lift_model,
            lift_oracle=lift_oracle,
        )

    def plot(
        self,
        result: LARResult,
        title: str = "Lift Area Ratio",
        figsize: tuple = (9, 5),
        save: bool = False,
        folder_path: str = "",
    ) -> None:
        """
        Plot oracle, model, and random lift curves with shaded areas.

        Parameters
        ----------
        result  : LARResult from .fit()
        title   : plot title
        figsize : figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        pct_o = result.lift_oracle.percentile
        pct_m = result.lift_model.percentile
        eer_o = result.lift_oracle.err
        eer_m = result.lift_model.err
        _COLOR_ORACLE = "#A0F2EA"
        _COLOR_ORACLE_LINE = "#71A39F"
        _COLOR_MODEL = "#2c6fad"
        _COLOR_MODEL_LINE = "#3362C4"

        # shaded areas
        ax.fill_between(
            pct_o, 1, eer_o, alpha=0.15, color=_COLOR_ORACLE, label="_nolegend_"
        )
        ax.fill_between(
            pct_m, 1, eer_m, alpha=0.30, color=_COLOR_MODEL, label="_nolegend_"
        )

        # annotate O and A
        ax.text(
            pct_o.iloc[len(pct_o) // 4],
            eer_o.iloc[len(eer_o) // 4] * 0.6,
            "O",
            fontsize=20,
            color=_COLOR_ORACLE_LINE,
            alpha=0.6,
            ha="center",
        )
        ax.text(
            pct_m.iloc[len(pct_m) // 3],
            eer_m.iloc[len(eer_m) // 3] * 0.55,
            "A",
            fontsize=20,
            color=_COLOR_MODEL_LINE,
            alpha=0.9,
            ha="center",
        )

        # lines
        ax.plot(
            pct_o,
            eer_o,
            color=_COLOR_ORACLE_LINE,
            linewidth=2,
            label=f"Oracle (area={result.area_oracle})",
        )
        ax.plot(
            pct_m,
            eer_m,
            color=_COLOR_MODEL_LINE,
            linewidth=2,
            label=f"Model  (area={result.area_model})",
        )
        ax.axhline(
            1, color="red", linewidth=1.0, linestyle="--", label="Random (err=1)"
        )

        # labels & formatting
        ax.set_xlabel("Percentile (%)", color="black")
        ax.set_ylabel("Event Rate Ratio (err)", color="black")
        ax.set_title(
            f"{title}\nLAR = A/O = {result.lar}  |  "
            f"N={result.N}, B={result.B}, ER={result.event_rate:.1%}",
            color="black",
            fontsize=11,
        )
        ax.tick_params(colors="black")
        ax.legend(
            facecolor="white",
            labelcolor="black",
            fontsize=9,
            framealpha=1,
            edgecolor="black",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

        plt.tight_layout()
        if save:
            plt.savefig(
                "{save_path}/lar_plot_white.png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.show()

    def summary(self, result: LARResult) -> None:
        """Print a readable summary of the LAR result."""
        print("=" * 45)
        print("  Lift Area Ratio — Summary")
        print("=" * 45)
        print(f"  N (observations) : {result.N}")
        print(f"  B (labels)         : {result.B}")
        print(f"  Event rate (ER)  : {result.event_rate:.2%}")
        print(f"  Max err (oracle) : {1/result.event_rate:.2f}")
        print("-" * 45)
        print(f"  Area model       : {result.area_model}")
        print(f"  Area oracle      : {result.area_oracle}")
        print(f"  LAR  = A / O     : {result.lar}")
        print("-" * 45)
        print(
            f"  Interpretation   : model achieves "
            f"{result.lar*100:.1f}% of perfect discrimination"
        )
        print("=" * 45)

    @staticmethod
    def _to_series(scores, labels):
        if not isinstance(scores, pd.Series):
            scores = pd.Series(scores)
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels)
        return scores, labels

    @staticmethod
    def _compute_lift(df: pd.DataFrame, N: int, ER: float) -> pd.DataFrame:
        """Sort by scores descending, compute cumulative err at each percentile."""
        dt = df.sort_values("scores", ascending=False).copy()
        dt["cum_bads"] = dt["labels"].cumsum()
        dt["cum_obs"] = dt["obs"].cumsum()
        dt["percentile"] = (dt["cum_obs"] / N * 100).round(2)
        dt["er"] = (dt["cum_bads"] / dt["cum_obs"]).round(4)
        dt["err"] = (dt["er"] / ER).round(4)

        # downsample to percentile-level (keep last row per percentile bucket)
        dt["pct_bucket"] = (dt["percentile"] * 1).round(0)
        dt = dt.groupby("pct_bucket", as_index=False).last()
        dt = dt.sort_values("pct_bucket")

        # prepend origin
        first_err = dt["err"].iloc[0]
        origin = pd.DataFrame({"percentile": [0], "er": [ER], "err": [first_err]})
        dt = pd.concat([origin, dt[["percentile", "er", "err"]]], ignore_index=True)

        return dt

    @staticmethod
    def _trapezoid_area(lift: pd.DataFrame) -> float:
        """Area between lift curve and random line (err=1) using trapezoid rule."""
        pct = lift["percentile"].values
        err = lift["err"].values - 1  # subtract random baseline
        return float(np.trapezoid(err, pct))

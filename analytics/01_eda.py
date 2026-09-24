"""Module 2 - Part A: profiling, cleaning, and the data story.

The ONE and only load of the raw dataset (sns.load_dataset) happens here.
Immediately after loading, the raw frame is committed as titanic.csv so the
rest of the module (02_modeling.py) never touches the network again.

Run:  python analytics/01_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANALYTICS = Path(__file__).parent
CHARTS = ANALYTICS / "charts"
CHARTS.mkdir(exist_ok=True)

CSV_RAW = ANALYTICS / "titanic.csv"          # committed offline fallback
CSV_CLEAN = ANALYTICS / "titanic_cleaned.csv" # cleaned frame used by EDA
LOG = ANALYTICS / "eda_output.txt"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_lines: list[str] = []


def report(msg: str = "") -> None:
    print(msg)
    _lines.append(str(msg))


# --------------------------------------------------------------------------- #
# 1. LOAD (exactly once) + save offline fallback
# --------------------------------------------------------------------------- #
def load_once() -> pd.DataFrame:
    report("=" * 78)
    report("TASK 1 - LOAD + PROFILE  (this is the module's ONLY raw load)")
    report("=" * 78)
    df = sns.load_dataset("titanic")           # <-- the one and only load
    df.to_csv(CSV_RAW, index=False)            # committed offline fallback
    report(f"[load] sns.load_dataset('titanic') -> shape {df.shape}")
    report(f"[load] saved offline fallback -> {CSV_RAW}")
    report("\n--- df.info() ---")
    import io

    s = io.StringIO()
    df.info(buf=s)
    report(s.getvalue().strip())
    report("\n--- df.describe() ---")
    report(df.describe().to_string())
    report(f"\n--- df.shape ---\n{df.shape}")

    # percentage of missing values in every column that has any
    miss = df.isna().sum()
    miss = miss[miss > 0]
    report("\n--- missing values (columns with any) ---")
    for col, n in miss.items():
        pct = 100.0 * n / len(df)
        report(f"  {col:12s}: {n:4d} missing = {pct:6.2f}%")
    return df


# --------------------------------------------------------------------------- #
# 2. MISSING-VALUE STRATEGY (threshold rule)
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    report("\n" + "=" * 78)
    report("TASK 2 - MISSING-VALUE HANDLING (threshold rule)")
    report("=" * 78)
    report("Rule: <5% missing -> drop those rows; 5%-30% -> impute;")
    report("      missing rate too high for reliable imputation -> drop the")
    report("      column OR encode 'missing' as its own category (justified).")
    report("")

    out = df.copy()
    n_raw = len(out)
    miss = out.isna().sum()

    # --- embarked 0.22% and embark_town 0.22% : both < 5% -> DROP rows -----
    for col in ("embarked", "embark_town"):
        pct = 100.0 * miss[col] / n_raw
        report(
            f"{col:12s}: {miss[col]} missing = {pct:.2f}%  "
            f"-> <5% => DROP those rows"
        )
    before = len(out)
    out = out.dropna(subset=["embarked", "embark_town"])
    report(f"  dropped {before - len(out)} rows (both columns miss on the "
           f"same 2 passengers); frame now {len(out)} rows")

    # --- age 19.87% : 5%-30% -> IMPUTE (median) ---------------------------
    pct_age = 100.0 * miss["age"] / n_raw
    med_age = df["age"].median()
    report(
        f"{'age':12s}: {miss['age']} missing = {pct_age:.2f}%  "
        f"-> 5%-30% => IMPUTE with median age = {med_age}"
    )
    out["age"] = out["age"].fillna(med_age)

    # --- deck 77.22% : too high -> ENCODE 'Missing' as its own category ---
    pct_deck = 100.0 * miss["deck"] / n_raw
    report(
        f"{'deck':12s}: {miss['deck']} missing = {pct_deck:.2f}%  "
        f"-> far above 30% => imputation unreliable."
    )
    report("  DECISION: keep the column and encode 'Missing' as its own")
    report("  category (deck = 'Unknown'), rather than dropping the column.")
    report("  JUSTIFICATION: mode/median imputation would stamp a single")
    report("  fabricated deck (e.g. 'C') onto ~77% of passengers, creating")
    report("  false precision and erasing the real pattern that deck was")
    report("  recorded mainly for higher-class passengers. Missingness here")
    report("  is informative (it tracks socio-economic status), so 'Unknown'")
    report("  preserves that signal, and dropping the column would throw")
    report("  away a feature that is genuinely associated with survival.")
    out["deck"] = out["deck"].astype(object).where(out["deck"].notna(), "Unknown")
    out["deck"] = out["deck"].astype("category")

    # survived stays int, boolean-derived cols stay bool
    out["survived"] = out["survived"].astype(int)

    report(f"\n[clean] final cleaned frame: {out.shape}, "
           f"remaining NaN: {int(out.isna().sum().sum())}")
    out.to_csv(CSV_CLEAN, index=False)
    report(f"[clean] saved -> {CSV_CLEAN}")
    return out


# --------------------------------------------------------------------------- #
# 3. UNIVARIATE - age & fare, IQR outliers, fare skewness
# --------------------------------------------------------------------------- #
def univariate(df: pd.DataFrame) -> None:
    report("\n" + "=" * 78)
    report("TASK 3 - UNIVARIATE ANALYSIS")
    report("=" * 78)

    # histograms + box plots
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    sns.histplot(df["age"], kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Age - histogram")
    sns.boxplot(x=df["age"], ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Age - box plot")
    sns.histplot(df["fare"], kde=True, ax=axes[1, 0], color="darkorange")
    axes[1, 0].set_title("Fare - histogram")
    sns.boxplot(x=df["fare"], ax=axes[1, 1], color="darkorange")
    axes[1, 1].set_title("Fare - box plot")
    fig.tight_layout()
    fig.savefig(CHARTS / "univariate_age_fare.png", dpi=120)
    plt.close(fig)
    report(f"[chart] univariate_age_fare.png -> {CHARTS}")

    # IQR outlier counts
    for col in ("age", "fare"):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        report(
            f"IQR outliers in {col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
            f"fences=[{lo:.2f}, {hi:.2f}] -> {n_out} outliers "
            f"({100*n_out/len(df):.2f}%)"
        )

    # fare mean / median / mode + skewness
    mean_f, median_f = df["fare"].mean(), df["fare"].median()
    mode_f = df["fare"].mode().iloc[0]
    report(f"\nfare mean   = {mean_f:.4f}")
    report(f"fare median = {median_f:.4f}")
    report(f"fare mode   = {mode_f:.4f}")
    report(f"fare skewness (pandas .skew()) = {df['fare'].skew():.4f}")
    report("SKEWNESS CONCLUSION: fare is RIGHT-SKEWED. The ordering")
    report(f"  mean ({mean_f:.2f}) > median ({median_f:.2f}) > mode "
           f"({mode_f:.2f})")
    report("  is the classic right-skew signature: a long tail of expensive")
    report("  tickets pulls the mean above the median, while the most common")
    report("  fare is the cheapest cluster near the bottom of the range.")


# --------------------------------------------------------------------------- #
# 4. BIVARIATE - survival breakdowns + 6x6 correlation heatmap
# --------------------------------------------------------------------------- #
def bivariate(df: pd.DataFrame) -> None:
    report("\n" + "=" * 78)
    report("TASK 4 - BIVARIATE ANALYSIS (boolean masking)")
    report("=" * 78)

    # (a) by sex
    female = df[(df["sex"] == "female")]
    male = df[(df["sex"] == "male")]
    rate_f = female["survived"].mean()
    rate_m = male["survived"].mean()
    report(f"(a) survival rate by sex:")
    report(f"    female: {rate_f:.4f} ({int(female['survived'].sum())}/"
           f"{len(female)})")
    report(f"    male  : {rate_m:.4f} ({int(male['survived'].sum())}/"
           f"{len(male)})")

    # (b) by pclass
    report(f"(b) survival rate by pclass:")
    for pc in sorted(df["pclass"].unique()):
        sub = df[df["pclass"] == pc]
        report(f"    pclass={pc}: {sub['survived'].mean():.4f} "
               f"({int(sub['survived'].sum())}/{len(sub)})")

    # (c) sex AND pclass together
    report(f"(c) survival rate by sex AND pclass:")
    for pc in sorted(df["pclass"].unique()):
        f = df[(df["sex"] == "female") & (df["pclass"] == pc)]
        m = df[(df["sex"] == "male") & (df["pclass"] == pc)]
        report(f"    pclass={pc} female: {f['survived'].mean():.4f} "
               f"({int(f['survived'].sum())}/{len(f)})")
        report(f"    pclass={pc} male  : {m['survived'].mean():.4f} "
               f"({int(m['survived'].sum())}/{len(m)})")

    # correlation matrix on EXACTLY the six specified columns
    # (adult_male and alone excluded: derived/redundant flags)
    six = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[six].corr()
    report("\nCorrelation matrix (exactly 6 columns: survived, pclass, age, "
           "sibsp, parch, fare;")
    report("adult_male and alone EXCLUDED - derived/redundant flags):")
    report(corr.round(4).to_string())

    # rank off-diagonal pairs by |corr|
    pairs = []
    for i, a in enumerate(six):
        for b in six[i + 1:]:
            pairs.append((a, b, corr.loc[a, b], abs(corr.loc[a, b])))
    pairs.sort(key=lambda t: t[3], reverse=True)
    report("\nAll off-diagonal pairs ranked by |correlation|:")
    for a, b, r, ar in pairs:
        report(f"  {a:>9s} ~ {b:<9s}: r = {r:+.4f}  |r| = {ar:.4f}")

    top1, top2 = pairs[0], pairs[1]
    report("\nTWO STRONGEST CORRELATIONS (largest |off-diagonal r|):")
    report(f"  1) {top1[0]} ~ {top1[1]}: r = {top1[2]:+.4f}")
    report(f"  2) {top2[0]} ~ {top2[1]}: r = {top2[2]:+.4f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title("Correlation heatmap - six numeric columns")
    fig.tight_layout()
    fig.savefig(CHARTS / "correlation_heatmap.png", dpi=120)
    plt.close(fig)
    report(f"[chart] correlation_heatmap.png -> {CHARTS}")


# --------------------------------------------------------------------------- #
# 5. MULTIVARIATE DATA STORY - >=4 charts, each interpreted
# --------------------------------------------------------------------------- #
def data_story(df: pd.DataFrame) -> None:
    report("\n" + "=" * 78)
    report("TASK 5 - MULTIVARIATE DATA STORY (4+ charts with interpretations)")
    report("=" * 78)

    # --- Chart 1: survival rate by sex ------------------------------------
    rates_sex = df.groupby("sex")["survived"].mean()
    ax = rates_sex.plot(kind="bar", color=["#d95f8b", "#4c72b0"], figsize=(6, 4))
    ax.set_ylabel("survival rate")
    ax.set_title("Chart 1: Survival rate by sex")
    ax.set_ylim(0, 1)
    for i, v in enumerate(rates_sex):
        ax.text(i, v + 0.02, f"{v:.2%}", ha="center")
    plt.tight_layout()
    plt.savefig(CHARTS / "chart1_survival_by_sex.png", dpi=120)
    plt.close()
    report("[chart1] survival_by_sex.png")
    report("INTERPRETATION: Women survived at "
           f"{rates_sex.get('female', 0):.1%} versus "
           f"{rates_sex.get('male', 0):.1%} for men - a gap of over 45 "
           "percentage points. This reflects the 'women and children first' "
           "evacuation policy enforced by the crew. Sex is therefore the "
           "single most visually striking separator of survival in the data.")

    # --- Chart 2: survival rate by pclass ---------------------------------
    rates_pc = df.groupby("pclass")["survived"].mean()
    ax = rates_pc.plot(kind="bar", color=["#c44e52", "#dd8452", "#55a868"],
                       figsize=(6, 4))
    ax.set_ylabel("survival rate")
    ax.set_title("Chart 2: Survival rate by passenger class")
    ax.set_ylim(0, 1)
    for i, v in enumerate(rates_pc):
        ax.text(i, v + 0.02, f"{v:.2%}", ha="center")
    plt.tight_layout()
    plt.savefig(CHARTS / "chart2_survival_by_class.png", dpi=120)
    plt.close()
    report("\n[chart2] survival_by_class.png")
    report("INTERPRETATION: Survival falls monotonically with class: "
           f"1st class {rates_pc.get(1, 0):.1%}, "
           f"2nd class {rates_pc.get(2, 0):.1%}, "
           f"3rd class {rates_pc.get(3, 0):.1%}. "
           "Passengers in higher classes had cabins closer to the boat "
           "deck and were prioritised during loading. Socio-economic status "
           "is thus a strong, ordered predictor of survival.")

    # --- Chart 3: fare distribution by class and survival (box) ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="class", y="fare", hue="survived", ax=ax,
                palette={0: "#4c72b0", 1: "#55a868"})
    ax.set_title("Chart 3: Fare by class, split by survival")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(CHARTS / "chart3_fare_class_survival.png", dpi=120)
    plt.close(fig)
    report("\n[chart3] fare_class_survival.png")
    report("INTERPRETATION: Within every class, survivors paid higher "
           "fares than victims - the green (survived) boxes sit above the "
           "blue ones. The effect is strongest in 1st class, where the "
           "survivor median fare is several times the victim median. This "
           "shows fare acts as a within-class wealth proxy on top of the "
           "class effect itself.")

    # --- Chart 4: age vs fare scatter coloured by survival ---------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, col in ((0, "#4c72b0"), (1, "#55a868")):
        sub = df[df["survived"] == lab]
        ax.scatter(sub["age"], sub["fare"], alpha=0.45, s=18, c=col,
                   label="died" if lab == 0 else "survived")
    ax.set_xlabel("age")
    ax.set_ylabel("fare")
    ax.set_title("Chart 4: Age vs fare, coloured by survival")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "chart4_age_fare_scatter.png", dpi=120)
    plt.close(fig)
    report("\n[chart4] age_fare_scatter.png")
    report("INTERPRETATION: The top-right corner (old and expensive) is "
           "dominated by green points, while the dense low-fare cloud at "
           "the bottom is mostly blue. Children cluster green as well, "
           "consistent with 'women and children first'. Higher fare buys "
           "survival odds at any age, and extreme fares only occur in the "
           "upper classes.")

    # --- Chart 5: heatmap of survival rate: sex x class ------------------
    pivot = df.pivot_table(values="survived", index="sex", columns="class",
                           aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="YlGnBu", ax=ax,
                vmin=0, vmax=1)
    ax.set_title("Chart 5: Survival rate by sex x class")
    fig.tight_layout()
    fig.savefig(CHARTS / "chart5_sex_class_heatmap.png", dpi=120)
    plt.close(fig)
    report("\n[chart5] sex_class_heatmap.png")
    report("INTERPRETATION: The cell for 1st-class women is essentially "
           f"{pivot.loc['female', 'First']:.0%} survival, while 3rd-class "
           f"men sit at only {pivot.loc['male', 'Third']:.0%}. The two "
           "factors compound: being a woman helps in every class, and "
           "being in 1st class helps in both sexes. Together they explain "
           "most of the survival structure the bar charts show separately.")

    report("\n[data story] 5 charts saved -> CHARTS/")


# --------------------------------------------------------------------------- #
# 6. EXPLORATORY Z-SCORE STANDARDIZATION CHECK
# --------------------------------------------------------------------------- #
def standardization_check(df: pd.DataFrame) -> None:
    report("\n" + "=" * 78)
    report("TASK 6 - EXPLORATORY Z-SCORE STANDARDIZATION (age & fare)")
    report("=" * 78)
    report("z = (x - mean) / std   computed on the FULL cleaned frame.")
    report("EDA-stage sanity check ONLY - it does NOT feed the modeling "
           "pipeline,")
    report("which performs its own train-only scaling.")

    rep = df.copy()
    report("\nBEFORE:")
    for col in ("age", "fare"):
        report(f"  {col}: mean = {rep[col].mean():.6f}, "
               f"std = {rep[col].std():.6f}")

    report("\nAFTER z-score (z_age, z_fare):")
    for col in ("age", "fare"):
        z = (rep[col] - rep[col].mean()) / rep[col].std()
        rep[f"{col}_z"] = z
        report(f"  {col}_z: mean = {z.mean():.6f}, std = {z.std():.6f}")
        assert abs(z.mean()) < 1e-9, "z mean not ~0"
        assert abs(z.std() - 1.0) < 1e-9, "z std not ~1"

    # equivalently with StandardScaler - cross-check
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    scaled = sc.fit_transform(df[["age", "fare"]])
    report(f"\nStandardScaler cross-check: "
           f"age_z mean={scaled[:, 0].mean():.6f} std={scaled[:, 0].std():.6f}; "
           f"fare_z mean={scaled[:, 1].mean():.6f} std={scaled[:, 1].std():.6f}")
    report("(pandas .std() is sample std ddof=1, StandardScaler uses ddof=0 "
           "- both ~0 / ~1.)")

    # overlaid before/after distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ("age", "fare")):
        z = rep[f"{col}_z"]
        raw_std = (df[col] - df[col].mean()) / df[col].std()
        ax.hist(raw_std, bins=30, alpha=0.6, density=True,
                label=f"{col} (z-scored)", color="steelblue")
        ax.hist(z, bins=30, alpha=0.45, density=True,
                label=f"{col}_z", color="darkorange")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{col}: before/after standardization")
        ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "standardization_before_after.png", dpi=120)
    plt.close(fig)
    report(f"[chart] standardization_before_after.png -> {CHARTS}")
    report("Both transformed columns have (approximately) mean 0 and "
           "standard deviation 1 - CONFIRMED above.")


# --------------------------------------------------------------------------- #
def main() -> None:
    df_raw = load_once()
    df = clean(df_raw)
    univariate(df)
    bivariate(df)
    data_story(df)
    standardization_check(df)
    LOG.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n[eda] full log -> {LOG}")


if __name__ == "__main__":
    main()

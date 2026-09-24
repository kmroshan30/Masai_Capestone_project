# Module 2 — Analytics Pipeline (`/analytics`)

One cohesive pipeline: the raw Titanic dataset is loaded **exactly once**
(`01_eda.py`), committed as `titanic.csv`, and every later step — EDA and
modeling — continues from that same data. `02_modeling.py` reads the CSV and
**never** calls `sns.load_dataset` again.

## Install / Run

```bash
pip install seaborn matplotlib scikit-learn pandas imbalanced-learn joblib
python analytics/01_eda.py     # load (once) -> titanic.csv -> cleaning -> EDA
python analytics/02_modeling.py  # reads titanic.csv -> modeling -> joblib
```

| File | What it is |
|---|---|
| `titanic.csv` | **Committed offline fallback** — raw load saved via `df.to_csv("titanic.csv", index=False)`; gradable with `pd.read_csv("titanic.csv")` with no network |
| `titanic_cleaned.csv` | Cleaned frame after the threshold-rule strategies |
| `01_eda.py` / `02_modeling.py` | The two ordered stages |
| `eda_output.txt` / `modeling_output.txt` | Full printed output of both runs |
| `charts/*.png` | All chart artifacts |
| `model_comparison.csv` | Final comparison table |
| `best_pipeline.joblib` | Saved **full pipeline** (preprocessor + estimator) |

---

# Part A — Profiling, cleaning, data story

## Task 1 — Load & profile

- `sns.load_dataset('titanic')` → **shape (891, 15)** — the module's only
  network/cache load; saved immediately as `titanic.csv`.
- `df.info()` / `df.describe()` printed in `eda_output.txt`.

**Missing values (every column that has any, % of 891):**

| Column | Missing | % |
|---|---|---|
| `age` | 177 | **19.87%** |
| `embarked` | 2 | **0.22%** |
| `deck` | 688 | **77.22%** |
| `embark_town` | 2 | **0.22%** |

## Task 2 — Missing-value strategies (threshold rule)

Rule: **<5% → drop rows · 5–30% → impute · too-high missing → drop column or
encode "missing" as its own category (justified).**

| Column | % | Strategy | Justification |
|---|---|---|---|
| `embarked` | 0.22% | **Drop rows** (2 rows) | <5% → rule says drop; both columns miss on the same 2 passengers, so a single `dropna(subset=[...])` removes exactly those 2 rows (891 → 889). |
| `embark_town` | 0.22% | **Drop rows** (same 2 rows) | <5% → drop; redundant with `embarked` anyway. |
| `age` | 19.87% | **Impute with median = 28.0** | In the 5–30% band → impute. Median (not mean) because age's right tail would drag a mean imputation upward; 19.87% is far too much to drop without hurting power. |
| `deck` | 77.22% | **Keep column, encode `"Unknown"` as its own category** | Imputation would be unreliable (>30%): stamping the mode deck onto ~77% of passengers creates false precision and erases the real pattern that deck was recorded mainly for higher-class passengers. Missingness itself is informative (it tracks socio-economic status), so `"Unknown"` preserves that signal; dropping the column would discard a feature genuinely associated with survival. |

Result: cleaned frame **(889, 15), 0 remaining NaN**.

## Task 3 — Univariate (age, fare)

Charts: `charts/univariate_age_fare.png` (histogram + box for each).

**IQR outlier counts** (fences = Q1 − 1.5·IQR, Q3 + 1.5·IQR):

| Column | Q1 | Q3 | IQR | Fences | Outliers |
|---|---|---|---|---|---|
| `age` | 22.00 | 35.00 | 13.00 | [2.50, 54.50] | **65 (7.31%)** |
| `fare` | 7.90 | 31.00 | 23.10 | [−26.76, 65.66] | **114 (12.82%)** |

**Fare mean / median / mode:**

- mean = **32.0967**, median = **14.4542**, mode = **8.0500**
  (pandas skew = +4.80)

**Skewness conclusion:** fare is **right-skewed**, because
**mean (32.10) > median (14.45) > mode (8.05)** — the classic right-skew
ordering. A long tail of expensive tickets pulls the mean above the median,
while the most common fare sits at the cheap bottom of the range.

## Task 4 — Bivariate (boolean masking)

**(a) Survival by sex**
- female: **0.7404** (231/312) · male: **0.1889** (109/577)

**(b) Survival by pclass**
- 1st: **0.6262** (134/214) · 2nd: **0.4728** (87/184) · 3rd: **0.2424** (119/491)

**(c) Survival by sex × pclass**

| | female | male |
|---|---|---|
| 1st | **0.9674** (89/92) | 0.3689 (45/122) |
| 2nd | **0.9211** (70/76) | 0.1574 (17/108) |
| 3rd | 0.5000 (72/144) | 0.1354 (47/347) |

**Correlation matrix — exactly the six specified columns**
(`survived, pclass, age, sibsp, parch, fare`; **`adult_male` and `alone`
excluded** — derived/redundant flags, directly computable from sex/age and
sibsp+parch respectively, not independent measured features):

```
          survived  pclass     age   sibsp   parch    fare
survived    1.0000 -0.3355 -0.0698 -0.0340  0.0832  0.2553
pclass     -0.3355  1.0000 -0.3365  0.0817  0.0168 -0.5482
age        -0.0698 -0.3365  1.0000 -0.2325 -0.1715  0.0937
sibsp      -0.0340  0.0817 -0.2325  1.0000  0.4145  0.1609
parch       0.0832  0.0168 -0.1715  0.4145  1.0000  0.2175
fare        0.2553 -0.5482  0.0937  0.1609  0.2175  1.0000
```

Heatmap: `charts/correlation_heatmap.png`.

**Two strongest correlations** (all off-diagonal pairs ranked by |r|):

1. **`pclass` ~ `fare`: r = −0.5482** — The strongest pair: ticket class and
   fare are nearly the same economic signal, since 1st-class tickets cost far
   more than 3rd-class ones. The negative sign just means higher class
   *number* (3rd) goes with lower fare. It warns us not to treat fare and
   pclass as independent evidence in a linear model — some of fare's
   apparent effect on survival is really the class effect.
2. **`sibsp` ~ `parch`: r = +0.4145** — Passengers travelling with many
   siblings/spouses also tended to bring many parents/children, i.e. family
   size travels together: these are the family-group tickets. They are
   related but not redundant (r well below 1), so both can stay as features;
   their joint signal is "travelling as a family" rather than two separate
   effects.

## Task 5 — Multivariate data story (5 charts, each interpreted)

1. **`chart1_survival_by_sex.png`** — Women survived at **74.0%** versus
   **18.9%** for men, a gap of over 45 percentage points. This reflects the
   "women and children first" evacuation policy enforced by the crew. Sex is
   the single most visually striking separator of survival in the data.
2. **`chart2_survival_by_class.png`** — Survival falls monotonically with
   class: 1st **62.6%**, 2nd **47.3%**, 3rd **24.2%**. Passengers in higher
   classes had cabins closer to the boat deck and were prioritised during
   loading. Socio-economic status is a strong, ordered predictor of
   survival.
3. **`chart3_fare_class_survival.png`** (box: fare by class, split by
   survival) — Within every class, survivors paid higher fares than victims
   — the survivor boxes sit above the victim boxes. The effect is strongest
   in 1st class, where the survivor median fare is several times the victim
   median. Fare therefore acts as a within-class wealth proxy on top of the
   class effect itself.
4. **`chart4_age_fare_scatter.png`** — The top-right corner (old and
   expensive) is dominated by surviving points, while the dense low-fare
   cloud at the bottom is mostly deaths. Children cluster green as well,
   consistent with "women and children first". Higher fare buys survival
   odds at any age, and extreme fares only occur in the upper classes.
5. **`chart5_sex_class_heatmap.png`** (survival rate: sex × class) —
   1st-class women survive at **97%** while 3rd-class men sit at only
   **14%**. The two factors compound: being a woman helps in every class,
   and being in 1st class helps in both sexes. Together they explain most
   of the survival structure the separate bar charts show.

Univariate panel: `univariate_age_fare.png` (histograms + box plots for
age and fare).

## Task 6 — Exploratory z-score standardization (EDA sanity check only)

z = (x − mean)/std on the **full cleaned** DataFrame:

| | before mean | before std | after mean | after std |
|---|---|---|---|---|
| age | 29.315152 | 12.984932 | **0.000000** | **1.000000** |
| fare | 32.096681 | 49.697504 | **0.000000** | **1.000000** |

Cross-checked with `StandardScaler` (age_z mean 0 / std 1, fare_z mean 0 /
std 1; pandas `.std()` uses ddof=1, `StandardScaler` uses ddof=0 — both ≈0/1).
Overlaid before/after plot: `charts/standardization_before_after.png`.
This check does **not** feed the modeling pipeline, which does its own
train-only scaling.

---

# Part B — Predictive modeling (continues from the same data)

## Task 7 — Stratified train/test split (before any preprocessing)

Class balance (full data): **died 549 (61.62%) / survived 342 (38.38%)**.

**Why stratify:** the classes are imbalanced (~38/62), so a plain random
split can put a noticeably different survival ratio in test than in train —
adding variance to every metric and making the three models' scores less
comparable. `stratify=y` guarantees train and test each preserve the ~38/62
split (observed: train 0.3834, test 0.3855). Split: **712 train / 179 test**
(`random_state=42`).

## Task 8 — Preprocessing (fit on train only, transform on test)

Implemented as a `ColumnTransformer` wrapped in a `Pipeline` with the final
estimator, so fit-on-train / transform-on-test is **structurally enforced**
— no step can be refit on test by accident.

- **Imputation:** numeric → `SimpleImputer(median)`, categorical →
  `SimpleImputer(most_frequent)` — both learned from the **train split
  only**. (Different from Task 2 by design: Task 2 dropped the 2
  missing-embarked rows for EDA; here every row is kept and the imputer
  fills them leak-safely.)
- **Encoding:** `OneHotEncoder(sex, embarked, handle_unknown='ignore')`.
- **Scaling:** `StandardScaler` on `pclass, age, sibsp, parch, fare`.
- Test data is only ever `.transform`ed — never fit.

## Task 9 — Three classifiers, same split, full metric suite

| model | confusion matrix [[TN,FP],[FN,TP]] | accuracy | precision | recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | [[98, 12], [23, 46]] | 0.8045 | 0.7931 | 0.6667 | 0.7244 | **0.8437** |
| Decision Tree | [[97, 13], [20, 49]] | **0.8156** | 0.7903 | **0.7101** | **0.7481** | 0.7904 |
| Random Forest | [[98, 12], [21, 48]] | **0.8156** | **0.8000** | 0.6957 | 0.7442 | 0.8300 |

Artifacts: `charts/confusion_matrices.png`, `charts/roc_curves.png`,
**`charts/decision_tree.png`** — the Decision Tree rendered with
`plot_tree`, labeled with **feature names** (from
`preprocessor.get_feature_names_out()`) and **class names**
(`died`/`survived`); top 3 levels rendered for readability, full tree fitted.

## Task 10 — Imbalance handling comparison (Logistic Regression)

Class balance: **not-survived 549 (0.6162) / survived 342 (0.3838)**.

| variant | precision | recall | F1 | accuracy | AUC |
|---|---|---|---|---|---|
| (a) baseline (no handling) | 0.7931 | 0.6667 | 0.7244 | 0.8045 | 0.8437 |
| (b) `class_weight='balanced'` | 0.7297 | 0.7826 | 0.7552 | 0.8045 | 0.8464 |
| (c) SMOTE (train fold only) | 0.7397 | 0.7826 | **0.7606** | **0.8101** | 0.8414 |

SMOTE is applied **only to the training fold** — it lives inside an
`imblearn.Pipeline` step, so resampling happens at `.fit()` time on train
data only and `.predict()` on test never resamples (no leakage).

**Conclusion:** SMOTE wins on F1 (0.7606 vs 0.7552 balanced vs 0.7244
baseline). Both imbalance strategies push the model to recognise more
survivors — recall jumps from 0.6667 → 0.7826 — at a small precision cost,
and F1 improves accordingly. SMOTE edges out class weighting because it
synthesises actual minority examples (giving the boundary more minority
support) rather than only re-weighting the same rows, and it also posts the
best accuracy (0.8101).

## Task 11 — Hyperparameter tuning + OOB

- `GridSearchCV` over `n_estimators ∈ {100,200,300}`,
  `max_depth ∈ {None,5,10,20}`, `max_features ∈ {sqrt,log2}`, cv=5,
  scoring accuracy.
- **Best parameters: `{'max_depth': 5, 'max_features': 'sqrt',
  'n_estimators': 100}`**
- Best CV accuracy: 0.8231 · test accuracy of best estimator: 0.8156
- Estimator constructed as `RandomForestClassifier(oob_score=True, ...)` at
  construction time → **OOB score = 0.8272**.

## Task 12 — Regression side-task: predict fare

Features: `pclass, sex, age, sibsp, parch, embarked` (all other available
non-fare columns; `who/alive/class/embark_town` are duplicates and
`adult_male/alone` are derived). Same leak-safe preprocessing pattern
(imputer + scaler + one-hot fit on train only).

| metric | value |
|---|---|
| MAE | **20.8094** |
| RMSE | **30.4731** |
| R² | **0.3999** |
| Adjusted R² | **0.3679** (n=179, k=9) |

Residual plot: `charts/regression_residuals.png`.
Residual std by predicted-fare tercile (low/mid/high): **7.54 / 12.82 /
44.82**.

**Heteroscedasticity conclusion:** the residual plot shows **non-random
(fan-shaped) spread — heteroscedasticity is present.** Residual variance
grows with predicted fare (tercile stds rise from ~7.5 to ~44.8): cheap
fares are predicted tightly while expensive 1st-class fares scatter widely.
This is expected for right-skewed fare data and means OLS standard errors
should be interpreted with care.

## Task 13 — Final model comparison table (two distinct metric groups)

Classification and regression metrics are on **different scales against
different targets** — presented as two separate metric groups, never merged
onto one shared scale (also saved as `model_comparison.csv`):

| model | metric_group | accuracy | precision | recall | f1 | auc | mae | rmse | r2 | adj_r2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | classification (target: survived) | 0.8045 | 0.7931 | 0.6667 | 0.7244 | **0.8437** | — | — | — | — |
| Decision Tree | classification (target: survived) | **0.8156** | 0.7903 | **0.7101** | **0.7481** | 0.7904 | — | — | — | — |
| Random Forest | classification (target: survived) | **0.8156** | **0.8000** | 0.6957 | 0.7442 | 0.8300 | — | — | — | — |
| Linear Regression | regression (target: fare) | — | — | — | — | — | 20.8094 | 30.4731 | 0.3999 | 0.3679 |

**Recommendation:** I would deploy **Logistic Regression**, which has the
highest ROC-AUC of the three classifiers (0.8437) at an accuracy of 0.8045,
meaning it ranks survivors above victims more consistently than the
alternatives and remains robust under the ~38/62 class imbalance. Random
Forest is a close second on discrimination (AUC 0.83, accuracy 0.8156, F1
0.7442) but adds complexity for no real AUC gain, while the bare Decision
Tree has the weakest AUC of the three (0.7904) despite its slightly higher
F1 (0.7481) — the classic sign of a fit to this particular split rather
than stable signal. Logistic Regression's recall can be traded up when
needed: with `class_weight='balanced'` the same model family already
reaches F1 0.7552, so its lower baseline recall (0.6667) is a tuning
choice, not a defect. If tree-based explanations were mandated, Random
Forest with its GridSearchCV-best parameters `{max_depth: 5,
max_features: sqrt, n_estimators: 100}` and OOB score 0.8272 would be the
fallback. The regression side-task (MAE 20.81, RMSE 30.47, R² 0.400) is
reported in its own metric group because it predicts a different target on
a different scale and is not comparable to the classification metrics.

## Task 14 — Saved artifact

`best_pipeline.joblib` = **the complete fitted pipeline**
(`preprocess` ColumnTransformer **+** final estimator in one sklearn
`Pipeline`), saved with `joblib.dump(full_pipeline, ...)` — not the bare
estimator.

Reload check (in `02_modeling.py`, output in `modeling_output.txt`):

```
joblib.load -> type=Pipeline, steps=['preprocess', 'clf']
raw input (unpreprocessed, contains NaNs):   pclass  sex   age  ...  embarked
                                              1 female  NaN  ...        S
                                              3   male 22.0  ...      NaN
predictions on raw input: [1, 0]
RELOAD CHECK PASSED: artifact works end-to-end on raw data.
```

The artifact imputes, encodes, scales and predicts on **raw new data**
(including NaNs) with no manual preprocessing.

## Reproducibility note

`sns.load_dataset('titanic')` runs **exactly once** in the whole module
(inside `01_eda.py`); `titanic.csv` is written immediately afterwards and
`02_modeling.py` reads only that CSV — so grading works fully offline via
`pd.read_csv("titanic.csv")`.

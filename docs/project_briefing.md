# CardioSurv — Report Writing Briefing
### AIT201 Applied Machine Learning · Xiamen University Malaysia
> **Purpose of this document:** This is NOT the final report. This is a full technical briefing so that whoever writes the report (or uses an LLM to help write it) has every fact, number, decision, and rationale they need — without needing to open the code files. Read this entirely before writing a single section.

---

## 0. Project Identity

| Field | Detail |
|---|---|
| Project name | CardioSurv |
| Scenario chosen | **Option A — Medical Data Analysis** |
| Course | AIT201 Applied Machine Learning, XMUM |
| Problem type | Multi-class classification + recommendation + survival analysis |
| Final deliverable | Web-deployed ML pipeline for cardiovascular risk triage |

---

## 1. Team Members & Roles

| Name | Role / Task |
|---|---|
| **Ang Jing Ru** | Feature Engineering — built `engineering.py`, produced `features.csv` |
| **Bougacha Mohamed** | ML Modelling — Part 1 risk classifier (`02_part1_modeling.ipynb`) + Part 2 recommender & survival model (`03_part2_modeling.ipynb`) |
| **Law Zi Ying** | Frontend — all HTML/CSS/JS pages (`index.html`, `predict.html`, `results.html`, `history.html`) + FastAPI scaffold |
| **Chiluba Chifunilo** | Backend/Database — DB models, session, API integration, production deployment to Render |
| **Tan Guan Han** | EDA — `01_eda.ipynb`, 5 publication figures, unit tests (`test_bed.py`, `test_grace.py`), evaluation plots |

---

## 2. Problem Statement

Cardiovascular disease is a leading cause of death globally, particularly among the elderly. A critical clinical gap is that elderly patients who maintained healthy lifestyles may still carry hidden serious cardiovascular risk due to the natural aging of the heart and blood vessels. Because these patients often appear generally healthy, sudden fatal cardiac events (cardiogenic shock, sudden heart failure) can occur unexpectedly.

This project addresses two clinical gaps in a **unified ML pipeline**:
1. Automatically classify a patient's cardiovascular risk (Low / Medium / High) from routine clinical vital signs.
2. Recommend a targeted treatment intervention and estimate the patient's two-year survival probability under that treatment.

---

## 3. Datasets

**Four data sources** were used across the two parts of the pipeline. This is an important point for the report — the team did not just take one ready-made dataset. They merged, harmonised, and augmented multiple sources.

### 3.1 Part 1 Datasets (Risk Classifier)

| Dataset | Source | Size | Usage |
|---|---|---|---|
| Heart Failure Prediction | Kaggle (fedesoriano) | 918 rows, 11 features | Primary classification labels and features |
| Cleveland Heart Disease | UCI ML Repository | 303 rows, 14 features | Supplementary set merged to add angiographic features (`ca`, `thal`) and increase training sample size |

**Why both datasets were merged:**
The Kaggle Heart Failure dataset has binary labels but lacks detailed angiographic features. The UCI Cleveland set adds clinically significant predictors such as `ca` (number of major vessels coloured by fluoroscopy) and `thal` (thalassemia type). Merging both datasets after harmonising column names and encoding schemes produced a richer, larger training set (~1,221 rows after merge and cleaning) and demonstrates data integration skills during preprocessing.

### 3.2 Part 2 Datasets (Recommender + Survival)

| Dataset | Source | Usage |
|---|---|---|
| MIMIC-IV Clinical Database | PhysioNet (Johnson et al., 2023) | Intended source for treatment history, medication records, and survival outcomes |
| SBRT/VT Dose Records | Literature-derived | BED formula calibration for the high-risk radiation path |

**Important note about Part 2 data (must be mentioned in the report):**
MIMIC-IV access was not guaranteed during development. Because the original heart disease datasets contain no longitudinal follow-up data (no survival times, no event records), the team made a deliberate data engineering decision: survival time durations and event labels were **synthetically simulated** using clinically grounded rules (GRACE score + risk category + arrhythmia flag). This is a **proof-of-concept** demonstration of the survival analysis pipeline, not a clinically validated predictor. The report should state this transparently in the Methodology section and frame it as an intentional choice to demonstrate the full pipeline architecture given dataset constraints.

---

## 4. Data Preprocessing & Feature Engineering

**Author: Ang Jing Ru**

Starting from `cleaned.csv`, the following engineered features were added to produce `features.csv`:

### 4.1 Engineered Features

| Feature | Description | Formula / Logic |
|---|---|---|
| `AgeBin` | Age group bucket | `<40`, `40-49`, `50-59`, `60-69`, `70+` |
| `BP_RiskLevel` | Blood pressure risk category | AHA categories: Normal / Elevated / Stage1 / Stage2 / Crisis |
| `HeartRateStressIndex` | Ratio of max heart rate to theoretical max | `MaxHR / (220 - Age)` — higher = more stressed cardiovascular response |
| `RiskCategory` | 3-class target label | Derived from clinical rules: Low / Medium / High |

### 4.2 Preprocessing Pipeline (in modeling notebooks)

Applied in a scikit-learn `ColumnTransformer`:
- **Numeric features** (`Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`, `HeartRateStressIndex`): `StandardScaler`
- **Categorical features** (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`, `AgeBin`, `BP_RiskLevel`): `OneHotEncoder` with `drop='first'` to avoid the dummy variable trap

### 4.3 Train/Test Split
- **80/20 stratified split** (stratified on `RiskCategory` to preserve class distribution across both sets)
- Train set: ~1,136 rows | Test set: ~285 rows
- `random_state=42` for reproducibility

---

## 5. Exploratory Data Analysis

**Author: Tan Guan Han** | Notebook: `01_eda.ipynb`

Five publication-quality figures were generated:

### Figure 1 — Class Balance
The dataset shows moderate imbalance: approximately 55% of patients diagnosed with heart disease, 45% without. The Medium risk category is the minority class (~8.4% of samples — only 119 of 1,421 rows). This justified using `class_weight='balanced'` in the Random Forest.

### Figure 2 — Numeric Distributions
- `Age`: Normally distributed, centred around 54 years (middle-aged cohort)
- `RestingBP` and `Cholesterol`: Right-skewed with some high outliers
- `MaxHR`: Left-skewed (decreases with age)
- `Oldpeak`: Heavy right skew (most patients have values near 0)

### Figure 3 — Categorical Breakdowns
Stacked bar charts for `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope` vs heart disease status. Key finding: ASY (asymptomatic) chest pain type is strongly associated with heart disease despite being asymptomatic — clinically important.

### Figure 4 — Spearman Correlation Heatmap
Computed on numeric features + `FastingBS` + `HeartDisease`. Reveals the strongest positive correlations with heart disease outcome.

### Figure 5 — Age Distribution by Heart Disease Status
Box plot showing patients with heart disease tend to be older. The interquartile range for disease-positive patients is shifted higher.

### Key EDA Observations (to include in report)
1. Class imbalance exists but is not severe; `class_weight='balanced'` is sufficient — no heavy resampling needed
2. Middle-aged to elderly cohort (centred ~54 years) aligns with clinical literature on peak cardiovascular risk age (50–70 years)
3. ASY chest pain type being the strongest categorical predictor is a clinically known phenomenon — asymptomatic ischaemia is a key risk factor
4. HeartRateStressIndex (engineered feature) is expected to contribute meaningfully as it normalises MaxHR by age

---

## 6. Part 1 — Risk Classifier

**Author: Bougacha Mohamed** | Notebook: `02_part1_modeling.ipynb`

### 6.1 Task Definition
Multi-class classification: predict `RiskCategory` ∈ {Low, Medium, High} from 14 input features (7 numeric + 7 categorical).

### 6.2 Models Trained

**Model 1: Random Forest**
```
n_estimators = 300
max_depth    = None (fully grown trees)
class_weight = 'balanced'
random_state = 42
```

**Model 2: XGBoost**
```
n_estimators  = 400
max_depth     = 6
learning_rate = 0.05
objective     = multi:softprob
num_class     = 3
eval_metric   = mlogloss
random_state  = 42
```

Both models used the same preprocessing pipeline (StandardScaler + OneHotEncoder) wrapped in a scikit-learn `Pipeline`.

### 6.3 Results

| Metric | Random Forest | XGBoost | Winner |
|---|---|---|---|
| Accuracy | **0.9614** | 0.9579 | Random Forest ✓ |
| F1-macro | **0.9321** | 0.9251 | Random Forest ✓ |
| AUC-OVR | **0.9880** | 0.9853 | Random Forest ✓ |

**Winner: Random Forest**

### 6.4 Why Random Forest Won (include this reasoning in the report)

The winner was selected by **F1-macro** rather than accuracy because the dataset has class imbalance (Medium class = 8.4% of samples). F1-macro weights all three classes equally and penalises poor recall on the minority class.

Random Forest outperformed XGBoost on this dataset for two reasons:
1. **Dataset size (~1,400 rows)**: Gradient boosting typically pulls ahead on larger datasets where it can iterate over many residual corrections. At ~1,400 rows, the variance introduced by sequential boosting adds noise rather than signal.
2. **Balanced class weights**: Random Forest's `class_weight='balanced'` naturally reweights minority class samples per tree, which suits the imbalanced Medium class better at this dataset size.

### 6.5 Feature Importance (Top features from Random Forest)
The top 15 features by mean decrease in impurity included:
- `Oldpeak` (ST depression)
- `MaxHR`
- `HeartRateStressIndex` (engineered feature)
- `Age`
- `ChestPainType_ASY`
- `ST_Slope_Flat` / `ST_Slope_Up`

This validates that the engineered `HeartRateStressIndex` is a meaningful predictor.

### 6.6 Confusion Matrix
Produced for both models. Random Forest shows strong diagonal dominance with most misclassifications occurring on the Medium (minority) class boundary — expected and acceptable.

---

## 7. Part 2 — Treatment Recommender + Survival Model

**Author: Bougacha Mohamed** | Notebook: `03_part2_modeling.ipynb`

### 7.1 Task Definition
Two sub-tasks:
1. **Intervention recommendation**: Given the Part 1 risk category + clinical routing, recommend an intervention type from 4 classes: `Medication-Standard (0)`, `Medication-Intensified (1)`, `Medication-Maximal (2)`, `SBRT (3)`
2. **Survival prediction**: Estimate 2-year survival probability with and without the recommended intervention

### 7.2 Clinical Routing Logic
Before the recommender runs, a rule-based clinical router assigns the patient to a treatment branch:

**High-risk path (SBRT)**: Triggered when `RiskCategory = High` AND `has_arrhythmia = True` (refractory ventricular tachycardia). Dose recommendation: 25 Gy single fraction, validated using the **Biologically Effective Dose (BED) formula**:

> BED = D × (1 + d / (α/β))

Where D = total dose (25 Gy), d = dose per fraction (25 Gy for single fraction), α/β = 10 Gy for cardiac tissue. BED = 25 × (1 + 25/10) = **87.5 Gy** — within published clinical window for cardiac radioablation.

**Low/Medium-risk path (Medication)**: Calibrated via **GRACE Risk Score** (a validated clinical scoring formula using age, heart rate, and systolic BP). Intensity: Standard / Intensified / Maximal depending on score band.

### 7.3 Part 2 Feature Set (how it was built)
Since `features.csv` has no survival data, Part 2 features were constructed by augmenting the existing feature set:

- `has_arrhythmia`: Simulated — 15% probability for High-risk patients, 2% for others (clinically grounded ratio)
- `grace_score`: Computed deterministically per patient using age, MaxHR, RestingBP via the GRACE formula
- `intervention_label`: Assigned by clinical routing rules (deterministic)
- `bed_gy`: 87.5 for SBRT patients, 0.0 otherwise
- `duration` (survival time): Synthetically simulated using clinical priors
- `event` (death/censoring): Synthetically simulated

### 7.4 Models Trained

**Model 1: XGBoost Intervention Recommender**
```
n_estimators = 300
max_depth    = 5
random_state = 42
objective    = multi:softprob (4 classes)
```

**Model 2: Survival Analysis**
- **Kaplan-Meier Fitter**: Stratified by treatment branch (SBRT vs Medication). Non-parametric survival curves with 95% confidence bands.
- **Cox Proportional Hazards (Cox PH)**: Parametric survival model with covariates: `age`, `risk_encoded`, `grace_score`, `bed_gy`. Produces per-patient 2-year survival probability.

### 7.5 Results

| Model | Metric | Value | Threshold | Status |
|---|---|---|---|---|
| XGBoost Recommender | Accuracy | 1.00 | ≥ 0.75 | ✅ |
| XGBoost Recommender | F1-macro | 1.00 | ≥ 0.75 | ✅ |
| Cox PH Survival | Concordance C-index | 0.85 | ≥ 0.65 | ✅ |

**Why XGBoost hits 1.00 — must be explained in the report:**
The intervention labels are deterministically derived from clinical routing rules (RiskCategory + has_arrhythmia + GRACE score). Because those rules are exact functions of features the model can observe, XGBoost learns the rule boundaries perfectly. This is expected and desirable in the context of this proof-of-concept: the model should agree exactly with the protocol. In a real MIMIC-IV deployment, labels would be noisier.

### 7.6 Log-Rank Test
A log-rank test was performed comparing survival curves between SBRT and Medication branches. The p-value from this test should be reported in the results section (it is computed in the notebook as `lr.p_value`).

---

## 8. System Architecture

The full pipeline is:

```
Patient Vital Signs Input
        ↓
Combine Dataset 1 (Kaggle) + Dataset 2 (UCI Cleveland)
        ↓
Feature Engineering: AgeBin, BP_RiskLevel, HeartRateStressIndex
        ↓
Preprocessing: StandardScaler + OneHotEncoder
        ↓
Part 1 — Train & Compare: Random Forest vs XGBoost
        ↓
Predict Risk Category (Low / Medium / High) + Confidence Score
        ↓
        ↓ ─────────────────── Risk Level? ─────────────────────
        ↓                                                       ↓
    High + Arrhythmia                               Low / Medium / High (no arrh.)
        ↓                                                       ↓
SBRT Path: BED Validation                         GRACE Score Calibration
BED = 25×(1+25/10) = 87.5 Gy                      Medication Intensity
        ↓                                                       ↓
        └──────────────── Part 2 XGBoost ────────────────────┘
                    Intervention Recommendation
                              ↓
              Survival Model: Kaplan-Meier + Cox PH
                              ↓
          Counterfactual Output:
    "Without intervention: X% → With treatment: Y% (2-year survival)"
```

### Backend Stack
- **FastAPI** (Python) — REST API with 5 endpoints: `/health`, `/predict`, `/recommend`, `/history`, `/patients/{id}`
- **SQLite** (`cardiosurv.db`) — stores patient records, predictions, recommendations
- **Deployed on Render** — production URL, PostgreSQL free tier
- Rate limiting: 30 requests/minute per IP via `slowapi`
- Model loaded once at startup via FastAPI `lifespan` event

### Frontend Stack
- Pure HTML/CSS/JS (no framework)
- 4 pages: `index.html` (landing), `predict.html` (form), `results.html` (output), `history.html` (log)
- Risk badge is both colour-coded AND shape-coded (circle=Low, triangle=Medium, square=High) for accessibility
- Chart.js survival comparison bar chart on results page

---

## 9. Example Cases (use these in Results & Discussion)

| Case | Patient | Part 1 Output | Part 2 Branch | Recommendation | Survival |
|---|---|---|---|---|---|
| A | 42 y/o M, healthy vitals | Low, 91% confidence | Medication (GRACE low) | Lifestyle + low-dose statin | 95% → 97% |
| B | 58 y/o M, ST abnormality | Medium, 83% confidence | Medication (GRACE mid) | Beta-blocker + statin + aspirin | 81% → 92% |
| C | 67 y/o M, exercise angina, no arrhythmia | High, 88% confidence | Medication (no arrhythmia) | Aggressive multi-drug regimen | 62% → 84% |
| D | 71 y/o M, refractory VT | High, 94% confidence | SBRT (arrhythmia present) | 25 Gy single-fraction radioablation, BED=87.5 Gy | 58% → 81% |

---

## 10. References (already compiled — use APA format)

1. Johnson, A. E. W. et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data, 10*, 1. https://www.nature.com/articles/s41597-022-01899-x

2. Mohan, S. et al. (2019). Effective heart disease prediction using hybrid machine learning techniques. *IEEE Access, 7*, 81542–81554. https://ieeexplore.ieee.org/document/8740989

3. Eagle, K. A. et al. (2004). A validated prediction model for all forms of acute coronary syndrome. *JAMA, 291*(22), 2727–2733. https://jamanetwork.com/journals/jama/fullarticle/198987

4. Joiner, M., & van der Kogel, A. (2009). *Basic clinical radiobiology* (4th ed.). Hodder Arnold. https://www.taylorfrancis.com/books/edit/10.1201/b15450/basic-clinical-radiobiology-michael-joiner-albert-van-der-kogel

5. Cuculich, P. S. et al. (2017). Noninvasive cardiac radiation for ablation of ventricular tachycardia. *New England Journal of Medicine, 377*, 2325–2336. https://www.nejm.org/doi/full/10.1056/NEJMoa1613773

6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*. https://dl.acm.org/doi/10.1145/2939672.2939785

7. Liaw, A., & Wiener, M. (2002). Classification and regression by randomForest. *R News, 2*(3), 18–22.

8. Wang, P. et al. (2019). Machine learning for survival analysis: A survey. *ACM Computing Surveys, 51*(6). https://dl.acm.org/doi/10.1145/3214306

9. fedesoriano. (2021). *Heart failure prediction dataset*. Kaggle. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

10. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart disease dataset* [UCI ML Repository]. https://archive.ics.uci.edu/dataset/45/heart+disease

---

## 11. Suggested Report Structure & What to Write in Each Section

### Title
*CardioSurv: A Unified Machine Learning Pipeline for Cardiovascular Risk Classification, Treatment Recommendation, and Survival Prediction*

### Abstract (~200 words)
Summarise: cardiovascular disease motivation → two-stage ML pipeline → Part 1 (RF vs XGBoost, winner RF at F1=0.9321) → Part 2 (XGBoost recommender + Cox PH survival) → 4-branch treatment routing → deployed as web app with FastAPI backend → proof-of-concept survival simulation due to dataset constraints.

### Introduction
Use Section 2 (Problem Statement) above. Mention the clinical gap, the peak risk age group (50–70 years), and how ML can support triage. State the project is Option A (Medical Data Analysis).

### Literature Review
Use references 2, 3, 5, 6, 8 from Section 10 above. Structure around:
- ML for heart disease prediction (Mohan et al., 2019 — IEEE Access)
- Survival analysis with ML (Wang et al., 2019 — ACM Computing Surveys)
- Clinical scoring systems in ML pipelines (Eagle et al., 2004 — GRACE score)
- Cardiac radioablation context (Cuculich et al., 2017 — NEJM)

### Methodology
Cover in order: (a) dataset description and why two datasets were merged, (b) feature engineering (AgeBin, BP_RiskLevel, HRSI), (c) preprocessing pipeline, (d) EDA key findings, (e) Part 1 model selection rationale, (f) Part 2 clinical routing logic + BED formula, (g) synthetic survival data rationale. Use Sections 3–7 above.

### Results and Discussion
Present the model comparison table (Section 6.3), explain why RF won (Section 6.4), feature importance findings (Section 6.5), Part 2 results table (Section 7.5), explain the 1.00 XGBoost score honestly (Section 7.5), walk through the 4 example cases (Section 9).

### Conclusion
Key findings: RF outperformed XGBoost at this dataset scale; engineered features (especially HRSI) contributed meaningfully; the pipeline successfully covers all 4 clinical routing branches; survival model is a proof-of-concept pending real longitudinal data. Limitations: synthetic survival data, dataset size (~1,400 rows), no external validation set. Future work: integrate real MIMIC-IV data, add external validation, explore LSTM/Transformer for temporal patient data.

---

## 12. Key Numbers to Remember

| Fact | Value |
|---|---|
| Final dataset size (after merge + clean) | ~1,221 rows (used for Part 1) |
| Train / Test split | 80% / 20%, stratified |
| Medium class proportion | 8.4% (minority class) |
| RF Accuracy | 0.9614 |
| RF F1-macro | 0.9321 |
| RF AUC-OVR | 0.9880 |
| XGB Accuracy | 0.9579 |
| XGB F1-macro | 0.9251 |
| XGB AUC-OVR | 0.9853 |
| XGBoost Recommender Accuracy | 1.00 (deterministic labels) |
| Cox PH C-index | 0.85 |
| BED for SBRT case | 87.5 Gy (25 Gy × (1 + 25/10)) |
| α/β for cardiac tissue | 10 Gy |
| Notebooks | 3 (EDA, Part1, Part2) |
| Saved model files | 3 (.pkl: part1 classifier, part2 recommender, Cox survival) |
| API endpoints | 5 (/health, /predict, /recommend, /history, /patients/{id}) |
| Frontend pages | 4 (index, predict, results, history) |

---

*End of briefing. All facts in this document were extracted directly from the project notebooks, PDFs, and source files. Do not invent numbers — use only what is listed here.*

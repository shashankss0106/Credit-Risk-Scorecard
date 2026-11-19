# Interpretable Credit Risk PD Scorecard & Cut-Off Optimization

An end-to-end, glass-box **Probability of Default (PD) Credit Scorecard** built with Python and Scikit-learn on the [All Lending Club Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) dataset. Designed for **regulatory explainability**, **chronological Out-of-Time (OOT) validation**, and **Expected Loss ($EL = PD \times LGD \times EAD$) cut-off optimization**.

## Business Problem
Retail lenders must balance loan approval growth against credit default losses while complying with fair-lending transparency requirements. Black-box models often struggle in credit committees because they cannot easily explain *why* an applicant was declined.

This project builds a fully auditable **300–850 additive points scorecard** using business-bucketed **Logistic Regression** paired with a **Depth-3 Decision Tree** risk policy—delivering clear decline reason codes and quantifying Expected Loss reduction across approval cut-offs.

## Dataset
- **Source:** [All Lending Club Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (`accepted_2007_to_2018Q4.csv.gz`)
- **Target Definition:** Matured loans only (`Fully Paid` = `0`, `Charged Off` / `Default` = `1`). Active (`Current`) loans are excluded to avoid outcome censoring.
- **Leakage Prevention:** Strictly restricted to application-time features (`loan_amnt`, `term`, `int_rate`, `home_ownership`, `annual_inc`, `dti`, `fico_range_low`, `revol_util`, `issue_d`). All post-origination columns (`total_pymnt`, `recoveries`, `out_prncp`, `last_pymnt_amnt`) are excluded.
- **Validation Split:** Chronological **Out-of-Time (OOT)** split—trained on `2013–2016` originations and evaluated on untouched `2017–2018` originations to reflect real credit-cycle drift.

## Methodology
1. **Business-Intuitive Bucketing:** Continuous risk drivers (`FICO`, `DTI`, `Revolving Utilization`, `Loan-to-Income`) are grouped into 4–5 transparent bands so every bucket maps directly to credit policy.
2. **Multicollinearity & Feature Pruning:** Computes **Variance Inflation Factor (VIF)** across encoded buckets and benchmarks **L1 (Lasso)** vs. **L2 (Ridge)** Logistic Regression.
3. **300–850 Additive Scorecard Scaling:** Converts Logistic Regression log-odds into integer scorecard points (`PDO = 20`, `Base Score = 600` at `50:1` good-to-bad odds) with automated **top decline reason codes**.
4. **Companion Depth-3 Decision Tree:** Generates human-readable `if-then` risk tiers (Low / Medium / High / Decline) for credit committee review.
5. **Banking Validation & Expected Loss Cut-Off:** Evaluates **OOT ROC-AUC**, **Gini** ($2 \times \text{AUC} - 1$), **KS-Statistic**, **10-Decile Lift**, and simulates **Expected Loss** ($EL = PD \times LGD \times EAD$, assuming `75%` unsecured LGD) across rejection thresholds.

## Repository Structure
```text
credit-risk-pd-scorecard/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── credit_risk_scorecard.ipynb   # Ready for Kaggle Notebooks or Jupyter
├── src/
│   └── scorecard_pipeline.py         # End-to-end runnable CLI script
└── outputs/                          # Generated charts & CSV tables (after running)
    ├── scorecard_points.csv
    ├── decile_lift_table.csv
    ├── cutoff_simulation.csv
    ├── vif_report.csv
    ├── ks_curve.png
    ├── score_distribution.png
    └── cutoff_expected_loss.png
```

## Quick Start

### Option A: Run on Kaggle Notebooks 
1. Open [All Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and click **New Notebook**.
2. Upload `notebooks/credit_risk_scorecard.ipynb` (File -> Import Notebook).
3. Click **Run All**—the notebook automatically locates the dataset in `/kaggle/input/`, computes all metrics, and exports the tables and PNG charts.

### Option B: Run Locally
```bash
pip install -r requirements.txt

# Run on downloaded LendingClub CSV
python src/scorecard_pipeline.py --data-path data/accepted_2007_to_2018Q4.csv.gz

# Or run a 2-second synthetic smoke test to verify setup
python src/scorecard_pipeline.py --demo
```

## Results Summary 
## Results Summary
| Metric | Train (2013–2016) | OOT Test (2017–2018) |
| :--- | :---: | :---: |
| Matured Loans (`N`) | 1,026,518 | 225,305 |
| Default Rate | 20.09% | 21.30% |
| ROC-AUC | 0.678 | 0.662 |
| Gini Coefficient | 0.356 | 0.323 |
| KS-Statistic | 25.4% | 23.3% |
| Expected Loss Avoided (Reject Bottom 20%, Score < 517) | — | 48.1% |
| Actual Defaults Captured (Reject Bottom 20%) | — | 33.1% |

## Key Visuals 
- `outputs/ks_curve.png` — Cumulative Good vs. Bad separation (KS-Statistic)
- `outputs/score_distribution.png` — 300–850 Score distribution for Paid vs. Defaulted loans
- `outputs/cutoff_expected_loss.png` — Approval Rate vs. Expected Loss Avoided frontier

## Author
**Shashank Singh** — PGDM – Big Data Analytics, Goa Institute of Management (GIM)
- LinkedIn: [linkedin.com/in/shashanksudhirsingh](https://www.linkedin.com/in/shashanksudhirsingh)

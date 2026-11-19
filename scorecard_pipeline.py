"""
Interpretable Credit Risk PD Scorecard & Cut-Off Optimization
=============================================================
End-to-end glass-box Probability of Default (PD) Scorecard using:
- Strict application-time feature allowlist (zero post-origination leakage)
- Chronological Out-of-Time (OOT) split (2013-2016 train, 2017-2018 OOT)
- Business-bucketed features + VIF multicollinearity audit
- L1 Lasso (feature check) & L2 Logistic Regression (300-850 Points Scorecard)
- Companion Depth-3 Decision Tree credit policy rules
- Banking validation (ROC-AUC, Gini, KS-Statistic, 10-Decile Lift)
- Expected Loss (EL = PD * LGD * EAD) cut-off simulation & Decline Reason Codes

Usage:
    # On Kaggle Notebook (auto-detects /kaggle/input/lending-club)
    python src/scorecard_pipeline.py

    # Locally with downloaded LendingClub CSV
    python src/scorecard_pipeline.py --data-path data/accepted_2007_to_2018Q4.csv.gz

    # Quick synthetic smoke test (verifies environment & chart generation)
    python src/scorecard_pipeline.py --demo
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_text

USE_COLS = [
    "issue_d",
    "loan_status",
    "loan_amnt",
    "term",
    "int_rate",
    "home_ownership",
    "annual_inc",
    "dti",
    "fico_range_low",
    "revol_util",
]

PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50
UNSECURED_LGD = 0.75


def generate_demo_data(n_rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic dataset with LendingClub schema for smoke testing."""
    rng = np.random.default_rng(seed)
    years = rng.choice([2013, 2014, 2015, 2016, 2017, 2018], size=n_rows, p=[0.12, 0.16, 0.20, 0.22, 0.16, 0.14])
    months = rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], size=n_rows)
    issue_d = [f"{m}-{y}" for m, y in zip(months, years)]

    fico = rng.normal(695, 38, size=n_rows).clip(600, 820)
    dti = rng.normal(19, 9, size=n_rows).clip(0, 50)
    annual_inc = rng.lognormal(mean=11.1, sigma=0.5, size=n_rows).clip(20000, 350000)
    loan_amnt = rng.choice([5000, 10000, 15000, 20000, 25000, 35000], size=n_rows)
    term = rng.choice([" 36 months", " 60 months"], size=n_rows, p=[0.74, 0.26])
    revol_util = rng.normal(52, 24, size=n_rows).clip(0, 100)
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], size=n_rows, p=[0.48, 0.40, 0.11, 0.01])
    int_rate = (18.0 - 0.015 * (fico - 600) + 0.12 * dti + rng.normal(0, 1.5, n_rows)).clip(5.5, 29.0)

    lti = loan_amnt / (annual_inc + 1.0)
    logits = (
        -2.0
        - 0.016 * (fico - 680)
        + 0.042 * (dti - 18)
        + 0.012 * (revol_util - 50)
        + 2.4 * (lti - 0.20)
        + 0.65 * (term == " 60 months")
        + 0.25 * (home == "RENT")
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    default_flag = rng.binomial(1, probs)
    loan_status = np.where(default_flag == 1, "Charged Off", "Fully Paid")

    return pd.DataFrame({
        "issue_d": issue_d,
        "loan_status": loan_status,
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "home_ownership": home,
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_range_low": fico,
        "revol_util": revol_util,
    })


def load_data(data_path: str | None, demo: bool) -> pd.DataFrame:
    """Load LendingClub accepted loans or synthetic demo data."""
    if demo:
        print("[DEMO MODE] Running on synthetic LendingClub-schema sample (12,000 rows).")
        return generate_demo_data()

    if data_path is None:
        kaggle_matches = glob.glob("/kaggle/input/**/accepted_2007_to_2018*", recursive=True)
        if not kaggle_matches:
            raise FileNotFoundError(
                "No LendingClub CSV found. Pass --data-path <path_to_csv> or run on Kaggle "
                "with wordsforthewise/lending-club attached (or use --demo to test)."
            )
        data_path = kaggle_matches[0]

    print(f"Loading dataset from: {data_path}")
    return pd.read_csv(data_path, usecols=USE_COLS, low_memory=False)


def prepare_matured_loans(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to matured loans, define binary target, and create chronological OOT split."""
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["target"] = (df["loan_status"] != "Fully Paid").astype(int)
    df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year
    df = df.dropna(subset=["issue_year", "fico_range_low", "dti", "annual_inc", "loan_amnt"])

    print(f"Total matured loans: {len(df):,} | Overall Default Rate: {df['target'].mean():.2%}")

    train = df[df["issue_year"].between(2013, 2016)].copy()
    oot = df[df["issue_year"].between(2017, 2018)].copy()
    print(f"Train (2013-2016): {len(train):,} (Default: {train['target'].mean():.2%})")
    print(f"OOT   (2017-2018): {len(oot):,} (Default: {oot['target'].mean():.2%})")
    return train, oot


def bucket_features(data: pd.DataFrame) -> pd.DataFrame:
    """Transform continuous credit features into transparent business buckets."""
    d = pd.DataFrame(index=data.index)
    d["fico"] = pd.cut(
        data["fico_range_low"],
        bins=[0, 659, 699, 739, 900],
        labels=["<660", "660-699", "700-739", "740+"],
    )
    d["dti"] = pd.cut(
        data["dti"],
        bins=[-1, 15, 25, 35, 1000],
        labels=["0-15%", "15-25%", "25-35%", "35%+"],
    )
    d["util"] = pd.cut(
        data["revol_util"].fillna(-1),
        bins=[-2, 0, 30, 60, 80, 1000],
        labels=["Missing/0", "1-30%", "31-60%", "61-80%", "80%+"],
    )
    lti = data["loan_amnt"] / (data["annual_inc"] + 1.0)
    d["lti"] = pd.cut(
        lti,
        bins=[-1, 0.15, 0.25, 0.35, 100],
        labels=["<15%", "15-25%", "25-35%", "35%+"],
    )
    d["term"] = data["term"].astype(str).str.strip()
    d["home"] = data["home_ownership"].where(
        data["home_ownership"].isin(["MORTGAGE", "RENT", "OWN"]), "OTHER"
    )
    return pd.get_dummies(d, drop_first=True, dtype=int)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) using pure scikit-learn."""
    X_vals = X.astype(float).values
    vif_rows = []
    for i, col in enumerate(X.columns):
        y_i = X_vals[:, i]
        X_other = np.delete(X_vals, i, axis=1)
        reg = LinearRegression().fit(X_other, y_i)
        r2 = reg.score(X_other, y_i)
        vif = float("inf") if r2 >= 0.9999 else 1.0 / (1.0 - r2)
        vif_rows.append({"Feature_Bucket": col, "VIF": round(vif, 2)})
    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)


def evaluate_banking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute ROC-AUC, Gini coefficient, and KS-Statistic."""
    auc = roc_auc_score(y_true, y_prob)
    gini = 2.0 * auc - 1.0
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_idx = int(np.argmax(tpr - fpr))
    ks = float(tpr[ks_idx] - fpr[ks_idx])
    return {"auc": auc, "gini": gini, "ks": ks, "fpr": fpr, "tpr": tpr, "ks_idx": ks_idx}


def build_decile_lift_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Create a 10-decile risk ranking table (Decile 1 = highest predicted risk)."""
    df = eval_df.copy()
    df["decile"] = pd.qcut(df["pd"].rank(method="first"), q=10, labels=range(10, 0, -1)).astype(int)
    agg = (
        df.groupby("decile", as_index=False)
        .agg(
            loans=("actual_default", "count"),
            defaults=("actual_default", "sum"),
            avg_score=("score", "mean"),
            avg_pd=("pd", "mean"),
            expected_loss=("expected_loss", "sum"),
        )
        .sort_values("decile")
    )
    agg["default_rate"] = agg["defaults"] / agg["loans"]
    agg["cum_defaults_pct"] = agg["defaults"].cumsum() / agg["defaults"].sum()
    agg["cum_loss_pct"] = agg["expected_loss"].cumsum() / agg["expected_loss"].sum()
    return agg


def build_cutoff_simulation(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate Expected Loss avoided across candidate rejection rates."""
    rows = []
    total_el = eval_df["expected_loss"].sum()
    total_def = eval_df["actual_default"].sum()
    for rej_pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        cutoff = eval_df["score"].quantile(rej_pct)
        rejected = eval_df[eval_df["score"] < cutoff]
        approved = eval_df[eval_df["score"] >= cutoff]
        rows.append({
            "Reject_Rate": rej_pct,
            "Approval_Rate": 1.0 - rej_pct,
            "Score_Cutoff": round(float(cutoff), 0),
            "Approved_Default_Rate": approved["actual_default"].mean(),
            "Actual_Defaults_Captured": rejected["actual_default"].sum() / total_def,
            "Expected_Loss_Avoided": rejected["expected_loss"].sum() / total_el,
        })
    return pd.DataFrame(rows)


def generate_charts(metrics: dict, eval_df: pd.DataFrame, cutoff_df: pd.DataFrame, out_dir: Path) -> None:
    """Save publication-ready KS curve, score distribution, and cut-off frontier charts."""
    # 1. KS Curve
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x_axis = np.linspace(0, 1, len(metrics["tpr"]))
    ax.plot(x_axis, metrics["tpr"], label="Cumulative Defaults (Bad)", color="#C0392B", lw=2.2)
    ax.plot(x_axis, metrics["fpr"], label="Cumulative Paid (Good)", color="#1F618D", lw=2.2)
    idx = metrics["ks_idx"]
    ax.vlines(x_axis[idx], metrics["fpr"][idx], metrics["tpr"][idx], colors="#27AE60", linestyles="--", lw=2,
              label=f"KS = {metrics['ks']:.1%}")
    ax.set_title("Out-of-Time KS Curve (Good vs. Bad Separation)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Population Ranked by Predicted Risk")
    ax.set_ylabel("Cumulative Proportion")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ks_curve.png", dpi=180)
    plt.close(fig)

    # 2. Score Distribution (Good vs Bad)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    good_scores = eval_df.loc[eval_df["actual_default"] == 0, "score"]
    bad_scores = eval_df.loc[eval_df["actual_default"] == 1, "score"]
    cutoff_20 = eval_df["score"].quantile(0.20)
    ax.hist(good_scores, bins=30, density=True, alpha=0.55, color="#1F618D", label="Fully Paid (Good)")
    ax.hist(bad_scores, bins=30, density=True, alpha=0.55, color="#C0392B", label="Charged Off / Default (Bad)")
    ax.axvline(cutoff_20, color="#111111", linestyle="--", lw=1.8, label=f"20% Reject Cut-Off ({cutoff_20:.0f})")
    ax.set_title("OOT Credit Score Distribution (300–850 Scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Additive Scorecard Points")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution.png", dpi=180)
    plt.close(fig)

    # 3. Approval Rate vs Expected Loss Avoided Frontier
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(cutoff_df["Approval_Rate"] * 100, cutoff_df["Expected_Loss_Avoided"] * 100,
            marker="o", color="#1F618D", lw=2.2, label="Expected Loss Avoided (%)")
    ax.plot(cutoff_df["Approval_Rate"] * 100, cutoff_df["Actual_Defaults_Captured"] * 100,
            marker="s", color="#C0392B", lw=2.0, linestyle="--", label="Actual Defaults Captured (%)")
    ax.invert_xaxis()
    ax.set_title("Credit Policy Cut-Off Trade-Off: Approval Rate vs. Loss Reduction", fontsize=12, fontweight="bold")
    ax.set_xlabel("Loan Approval Rate (%)")
    ax.set_ylabel("Share of Risk Avoided (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cutoff_expected_loss.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpretable Credit Risk PD Scorecard Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to LendingClub accepted CSV(.gz)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for CSVs and PNG charts")
    parser.add_argument("--demo", action="store_true", help="Run fast synthetic smoke test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & Split
    df = load_data(args.data_path, args.demo)
    train, oot = prepare_matured_loans(df)

    # 2. Business Bucketing
    X_train = bucket_features(train)
    y_train = train["target"].values
    X_oot = bucket_features(oot).reindex(columns=X_train.columns, fill_value=0)
    y_oot = oot["target"].values

    # 3. VIF Multicollinearity Audit
    vif_df = compute_vif(X_train)
    vif_df.to_csv(out_dir / "vif_report.csv", index=False)
    print(f"\nMax VIF across feature buckets: {vif_df['VIF'].max():.2f}")

    # 4. L1 Lasso Feature Check vs. L2 Logistic Regression Champion
    lr_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
    lr_l1.fit(X_train, y_train)
    nonzero_l1 = int(np.sum(np.abs(lr_l1.coef_[0]) > 1e-4))
    print(f"L1 Lasso retained {nonzero_l1}/{X_train.shape[1]} feature buckets.")

    lr_l2 = LogisticRegression(C=0.5, max_iter=1000)
    lr_l2.fit(X_train, y_train)

    # 5. Additive 300-850 Scorecard Points Table
    factor = PDO / np.log(2.0)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    base_points = int(round(offset - factor * lr_l2.intercept_[0]))
    bucket_points = np.round(-factor * lr_l2.coef_[0]).astype(int)

    points_table = pd.DataFrame({
        "Feature_Bucket": X_train.columns,
        "LogOdds_Coef": np.round(lr_l2.coef_[0], 4),
        "Scorecard_Points": bucket_points,
    }).sort_values("Scorecard_Points", ascending=False)
    points_table.to_csv(out_dir / "scorecard_points.csv", index=False)

    print(f"\n--- SCORECARD POINTS TABLE (Base Score = {base_points}) ---")
    print(points_table.to_string(index=False))

    # 6. OOT Validation Metrics
    pd_train = np.clip(lr_l2.predict_proba(X_train)[:, 1], 1e-6, 1 - 1e-6)
    pd_oot = np.clip(lr_l2.predict_proba(X_oot)[:, 1], 1e-6, 1 - 1e-6)
    m_train = evaluate_banking_metrics(y_train, pd_train)
    m_oot = evaluate_banking_metrics(y_oot, pd_oot)
    print("\n--- MODEL VALIDATION ---")
    print(f"Train | AUC: {m_train['auc']:.3f} | Gini: {m_train['gini']:.3f} | KS: {m_train['ks']:.1%}")
    print(f"OOT   | AUC: {m_oot['auc']:.3f} | Gini: {m_oot['gini']:.3f} | KS: {m_oot['ks']:.1%}")

    # 7. Scores, Decile Lift & Expected Loss Cut-Off Simulation
    oot_scores = np.clip(base_points + X_oot.values.dot(bucket_points), 300, 850)
    eval_df = pd.DataFrame({
        "score": oot_scores,
        "pd": pd_oot,
        "actual_default": y_oot,
        "ead": oot["loan_amnt"].values,
    })
    eval_df["expected_loss"] = eval_df["pd"] * UNSECURED_LGD * eval_df["ead"]

    decile_df = build_decile_lift_table(eval_df)
    decile_df.to_csv(out_dir / "decile_lift_table.csv", index=False)

    cutoff_df = build_cutoff_simulation(eval_df)
    cutoff_df.to_csv(out_dir / "cutoff_simulation.csv", index=False)
    print("\n--- EXPECTED LOSS CUT-OFF SIMULATION ---")
    print(cutoff_df.to_string(index=False))

    # 8. Sample Automated Decline Reason Codes
    worst_pos = int(eval_df["score"].idxmin())
    active = X_oot.iloc[worst_pos]
    reasons = points_table[points_table["Feature_Bucket"].isin(active[active == 1].index)]
    print(f"\nSample Declined Applicant Score: {eval_df.loc[worst_pos, 'score']:.0f}")
    print("Top Decline Reason Codes:\n", reasons.sort_values("Scorecard_Points").head(3).to_string(index=False))

    # 9. Companion Depth-3 Decision Tree Policy Rules
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, class_weight="balanced", random_state=42)
    tree.fit(X_train, y_train)
    print("\n--- COMPANION DEPTH-3 CREDIT POLICY TREE ---")
    print(export_text(tree, feature_names=list(X_train.columns)))

    # 10. Save Charts
    generate_charts(m_oot, eval_df, cutoff_df, out_dir)
    print(f"\nAll tables and PNG charts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

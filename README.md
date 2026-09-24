# 💰 AI-Powered Loan Approval & Credit Risk Analytics

**AICTE | IBM SkillsBuild — Data Analytics with AI Internship 2026 (BharatCares)**
Author: Sakshi Jaiswal

## 📌 Project Description
Lenders receive thousands of loan applications with different financial, demographic and credit characteristics.
This project analyses historical loan-application data to find **what drives approval**, **which applicant segments are riskiest**, and how a
**machine-learning model** can support (not replace) lending decisions.

Pipeline: `Data cleaning → Feature engineering → KPIs → EDA → Risk segmentation → ML model → Insights → Recommendations`

## 📊 Dataset
- **Kaggle – Loan Approval Prediction Dataset** (4,269 rows, 13 columns):
  https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
- To use it, download `loan_approval_dataset.csv` and place it in the project folder.
- If that file is not found, the notebook auto-generates a **schema-compatible synthetic dataset**
  (`loan_approval_dataset_synthetic.csv`) so it always runs end-to-end. Check the `data_source` line printed in Section 2 to see which one was used.

| Column | Description |
|---|---|
| `no_of_dependents`, `education`, `self_employed` | Applicant profile |
| `income_annum`, `loan_amount`, `loan_term` | Income, requested loan (INR), tenure (years) |
| `cibil_score` | Credit score (300–900) |
| `residential / commercial / luxury / bank asset values` | Applicant assets (INR) |
| `loan_status` | **Target:** Approved / Rejected |

## 🧪 What the notebook does
1. **Cleaning** – standardise text, duplicates, missing/invalid values, outlier flagging
2. **Feature engineering** – `loan_to_income`, `total_assets`, `asset_coverage`, CIBIL / income / loan-size bands
3. **KPIs** – total applications, approval & rejection rate, average loan & income, low-credit share, approved loan value
4. **EDA (11 figures)** – distributions, approval rate by segment, correlation, CIBIL × affordability heatmap
5. **Risk segmentation** – transparent Low / Medium / High score
6. **ML models** – Logistic Regression, Decision Tree, Random Forest (5-fold CV + grid-search tuning), ROC, confusion matrix, permutation importance, probability calibration
7. **Insights, risks, opportunities & recommended actions**

## 🔎 Key results (from the run included in this repo)
- Approval rate **57.5%** across 4,269 applications
- **CIBIL score is the dominant driver**: approval rises from ~11% (Poor) to ~100% (Excellent)
- Loan-to-income is a secondary factor; education, employment type and dependents have little effect
- Tuned Random Forest: **~92% accuracy, ROC-AUC ≈ 0.97**

> ⚠️ The numbers above come from the synthetic dataset. Re-run the notebook with the real Kaggle file to reproduce results on real data.

## 🛠 Technologies
Python 3.9+ · pandas · NumPy · matplotlib · seaborn · scikit-learn · Jupyter Notebook

## ▶️ Setup & Run
```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# (optional) place loan_approval_dataset.csv in this folder
jupyter notebook SakshiJaiswal_LoanApproval_CreditRisk_Analytics.ipynb
```
Run **Kernel → Restart & Run All**. Outputs are written to `figures/`, `loan_data_cleaned.csv` and `results_summary.json`.

## 📁 Project Structure
```
├── SakshiJaiswal_LoanApproval_CreditRisk_Analytics.ipynb   # main analysis
├── SakshiJaiswal_ProjectReport.docx                        # project report
├── requirements.txt
├── README.md
├── loan_approval_dataset_synthetic.csv                     # fallback dataset
├── loan_data_cleaned.csv                                   # generated
├── results_summary.json                                    # generated
└── figures/                                                # generated charts
```

## ⚖️ Disclaimer
Academic project for demonstrating data-driven decision support. The model imitates historical approval decisions and
is **not** a substitute for professional credit assessment or a tool for approving real borrowers.

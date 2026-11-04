"""Dataset contract, cost policy, and labels shared by training and the app."""

from __future__ import annotations

import math
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PACKAGE_ROOT / "data" / "lending_club_sample.csv"
MODEL_PATH = PACKAGE_ROOT / "models" / "credit_pipeline.joblib"
MODEL_CARD_PATH = PACKAGE_ROOT / "models" / "model_card.json"
FIGURE_DIR = PACKAGE_ROOT / "reports" / "figures"

OPENML_DATA_ID = 43729
OPENML_NAME = "Lending-Club-Loan-Data"

DATASET = {
    "name": "Lending Club loan sample (OpenML Lending-Club-Loan-Data)",
    "openml_name": OPENML_NAME,
    "openml_id": OPENML_DATA_ID,
    "openml_url": "https://www.openml.org/d/43729",
    "parquet_url": "https://data.openml.org/datasets/0004/43729/dataset_43729.pq",
    "license": "CC0: Public Domain",
    "license_notes": (
        "OpenML data id 43729 records its licence field as 'CC0: Public Domain'. "
        "The table is a public 9,578-loan extract with a repayment label. It is not "
        "the full Lending Club servicing tape. This project caches that entire extract "
        "at data/lending_club_sample.csv so training, tests, CI, and the app run offline, "
        "with no account and no network."
    ),
    "citation": (
        "Lending-Club-Loan-Data. OpenML data id 43729. "
        "https://www.openml.org/d/43729. Licence recorded by OpenML: CC0: Public Domain."
    ),
    "fetched_with": "sklearn.datasets.fetch_openml(data_id=43729, as_frame=True)",
    "target_definition": (
        "not_fully_paid is 1 when the borrower did not fully repay and 0 when the loan "
        "was fully paid. The model scores the chance the loan is fully paid and approves "
        "when that chance clears the cost cutoff."
    ),
}

TARGET = "not_fully_paid"
# Encoded positive class: fully paid = 1 = approve. Raw 1 means the loan was not fully paid.
POSITIVE_CLASS = 1

# Planning dollars, not observed charge-offs. See COST_SOURCE.
# At the sample's mean rate (12.264%) and mean installment ($319.09), a 36-month
# amortizing loan implies about $9,571 of principal and about $1,917 of gross interest.
# The file does not record term or loss, so these are round weights near that sketch:
# half of the sketched principal, and a net-contribution stand-in below the gross interest.
FP_COST = 5000.0
FN_COST = 600.0
COST_SOURCE = (
    "Planning scenario in dollars, not observed recoveries. The extract has no loan term "
    "and no charge-off balance. At the sample mean interest rate (12.264%) and mean "
    "monthly installment ($319.09), a 36-month amortizing loan implies about $9,571 of "
    "principal and about $1,917 of gross interest. Funding a loan that is not fully paid "
    "is priced at $5,000, near half of that principal sketch. Declining a loan that would "
    "have been fully paid is priced at $600, below that gross interest, as a stand-in for "
    "forgone net contribution after funding costs the file does not record. A different "
    "pair of weights needs a different cutoff. The probability is saved separately so the "
    "desk can move the cutoff without refitting."
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER = 16
REVIEW_BAND = 0.05

# Removed in a side refit. It stays in the deployed score; the refit only measures it.
ABLATION_FEATURE = "fico"

NUMERIC_FEATURES = [
    "credit_policy",
    "int_rate",
    "installment",
    "log_annual_inc",
    "dti",
    "fico",
    "days_with_cr_line",
    "revol_bal",
    "revol_util",
    "inq_last_6mths",
    "delinq_2yrs",
    "pub_rec",
]

CATEGORICAL_FEATURES = ["purpose"]

FEATURE_COLUMNS = [
    "credit_policy",
    "purpose",
    "int_rate",
    "installment",
    "log_annual_inc",
    "dti",
    "fico",
    "days_with_cr_line",
    "revol_bal",
    "revol_util",
    "inq_last_6mths",
    "delinq_2yrs",
    "pub_rec",
]

# Withheld from the deployed score. credit_policy is the platform's own underwriting
# flag. int_rate is the price that platform assigned from its risk grade. Both are
# put back in a side comparison that is not allowed to choose the model.
EXCLUDED_FEATURES = ["credit_policy", "int_rate"]

POLICY_REASON = (
    "credit_policy is Lending Club's own underwriting flag, and int_rate is the interest "
    "rate the platform assigned from its risk grade. Both already contain a lender's "
    "judgment. The deployed score does not see them, so the desk can explain the "
    "application rather than repeat the old price. A side comparison puts the two fields "
    "back at the winning hyperparameters and does not get to pick the model. FICO stays "
    "in the score because it is known from the credit file at application time. A second "
    "refit drops FICO so the report can show how much of the ranking the score itself carries."
)

REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET]

FAIRNESS_REFERENCE = {
    "credit_policy": "Meets policy",
    "fico_band": "760 and over",
    "purpose": "Debt consolidation",
}

GROUP_ORDER = {
    "credit_policy": ["Outside policy", "Meets policy"],
    "fico_band": ["Under 680", "680–719", "720–759", "760 and over"],
    "purpose": [
        "Debt consolidation",
        "Credit card payoff",
        "All other",
        "Home improvement",
        "Major purchase",
        "Small business",
        "Education",
    ],
}

ATTRIBUTE_LABELS = {
    "credit_policy": "Existing credit policy",
    "fico_band": "FICO band",
    "purpose": "Loan purpose",
}

MODEL_ORDER = ("logistic_regression", "random_forest", "gradient_boosting")

DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}

FEATURE_LABELS = {
    "credit_policy": "Lending Club credit policy",
    "purpose": "Loan purpose",
    "int_rate": "Interest rate",
    "installment": "Monthly payment",
    "log_annual_inc": "Annual income",
    "dti": "Debt-to-income ratio",
    "fico": "FICO score",
    "days_with_cr_line": "Length of credit history",
    "revol_bal": "Revolving balance",
    "revol_util": "Revolving utilization",
    "inq_last_6mths": "Credit inquiries in the last 6 months",
    "delinq_2yrs": "Recent delinquencies",
    "pub_rec": "Derogatory public records",
}

CATEGORY_OPTIONS: dict[str, list[str]] = {
    "purpose": [
        "debt_consolidation",
        "credit_card",
        "all_other",
        "home_improvement",
        "small_business",
        "major_purchase",
        "educational",
    ],
}

VALUE_LABELS: dict[str, dict[str, str]] = {
    "purpose": {
        "debt_consolidation": "Debt consolidation",
        "credit_card": "Credit card payoff",
        "all_other": "All other",
        "home_improvement": "Home improvement",
        "small_business": "Small business",
        "major_purchase": "Major purchase",
        "educational": "Education",
    },
    "credit_policy": {
        "0": "Outside policy",
        "1": "Meets policy",
    },
}

# Illustrative files a reviewer can load. They are not real applicants.
# Income is stored as the natural log, which is the column the model sees.
PRESETS: dict[str, dict[str, object]] = {
    "Stronger file": {
        "purpose": "credit_card",
        "installment": 220.0,
        "log_annual_inc": math.log(95000),
        "dti": 6.5,
        "fico": 770,
        "days_with_cr_line": 8000.0,
        "revol_bal": 2500,
        "revol_util": 12.0,
        "inq_last_6mths": 0,
        "delinq_2yrs": 0,
        "pub_rec": 0,
    },
    "Higher-risk file": {
        "purpose": "small_business",
        "installment": 720.0,
        "log_annual_inc": math.log(34000),
        "dti": 26.0,
        "fico": 640,
        "days_with_cr_line": 700.0,
        "revol_bal": 28000,
        "revol_util": 92.0,
        "inq_last_6mths": 6,
        "delinq_2yrs": 2,
        "pub_rec": 1,
    },
}

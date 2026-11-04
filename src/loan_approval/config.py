"""Dataset contract, cost policy, and labels shared by training and the app."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PACKAGE_ROOT / "data" / "credit-g.csv"
MODEL_PATH = PACKAGE_ROOT / "models" / "credit_pipeline.joblib"
MODEL_CARD_PATH = PACKAGE_ROOT / "models" / "model_card.json"
FIGURE_DIR = PACKAGE_ROOT / "reports" / "figures"

OPENML_DATA_ID = 31
OPENML_NAME = "credit-g"

DATASET = {
    "name": "Statlog (German Credit Data)",
    "openml_name": OPENML_NAME,
    "openml_id": OPENML_DATA_ID,
    "openml_url": "https://www.openml.org/d/31",
    "uci_url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    "doi": "https://doi.org/10.24432/C5NC77",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "license_notes": (
        "The UCI Machine Learning Repository distributes Statlog (German Credit Data) "
        "under CC BY 4.0. OpenML redistributes the same table as data id 31 (credit-g) "
        "and records its licence field as 'Public'. This project follows the UCI CC BY 4.0 terms."
    ),
    "citation": (
        "Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77."
    ),
    "fetched_with": "sklearn.datasets.fetch_openml(data_id=31, as_frame=True)",
}

TARGET = "class"
POSITIVE_LABEL = "good"
NEGATIVE_LABEL = "bad"
# 1 = good credit = approve. Matches the published Statlog cost matrix, where class 1 is good.
POSITIVE_CLASS = 1

# Published with the UCI file: approving a bad credit costs 5, declining a good credit costs 1.
FP_COST = 5.0
FN_COST = 1.0

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER = 16
REVIEW_BAND = 0.05

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

FEATURE_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
]

# Dropped from the deployed scorer. They remain available for the fairness audit
# and for the side-by-side "what if we had used them" CV comparison.
EXCLUDED_FEATURES = ["personal_status", "foreign_worker"]

REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET]

# personal_status mixes sex and marital status, and the file has no "female single" code.
FAIRNESS_REFERENCE = {
    "sex": "male",
    "age_band": "25 and over",
    "foreign_worker": "no",
}

GROUP_ORDER = {
    "sex": ["female", "male"],
    "age_band": ["Under 25", "25 and over"],
    "foreign_worker": ["yes", "no"],
}

ATTRIBUTE_LABELS = {
    "sex": "Sex (derived from personal status)",
    "age_band": "Age band",
    "foreign_worker": "Foreign worker",
}

MODEL_ORDER = ("logistic_regression", "random_forest", "gradient_boosting")

DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}

FEATURE_LABELS = {
    "checking_status": "Checking account status",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount (DM)",
    "savings_status": "Savings account / bonds",
    "employment": "Time with current employer",
    "installment_commitment": "Installment rate (1–4 scale)",
    "personal_status": "Personal status and sex",
    "other_parties": "Other debtors or guarantors",
    "residence_since": "Years at current residence",
    "property_magnitude": "Property",
    "age": "Age (years)",
    "other_payment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits at this bank",
    "job": "Job type",
    "num_dependents": "Dependents",
    "own_telephone": "Telephone in the applicant's name",
    "foreign_worker": "Foreign worker",
}

CATEGORY_OPTIONS: dict[str, list[str]] = {
    "checking_status": ["no checking", "<0", "0<=X<200", ">=200"],
    "credit_history": [
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit",
    ],
    "purpose": [
        "new car",
        "used car",
        "furniture/equipment",
        "radio/tv",
        "domestic appliance",
        "repairs",
        "education",
        "retraining",
        "business",
        "other",
    ],
    "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "personal_status": ["male div/sep", "female div/dep/mar", "male single", "male mar/wid"],
    "other_parties": ["none", "co applicant", "guarantor"],
    "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
    "other_payment_plans": ["bank", "stores", "none"],
    "housing": ["rent", "own", "for free"],
    "job": [
        "unemp/unskilled non res",
        "unskilled resident",
        "skilled",
        "high qualif/self emp/mgmt",
    ],
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["yes", "no"],
}

VALUE_LABELS: dict[str, dict[str, str]] = {
    "checking_status": {
        "no checking": "No checking account",
        "<0": "Overdrawn (under 0 DM)",
        "0<=X<200": "Balance from 0 to under 200 DM",
        ">=200": "Balance of 200 DM or more",
    },
    "credit_history": {
        "no credits/all paid": "No credits taken, or all paid duly",
        "all paid": "All credits at this bank paid duly",
        "existing paid": "Existing credits paid duly so far",
        "delayed previously": "Past payment delays",
        "critical/other existing credit": "Critical account, or credits at other banks",
    },
    "purpose": {
        "new car": "New car",
        "used car": "Used car",
        "furniture/equipment": "Furniture or equipment",
        "radio/tv": "Radio or TV",
        "domestic appliance": "Domestic appliance",
        "repairs": "Repairs",
        "education": "Education",
        "retraining": "Retraining",
        "business": "Business",
        "other": "Other",
    },
    "savings_status": {
        "<100": "Under 100 DM",
        "100<=X<500": "100 to under 500 DM",
        "500<=X<1000": "500 to under 1,000 DM",
        ">=1000": "1,000 DM or more",
        "no known savings": "No known savings account",
    },
    "employment": {
        "unemployed": "Unemployed",
        "<1": "Less than 1 year",
        "1<=X<4": "1 to under 4 years",
        "4<=X<7": "4 to under 7 years",
        ">=7": "7 years or more",
    },
    "personal_status": {
        "male div/sep": "Male, divorced or separated",
        "female div/dep/mar": "Female, divorced, separated, or married",
        "male single": "Male, single",
        "male mar/wid": "Male, married or widowed",
    },
    "other_parties": {
        "none": "None",
        "co applicant": "Co-applicant",
        "guarantor": "Guarantor",
    },
    "property_magnitude": {
        "real estate": "Real estate",
        "life insurance": "Life insurance or building-society savings",
        "car": "Car or other property",
        "no known property": "No known property",
    },
    "other_payment_plans": {
        "bank": "Bank",
        "stores": "Stores",
        "none": "None",
    },
    "housing": {
        "rent": "Rent",
        "own": "Own",
        "for free": "Rent-free",
    },
    "job": {
        "unemp/unskilled non res": "Unemployed or unskilled, non-resident",
        "unskilled resident": "Unskilled resident",
        "skilled": "Skilled employee or official",
        "high qualif/self emp/mgmt": "Management, self-employed, or highly qualified",
    },
    "own_telephone": {
        "none": "No telephone in the applicant's name",
        "yes": "Yes, registered in the applicant's name",
    },
    "foreign_worker": {
        "yes": "Yes",
        "no": "No",
    },
}

# Illustrative files a reviewer can load. They are not real applicants.
PRESETS: dict[str, dict[str, object]] = {
    "Stronger file": {
        # "No checking account" is a favorable code in this historical file, so the
        # stronger example keeps it and improves savings, term, purpose, and support.
        "checking_status": "no checking",
        "duration": 12,
        "credit_history": "critical/other existing credit",
        "purpose": "used car",
        "credit_amount": 1200,
        "savings_status": ">=1000",
        "employment": ">=7",
        "installment_commitment": 1,
        "other_parties": "guarantor",
        "residence_since": 4,
        "property_magnitude": "real estate",
        "age": 40,
        "other_payment_plans": "none",
        "housing": "own",
        "existing_credits": 2,
        "job": "skilled",
        "num_dependents": 1,
        "own_telephone": "yes",
    },
    "Higher-risk file": {
        "checking_status": "<0",
        "duration": 48,
        "credit_history": "delayed previously",
        "purpose": "education",
        "credit_amount": 9000,
        "savings_status": "<100",
        "employment": "<1",
        "installment_commitment": 4,
        "other_parties": "none",
        "residence_since": 1,
        "property_magnitude": "no known property",
        "age": 22,
        "other_payment_plans": "bank",
        "housing": "rent",
        "existing_credits": 1,
        "job": "unskilled resident",
        "num_dependents": 1,
        "own_telephone": "none",
    },
}

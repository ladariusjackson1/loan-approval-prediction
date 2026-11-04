# Loan approval prediction

Educational credit-decision project by **Ladarius Jackson** for a business analytics portfolio. A scikit-learn pipeline scores the UCI Statlog German Credit file, compares three classifiers under cross-validation, and turns the published cost matrix into an approve / decline cutoff. A Streamlit desk lets you enter an application and read the fields that moved the score.

This is a teaching model on a public historical file of 1,000 German applications, denominated in Deutsche Marks and donated in 1994. It does not issue a credit offer or a compliance determination.

## The decision

The dataset ships with a cost matrix. Approving a bad credit costs 5. Declining a good credit costs 1. The question is whether a scored cutoff beats the two rules that ignore the model — approve everyone, or decline everyone — on applicants the model never saw during training or model choice.

`good` is the positive class (coded 1). A predicted positive is an approval.

## Data

| | |
| --- | --- |
| Name | Statlog (German Credit Data) |
| Size | 1,000 applicants, 20 attributes, no missing values in the cached file |
| Outcomes | 700 good credits, 300 bad credits |
| How this project fetches it | `sklearn.datasets.fetch_openml(data_id=31, as_frame=True)` |
| OpenML | [credit-g, data id 31](https://www.openml.org/d/31). OpenML records the licence field as `Public`. |
| UCI record | [Statlog German Credit Data, id 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| DOI | [10.24432/C5NC77](https://doi.org/10.24432/C5NC77) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), the license stated by the UCI repository. This project follows those terms. |
| Citation | Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77. |

The table used for training, tests, and CI is cached at `data/credit-g.csv`. `PYTHONPATH=src python -m loan_approval.train --download` refreshes it from OpenML.

`personal_status` mixes sex and marital status, and the file has no code for single women. The fairness report derives a sex label from that field. It is an audit column, and it is a blunt one.

![Seven in ten files are good credits](reports/figures/class_balance.png)

## Method

The holdout is a stratified 20% split (200 rows, seed 42). Model choice and the cutoff are fit on the other 800 rows only.

Preprocessing lives inside one `sklearn` `Pipeline`, so each cross-validation fold refits it:

- numeric fields: median imputation, then `StandardScaler`
- categorical fields: most-frequent imputation, then one-hot encoding with `handle_unknown="ignore"`

Scaling gives the logistic penalty comparable units. A per-feature rescaling does not change tree splits, so the forest and the booster neither gain nor lose from sharing that step.

The deployed score drops `personal_status` and `foreign_worker`. Age stays in the model and is audited by band. A side comparison puts the two dropped fields back, and another refit removes age, both at the winning hyperparameters. Neither comparison is allowed to pick the deployed model.

Three classifiers share the same shuffled `StratifiedKFold` (5 folds, seed 42):

- logistic regression (`lbfgs`, L2)
- random forest
- gradient boosting

Each model is scored once with library defaults, then with a randomized search (`n_iter=16`, or the full 12-cell logistic grid when that grid is smaller). The search metric is ROC-AUC. The winner is the highest mean cross-validated ROC-AUC, then the smaller fold standard deviation, then the simpler model. The cutoff minimizes average Statlog cost on out-of-fold training probabilities. A tie takes the higher cutoff. The winner is refit on all 800 training rows and scored once on the holdout.

The desk explains a single file by replacing each field with the training median or the most common category and recording the change in predicted probability. That is an ablation. Correlated fields share credit.

## Results

The deployed model is a **random forest**. Mean cross-validated ROC-AUC is 0.795. Gradient boosting rounds to the same 0.795 (0.7948 before rounding). Logistic regression reaches 0.789. Fold standard deviations are about 0.07, so this is a close race. The selection rule still has to name one model. The holdout ranks them the same way: forest ROC-AUC **0.782**, boosting 0.777, logistic regression 0.761.

| Model | CV ROC-AUC, default | CV ROC-AUC, tuned | Holdout ROC-AUC | Holdout F1 | Holdout cost / applicant | Cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.783 ± 0.079 | 0.789 ± 0.079 | 0.761 | 0.645 | 0.525 | 0.80 |
| **Random forest** | 0.790 ± 0.076 | **0.795 ± 0.070** | **0.782** | **0.632** | **0.490** | **0.69** |
| Gradient boosting | 0.778 ± 0.061 | 0.795 ± 0.069 | 0.777 | 0.696 | 0.505 | 0.77 |

Tuned settings: logistic `C=0.1` with no class weight; forest with 400 trees, depth 16, `min_samples_leaf=1`, `max_features="sqrt"`, `class_weight="balanced"`; boosting with 200 trees, learning rate 0.05, depth 2, subsample 0.8, `min_samples_leaf=5`.

![Cross-validated ROC-AUC before and after search](reports/figures/model_comparison.png)

![Holdout ROC curves](reports/figures/roc_curves.png)

F1 of 0.632 belongs to a strict cutoff. At 0.69 the holdout approves 72 of 200 files (36%).

|  | Decline | Approve |
| --- | ---: | ---: |
| Bad credit | 55 | 5 |
| Good credit | 73 | 67 |

Of those 72 approvals, 67 are good credits (precision 93.1%). The rule approves 67 of 140 good credits (recall 47.9%) and declines 55 of 60 bad credits (91.7%). Holdout cost is 98, against 140 for declining everyone and 300 for approving everyone. Per applicant that is **0.490**, against 0.700 and 1.500: a 30% reduction versus the better naive rule.

![Holdout confusion matrix](reports/figures/confusion_matrix.png)

![Training out-of-fold cost by cutoff](reports/figures/threshold_cost.png)

Holdout probabilities track observed frequencies with some wobble on 200 rows (Brier score 0.177). Treat the output as a rank with a cutoff.

![Holdout calibration](reports/figures/calibration.png)

### What the score uses

On the holdout, shuffling checking-account status drops ROC-AUC by about 0.15. Savings, loan duration, credit history, and credit amount are the next fields whose shuffle costs are larger than their noise. Shuffling age does not move holdout AUC.

![Permutation importance](reports/figures/feature_importance.png)

Putting `personal_status` and `foreign_worker` back, at the same forest hyperparameters, changes cross-validated ROC-AUC by less than 0.001. Removing age and refitting changes it by about −0.011, which is small beside a fold standard deviation of 0.07.

The logistic model is kept as a directional companion. It is not the deployed score. In that fit, no checking account and a used-car purpose raise the log-odds of good credit. An overdrawn checking account, a new-car purpose, an education purpose, savings under 100 DM, and a longer term lower them. The code `critical/other existing credit` also raises log-odds in this file. That category often means the applicant already has credit at other banks, and the name is easy to misread.

![Logistic coefficients](reports/figures/logistic_coefficients.png)

### Fairness audit

Rates below use out-of-fold training scores, so each row was scored by a model that did not train on it, at the deployed cutoff of 0.69. The four-fifths ratio is a US-style screening heuristic. It is not a legal conclusion about this German teaching set. The reference groups are male, age 25 and over, and `foreign_worker = no`.

| Group | n | Historical good rate | Approval rate | True approval rate | Disparate impact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Female | 250 | 64.4% | 35.6% | 50.3% | 0.89 |
| Male | 550 | 72.5% | 40.0% | 50.4% | 1.00 |
| Under 25 | 121 | 59.5% | 18.2% | 26.4% | 0.43 |
| 25 and over | 679 | 71.9% | 42.3% | 53.9% | 1.00 |
| Foreign worker, yes | 772 | 69.3% | 38.3% | 50.3% | 0.83 |
| Foreign worker, no | 28 | 89.3% | 46.4% | 52.0% | 1.00 |

![Approval rate against the historical good-credit rate](reports/figures/fairness.png)

For sex, the approval gap is about 4 points, smaller than the gap in historical good rates, and the true-approval rates match. Disparate impact of 0.89 sits inside an 80% screen. A holdout slice of 60 women is wider (disparate impact 0.67). That slice is the noisy one. The out-of-fold table is the estimate to read.

For age, applicants under 25 are approved at 18% against 42% for everyone else. Disparate impact is 0.43, and the true-approval rate is 26% against 54%. The deployed forest barely uses the age column itself, so the gap is carried by correlated fields: checking status, savings, term, amount, employment. Deleting age from a refit is a small AUC change. A policy that wants closer approval rates by age has to constrain those proxies and measure the gap again.

The group `foreign_worker = no` has 28 training rows. The 0.83 ratio is inside the screen and too thin to manage to.

## Credit desk

```bash
streamlit run app/streamlit_app.py
```

The form scores as you edit. It shows P(good credit), the cutoff, a referral flag when the probability sits within 5 points of the cutoff, and the fields that move the probability relative to a typical training file. The slider changes risk appetite. The probability stays put. Personal status and foreign-worker status are withheld from the form for the same reason they are withheld from the model.

Example presets cover a typical training profile, a stronger file, and a higher-risk file.

## How to run

Python 3.11 or newer. Development here used 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m loan_approval.train
PYTHONPATH=src python -m loan_approval.train --download   # refresh data/credit-g.csv from OpenML

pytest
ruff check .
ruff format --check .
streamlit run app/streamlit_app.py
```

`pytest` reads `pythonpath = ["src"]` from `pyproject.toml` and fails if coverage of `loan_approval` drops under 85%. GitHub Actions (`.github/workflows/ci.yml`) runs Ruff and pytest on Python 3.12 using the cached CSV. It does not retrain.

Saved artifacts: `models/credit_pipeline.joblib` and `models/model_card.json`. They were fit with scikit-learn 1.9.1. Retrain after an upgrade.

## Layout

```
src/loan_approval/    data, pipeline, search, cost cutoff, fairness, explanations
app/streamlit_app.py  credit desk
data/credit-g.csv     cached OpenML / UCI table
models/               fitted pipeline and model card
reports/figures/      charts in this readme
tests/                pytest suite
```

## Business takeaways

1. Under the published 5:1 loss, the scored cutoff earns its keep. Holdout cost is 0.490 per applicant, against 0.700 for declining everyone and 1.500 for approving everyone. The policy approves about one file in three, and about 93% of those approvals are good credits. It declines 55 of 60 bad credits in the holdout.
2. F1 of 0.632 is what a strict cutoff looks like when many good credits are turned away on purpose. Cost, the false-approval count, and ROC-AUC are the figures that match the decision.
3. Algorithm choice is a small lever. Tuning moved cross-validated AUC by less than a point, and the forest-versus-boosting gap is inside the fold noise. The selection rule names the forest. Holdout cost agrees.
4. The checking account is the field worth getting right. Savings, term, credit history, and amount follow. Sex and foreign-worker status add no measurable ranking accuracy. Age adds little.
5. The disparity that remains is age, and it is mostly indirect. Under-25 approval sits far outside a four-fifths screen even though shuffling the age column does not hurt the deployed forest. Equal treatment by age means constraining the correlated fields and measuring again.
6. Send files within 5 points of the cutoff to an analyst. The cutoff is optimal for the 5:1 matrix only. A different loss ratio needs a different threshold, which is why the desk shows the probability separately from the stamp.
7. Quote the probability as a rank, with the calibration chart in view. Two hundred holdout rows are enough to see that the forest is roughly calibrated, and not enough to treat every percentage point as a frequency.

## Limitations

The book is 1,000 historical applications. Labels are the dataset's good/bad marks, with no reject inference and no through-the-door population from a live lender. Amounts are in Deutsche Marks. `personal_status` cannot separate sex from marital status. The foreign-worker reference group and the under-25 holdout slice are small. Explanations are one-field ablations, so correlated inputs share credit. The saved pipeline is tied to the scikit-learn version in `requirements.txt`.

## License

Code is released under the MIT license. The dataset remains under CC BY 4.0. Cite Hofmann (1994) when you reuse the file.

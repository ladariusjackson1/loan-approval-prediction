# Loan approval prediction

[![Live demo](https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://loan-approval-prediction.streamlit.app)

Educational loan-decision project by **Ladarius Jackson**. A scikit-learn pipeline scores a public Lending Club sample, compares three classifiers under cross-validation, and turns a stated dollar loss into an approve / decline cutoff. A Streamlit desk scores one application and says, in plain English, which fields pushed the decision.

This is a teaching model on 9,578 historical loans. It does not issue a credit offer, a denial, or a compliance determination. The dollar figures below are a planning scenario. The file has no charge-off balance, so they are not a lender's profit and loss.

## The business question

A consumer lender has to fund a loan or turn it away before anyone knows whether it will be repaid. Approving everyone books the loans that never fully repay. Declining everyone protects that loss and gives up the loans that would have paid. The useful question is whether a score, with a cutoff chosen only on the training book, beats both of those rules on loans it has not seen.

In this file, **fully paid** is the positive class. A predicted positive is an approval.

The planning weights, used only to pick the cutoff and to talk about money:

| Error | What happened | Planning cost |
| --- | --- | ---: |
| False approval | Fund a loan that is not fully paid | $5,000 |
| False decline | Turn away a loan that is fully paid | $600 |

Why these two numbers. The extract has no term and no loss amount. At the sample's mean interest rate (12.264%) and mean monthly installment ($319.09), a 36-month amortizing loan implies about $9,571 of principal and about $1,917 of gross interest. Five thousand dollars is a round weight near half of that principal sketch. Six hundred dollars is a round weight below that gross interest, standing in for net contribution after funding costs the file does not record. Change the weights and the cutoff moves. The desk keeps the score and the cutoff separate for that reason.

## Data

| | |
| --- | --- |
| Name | Lending Club loan sample, OpenML `Lending-Club-Loan-Data` |
| OpenML | [data id 43729](https://www.openml.org/d/43729) |
| Size | 9,578 loans, 13 predictors, no missing values in the cached file |
| Outcomes | 8,045 fully paid (84.0%), 1,533 not fully paid (16.0%) |
| How this project fetches it | `sklearn.datasets.fetch_openml(data_id=43729, as_frame=True)` |
| License | CC0: Public Domain, the licence field OpenML records for data id 43729 |
| What this file is | The entire OpenML extract. It is already a public sample, not the multi-million-row Lending Club servicing tape |

No account and no API key. The table used for training, tests, CI, and the app is cached at `data/lending_club_sample.csv`, so those all run offline. `PYTHONPATH=src python -m loan_approval.train --download` refreshes the cache from OpenML.

The target column in the file is `not_fully_paid` (1 = the borrower did not fully repay). The model scores the complementary event.

The deployed score does **not** see `credit_policy` (Lending Club's own underwriting flag) or `int_rate` (the price the platform assigned from its risk grade). Both already contain a lender's judgment. They stay in the cache for a side comparison. The extract has no sex, age, or race fields, so this project cannot run a protected-class audit.

![Eighty-four percent of the sample was fully paid](reports/figures/class_balance.png)

## Method

The holdout is a stratified 20% split (1,916 loans, seed 42). Model choice and the cutoff are fit on the other 7,662 loans only.

Preprocessing lives inside one `sklearn` pipeline, so each cross-validation fold refits it:

- numeric fields: median imputation, then `StandardScaler`
- loan purpose: most-frequent imputation, then one-hot encoding with `handle_unknown="ignore"`

Scaling gives the logistic penalty comparable units. A per-feature rescaling does not change tree splits, so the forest and the booster neither gain nor lose from sharing that step.

Three classifiers share the same shuffled `StratifiedKFold` (5 folds, seed 42):

- logistic regression (`lbfgs`, L2)
- random forest
- gradient boosting

Each model is scored once with library defaults, then with a randomized search (`n_iter=16`, or the full 12-cell logistic grid, which is smaller). The search metric is ROC-AUC. The winner is the highest mean cross-validated ROC-AUC, then the smaller fold standard deviation, then the simpler model. The cutoff minimizes average planning cost on out-of-fold training scores. A tie takes the higher cutoff. The winner is refit on all 7,662 training loans and scored once on the holdout.

A second refit, at the winning hyperparameters, puts `credit_policy` and `int_rate` back. A third refit drops FICO. Neither is allowed to pick the deployed model.

The desk explains one application by replacing each field with the training median or the most common purpose and recording the change in score. That is an ablation. Correlated fields share credit.

## Results

The deployed model is **logistic regression**. Mean cross-validated ROC-AUC is 0.670 ± 0.015. Gradient boosting reaches 0.664 ± 0.015. The random forest reaches 0.662 ± 0.015. The fold standard deviation is about 0.015, so this is a close race. The selection rule still has to name one model. The holdout ranks them the same way: logistic ROC-AUC **0.704**, boosting 0.699, forest 0.695. The holdout is one split. It is higher than the cross-validated number, and the cross-validated number is the one the selection rule used.

| Model | CV ROC-AUC, default | CV ROC-AUC, tuned | Holdout ROC-AUC | Holdout F1 | Holdout cost / loan | Cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Logistic regression** | 0.669 ± 0.017 | **0.670 ± 0.015** | **0.704** | 0.409 | **$432.36** | **0.64** |
| Random forest | 0.643 ± 0.018 | 0.662 ± 0.015 | 0.695 | 0.391 | $439.87 | 0.68 |
| Gradient boosting | 0.654 ± 0.014 | 0.664 ± 0.015 | 0.699 | 0.438 | $433.19 | 0.89 |

Tuned settings: logistic `C=10` with `class_weight="balanced"`; forest with 400 trees, depth 16, `min_samples_leaf=10`, `max_features="sqrt"`, `class_weight="balanced"`; boosting with 200 trees, learning rate 0.05, depth 2, subsample 0.8, `min_samples_leaf=5`.

Tuning barely moved logistic regression (0.669 to 0.670). It moved the forest from 0.643 to 0.662 and boosting from 0.654 to 0.664. After that search, the dollar gap between the three models on the holdout is about $7 a loan. The gap that matters is the one against the rules that ignore the model.

![Cross-validated ROC-AUC before and after search](reports/figures/model_comparison.png)

![Holdout ROC curves](reports/figures/roc_curves.png)

F1 of 0.409 belongs to a strict cutoff. At 0.64 the holdout approves 443 of 1,916 loans (23.1%).

|  | Decline | Approve |
| --- | ---: | ---: |
| Not fully paid | 284 | 23 |
| Fully paid | 1,189 | 420 |

Of those 443 approvals, 420 are fully paid (precision 94.8%). The rule approves 420 of 1,609 fully paid loans (recall 26.1%) and declines 284 of 307 loans that were not fully paid (92.5%).

![Holdout confusion matrix](reports/figures/confusion_matrix.png)

![Training out-of-fold planning cost by cutoff](reports/figures/threshold_cost.png)

### Money on the holdout book

Same 1,916 loans, same planning weights. These totals are that book only. They are not annualized and they are not recoveries.

| Rule | Planning cost | Versus the scored cutoff |
| --- | ---: | ---: |
| Approve everyone | $1,535,000.00 | scored cutoff costs $706,600.00 less |
| Decline everyone | $965,400.00 | scored cutoff costs $137,000.00 less |
| **Scored cutoff (0.64)** | **$828,400.00** | **$432.36 per loan** |

Decline-everyone is the cheaper rule that ignores the model ($503.86 per loan). The scored cutoff saves **$71.50 per loan** against it, **$137,000.00** on this holdout. The arithmetic is the 420 good approvals the rule is willing to book, minus the 23 loans it funds that are not fully paid: 420 × $600 − 23 × $5,000 = $137,000.

### The score is a rank, not a frequency

Holdout Brier score is 0.223. A constant guess at the holdout fully-paid rate (84.0%) would score about 0.135. The balanced class weight separates the classes and pulls the raw output away from the observed rate, so 0.64 is the cutoff on this model's score, not a statement that 64% of similar loans repay. The calibration chart is the picture of that. Quote the score against the cutoff.

![Holdout calibration](reports/figures/calibration.png)

### What pushed the score

On the holdout, shuffling FICO drops ROC-AUC by about 0.078. Inquiries in the last six months, annual income, loan purpose, and the monthly payment are the next fields whose shuffle costs are larger than their noise. Debt-to-income, length of credit history, recent delinquencies, and public records do not move holdout AUC once the other fields are present. Public records are rare in this extract, so a near-zero shuffle cost is a thin sample, not a finding that public records never matter.

![Permutation importance](reports/figures/feature_importance.png)

Dropping FICO and refitting, at the same logistic hyperparameters, lowers cross-validated ROC-AUC by about 0.012 (0.670 to 0.658). That is inside a fold standard deviation of 0.015. The shuffle hurts more than the refit because the refit can lean on income, inquiries, and purpose, which move with FICO. Putting `credit_policy` and `int_rate` back changes cross-validated ROC-AUC by about +0.001. The old policy flag and the platform's price do not buy ranking accuracy on top of the application fields.

The logistic coefficients are the deployed score, on standardized inputs. A small-business purpose lowers the log-odds of a fully paid loan the most. A credit-card payoff, a higher FICO, a major-purchase purpose, and higher income raise them. More inquiries, a larger monthly payment, and a higher revolving balance lower them.

![Logistic coefficients](reports/figures/logistic_coefficients.png)

### Where the approvals land

Rates below use out-of-fold training scores, so each loan was scored by a model that did not train on it, at the deployed cutoff of 0.64. The four-fifths ratio is a screening arithmetic. It is not a legal conclusion. This extract has no sex, age, or race, and the reference groups are "meets the old credit policy," FICO 760 and over, and debt consolidation.

| Group | n | Fully paid | Approval rate | True approval rate | Ratio vs reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Meets old policy | 6,199 | 86.6% | 30.9% | 33.5% | 1.00 |
| Outside old policy | 1,463 | 72.8% | 2.9% | 3.8% | 0.09 |
| FICO under 680 | 1,735 | 77.2% | 1.7% | 2.2% | 0.02 |
| FICO 680–719 | 3,007 | 82.8% | 8.8% | 9.6% | 0.11 |
| FICO 720–759 | 1,974 | 87.2% | 44.2% | 47.2% | 0.53 |
| FICO 760 and over | 946 | 93.3% | 83.8% | 85.8% | 1.00 |
| Debt consolidation | 3,151 | 84.4% | 20.3% | 22.2% | 1.00 |
| Credit card payoff | 1,028 | 88.1% | 43.1% | 46.7% | 2.12 |
| All other | 1,878 | 83.7% | 24.7% | 28.3% | 1.21 |
| Home improvement | 498 | 84.3% | 31.7% | 34.5% | 1.56 |
| Major purchase | 338 | 89.6% | 58.6% | 61.1% | 2.88 |
| Education | 267 | 80.9% | 17.2% | 19.9% | 0.85 |
| Small business | 502 | 71.9% | 2.0% | 2.5% | 0.10 |

![Approval rate against the historical fully-paid rate](reports/figures/fairness.png)

The model never sees the old policy flag, and it still approves 2.9% of outside-policy loans against 30.9% of loans that met the old policy. That gap is carried by the fields it does see, mostly FICO and inquiries. FICO is in the score on purpose, so approval rising from 1.7% under 680 to 83.8% at 760 and over is the cutoff doing what it was asked to do. The fully-paid rate also rises with FICO, from 77.2% to 93.3%, and the approval gap is much steeper than that repayment gap because the cutoff is strict.

Small business is the purpose with the lowest fully-paid rate in the training book (71.9%) and the lowest approval rate (2.0%). Credit-card payoff and major purchase are approved well above the debt-consolidation reference. Those purpose gaps are large enough to show a reviewer, and too entangled with FICO and loan size to treat as a pure purpose effect.

## Credit desk

```bash
streamlit run streamlit_app.py
```

Streamlit Community Cloud uses that same command from the repository root. The entry point is `streamlit_app.py`. Dependencies are pinned in `requirements.txt`. Python 3.12 is pinned in `runtime.txt`. Theme and browser settings are in `.streamlit/config.toml`.

The form scores as you edit. It shows the model score, the cutoff, a referral flag when the score sits within 5 points of the cutoff, and a plain-English note of what pushed the decision toward approval or toward a decline. The slider changes risk appetite. The score stays put. Interest rate and the old credit-policy flag are not on the form.

![Desk: a higher-risk file, the decline, and the fields that pushed it](docs/demo.gif)

Example presets cover a typical training profile, a stronger file, and a higher-risk file.

## How to run

Python 3.11 or newer. Development here used 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

PYTHONPATH=src python -m loan_approval.train
PYTHONPATH=src python -m loan_approval.train --download   # refresh the OpenML cache

pytest
ruff check .
ruff format --check .
streamlit run streamlit_app.py
```

`pytest` reads `pythonpath = ["src"]` from `pyproject.toml` and fails if coverage of `loan_approval` drops under 85%. GitHub Actions (`.github/workflows/ci.yml`) runs Ruff and pytest on Python 3.12 using the cached CSV. It does not retrain and it does not call OpenML.

Saved artifacts: `models/credit_pipeline.joblib` and `models/model_card.json`. They were fit with scikit-learn 1.9.1. Retrain after an upgrade.

## Layout

```
streamlit_app.py      Community Cloud entry point, repo root
app/streamlit_app.py  loan desk
.streamlit/config.toml
runtime.txt           Python 3.12 for Community Cloud
requirements.txt      pinned runtime dependencies
src/loan_approval/    data, pipeline, search, dollar cutoff, segments, explanations
data/lending_club_sample.csv
models/               fitted pipeline and model card
reports/figures/      charts in this readme
docs/demo.gif         short desk recording
tests/                pytest suite
```

## Business findings

1. Under the stated weights, the scored cutoff earns its keep on loans it did not see. Holdout planning cost is $828,400.00, against $965,400.00 for declining everyone and $1,535,000.00 for approving everyone. That is $137,000.00 saved versus the better naive rule, $71.50 a loan, by approving 23.1% of the holdout and keeping 94.8% of those approvals fully paid.
2. F1 of 0.409 is what a strict cutoff looks like when most fully paid loans are turned away on purpose. Cost, the false-approval count (23 of 307), and ROC-AUC are the figures that match the decision.
3. Algorithm choice is a small lever once each model has been tuned. The cross-validated gap from logistic regression to boosting is 0.006, inside a fold standard deviation of 0.015. The selection rule names logistic regression. Holdout cost agrees, by about $7 a loan.
4. FICO is the field worth getting right, then recent inquiries, income, purpose, and the monthly payment. Debt-to-income adds nothing measurable once those are in the score. The old credit-policy flag and the platform's interest rate add about 0.001 of cross-validated AUC. They are not worth feeding back into a new screen.
5. The cutoff concentrates approvals in high FICO, credit-card payoff, and major purchase, and it almost never approves a small-business purpose or a file the old policy would have failed. That is the score and the $5,000 / $600 weights, not a protected-class result. This extract cannot support a sex, age, or race audit.
6. Send files within 5 points of the cutoff to an analyst. The cutoff is optimal for these two weights only. A different loss ratio needs a different threshold.
7. Do not quote the score as the share of similar loans that repay. Balanced class weights put the Brier score at 0.223, worse than guessing the 84% base rate (about 0.135). Use the score to rank the file against the cutoff, with the calibration chart in view.

## Limitations

The book is a 9,578-loan public extract, not a lender's through-the-door population, and there is no reject inference. Labels are "fully paid" versus "not fully paid," which is broader than a charge-off. The file has no term, no principal, and no loss amount, so the $5,000 and $600 weights are a scenario tied to a 36-month sketch at the sample means. The monthly payment still carries some of the offer, because installment depends on amount, rate, and term, even though the rate itself is withheld. Explanations are one-field ablations, so correlated inputs share credit. A small swap can point the surprising way: on the higher-risk example, two recent delinquencies raise the score by about one point versus none, because that column is weak once FICO, inquiries, and purpose are held fixed. Treat a move of a point or two as noise. The balanced class weight means the score is not a calibrated probability. Segment rates for education, major purchase, and outside-policy loans are smaller slices. The saved pipeline is tied to the scikit-learn version in `requirements.txt`.

## License

Code is released under the MIT license. The cached extract is used under the CC0 licence recorded by OpenML for data id 43729. Cite that OpenML record when you reuse the file.

# Model card: loan approval score

Educational score on a public Lending Club sample. Not a credit offer, a denial, or a compliance determination.

## Model details

- **Deployed model:** Logistic regression
- **Provisional leader on cross-validated ROC-AUC:** Logistic regression
- **Selection:** Highest mean stratified cross-validated ROC-AUC on uncalibrated training scores. Ties in that mean break toward the smaller fold standard deviation, then toward the simpler model (logistic regression, then histogram gradient boosting). A more complex leader is kept only when the paired bootstrap 95% interval for its out-of-fold ROC-AUC advantage excludes zero. If the interval includes zero, the simpler model is deployed. The holdout set is not used to choose the model or the threshold. After the model is chosen, an isotonic map is fit on out-of-fold scores and the cost cutoff is chosen on the calibrated out-of-fold probabilities.
- **What the run actually did:** Logistic regression is deployed. It led on cross-validated ROC-AUC, and the paired out-of-fold bootstrap 95% interval for its ROC-AUC advantage over Histogram gradient boosting excludes zero.
- **Person responsible:** Ladarius Jackson
- **Library:** scikit-learn 1.9.1
- **Positive class:** fully paid, coded as 1, decision = approve
- **Inputs:** purpose, installment, log_annual_inc, dti, fico, days_with_cr_line, revol_bal, revol_util, inq_last_6mths, delinq_2yrs, pub_rec
- **Withheld:** credit_policy, int_rate. credit_policy is Lending Club's own underwriting flag, and int_rate is the interest rate the platform assigned from its risk grade. Both already contain a lender's judgment. The deployed score does not see them, so the desk can explain the application rather than repeat the old price. A side comparison puts the two fields back at the winning hyperparameters and does not get to pick the model. FICO stays in the score because it is known from the credit file at application time. A second refit drops FICO so the report can show how much of the ranking the score itself carries.

## Intended use

Compare a cost-weighted approve / decline cutoff with two rules that ignore the model, on a public sample a reviewer can rerun. The Streamlit desk scores one file, simulates a book at a chosen cutoff, and scores a CSV.

Out of scope: production underwriting, pricing, adverse-action letters, fair-lending certification, and any decision about a real applicant.

## Factors

The score sees the application fields listed above. Segment rates are reported for the old credit-policy flag, FICO band, and loan purpose. Those segments are evaluation slices. FICO and purpose are also inputs. The old policy flag is not an input.

## Metrics

Model choice uses stratified cross-validated ROC-AUC on uncalibrated scores, with a paired percentile bootstrap of the out-of-fold gap. The holdout bootstrap interval is reported and is not used to pick the model or the cutoff.

After the model is chosen, isotonic regression maps the score to a probability. The cutoff minimizes average planning cost on cross-fit calibrated out-of-fold scores. Ties break toward the higher cutoff.

Planning weights: $5,000.00 for funding a loan that is not fully paid, $600.00 for declining a loan that is fully paid. Planning scenario in dollars, not observed recoveries. The extract has no loan term and no charge-off balance. At the sample mean interest rate (12.264%) and mean monthly installment ($319.09), a 36-month amortizing loan implies about $9,571 of principal and about $1,917 of gross interest. Funding a loan that is not fully paid is priced at $5,000, near half of that principal sketch. Declining a loan that would have been fully paid is priced at $600, below that gross interest, as a stand-in for forgone net contribution after funding costs the file does not record. A different pair of weights needs a different cutoff. The probability is saved separately so the desk can move the cutoff without refitting.

Brier score is reported for the uncalibrated score, the calibrated probability, and a constant prediction at the holdout fully-paid rate.

## Training and evaluation data

- **Dataset:** Lending Club loan sample (OpenML Lending-Club-Loan-Data)
- **OpenML:** [43729](https://www.openml.org/d/43729)
- **Licence:** CC0: Public Domain
- **How it was fetched:** sklearn.datasets.fetch_openml(data_id=43729, as_frame=True)
- **Target:** not_fully_paid is 1 when the borrower did not fully repay and 0 when the loan was fully paid. The model scores the chance the loan is fully paid and approves when that chance clears the cost cutoff.
- **Fully-paid rate in the file:** 84.0%
- **Split:** stratified holdout of 20.0%, seed 42. Train 7662 / holdout 1916.
- **Cross-validation:** 5 stratified folds, seed 42. Search iterations requested: 16.
- **Bootstrap draws:** 1000.
- **Cache hash:** `623b858479ac7886fee10e9147c10b2e85a2a6cf8dbabca5ce41e89969f63bcc`

OpenML data id 43729 records its licence field as 'CC0: Public Domain'. The table is a public 9,578-loan extract with a repayment label. It is not the full Lending Club servicing tape. This project caches that entire extract at data/lending_club_sample.csv so training, tests, CI, and the app run offline, with no account and no network.

## Quantitative analysis

### Ranking comparison

Cross-validated ROC-AUC is the selection metric. Holdout ROC-AUC below is the uncalibrated ranking score. The interval is a percentile bootstrap on the holdout rows.

| Model | Stage | CV ROC-AUC | Holdout ROC-AUC | Holdout 95% interval |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | default | 0.669 ± 0.017 | — | — |
| Logistic regression | tuned | 0.670 ± 0.015 | 0.704 | 0.673 to 0.736 |
| Histogram gradient boosting | default | 0.630 ± 0.013 | — | — |
| Histogram gradient boosting | tuned | 0.658 ± 0.012 | 0.697 | 0.666 to 0.729 |

### Bootstrap gaps

| Sample | Reference minus other | Difference | 95% interval | Includes zero | Used to choose |
| --- | --- | ---: | ---: | --- | --- |
| out_of_fold | logistic_regression − hist_gradient_boosting | 0.013 | 0.003 to 0.021 | no | yes |
| holdout | logistic_regression − hist_gradient_boosting | 0.006 | -0.010 to 0.022 | yes | no |

### Deployed policy, after calibration

- **Cutoff:** 0.89
- **Holdout ROC-AUC (calibrated scores):** 0.703 (95% bootstrap interval 0.674 to 0.734)
- **Uncalibrated holdout ROC-AUC:** 0.704
- **Holdout average precision:** 0.915
- **Holdout F1 at the cutoff:** 0.406
- **Precision / recall:** 0.950 / 0.258
- **Approval rate:** 22.8%
- **Confusion matrix** (rows are not fully paid, fully paid; columns are decline, approve): [[285, 22], [1194, 415]]
- **Counts:** true decline 285, false approval 22, false decline 1194, true approval 415, n = 1916
- **Brier, calibrated:** 0.124
- **Brier, uncalibrated:** 0.223
- **Brier, constant holdout fully-paid rate:** 0.135
- **Planning cost per holdout loan:** $431.32
- **Planning cost on the holdout book:** $826,400.00
- **Approve-everyone cost:** $1,535,000.00
- **Decline-everyone cost:** $965,400.00
- **Saved versus the cheaper naive rule (decline_all):** $139,000.00 ($72.55 per loan)

### Side refits, not used to choose the model

**Withheld fields put back.** Cross-validated ROC-AUC 0.671 ± 0.015, a difference of 0.001 versus the policy feature set. Same model family and hyperparameters as the deployed scorer, refit in cross-validation with credit_policy, int_rate added back. This is not a second search, and it is not used to pick the deployed model.

**FICO removed.** Cross-validated ROC-AUC 0.658 ± 0.015, a difference of -0.012 versus the policy feature set. Same model family and hyperparameters, with fico removed from an otherwise unchanged policy feature set. Not used to choose the deployed model.

### Where approvals land

Rates use out-of-fold scores from the deployed model, at the deployed cutoff, so each training loan was scored by a model that did not train on it. The reference groups are meets-policy, FICO 760 and over, and debt consolidation.

| Segment | Group | n | Fully paid | Approval rate | Ratio vs reference |
| --- | --- | ---: | ---: | ---: | ---: |
| credit_policy | Meets policy | 6199 | 86.6% | 30.5% | 1.00 |
| credit_policy | Outside policy | 1463 | 72.8% | 2.9% | 0.10 |
| fico_band | 680–719 | 3007 | 82.8% | 8.5% | 0.10 |
| fico_band | 720–759 | 1974 | 87.2% | 43.8% | 0.52 |
| fico_band | 760 and over | 946 | 93.3% | 83.5% | 1.00 |
| fico_band | Under 680 | 1735 | 77.2% | 1.4% | 0.02 |
| purpose | All other | 1878 | 83.7% | 24.5% | 1.23 |
| purpose | Credit card payoff | 1028 | 88.1% | 42.6% | 2.13 |
| purpose | Debt consolidation | 3151 | 84.4% | 20.0% | 1.00 |
| purpose | Education | 267 | 80.9% | 16.1% | 0.81 |
| purpose | Home improvement | 498 | 84.3% | 31.3% | 1.57 |
| purpose | Major purchase | 338 | 89.6% | 58.3% | 2.92 |
| purpose | Small business | 502 | 71.9% | 2.2% | 0.11 |

## Ethical considerations

A strict cutoff concentrates approvals in higher FICO bands and in purposes the training book repaid more often. That is the score and the planning weights. It is not evidence about a protected class, because this extract does not contain one.

The plain-English decline reasons and the one-field counterfactual are review aids. They are not reasons that satisfy the Fair Credit Reporting Act, and they are not instructions to alter a credit file.

## Caveats and recommendations

- This is a teaching model on a public 9,578-loan extract. It does not issue a credit offer, a denial, or a compliance determination.
- The extract has no sex, age, or race fields, so it cannot support a protected-class audit. FICO band, loan purpose, and the old credit-policy flag are not protected classes.
- The four-fifths ratio is a screening arithmetic on those segments. It is not a legal conclusion under the Equal Credit Opportunity Act or any other statute.
- There is no reject inference. The file is loans that were made, not applications that were turned away.
- The label is fully paid versus not fully paid, which is broader than a charge-off.
- The file has no term, no principal, and no loss amount. The $5,000 and $600 figures are planning weights tied to a 36-month sketch at the sample means, not observed recoveries.
- The monthly payment still carries some of the offer, because installment depends on amount, rate, and term, even though the interest rate itself is withheld.
- Decline reasons are one-field ablations against a typical training file. A counterfactual hint is the smallest one-field change, inside the training range, that the model scores above the cutoff. Correlated fields share credit. Neither is a FCRA adverse-action notice.
- The isotonic map makes the deployed score a probability for planning. It was fit on out-of-fold training scores only. It is not a guarantee that a future book will match these frequencies.
- The saved pipeline is tied to the scikit-learn version recorded in the model card.

Quote the cross-validated ROC-AUC, with its bootstrap interval, when comparing models. Quote the calibrated probability against the cutoff when talking about a decision. Quote realized holdout counts separately from expected defaults, which use only the probabilities.

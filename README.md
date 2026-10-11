# Churn Risk Advisor

A web app that scores telecom customer churn risk, shows which changes would
lower it, and scores a whole customer file at once. Built as Project 1 of
Introduction to Applied AI (Department of Electronics).

**Live app:** >>> PASTE YOUR STREAMLIT URL HERE after deployment

## Screenshots

### One customer
![One customer](screenshot1.png)

### Batch scoring
![Batch scoring](screenshot2.png)

## Model card: Churn Risk Advisor v1.0

**Intended use:** rank existing telecom customers by churn risk so a
retention team can prioritize calls. Decision support, not automation.

**Not for:** credit, pricing, or any decision that denies a service.

**Data:** IBM Telco Customer Churn, 7,043 customers, 26.5% churn.

**Model:** XGBoost (tuned), chosen by 5-fold CV. Features: 30 encoded columns.

**Performance:** CV AUC 0.850 +/- 0.012 (0.8502 +/- 0.0117); test AUC 0.848
(0.8483, test set used once). The majority-class baseline accuracy is 73.5%.
XGBoost beat Logistic Regression (0.8464 +/- 0.0129) by only 0.004 AUC, which
is within noise, so the choice between them is not strongly supported by the
data.

**Threshold:** 0.15, from a cost analysis (FN = PKR 6,000, FP = PKR 1,000).

**Limitations:** one US dataset from one period; no Pakistan data;
associations, not causes; performance may drift as plans change. The
what-if table shows how the model's score moves, not what a customer would
do if offered a change.

**Fairness check (Week 15):** compare recall across gender and seniors.

**Owner and version:** Nosharwan Khan, v1.0, 11 October 2026.

## What the app does

- **One customer:** sidebar form (impossible combinations are blocked, and
  TotalCharges is derived, not typed), churn probability, risk band
  (HIGH / WATCH / LOW) and action, and a what-if table.
- **Batch scoring:** upload a CSV with the original Telco columns, get a
  scored table and a downloadable CSV. Missing columns give a clear error.
- **About:** model metadata and limitations.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `churn_xgb.json` | Model in XGBoost's native format (used by the app) |
| `churn_model.joblib` | Original pickled model from the notebook |
| `model_meta.json` | Feature columns, versions, threshold, metrics |
| `sample_customers.csv` | 50 customers for the batch demo |
| `requirements.txt` | Pinned dependencies |

## Run locally

Python 3.12 is required (xgboost 3.4.1 does not support Python 3.11).

    py -3.12 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

(Mac/Linux: use `python3.12 -m venv .venv` and `source .venv/bin/activate`.)

## Notebooks

- Week 1: >>> PASTE KAGGLE LINK
- Week 2: >>> PASTE KAGGLE LINK
- Week 3 and Lab 4 (deployment prep): >>> PASTE KAGGLE LINK

## Write it answers

### Part 1

**Task 1.3 (skew bug):** Scoring one Two year contract customer the Week 3
way (`build_features` with `drop_first=True`) gave P(churn) = 0.136, while the
correct serving encoder (`prepare_input`) gave P(churn) = 0.012. The two differ
because `drop_first=True` drops the first category it finds in the data. In a
single row the only Contract value present is "Two year", so it is dropped and
both Contract columns become 0, which the model reads as Month-to-month. No
error is raised, so the bug is silent: the app would have told a loyal
customer they were about 11 times more likely to leave than they really are.

**Parity test:** The largest difference between `p_train` (how the model was
evaluated in the notebook) and `p_serve` (how the app calls it) was 0.0, and
the test passed on all 7,043 customers. I checked every row, not a sample,
because encoding bugs hide in rare categories. A second parity test showed the
JSON model file gives the same probabilities as the notebook model (largest
difference 0.0 on 7,043 customers).

**What-if:** For the riskiest customer (baseline P = 0.935), the change that
lowers risk the most is Contract = Two year (delta -0.170, to 0.765); One year
gives -0.073 and automatic card payment only -0.022. This does not prove the
customer would stay. The model learned associations from past data, not
causes. Customers who already chose longer contracts may differ in other ways
from customers who are offered one. Even after the change this customer is
still high-risk (0.765). A randomized retention offer (A/B test) would be
needed to find the real effect.

### Part 2

**One thing that surprised me:** With tenure 60 and a Two year contract the
app showed 9% churn probability, and the risk band was WATCH, not LOW as the
lab expected. My threshold is 0.15 (from my cost analysis), so WATCH starts at
0.075. The band depends on my own threshold, not a fixed number. Also,
lowering Monthly charges from 85 to 40 did not lower the risk (9% went to 11%).

**An input the app still handles badly:** A customer file with a text value in
a numeric column (for example "N/A" in MonthlyCharges) or a category the model
never saw (for example a new payment method). The app checks that columns
exist, but not that the values are valid. An unseen category becomes all zeros
silently, so the model gives a confident answer without any warning.

### Part 3

>>> WRITE AFTER DEPLOYMENT: your public URL; did the first deployment
>>> succeed; if not, quote the error line and say what fixed it.

Local problem I fixed before deploying: `streamlit run app.py` failed with
`XGBoostError: input stream corrupted` when loading `churn_model.joblib`,
although the file hash matched the Kaggle original, so the file was not
damaged. xgboost 3.4.1 could not read the pickled model on Windows. I saved
the model in XGBoost's native JSON format, load it in the app through a small
wrapper class, and proved with a parity test that it gives identical
probabilities. I also had to install Python 3.12 because xgboost 3.4.1
requires Python 3.12 or newer.

## Check your understanding

1. **Why does `get_dummies(drop_first=True)` break predictions for one
   customer?** It drops the first category it sees in the data. With one row
   that is the customer's own category, so all of that feature's columns
   become 0 and the model reads the baseline category instead.

2. **What does the parity test compare, and why every row?** It compares the
   probabilities from the notebook's evaluation path and from the app's input
   path. Bugs hide in rare categories, so checking 5 rows can pass while 50
   customers are wrong.

3. **`st.cache_resource` or `st.cache_data` for the model?**
   `st.cache_resource`, because the model is a shared object that should be
   loaded once per server and reused, not copied on every rerun.

4. **Why derive TotalCharges instead of asking the user?** TotalCharges is
   roughly tenure times MonthlyCharges. Asking the user could produce
   impossible combinations that the model never saw, and its answers for them
   would be meaningless.

5. **Why pin the exact scikit-learn version?** The saved model stores
   library objects by their internal structure, so loading it needs the same
   version. A different version may warn, fail or quietly behave differently.
   The same applies to xgboost.

6. **The what-if shows a two-year contract lowers risk (0.79 to 0.27). Would
   offering one make the customer stay? How to find out?** Not necessarily:
   the model shows an association, and customers who choose long contracts may
   already be more loyal. The company can run a randomized test: offer a
   contract to a random half of at-risk customers and compare how many stay.

## Segment and action

The riskiest named segment is **At-Risk High-Spenders** (churn 43.1%). Action:
offer a contract lock-in discount (15% off for 12 months) and a technical
onboarding check.

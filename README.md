# Bank Marketing Term Deposit Prediction

Predicting which bank clients are likely to subscribe to a term deposit, using demographic, campaign, and macroeconomic data from a Portuguese bank's direct marketing campaign — and catching a data leakage bug along the way.

## The short version

An early pass at this model reported **100% accuracy**. That's not a win — it's a bug. The model was leaking the outcome through a feature (`duration`, the length of the sales call) that isn't known until *after* you've already decided to call someone. Once removed and properly evaluated on held-out data, the honest results are:

| Model | Accuracy | ROC AUC | Recall (subscribers) |
|---|---|---|---|
| Random Forest (class-weighted) | 90% | 0.76 | 18-30% |
| Logistic Regression (class-weighted) | 83% | **0.78** | **59%** |

**Takeaway:** the "less accurate" model is the better business choice. Logistic regression catches roughly 3x more actual subscribers, at the cost of some extra unproductive calls — a trade the bank should be willing to make, since a missed subscriber costs more than a wasted call.

The strongest real predictor of subscription is `euribor3m` (the prevailing interest rate environment) — timing the campaign matters more than who specifically gets called.

**[→ View the interactive dashboard](https://edward-henderson.github.io/bank-marketing-prediction/dashboard.html)** *(update this link once GitHub Pages is enabled — see below)*

## Repo contents

- [`bank_marketing_analysis.ipynb`](bank_marketing_analysis.ipynb) — full analysis: EDA, the leakage diagnosis, model comparison, feature importance, and recommendations
- [`dashboard.html`](dashboard.html) — standalone interactive dashboard (filterable client segments, model comparison, ROC curves)
- `data/bank.csv` — dataset (4,119 clients, 21 features)
- `outputs/` — exported charts

## Viewing the dashboard

The dashboard is a single self-contained HTML file — open it directly in any browser, or enable GitHub Pages (Settings → Pages → deploy from main branch) to get a shareable link.

## Dataset

Bank Marketing dataset (UCI Machine Learning Repository), 4,119 records from a Portuguese bank's phone-based term deposit campaign. Features include client demographics, prior campaign history, and macroeconomic indicators (interest rates, employment, consumer confidence).

## Methods

- Exploratory data analysis on demographic and campaign variables
- Label encoding of categorical features
- Random Forest and Logistic Regression classifiers, with class-weighting to address the ~89/11 class imbalance
- Data leakage diagnosis: comparing model performance with and without the `duration` feature, and identifying evaluation-on-training-data as the likely source of an earlier inflated 100% accuracy result
- Model comparison via ROC AUC, precision/recall, and confusion matrices rather than accuracy alone

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook bank_marketing_analysis.ipynb
```

## Next steps

- Tune the classification threshold directly around precision/recall tradeoffs rather than accuracy
- Test on more recent data, given how much `euribor3m` has moved since this dataset was collected
- Try gradient boosting (XGBoost/LightGBM) and compare against the Random Forest/Logistic Regression baselines

# Job Offer Acceptance Predictor (Streamlit)

Frontend for the MongoDB + ML project. A user enters candidate details and the offer; the app says whether the
candidate is likely to **accept**, **decline**, or is **uncertain**, and shows which factors drive that.

## Run

```bash
pip install -r requirements.txt
python train_model.py --data job_offer_acceptance_prediction_dataset.xlsx   # or train from the sidebar
streamlit run app.py
```

No dataset yet? The app still runs using the rule-based factors only; train a model any time from the sidebar
(upload the Excel/CSV, or pull from MongoDB Atlas).

## MongoDB (optional)
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your Atlas URI (or set `MONGODB_URI`).
Then you can train from `job_offer_prediction.candidates` and save every prediction to `job_offer_prediction.predictions`.

## How the decision is made
* **ML layer** - same cleaning/features as the notebook (Age, Experience, Current/Expected Salary, Companies Changed,
  Gender, Education, Skills, Job Sector, Has Another Offer). Logistic Regression, Random Forest and Gradient Boosting are
  trained; the best by ROC-AUC is kept.
* **Rule layer** - the dataset has no location, offered salary, work mode, etc., so these are scored by transparent rules in
  `model_utils.py` (`score_factors`): offer vs expectation, cost-of-living-adjusted raise, location/relocation, work mode,
  career growth, brand, benefits, notice period, competing offer, employment type. Weights are adjustable in the sidebar.
* The two are blended (the ML share is set automatically from the model's ROC-AUC, adjustable), then caps limit the score for deal-breakers
  (offer far below expectation, a pay cut, or a required relocation the candidate refuses).

Edit `COST_INDEX`, `CITIES`, and the score tables in `model_utils.py` to tune the assumptions.

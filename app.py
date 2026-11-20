"""Job Offer Acceptance Predictor - Streamlit frontend.   Run:  streamlit run app.py"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import model_utils as mu

st.set_page_config(page_title="Job Offer Acceptance Predictor", page_icon="💼", layout="wide")


# ----------------------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle(mtime: float):  # mtime is only a cache key so retraining refreshes the model
    return mu.load_bundle()


def current_bundle():
    mtime = mu.MODEL_PATH.stat().st_mtime if mu.MODEL_PATH.exists() else 0.0
    return get_bundle(mtime)


bundle = current_bundle()
mongo_uri = mu.get_mongo_uri(st.secrets)

DEFAULT_CHOICES = {
    "Gender": ["Male", "Female", "Other"],
    "Education": ["High School", "Bachelor's", "Master's", "PhD"],
    "Job Sector": ["IT", "Healthcare", "Finance", "Education", "Manufacturing", "Retail", "Other"],
}


def choices_for(col: str) -> list[str] | None:
    if bundle and bundle["choices"].get(col):
        return bundle["choices"][col]
    return DEFAULT_CHOICES.get(col)


# ----------------------------------------------------------------------------------------
# Sidebar: model status, blending, training, MongoDB
# ----------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model & settings")

    if bundle:
        m = mu.best_metrics(bundle)
        st.success(f"Trained model loaded: **{bundle['model_name']}**")
        st.caption(
            f"Trained on {bundle['n_rows']:,} rows · {bundle['trained_at']}  \n"
            f"Test ROC-AUC **{m['ROC-AUC']:.2f}** · balanced accuracy **{m['Balanced Accuracy']:.2f}** · "
            f"accuracy **{m['Accuracy']:.2f}** (always guessing 'Yes' scores {bundle['majority_baseline_accuracy']:.2f})"
        )
        if m["ROC-AUC"] < 0.65:
            st.warning("This model is only weakly better than guessing, so the rule-based factors carry most of the decision.")
        ml_weight = st.slider(
            "How much to trust the trained model", 0, 100, int(mu.default_ml_weight(bundle) * 100), 5,
            format="%d%%", help="The rest comes from the rule-based factors (location, offer vs expectation, ...). "
                                "Default is set automatically from the model's ROC-AUC.",
        ) / 100
    else:
        st.warning("No trained model yet - running on **rule-based factors only**. "
                   "Train one below using your dataset.")
        ml_weight = 0.0

    with st.expander("Train / retrain model"):
        st.caption("Uses the same cleaning and features as the Colab notebook.")
        src = st.radio("Data source", ["Upload Excel/CSV", "MongoDB Atlas"], horizontal=True)
        up = st.file_uploader("Dataset (.xlsx / .csv)", type=["xlsx", "csv"]) if src == "Upload Excel/CSV" else None
        if src == "MongoDB Atlas" and not mongo_uri:
            st.info("Set `MONGODB_URI` in `.streamlit/secrets.toml` or as an environment variable.")
        if st.button("Train model", width="stretch"):
            try:
                with st.spinner("Training..."):
                    if src == "Upload Excel/CSV":
                        if up is None:
                            raise ValueError("Please upload a dataset first.")
                        raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up, sheet_name=0)
                    else:
                        if not mongo_uri:
                            raise ValueError("MongoDB URI not configured.")
                        raw = mu.load_from_mongo(mongo_uri)
                    mu.train_and_save(raw)
                get_bundle.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

    with st.expander("Factor importance (advanced)"):
        st.caption("Relative weight of each rule-based factor. Values are normalised automatically.")
        weights = {k: st.slider(k, 0, 30, int(v * 100), 1) / 100 for k, v in mu.DEFAULT_WEIGHTS.items()}
        if sum(weights.values()) == 0:
            weights = dict(mu.DEFAULT_WEIGHTS)

    save_to_db = False
    if mongo_uri:
        save_to_db = st.checkbox("Save each prediction to MongoDB", value=False,
                                 help=f"Writes to {mu.DB_NAME}.{mu.PREDICTIONS}")

# ----------------------------------------------------------------------------------------
# Main form
# ----------------------------------------------------------------------------------------
st.title("💼 Job Offer Acceptance Predictor")
st.write("Enter the candidate's details and the offer on the table. The app estimates how likely the candidate is "
         "to **accept** it, and explains which factors help or hurt.")

city_list = sorted(mu.CITIES)
r = bundle["ranges"] if bundle else None

# --- 1. About the candidate -------------------------------------------------------------
with st.container(border=True):
    st.subheader("1 · About the candidate")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 70, 28)
    gender_opts = choices_for("Gender") or ["Male", "Female", "Other"]
    gender = c2.selectbox("Gender", gender_opts)
    edu_opts = choices_for("Education") or ["Bachelor's"]
    education = c3.selectbox("Highest education", edu_opts)

    c1, c2, c3 = st.columns(3)
    experience = c1.number_input("Years of experience", 0.0, 50.0, 4.0, 0.5)
    companies_changed = c2.number_input("Companies changed so far", 0, 30, 2)
    sector_opts = choices_for("Job Sector")
    sector = c3.selectbox("Job sector", sector_opts) if sector_opts else c3.text_input("Job sector", "IT")

    skill_opts = choices_for("Skills")
    skills = st.selectbox("Skill group", skill_opts) if skill_opts else st.text_input("Skills (comma-separated)", "Python, SQL")

# --- 2. Current job ---------------------------------------------------------------------
with st.container(border=True):
    st.subheader("2 · Current job & expectations")
    c1, c2 = st.columns(2)
    not_earning = c1.checkbox("Not currently earning (fresher / between jobs)")
    current_salary = c1.number_input("Current salary", 0, 100_000_000, 0 if not_earning else 600_000, 10_000,
                                     disabled=not_earning, help="Use the same unit/period as your training data (e.g. annual).")
    if not_earning:
        current_salary = 0
    expected_salary = c2.number_input("Expected salary", 0, 100_000_000, 800_000, 10_000)
    if r:
        c2.caption(f"Training data ranged from {r['Current Salary'][0]:,.0f} to {r['Expected Salary'][1]:,.0f}. "
                   "Values far outside this make the ML part less reliable.")

    cur_city_opts = city_list + [mu.OTHER]
    current_city = c1.selectbox("Current location", cur_city_opts, index=cur_city_opts.index("Bengaluru"))
    current_manual_tier = 2
    if current_city == mu.OTHER:
        current_manual_tier = c1.selectbox("Cost of living there", [1, 2, 3], index=1,
                                           format_func=lambda t: {1: "High (metro)", 2: "Medium", 3: "Lower"}[t],
                                           key="cur_tier")
    notice_days = c2.number_input("Notice period (days)", 0, 180, 30, 5)

# --- 3. The offer -----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("3 · The new offer")
    c1, c2, c3 = st.columns(3)
    offered_salary = c1.number_input("Offered salary", 0, 100_000_000, 750_000, 10_000)
    offered_mode = c2.selectbox("Work mode offered", ["On-site", "Hybrid", "Remote"])
    employment_type = c3.selectbox("Employment type", list(mu.EMPLOYMENT_TYPE_SCORE))

    new_city_opts = city_list + [mu.REMOTE, mu.OTHER]
    c1, c2, c3 = st.columns(3)
    new_city = c1.selectbox("New job location", new_city_opts, index=new_city_opts.index("Pune"))
    new_manual_tier = 2
    manual_distance = 0
    if new_city == mu.OTHER:
        new_manual_tier = c2.selectbox("Cost of living there", [1, 2, 3], index=1,
                                       format_func=lambda t: {1: "High (metro)", 2: "Medium", 3: "Lower"}[t],
                                       key="new_tier")
    if mu.OTHER in (current_city, new_city) and new_city != mu.REMOTE and current_city != new_city:
        manual_distance = c3.number_input("Approx. distance between the two (km)", 0, 20000, 300, 50)
    relocation_assistance = c3.checkbox("Employer offers relocation support", value=False) if new_city != mu.REMOTE else False
    joining_days = c1.number_input("Employer wants you to join within (days)", 0, 180, 45, 5)

    c1, c2, c3 = st.columns(3)
    level_change = c1.selectbox("Role level vs current", list(mu.LEVEL_SCORE), index=1)
    brand = c2.selectbox("Company vs current employer", list(mu.BRAND_SCORE), index=1)
    benefits = c3.selectbox("Benefits (insurance, bonus, leave...)", list(mu.BENEFITS_SCORE), index=1)

# --- 4. Preferences & competition -------------------------------------------------------
with st.container(border=True):
    st.subheader("4 · Candidate preferences & competing offers")
    c1, c2, c3 = st.columns(3)
    preferred_mode = c1.selectbox("Preferred work mode", ["No preference", "Remote", "Hybrid", "On-site"])
    willingness = c2.selectbox("Willing to relocate?", list(mu.RELOCATE_BASE))
    has_other_offer = c3.radio("Holding another offer?", ["No", "Yes"], horizontal=True)
    other_offer_salary = 0
    if has_other_offer == "Yes":
        other_offer_salary = c3.number_input("Other offer's salary (0 = unknown)", 0, 100_000_000, 0, 10_000)

inputs = {
    "age": age, "gender": gender, "education": education, "experience": experience,
    "companies_changed": companies_changed, "sector": sector, "skills": skills,
    "current_salary": current_salary, "expected_salary": expected_salary, "current_city": current_city,
    "current_manual_tier": current_manual_tier, "notice_days": notice_days,
    "offered_salary": offered_salary, "offered_mode": offered_mode, "employment_type": employment_type,
    "new_city": new_city, "new_manual_tier": new_manual_tier, "manual_distance_km": manual_distance,
    "relocation_assistance": relocation_assistance, "joining_days": joining_days,
    "level_change": level_change, "brand": brand, "benefits": benefits,
    "preferred_mode": preferred_mode, "relocation_willingness": willingness,
    "has_other_offer": has_other_offer, "other_offer_salary": other_offer_salary,
}
settings = {"ml_weight": ml_weight, "weights": weights}

if st.button("🔮 Predict", type="primary", width="stretch"):
    res = mu.evaluate(inputs, bundle, ml_weight, weights)
    st.session_state["result"] = {
        "inputs": dict(inputs), "settings": dict(settings), "res": res,
        "min_offer": mu.minimum_offer_for_accept(inputs, bundle, ml_weight, weights),
    }
    if save_to_db:
        try:
            mu.save_prediction(mongo_uri, inputs, res)
            st.toast("Prediction saved to MongoDB ✅")
        except Exception as e:
            st.toast(f"Could not save to MongoDB: {e}", icon="⚠️")

# ----------------------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------------------
saved = st.session_state.get("result")
if saved:
    res: mu.Result = saved["res"]
    if saved["inputs"] != inputs or saved["settings"] != settings:
        st.info("Inputs or settings changed since the last prediction - press **Predict** to refresh.")

    st.divider()
    p = res.probability
    if p >= mu.LIKELY_ACCEPT:
        st.success(f"## ✅ {res.verdict}")
    elif p < mu.LIKELY_DECLINE:
        st.error(f"## ❌ {res.verdict}")
    else:
        st.warning(f"## ⚖️ {res.verdict}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Chance of accepting", f"{p:.0%}")
    m2.metric("Rule-based factors", f"{res.rule_probability:.0%}")
    m3.metric("Trained ML model", "n/a" if res.ml_probability is None else f"{res.ml_probability:.0%}",
              help=f"Weight in final score: {res.ml_weight:.0%}")
    st.progress(min(max(p, 0.0), 1.0))

    for cap, msg in res.caps:
        (st.error if cap <= 0.12 else st.warning)(f"🚫 {msg} The chance is capped at {cap:.0%}.")

    df = pd.DataFrame([{
        "Factor": x.name, "Score": round(x.score), "Weight": x.weight, "Detail": x.detail,
    } for x in res.factors if x.applicable])
    df["Weight"] = df["Weight"] / df["Weight"].sum()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Why this result")
        chart = (
            alt.Chart(df).mark_bar().encode(
                x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Pull towards accepting (0–100)"),
                y=alt.Y("Factor:N", sort="-x", title=None),
                color=alt.Color("Score:Q", scale=alt.Scale(domain=[0, 50, 100], range=["#d9534f", "#f0ad4e", "#3c9d5d"]),
                                legend=None),
                tooltip=["Factor", "Score", "Detail"],
            ).properties(height=300)
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("#### Biggest levers")
        df["Impact"] = (df["Score"] - 55) * df["Weight"]  # vs. a neutral 55
        for _, row in df.sort_values("Impact").head(3).iterrows():
            if row["Impact"] < 0:
                st.markdown(f"🔻 **{row['Factor']}** - {row['Detail']}")
        for _, row in df.sort_values("Impact", ascending=False).head(3).iterrows():
            if row["Impact"] > 0:
                st.markdown(f"🔺 **{row['Factor']}** - {row['Detail']}")

        mo = saved["min_offer"]
        if p < mu.LIKELY_ACCEPT:
            if mo is not None:
                st.info(f"💡 With these inputs, an offer of about **{mo:,.0f}** would reach 'likely to accept' "
                        f"({mo / max(inputs['offered_salary'], 1) - 1:+.0%} vs the current offer).")
            else:
                st.info("💡 Raising the salary alone would not be enough - other factors (see red bars) need to change.")

    with st.expander("Full factor table"):
        st.dataframe(df[["Factor", "Score", "Weight", "Detail"]].assign(Weight=lambda d: (d["Weight"] * 100).round(1).astype(str) + "%"),
                     hide_index=True, width="stretch")

    st.caption(
        "⚠️ This is an estimate for decision support, not a guarantee. The training data described in the project report "
        "is synthetic, and location, offered salary, work mode and the other rule-based factors are scored with "
        "transparent assumptions rather than learned from data."
    )

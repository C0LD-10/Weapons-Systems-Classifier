"""
streamlit_app.py — Global Weapons Systems dashboard.

Three tabs:
  1. Data Explorer   — filterable overview of the raw dataset.
  2. Classify a System — interactive form -> live prediction via the
     trained pipeline (src/predict.py), with the full probability
     distribution and an explicit reminder of the model's known
     limitations (see notebooks/01_eda.ipynb, Section 5).
  3. Model Performance — the metrics, confusion matrix, and permutation
     importance produced by src/train.py and src/evaluate.py.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from predict import CategoryPredictor  # noqa: E402

st.set_page_config(page_title="Global Weapons Systems Classifier", layout="wide")


@st.cache_resource
def get_predictor() -> CategoryPredictor:
    return CategoryPredictor(config_path=ROOT / "config.yaml")


@st.cache_data
def get_raw_data() -> pd.DataFrame:
    cfg = load_config(ROOT / "config.yaml")
    return pd.read_csv(cfg["paths"]["raw_data"])


@st.cache_data
def get_metrics() -> dict:
    cfg = load_config(ROOT / "config.yaml")
    with open(cfg["paths"]["metrics_file"]) as f:
        return json.load(f)


cfg = load_config(ROOT / "config.yaml")
df = get_raw_data()

st.title("🎯 " + cfg["app"]["streamlit_title"])
st.caption(
    "A Category classifier trained on structured weapons-system specs. "
    "See the **Model Performance** tab for an honest account of what "
    "this model has and hasn't actually learned."
)

tab_explore, tab_classify, tab_performance = st.tabs(
    ["📊 Data Explorer", "🔮 Classify a System", "📈 Model Performance"]
)

# =============================================================================
# TAB 1 — Data Explorer
# =============================================================================
with tab_explore:
    st.subheader("Dataset overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Categories", df["Category"].nunique())
    col3.metric("Countries of origin", df["Country_of_Origin"].nunique())
    col4.metric("Year range", f"{df['Year_Introduced'].min()}–{df['Year_Introduced'].max()}")

    st.divider()

    filt_col1, filt_col2, filt_col3 = st.columns(3)
    with filt_col1:
        categories = st.multiselect(
            "Category", sorted(df["Category"].unique()), default=[]
        )
    with filt_col2:
        countries = st.multiselect(
            "Country of origin", sorted(df["Country_of_Origin"].unique()), default=[]
        )
    with filt_col3:
        status = st.multiselect(
            "Service status", sorted(df["Service_Status"].unique()), default=[]
        )

    filtered = df.copy()
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if countries:
        filtered = filtered[filtered["Country_of_Origin"].isin(countries)]
    if status:
        filtered = filtered[filtered["Service_Status"].isin(status)]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} rows")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Category distribution (filtered)**")
        st.bar_chart(filtered["Category"].value_counts())
    with chart_col2:
        st.markdown("**Median Unit Cost (USD, log scale) by Category**")
        cost_by_cat = filtered.groupby("Category")["Unit_Cost_USD"].median().sort_values()
        st.bar_chart(cost_by_cat)

    st.markdown("**Browse rows**")
    display_cols = [
        "Weapon_Name", "Country_of_Origin", "Category", "Subcategory",
        "Year_Introduced", "Service_Status", "Unit_Cost_USD", "Primary_Users",
    ]
    st.dataframe(filtered[display_cols].head(500), use_container_width=True, height=400)

# =============================================================================
# TAB 2 — Classify a System
# =============================================================================
with tab_classify:
    st.subheader("Predict a system's Category from its specs")
    st.info(
        "💡 The trained model relies partly on *which fields are filled in*, "
        "not just their values — several columns are only populated for "
        "certain categories in the training data (e.g. `Warhead_Weight_kg` "
        "is essentially only present for munitions). Leaving a field blank "
        "is informative to the model, not neutral. Fill in what's realistic "
        "for the kind of system you have in mind.",
        icon="💡",
    )

    predictor = get_predictor()

    with st.form("classify_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Identity & origin**")
            country = st.text_input("Country of Origin", value="USA")
            manufacturer = st.text_input("Manufacturer", value="")
            year = st.number_input("Year Introduced", min_value=1900, max_value=2030, value=2015)
            caliber = st.text_input("Caliber / Warhead type", value="5.56x45mm NATO")
            action_type = st.text_input("Action Type", value="Gas-operated short stroke piston")

        with c2:
            st.markdown("**Physical specs**")
            weight = st.number_input("Weight (kg)", min_value=0.0, value=3.5, format="%.2f")
            length = st.number_input("Length (mm)", min_value=0.0, value=0.0, format="%.1f")
            eff_range = st.number_input("Effective Range (m)", min_value=0.0, value=500.0)
            max_range = st.number_input("Max Range (m)", min_value=0.0, value=800.0)
            crew = st.number_input("Crew Size", min_value=0, max_value=200, value=1)

        with c3:
            st.markdown("**Operational profile**")
            cost = st.number_input("Unit Cost (USD)", min_value=0.0, value=1200.0)
            n_operators = st.number_input("Num Operator Nations", min_value=0, value=10)
            combat_proven = st.selectbox("Combat Proven", ["Yes", "No", "Limited"], index=0)
            nato = st.selectbox("NATO Compatible", ["Yes", "No", "Partial"], index=0)
            environment = st.text_input("Operating Environment", value="All-terrain")

        submitted = st.form_submit_button("Predict Category", type="primary")

    if submitted:
        raw_row = {
            "Country_of_Origin": country or None,
            "Manufacturer": manufacturer or None,
            "Year_Introduced": year,
            "Caliber": caliber or None,
            "Action_Type": action_type or None,
            "Weight_kg": weight or None,
            "Length_mm": length or None,
            "Effective_Range_m": eff_range or None,
            "Max_Range_m": max_range or None,
            "Crew_Size": crew,
            "Unit_Cost_USD": cost or None,
            "Num_Operator_Nations": n_operators,
            "Combat_Proven": combat_proven,
            "NATO_Compatible": nato,
            "Operating_Environment": environment or None,
        }
        result = predictor.predict_one(raw_row)

        st.success(f"**Predicted Category: {result['predicted_category']}**")

        proba_df = pd.DataFrame(
            list(result["probabilities"].items()), columns=["Category", "Probability"]
        )
        st.bar_chart(proba_df.set_index("Category"))
        st.dataframe(proba_df.style.format({"Probability": "{:.3f}"}), use_container_width=True)

# =============================================================================
# TAB 3 — Model Performance
# =============================================================================
with tab_performance:
    st.subheader("Model performance & honesty check")

    metrics = get_metrics()

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected model", metrics["selected_model"])
    m2.metric("Test macro-F1", f"{metrics['test_macro_f1']:.4f}")
    m3.metric("Test rows", metrics["test_rows"])

    st.warning(
        "**Read this before trusting the number above.** Test macro-F1 is "
        "~0.9995 — essentially ceiling performance. The EDA notebook "
        "(`notebooks/01_eda.ipynb`, Section 5) shows several categorical "
        "fields (e.g. `Action_Type`) are ~90-98% 'pure' with respect to "
        "`Category` in this dataset, consistent with categorical fields "
        "having been generated *conditionally on* Category. This model "
        "most likely recovers that generative template rather than a "
        "generalizable domain pattern, and should not be assumed to "
        "transfer to real-world weapons specs from a different source.",
        icon="⚠️",
    )

    st.markdown("### Cross-validated model comparison")
    cv_rows = []
    for name, res in metrics["cv_results"].items():
        cv_rows.append({
            "Model": name,
            "CV macro-F1 (mean)": res["cv_mean"],
            "CV macro-F1 (std)": res["cv_std"],
            "5-fold fit time (s)": res["fit_seconds_for_5fold"],
        })
    st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)

    st.markdown("### Confusion matrix (held-out test set)")
    cm_path = Path(cfg["paths"]["figures_dir"]) / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)

    st.markdown("### Permutation feature importance")
    st.caption(
        "Mean drop in held-out macro-F1 when a feature's values are "
        "shuffled — model-agnostic and unbiased toward high-cardinality "
        "columns, unlike impurity-based importance."
    )
    imp_path = Path(cfg["paths"]["figures_dir"]) / "permutation_importance.png"
    if imp_path.exists():
        st.image(str(imp_path), use_container_width=True)

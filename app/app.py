
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Micro-Lender Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "micro_lender_random_forest.pkl"
)

features_path = os.path.join(
    BASE_DIR,
    "behavior_features.pkl"
)

model = joblib.load(model_path)
behavior_features = joblib.load(features_path)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("💳 Micro-Lender Credit Risk Assessment")

st.write(
    """
    A machine-learning decision-support prototype for estimating
    borrower default risk using credit limits, repayment behaviour,
    billing history and payment history.
    """
)

st.info(
    "This prototype is intended for educational and decision-support "
    "purposes and should not be used as the sole basis for a real lending decision."
)

# --------------------------------------------------
# BORROWER INPUTS
# --------------------------------------------------
st.header("1. Borrower Credit Information")

limit_bal = st.number_input(
    "Credit Limit",
    min_value=1.0,
    value=100000.0,
    step=10000.0
)

st.subheader("Repayment Status — Last Six Months")

st.caption(
    "Positive values indicate repayment delay. "
    "Zero or negative values indicate no recorded positive delay."
)

pay_0 = st.number_input("Most Recent Repayment Status (PAY_0)", value=0)
pay_2 = st.number_input("PAY_2", value=0)
pay_3 = st.number_input("PAY_3", value=0)
pay_4 = st.number_input("PAY_4", value=0)
pay_5 = st.number_input("PAY_5", value=0)
pay_6 = st.number_input("PAY_6", value=0)

# --------------------------------------------------
# BILL AMOUNTS
# --------------------------------------------------
st.header("2. Monthly Bill Amounts")

col1, col2, col3 = st.columns(3)

with col1:
    bill_amt1 = st.number_input("BILL_AMT1", value=50000.0)
    bill_amt2 = st.number_input("BILL_AMT2", value=45000.0)

with col2:
    bill_amt3 = st.number_input("BILL_AMT3", value=40000.0)
    bill_amt4 = st.number_input("BILL_AMT4", value=35000.0)

with col3:
    bill_amt5 = st.number_input("BILL_AMT5", value=30000.0)
    bill_amt6 = st.number_input("BILL_AMT6", value=25000.0)

# --------------------------------------------------
# PAYMENT AMOUNTS
# --------------------------------------------------
st.header("3. Monthly Payment Amounts")

col4, col5, col6 = st.columns(3)

with col4:
    pay_amt1 = st.number_input("PAY_AMT1", value=5000.0)
    pay_amt2 = st.number_input("PAY_AMT2", value=5000.0)

with col5:
    pay_amt3 = st.number_input("PAY_AMT3", value=5000.0)
    pay_amt4 = st.number_input("PAY_AMT4", value=5000.0)

with col6:
    pay_amt5 = st.number_input("PAY_AMT5", value=5000.0)
    pay_amt6 = st.number_input("PAY_AMT6", value=5000.0)

# --------------------------------------------------
# ASSESSMENT
# --------------------------------------------------
st.header("4. Credit Risk Assessment")

if st.button("Assess Credit Risk", type="primary"):

    repayment_status = [
        pay_0, pay_2, pay_3,
        pay_4, pay_5, pay_6
    ]

    # Engineered feature 1
    delay_months = sum(status > 0 for status in repayment_status)

    # Engineered feature 2
    credit_utilization = bill_amt1 / limit_bal

    borrower = {
        "LIMIT_BAL": limit_bal,

        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,

        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,

        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6,

        "DELAY_MONTHS": delay_months,
        "CREDIT_UTILIZATION": credit_utilization
    }

    input_df = pd.DataFrame([borrower])

    # Guarantee same order used during model training
    input_df = input_df[behavior_features]

    # Prediction
    default_probability = model.predict_proba(input_df)[0, 1]

    # Credit score transformation used in project
    credit_score = int(
        round((1 - default_probability) * 100)
    )

    # Risk categories
    if credit_score >= 70:
        risk_category = "Low Risk"
        recommendation = "Consider for Approval"

    elif credit_score >= 40:
        risk_category = "Medium Risk"
        recommendation = "Manual Review"

    else:
        risk_category = "High Risk"
        recommendation = "Further Assessment Required"

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    st.success("Assessment completed successfully.")

    st.subheader("Assessment Result")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Default Probability",
            f"{default_probability * 100:.2f}%"
        )

    with r2:
        st.metric(
            "Credit Score",
            f"{credit_score} / 100"
        )

    with r3:
        st.metric(
            "Months with Repayment Delay",
            delay_months
        )

    st.write("### Risk Category")

    if risk_category == "Low Risk":
        st.success("🟢 Low Risk")

    elif risk_category == "Medium Risk":
        st.warning("🟠 Medium Risk")

    else:
        st.error("🔴 High Risk")

    st.write("### Lending Recommendation")
    st.write(f"**{recommendation}**")

    st.write("### Supporting Indicators")

    st.write(
        f"Credit utilization: **{credit_utilization:.2%}**"
    )

    st.write(
        f"Positive repayment-delay months: "
        f"**{delay_months} out of 6**"
    )

    st.caption(
        "The risk category and recommendation are decision-support "
        "outputs derived from the model probability and project-defined "
        "score thresholds. Human review remains appropriate for lending decisions."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Micro-Lender Credit Scoring — Machine Learning for Credit "
    "Default Risk Assessment | 3MTT Data Science Capstone Project"
)

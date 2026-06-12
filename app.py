import streamlit as st
import joblib
import numpy as np

model = joblib.load("credit_model.pkl")

st.title("Credit Scoring Model")
st.markdown("""
This application predicts whether a customer is a **Good Credit Risk** or **Bad Credit Risk**
using a Random Forest machine learning model trained on the German Credit Dataset.
""")

f1 = st.selectbox(
    "Status of Existing Checking Account",
    [1, 2, 3, 4]
)

f2 = st.number_input(
    "Duration in Months",
    min_value=0
)

f3 = st.selectbox(
    "Credit History",
    [0, 1, 2, 3, 4]
)

f4 = st.number_input(
    "Purpose",
    min_value=0
)

f5 = st.number_input(
    "Credit Amount",
    min_value=0
)

f6 = st.selectbox(
    "Savings Account / Bonds",
    [1, 2, 3, 4, 5]
)

f7 = st.selectbox(
    "Present Employment Since",
    [1, 2, 3, 4, 5]
)

f8 = st.number_input(
    "Installment Rate (% Income)",
    min_value=0
)

f9 = st.number_input(
    "Personal Status and Sex",
    min_value=0
)

f10 = st.number_input(
    "Other Debtors / Guarantors",
    min_value=0
)

f11 = st.number_input(
    "Present Residence Since",
    min_value=0
)

f12 = st.number_input(
    "Property",
    min_value=0
)

f13 = st.number_input(
    "Age in Years",
    min_value=18
)

f14 = st.number_input(
    "Other Installment Plans",
    min_value=0
)

f15 = st.selectbox(
    "Housing",
    [1, 2, 3]
)

f16 = st.number_input(
    "Number of Existing Credits",
    min_value=0
)

f17 = st.selectbox(
    "Job",
    [1, 2, 3, 4]
)

f18 = st.number_input(
    "Number of Dependents",
    min_value=0
)

f19 = st.number_input(
    "Telephone",
    min_value=0
)

f20 = st.selectbox(
    "Foreign Worker",
    [0, 1]
)

f21 = st.number_input(
    "Credit History Flag 1",
    min_value=0
)

f22 = st.number_input(
    "Credit History Flag 2",
    min_value=0
)

f23 = st.number_input(
    "Housing Flag",
    min_value=0
)

f24 = st.number_input(
    "Foreign Worker Flag",
    min_value=0
)

if st.button("Predict"):

    features = np.array([
        [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,
         f11,f12,f13,f14,f15,f16,f17,f18,
         f19,f20,f21,f22,f23,f24]
    ])

    st.write("Input Features:")
    st.write(features)

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    st.write("Prediction Value:", prediction[0])
    good_prob = probability[0][1]
    st.write(f"Credit Approval Probability:{good_prob:2%}")

    if prediction[0] == 1:
        st.success("Good Credit Risk")
        st.info(f"Confidence: {good_prob:.2%}")

    else:
        st.error("Bad Credit Risk")
        st.info(f"Confidence: {(1-good_prob):.2%}")
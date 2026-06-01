import streamlit as st
import pandas as pd
df = pd.read_csv("Salary Data.csv")
import pickle

st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💰"
)

st.title("💰 Salary Prediction System")
st.write("Predict employee salaries using a Random Forest Machine Learning model")

st.sidebar.title("About Project")

st.sidebar.info("""
Salary Prediction System

Model: Random Forest

Technologies:
- Python
- Pandas
- Scikit-Learn
- Streamlit

Developed by Saheb Chhabra
""")

model = pickle.load(open("salary_model.pkl", "rb"))

gender_encoder = pickle.load(open("gender_encoder.pkl", "rb"))
education_encoder = pickle.load(open("education_encoder.pkl", "rb"))
job_encoder = pickle.load(open("job_encoder.pkl", "rb"))

st.title("Salary Prediction System")

age = st.number_input("Age", min_value=18, max_value=70)

gender = st.selectbox("Gender", gender_encoder.classes_)

education = st.selectbox(
    "Education Level",
    education_encoder.classes_
)

job_title = st.selectbox(
    "Job Title",
    job_encoder.classes_
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0
)

if st.button("Predict Salary"):

    gender_encoded = gender_encoder.transform([gender])[0]
    education_encoded = education_encoder.transform([education])[0]
    job_encoded = job_encoder.transform([job_title])[0]

    input_data = pd.DataFrame(
        [[age, gender_encoded, education_encoded,
          job_encoded, experience]],
        columns=[
            "Age",
            "Gender",
            "Education Level",
            "Job Title",
            "Years of Experience"
        ]
    )

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Salary: ₹{prediction[0]:,.0f}"
    )

st.subheader("📊 Dataset Preview")

st.dataframe(df.head())

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", "0.915")

with col2:
    st.metric("RMSE", "12,685")
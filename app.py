import streamlit as st
import numpy as np
import joblib 
import warnings

warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")
st.title("Student Exam Predictor Score")
studying_hours = st.slider("Studting Hours per day", 0.0, 12.0, 2.0) 
attendance = st.slider("Attendance Perfomance", 0.0, 100.0, 80.0) 
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 5) 
sleep_hours = st.slider("Sleeping Hours per Night", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-Time job", ["No", "Yes"])

ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Score"):

    input_data = np.array([[studying_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Exam Score: {prediction:.2f}")
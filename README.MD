# Student Exam Score Prediction System

A production-ready **Machine Learning web application** developed using **Python** and **Streamlit** to predict student examination performance based on academic habits and lifestyle indicators.

This project demonstrates the complete pipeline of **model deployment**, **user interaction**, and **real-time prediction** using a pre-trained regression model.

---

## 1. Project Overview

The Student Exam Score Prediction System is designed to estimate a student's expected exam score by analyzing multiple contributing factors such as study time, attendance, mental health, sleep patterns, and employment status.

The application provides an intuitive web interface that allows users to input student data and receive an instant prediction.

---

## 2. System Architecture

User Input (Streamlit UI)
↓
Data Preprocessing (Encoding & Formatting)
↓
Pre-trained Machine Learning Model
↓
Score Prediction (0 – 100)
↓
Result Visualization


---

## 3. Key Features

- Interactive and responsive web interface  
- Real-time exam score prediction  
- Input validation and score normalization  
- Clean and modular Python implementation  
- Easily extensible for additional features or models  

---

## 4. Input Variables

| Variable | Description | Type |
|--------|------------|------|
| Studying Hours | Daily study duration | Float |
| Attendance Performance | Class attendance percentage | Float |
| Mental Health Rating | Self-rated mental health score (1–10) | Integer |
| Sleep Hours | Average sleep per night | Float |
| Part-Time Job | Employment status | Categorical |

Categorical features are encoded internally before prediction.

---

## 5. Machine Learning Model

- **Model Type:** Regression Model  
- **Storage Format:** `joblib`  
- **Model File:** `best_model.pkl`  
- **Input Shape:**  
(n_samples, 5)


The model is loaded once at runtime to ensure optimal performance.

---

## 6. Technology Stack

- **Programming Language:** Python  
- **Framework:** Streamlit  
- **Libraries:**
- NumPy  
- Joblib  
- Warnings (for clean execution)  
- **Deployment:** Local or Cloud-ready  

---

## 7. Installation & Setup

### Prerequisites
Ensure **Python 3.8+** is installed.

### Install Dependencies
```bash
pip install streamlit numpy joblib
8. Running the Application
streamlit run app.py
The application will be accessible at:

http://localhost:8501
9. Project Structure
student_exam_predictor/
│
├── app.py                # Streamlit application
├── best_model.pkl        # Trained ML model
├── README.md             # Project documentation
10. Use Cases
Academic performance evaluation

Educational data analysis

Student advisory systems

Machine Learning deployment practice

Portfolio and demonstration projects

11. Limitations
Predictions depend on the quality of training data

Results should be used as guidance, not absolute outcomes

Model performance may vary across different student populations

12. Future Improvements
Add model explainability (SHAP / feature importance)

Support batch predictions

Deploy on cloud platforms (Streamlit Cloud, AWS, Azure)

Integrate database for prediction history

Add authentication and user roles

13. Disclaimer
This system is developed strictly for educational and demonstration purposes.
It should not be used as the sole determinant of academic performance or decision-making.

14. Author
Jonathan Petro
Machine Learning & Python Developer
Focused on building data-driven educational solutions.

15. License
This project is released under the MIT License.

16. Acknowledgements
Streamlit Community

Scikit-Learn Ecosystem

Open-source contributors
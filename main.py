import pickle
import streamlit as st

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Load Trained Models
# -----------------------------
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# -----------------------------
# App Title
# -----------------------------
st.title("🩺 Multiple Disease Prediction System")
st.caption("**Project under development – results may not be 100% accurate.**")

# -----------------------------
# Navigation Tabs
# -----------------------------
tabs = st.tabs([
    "📘 About Project",
    "🩸 Diabetes Prediction",
    "❤️ Heart Disease Prediction",
    "🧠 Parkinson’s Prediction",
    "📰 Blog / Insights"
])

# -----------------------------
# About Project
# -----------------------------
with tabs[0]:
    st.header("About This Project")
    st.write("""
    This web application predicts the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson’s Disease** 
    using pre-trained Machine Learning models.  
    The system demonstrates how AI can assist healthcare professionals by providing early predictions 
    based on medical parameters.  
    However, these predictions are **not diagnostic** and should not replace medical advice.
    """)
    st.subheader("Project Highlights")
    st.markdown("""
    - 🧩 Built using **Streamlit** and **Scikit-learn**  
    - 💾 Models saved as `.sav` files for lightweight deployment  
    - ⚙️ Backend powered by pre-trained supervised ML models  
    - 🌐 Deployed on Streamlit Cloud  
    - 🧑‍💻 Designed and developed by *Japanjot Singh*
    """)

# -----------------------------
# Diabetes Prediction Tab
# -----------------------------
with tabs[1]:
    st.header("🩸 Diabetes Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies (0–17)')
    with col2:
        Glucose = st.text_input('Glucose Level (0–199)')
    with col3:
        BloodPressure = st.text_input('Blood Pressure (0–122)')
    with col1:
        SkinThickness = st.text_input('Skin Thickness (0–99)')
    with col2:
        Insulin = st.text_input('Insulin Level (0–846)')
    with col3:
        BMI = st.text_input('BMI (0–67.1)')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function (0.078–2.42)')
    with col2:
        Age = st.text_input('Age (21–81)')

    diab_diagnosis = ''

    if st.button('🔍 Predict Diabetes'):
        try:
            if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
                st.error("⚠️ Please fill in all fields before prediction.")
            else:
                inputs = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]

                diab_prediction = diabetes_model.predict([inputs])
                if diab_prediction[0] == 1:
                    diab_diagnosis = '🩸 The person is diabetic.'
                else:
                    diab_diagnosis = '✅ The person is not diabetic.'
                st.success(diab_diagnosis)
        except ValueError:
            st.error("❌ Please enter valid numeric values only.")

# -----------------------------
# Heart Disease Prediction Tab
# -----------------------------
with tabs[2]:
    st.header("❤️ Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
    with col3:
        cp = st.text_input('Chest Pain Type (0–3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dl)')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1/0)')
    with col1:
        restecg = st.text_input('Resting ECG Result (0–2)')
    with col2:
        thalach = st.text_input('Max Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1/0)')
    with col1:
        oldpeak = st.text_input('ST Depression')
    with col2:
        slope = st.text_input('Slope (0–2)')
    with col3:
        ca = st.text_input('Major Vessels (0–3)')
    with col1:
        thal = st.text_input('Thal (0–2)')

    heart_diagnosis = ''

    if st.button('🔍 Predict Heart Disease'):
        try:
            if not all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
                st.error("⚠️ Please fill in all fields before prediction.")
            else:
                inputs = [float(age), float(sex), float(cp), float(trestbps), float(chol),
                          float(fbs), float(restecg), float(thalach), float(exang),
                          float(oldpeak), float(slope), float(ca), float(thal)]

                heart_prediction = heart_disease_model.predict([inputs])
                if heart_prediction[0] == 1:
                    heart_diagnosis = '❤️ The person has heart disease.'
                else:
                    heart_diagnosis = '✅ The person does not have heart disease.'
                st.success(heart_diagnosis)
        except ValueError:
            st.error("❌ Please enter valid numeric values only.")

# -----------------------------
# Parkinson’s Prediction Tab
# -----------------------------
with tabs[3]:
    st.header("🧠 Parkinson’s Disease Prediction using ML")
    st.caption("Enter values within the given ranges for accurate predictions.")

    # Define input layout
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz) (88–260)')
        RAP = st.text_input('MDVP:RAP (0.00068–0.02144)')
        APQ3 = st.text_input('Shimmer:APQ3 (0.01026–0.03134)')
        HNR = st.text_input('HNR (8.44–33.04)')
        D2 = st.text_input('D2 (1.42–3.67)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz) (102–592)')
        PPQ = st.text_input('MDVP:PPQ (0.00092–0.01958)')
        APQ5 = st.text_input('Shimmer:APQ5 (0.01161–0.04518)')
        RPDE = st.text_input('RPDE (0.256–0.685)')
        PPE = st.text_input('PPE (0.044–0.527)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz) (65–239)')
        DDP = st.text_input('Jitter:DDP (0.002–0.064)')
        APQ = st.text_input('MDVP:APQ (0.013–0.043)')
        DFA = st.text_input('DFA (0.574–0.825)')
        spread1 = st.text_input('spread1 (-7.96 – -2.43)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%) (0.0016–0.033)')
        Shimmer = st.text_input('MDVP:Shimmer (0.009–0.119)')
        DDA = st.text_input('Shimmer:DDA (0.013–0.169)')
        spread2 = st.text_input('spread2 (0.006–0.450)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs) (0.000007–0.00026)')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB) (0.085–1.30)')
        NHR = st.text_input('NHR (0.0006–0.314)')

    parkinsons_diagnosis = ''

    if st.button("🔍 Predict Parkinson’s Disease"):
        try:
            inputs = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                      Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                      RPDE, DFA, spread1, spread2, D2, PPE]

            if not all(inputs):
                st.error("⚠️ Please fill in all fields before prediction.")
            else:
                inputs = [float(x) for x in inputs]
                parkinsons_prediction = parkinsons_model.predict([inputs])

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "🧠 The person has Parkinson’s disease."
                else:
                    parkinsons_diagnosis = "✅ The person does not have Parkinson’s disease."
                st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("❌ Please enter valid numeric values only.")

# -----------------------------
# Blog / Insights Tab
# -----------------------------
with tabs[4]:
    st.header("📰 Blog & Insights (Coming Soon)")
    st.info("""
    This section will include:
    - Detailed explanation of each ML model  
    - Visualizations of dataset distributions  
    - Limitations and future improvements  
    - Research references and external resources  
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr style="border: 1px solid #ddd;">
<p style="text-align:center;">🌐 Designed and Developed by <b>Japanjot Singh</b></p>
""", unsafe_allow_html=True)
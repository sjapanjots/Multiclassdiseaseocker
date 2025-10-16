import pickle
import streamlit as st

# --------------------------- PAGE CONFIG & LIGHT THEME ---------------------------
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Custom light theme styling
light_theme = """
<style>
body {
    background-color: #fdfdfd;  /* light background */
    color: #000000;             /* dark text */
}
.stButton>button {
    background-color: #e0f7fa;  /* light cyan button */
    color: #000;
    font-weight: bold;
}
.stMarkdown h3 {
    color: #007ACC;
}
</style>
"""
st.markdown(light_theme, unsafe_allow_html=True)

# --------------------------- LOAD MODELS ---------------------------
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# --------------------------- TABS NAVIGATION ---------------------------
tab1, tab2, tab3 = st.tabs(["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"])

# --------------------------- DIABETES PAGE ---------------------------
with tab1:
    st.title('Diabetes Prediction using ML')
    st.markdown("""
    **Problem Statement:** Diabetes is a chronic disease that occurs when blood sugar (glucose) is too high.
    This tool predicts the likelihood of diabetes based on key medical parameters.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input('Number of Pregnancies', placeholder="0-17")
    with col2: Glucose = st.text_input('Glucose Level', placeholder="0-199")
    with col3: BloodPressure = st.text_input('Blood Pressure', placeholder="0-122")
    with col1: SkinThickness = st.text_input('Skin Thickness', placeholder="0-99")
    with col2: Insulin = st.text_input('Insulin Level', placeholder="0-846")
    with col3: BMI = st.text_input('BMI', placeholder="0-67.1")
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', placeholder="0.078-2.42")
    with col2: Age = st.text_input('Age', placeholder="21-81")
    
    if st.button('Predict Diabetes', key='diabetes'):
        errors = []
        fields = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        field_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        min_max = [(0,17),(0,199),(0,122),(0,99),(0,846),(0,67.1),(0.078,2.42),(21,81)]
        for i, val in enumerate(fields):
            if not val.strip(): errors.append(f"{field_names[i]} is required.")
        try:
            vals = [float(f) for f in fields]
        except ValueError:
            errors.append("All input values must be numeric.")
        if 'vals' in locals():
            for i, v in enumerate(vals):
                if v < min_max[i][0] or v > min_max[i][1]:
                    st.warning(f"{field_names[i]} is outside normal range ({min_max[i][0]}-{min_max[i][1]})")
        if errors:
            for e in errors: st.error(e)
        else:
            pred = diabetes_model.predict([vals])[0]
            if pred==1:
                st.error("⚠ The person is diabetic")
            else:
                st.success("✅ The person is not diabetic")
    st.markdown("[Learn more about Diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)")

# --------------------------- HEART DISEASE PAGE ---------------------------
with tab2:
    st.title('Heart Disease Prediction using ML')
    st.markdown("""
    **Problem Statement:** Heart disease is the leading cause of death worldwide.
    This tool predicts the likelihood of heart disease based on vital health parameters.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age', placeholder="29-77")
    with col2: sex = st.text_input('Sex', placeholder="0=Female,1=Male")
    with col3: cp = st.text_input('Chest Pain type', placeholder="0-3")
    with col1: trestbps = st.text_input('Resting Blood Pressure', placeholder="94-200")
    with col2: chol = st.text_input('Serum Cholesterol', placeholder="126-564")
    with col3: fbs = st.text_input('Fasting Blood Sugar >120 mg/dl', placeholder="0/1")
    with col1: restecg = st.text_input('Resting ECG', placeholder="0-2")
    with col2: thalach = st.text_input('Max Heart Rate', placeholder="71-202")
    with col3: exang = st.text_input('Exercise Induced Angina', placeholder="0/1")
    with col1: oldpeak = st.text_input('ST depression', placeholder="0-6.2")
    with col2: slope = st.text_input('Slope of ST segment', placeholder="0-2")
    with col3: ca = st.text_input('Major vessels colored', placeholder="0-3")
    with col1: thal = st.text_input('Thalassemia', placeholder="0=normal,1=fixed,2=reversable")
    
    if st.button('Predict Heart Disease', key='heart'):
        errors = []
        fields = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        field_names = ["Age","Sex","Chest Pain","Rest BP","Cholesterol","Fasting BS","Rest ECG","Max HR","ExAng","Oldpeak","Slope","Ca","Thal"]
        for i, val in enumerate(fields):
            if not val.strip(): errors.append(f"{field_names[i]} is required.")
        try: vals = [float(f) for f in fields]
        except ValueError:
            errors.append("All input values must be numeric.")
        if errors:
            for e in errors: st.error(e)
        else:
            pred = heart_disease_model.predict([vals])[0]
            if pred==1:
                st.error("⚠ The person has heart disease")
            else:
                st.success("✅ The person does not have heart disease")
    st.markdown("[Learn more about Heart Disease](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))")

# --------------------------- PARKINSON'S PAGE ---------------------------
with tab3:
    st.title("Parkinson's Disease Prediction using ML")
    st.markdown("""
    **Problem Statement:** Parkinson’s disease is a neurodegenerative disorder affecting movement.
    This tool predicts the likelihood of Parkinson’s disease using voice measurements.
    """)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    parkinsons_fields = {
        'fo': st.text_input('MDVP:Fo(Hz)'), 'fhi': st.text_input('MDVP:Fhi(Hz)'),
        'flo': st.text_input('MDVP:Flo(Hz)'), 'Jitter_percent': st.text_input('MDVP:Jitter(%)'),
        'Jitter_Abs': st.text_input('MDVP:Jitter(Abs)'), 'RAP': st.text_input('MDVP:RAP'),
        'PPQ': st.text_input('MDVP:PPQ'), 'DDP': st.text_input('Jitter:DDP'),
        'Shimmer': st.text_input('MDVP:Shimmer'), 'Shimmer_dB': st.text_input('MDVP:Shimmer(dB)'),
        'APQ3': st.text_input('Shimmer:APQ3'), 'APQ5': st.text_input('Shimmer:APQ5'),
        'APQ': st.text_input('MDVP:APQ'), 'DDA': st.text_input('Shimmer:DDA'),
        'NHR': st.text_input('NHR'), 'HNR': st.text_input('HNR'),
        'RPDE': st.text_input('RPDE'), 'DFA': st.text_input('DFA'),
        'spread1': st.text_input('spread1'), 'spread2': st.text_input('spread2'),
        'D2': st.text_input('D2'), 'PPE': st.text_input('PPE')
    }
    
    if st.button("Predict Parkinson's", key='parkinson'):
        errors = []
        for k,v in parkinsons_fields.items():
            if not v.strip(): errors.append(f"{k} is required.")
        try: vals = [float(v) for v in parkinsons_fields.values()]
        except ValueError: errors.append("All input values must be numeric.")
        if errors:
            for e in errors: st.error(e)
        else:
            pred = parkinsons_model.predict([vals])[0]
            if pred==1:
                st.error("⚠ The person has Parkinson's disease")
            else:
                st.success("✅ The person does not have Parkinson's disease")
    st.markdown("[Learn more about Parkinson's Disease](https://www.parkinson.org/Understanding-Parkinsons)")

# --------------------------- FOOTER ---------------------------
st.markdown("---")
st.markdown("⚠ This is a predictive tool; not a substitute for medical advice.")
st.markdown("Designed & Developed by Japanjot Singh")
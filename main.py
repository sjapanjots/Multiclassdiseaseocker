import pickle
import streamlit as st
import pandas as pd

# loading the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:    
    st.title("Multiple Disease Prediction System")
    choice = st.radio("Navigation", ["Diabetes Prediction","Heart Disease Prediction","Parkinsons Prediction"])
    st.info("Project is not 100% accurate")

# ------------------------------------ DIABETES -----------------------------------
if choice == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies (0 - 17)')
    with col2:
        Glucose = st.text_input('Glucose Level (0 - 199)')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value (0 - 122)')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value (0 - 99)')
    with col2:
        Insulin = st.text_input('Insulin Level (0 - 846)')
    with col3:
        BMI = st.text_input('BMI value (0 - 67.1)')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value (0.078 - 2.42)')
    with col2:
        Age = st.text_input('Age of the Person (21 - 81)')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        errors = []

        # Check empty fields
        fields = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        field_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        for i, val in enumerate(fields):
            if not val.strip():
                errors.append(f"{field_names[i]} is required.")

        # Convert to float and validate
        try:
            Pregnancies_val = float(Pregnancies)
        except ValueError:
            errors.append("Pregnancies must be a number.")

        try:
            Glucose_val = float(Glucose)
        except ValueError:
            errors.append("Glucose must be a number.")

        try:
            BloodPressure_val = float(BloodPressure)
        except ValueError:
            errors.append("Blood Pressure must be a number.")

        try:
            SkinThickness_val = float(SkinThickness)
        except ValueError:
            errors.append("Skin Thickness must be a number.")

        try:
            Insulin_val = float(Insulin)
        except ValueError:
            errors.append("Insulin must be a number.")

        try:
            BMI_val = float(BMI)
        except ValueError:
            errors.append("BMI must be a number.")

        try:
            DiabetesPedigreeFunction_val = float(DiabetesPedigreeFunction)
        except ValueError:
            errors.append("Diabetes Pedigree Function must be a number.")

        try:
            Age_val = float(Age)
        except ValueError:
            errors.append("Age must be a number.")

        # Show errors or make prediction
        if errors:
            for e in errors:
                st.error(e)
        else:
            diab_prediction = diabetes_model.predict([[Pregnancies_val, Glucose_val, BloodPressure_val, SkinThickness_val,
                                                      Insulin_val, BMI_val, DiabetesPedigreeFunction_val, Age_val]])
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0]==1 else 'The person is not diabetic'
            st.success(diab_diagnosis)

# ------------------------------------ HEART DISEASE -----------------------------------
if choice == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (0=Female,1=Male)')
    with col3:
        cp = st.text_input('Chest Pain types (0-3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0/1)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0-2)')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (0/1)')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy (0-3)')
    with col1:
        thal = st.text_input('thal: 0=normal,1=fixed,2=reversable')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        errors = []

        heart_fields = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_names = ["Age","Sex","Chest Pain","Resting BP","Cholesterol","Fasting BS","Rest ECG","Max HR","ExAng","Oldpeak","Slope","Ca","Thal"]
        for i, val in enumerate(heart_fields):
            if not val.strip():
                errors.append(f"{heart_names[i]} is required.")

        # Convert to float
        try:
            vals = [float(x) for x in heart_fields]
        except ValueError:
            errors.append("All numeric fields must be numbers.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            heart_prediction = heart_disease_model.predict([vals])
            heart_diagnosis = 'The person is having heart disease' if heart_prediction[0]==1 else 'The person does not have any heart disease'
            st.success(heart_diagnosis)

# ------------------------------------ PARKINSON'S DISEASE -----------------------------------
if choice == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)  

    # List of fields
    parkinsons_fields = {
        'fo': st.text_input('MDVP:Fo(Hz) (88-260)'), 
        'fhi': st.text_input('MDVP:Fhi(Hz) (102-592)'),
        'flo': st.text_input('MDVP:Flo(Hz) (65-239)'),
        'Jitter_percent': st.text_input('MDVP:Jitter(%) (0.00168-0.03316)'),
        'Jitter_Abs': st.text_input('MDVP:Jitter(Abs) (0.000007-0.00026)'),
        'RAP': st.text_input('MDVP:RAP (0.00068-0.02144)'),
        'PPQ': st.text_input('MDVP:PPQ (0.00092-0.01958)'),
        'DDP': st.text_input('Jitter:DDP (0.00204-0.06433)'),
        'Shimmer': st.text_input('MDVP:Shimmer (0.00954-0.11908)'),
        'Shimmer_dB': st.text_input('MDVP:Shimmer(dB) (0.085-1.302)'),
        'APQ3': st.text_input('Shimmer:APQ3 (0.01026-0.03134)'),
        'APQ5': st.text_input('Shimmer:APQ5 (0.01161-0.04518)'),
        'APQ': st.text_input('MDVP:APQ (0.01373-0.04368)'),
        'DDA': st.text_input('Shimmer:DDA (0.01364-0.16942)'),
        'NHR': st.text_input('NHR (0.00065-0.31482)'),
        'HNR': st.text_input('HNR (8.441-33.047)'),
        'RPDE': st.text_input('RPDE (0.25657-0.68515)'),
        'DFA': st.text_input('DFA (0.574282-0.825288)'),
        'spread1': st.text_input('spread1 (-7.964984--2.434031)'),
        'spread2': st.text_input('spread2 (0.006274-0.450493)'),
        'D2': st.text_input('D2 (1.423287-3.671155)'),
        'PPE': st.text_input('PPE (0.044539-0.527367)')
    }

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        errors = []

        # Empty check
        for k, v in parkinsons_fields.items():
            if not v.strip():
                errors.append(f"{k} is required.")

        # Numeric conversion
        try:
            parkinsons_vals = [float(v) for v in parkinsons_fields.values()]
        except ValueError:
            errors.append("All input values must be numeric.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            parkinsons_prediction = parkinsons_model.predict([parkinsons_vals])
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0]==1 else "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)

# Footer
hide_st_style = """
<style>
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
Design and Developed by Japanjot Singh
"""
st.markdown(hide_st_style , unsafe_allow_html=True)
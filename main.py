import pickle
import streamlit as st

# --------------------------- PAGE CONFIG & LIGHT THEME ---------------------------
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
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
.stSidebar {
    background-color: #f0f0f0;  /* light gray sidebar */
}
</style>
"""
st.markdown(light_theme, unsafe_allow_html=True)

# --------------------------- LOAD MODELS ---------------------------
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# --------------------------- SIDEBAR NAVIGATION ---------------------------
st.sidebar.title("Multiple Disease Prediction System")
choice = st.sidebar.radio("Navigation", ["Diabetes Prediction","Heart Disease Prediction","Parkinsons Prediction"])
st.sidebar.info("Project is not 100% accurate")

# --------------------------- DIABETES PAGE ---------------------------
if choice == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input('Number of Pregnancies (0-17)')
    with col2: Glucose = st.text_input('Glucose Level (0-199)')
    with col3: BloodPressure = st.text_input('Blood Pressure (0-122)')
    with col1: SkinThickness = st.text_input('Skin Thickness (0-99)')
    with col2: Insulin = st.text_input('Insulin Level (0-846)')
    with col3: BMI = st.text_input('BMI (0-67.1)')
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function (0.078-2.42)')
    with col2: Age = st.text_input('Age (21-81)')

    if st.button('Diabetes Test Result'):
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

        # Min-max warning
        if 'vals' in locals():
            for i, v in enumerate(vals):
                if v < min_max[i][0] or v > min_max[i][1]:
                    st.warning(f"{field_names[i]} is outside normal range ({min_max[i][0]}-{min_max[i][1]})")

        if errors:
            for e in errors: st.error(e)
        else:
            pred = diabetes_model.predict([vals])[0]
            st.success("The person is diabetic" if pred==1 else "The person is not diabetic")

# --------------------------- HEART DISEASE PAGE ---------------------------
if choice == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age (29-77)')
    with col2: sex = st.text_input('Sex (0=Female,1=Male)')
    with col3: cp = st.text_input('Chest Pain (0-3)')
    with col1: trestbps = st.text_input('Resting Blood Pressure (94-200)')
    with col2: chol = st.text_input('Serum Cholestoral in mg/dl (126-564)')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0/1)')
    with col1: restecg = st.text_input('Resting ECG (0-2)')
    with col2: thalach = st.text_input('Maximum Heart Rate achieved (71-202)')
    with col3: exang = st.text_input('Exercise Induced Angina (0/1)')
    with col1: oldpeak = st.text_input('ST depression induced by exercise (0-6.2)')
    with col2: slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
    with col3: ca = st.text_input('Major vessels colored by flourosopy (0-3)')
    with col1: thal = st.text_input('thal (0=normal,1=fixed,2=reversable)')

    if st.button('Heart Disease Test Result'):
        errors = []
        fields = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        field_names = ["Age","Sex","Chest Pain","Rest BP","Cholesterol","Fasting BS","Rest ECG","Max HR","ExAng","Oldpeak","Slope","Ca","Thal"]
        min_max = [(29,77),(0,1),(0,3),(94,200),(126,564),(0,1),(0,2),(71,202),(0,1),(0,6.2),(0,2),(0,3),(0,2)]

        for i, val in enumerate(fields):
            if not val.strip(): errors.append(f"{field_names[i]} is required.")
        
        try: vals = [float(f) for f in fields]
        except ValueError:
            errors.append("All input values must be numeric.")

        if 'vals' in locals():
            for i, v in enumerate(vals):
                if v < min_max[i][0] or v > min_max[i][1]:
                    st.warning(f"{field_names[i]} is outside normal range ({min_max[i][0]}-{min_max[i][1]})")

        if errors:
            for e in errors: st.error(e)
        else:
            pred = heart_disease_model.predict([vals])[0]
            st.success("The person has heart disease" if pred==1 else "The person does not have heart disease")

# --------------------------- PARKINSON'S PAGE ---------------------------
if choice == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
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

    if st.button("Parkinson's Test Result"):
        errors = []
        for k, v in parkinsons_fields.items():
            if not v.strip(): errors.append(f"{k} is required.")
        
        try: vals = [float(v) for v in parkinsons_fields.values()]
        except ValueError: errors.append("All input values must be numeric.")

        # Min-max warning (optional realistic ranges)
        parkinsons_ranges = {
            'fo':(88,260),'fhi':(102,592),'flo':(65,239),'Jitter_percent':(0.00168,0.03316),
            'Jitter_Abs':(0.000007,0.00026),'RAP':(0.00068,0.02144),'PPQ':(0.00092,0.01958),
            'DDP':(0.00204,0.06433),'Shimmer':(0.00954,0.11908),'Shimmer_dB':(0.085,1.302),
            'APQ3':(0.01026,0.03134),'APQ5':(0.01161,0.04518),'APQ':(0.01373,0.04368),
            'DDA':(0.01364,0.16942),'NHR':(0.00065,0.31482),'HNR':(8.441,33.047),
            'RPDE':(0.25657,0.68515),'DFA':(0.574282,0.825288),'spread1':(-7.964984,-2.434031),
            'spread2':(0.006274,0.450493),'D2':(1.423287,3.671155),'PPE':(0.044539,0.527367)
        }
        if 'vals' in locals():
            for i, (k,v) in enumerate(zip(parkinsons_fields.keys(),vals)):
                if v < parkinsons_ranges[k][0] or v > parkinsons_ranges[k][1]:
                    st.warning(f"{k} is outside normal range ({parkinsons_ranges[k][0]}-{parkinsons_ranges[k][1]})")

        if errors:
            for e in errors: st.error(e)
        else:
            pred = parkinsons_model.predict([vals])[0]
            st.success("The person has Parkinson's disease" if pred==1 else "The person does not have Parkinson's disease")

# --------------------------- FOOTER ---------------------------
hide_st_style = """
<style>
footer {visibility: hidden;}
</style>
Design and Developed by Japanjot Singh
"""
st.markdown(hide_st_style , unsafe_allow_html=True)
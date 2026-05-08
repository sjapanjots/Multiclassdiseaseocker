import pickle
import streamlit as st
import pandas as pd
import nbformat
from nbconvert import HTMLExporter
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AI Health Prediction Dashboard",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Feature 5: Dark / Light Mode Toggle via CSS
# -----------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

base_css = """
<style>
    :root {
        --accent: #0f9f8f;
        --accent-2: #d95f59;
        --panel: rgba(255, 255, 255, 0.88);
        --line: rgba(16, 24, 40, 0.12);
        --text-soft: #667085;
    }
    .stApp {
        background:
            linear-gradient(135deg, rgba(15, 159, 143, 0.10), transparent 34%),
            linear-gradient(225deg, rgba(217, 95, 89, 0.10), transparent 32%),
            #f7faf9;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }
    h1, h2, h3 { letter-spacing: 0; }
    .app-hero {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        background: var(--panel);
        box-shadow: 0 14px 34px rgba(16, 24, 40, 0.08);
        margin-bottom: 1rem;
    }
    .app-kicker {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .app-hero h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.05;
    }
    .app-hero p {
        color: var(--text-soft);
        margin: 0.65rem 0 0;
        max-width: 780px;
        font-size: 1rem;
    }
    .section-lead {
        border-left: 4px solid var(--accent);
        padding: 0.1rem 0 0.1rem 0.85rem;
        margin: 0.2rem 0 1rem;
    }
    .section-lead h2 {
        margin: 0 0 0.2rem;
        font-size: 1.35rem;
    }
    .section-lead p {
        margin: 0;
        color: var(--text-soft);
    }
    div[data-testid="stTabs"] button {
        border-radius: 8px 8px 0 0;
        font-weight: 650;
    }
    div[data-testid="metric-container"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 8px 22px rgba(16, 24, 40, 0.05);
    }
    .stTextInput input, .stNumberInput input { border-radius: 8px; }
    .stSelectbox div[data-baseweb="select"] > div { border-radius: 8px; }
    .stButton button {
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid rgba(15, 159, 143, 0.35);
    }
    .stButton button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }
</style>
"""
dark_css = """
<style>
    :root {
        --panel: rgba(24, 28, 31, 0.92);
        --line: rgba(255, 255, 255, 0.13);
        --text-soft: #b8c0bd;
    }
    .stApp {
        background:
            linear-gradient(135deg, rgba(15, 159, 143, 0.14), transparent 34%),
            linear-gradient(225deg, rgba(217, 95, 89, 0.13), transparent 32%),
            #111514;
        color: #f7faf9;
    }
    .app-hero, div[data-testid="metric-container"] { background: var(--panel); }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #1c2220;
        color: #f7faf9;
    }
</style>
"""
light_css = "<style></style>"
hide_st_style = """
<style>
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(base_css, unsafe_allow_html=True)
st.markdown(dark_css if st.session_state.dark_mode else light_css, unsafe_allow_html=True)

# -----------------------------
# Load Trained Models
# -----------------------------
diabetes_model        = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model   = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model      = pickle.load(open('parkinsons_model.sav', 'rb'))
burnout_model         = pickle.load(open('burnout_model.sav', 'rb'))
burnout_preprocessing = pickle.load(open('burnout_preprocessing.sav', 'rb'))
burnout_label_encoder = pickle.load(open('burnout_label_encoder.sav', 'rb'))

# -----------------------------
# Feature 1: Session State — Prediction History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Header Row: Title + Dark Mode Toggle
# -----------------------------
title_col, toggle_col = st.columns([6, 1])
with title_col:
    st.markdown("""
    <div class="app-hero">
        <div class="app-kicker">Machine learning assistant</div>
        <h1>AI Health Prediction Dashboard</h1>
        <p>Run quick prediction checks for diabetes, heart disease, Parkinson's disease, and employee burnout from one focused workspace.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Project under development. Results are informational and should not replace professional medical advice.")
with toggle_col:
    st.write("")
    st.write("")
    toggle_label = "Dark mode" if not st.session_state.dark_mode else "Light mode"
    if st.button(toggle_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# -----------------------------
# Navigation Tabs
# -----------------------------
tabs = st.tabs([
    "About",
    "Diabetes",
    "Heart Disease",
    "Parkinson's",
    "Burnout",
    "History",
    "Insights",
    "Models"
])

# -----------------------------
# About Project
# -----------------------------
with tabs[0]:
    st.markdown("""
    <div class="section-lead">
        <h2>Project Overview</h2>
        <p>A compact Streamlit dashboard for exploring multiple trained prediction models.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("""
    This web application predicts the likelihood of **Diabetes**, **Heart Disease**, **Parkinson's Disease**,
    and **Employee Burnout** using pre-trained Machine Learning models.
    The system demonstrates how AI can assist healthcare professionals by providing early predictions
    based on medical parameters.
    However, these predictions are **not diagnostic** and should not replace medical advice.
    """)
    st.markdown("""
    ### Project Highlights
    - Built using **Streamlit** and **Scikit-learn**
    - Models saved as `.sav` files for lightweight deployment
    - Backend powered by pre-trained supervised ML models
    - Designed for Streamlit Cloud deployment
    - Prediction history tracking across the session
    - Dark and light mode support
    - Designed and developed by *Japanjot Singh*
    """)

    # Quick stats from history
    if st.session_state.history:
        st.markdown("### Session Stats")
        sc1, sc2, sc3, sc4 = st.columns(4)
        total = len(st.session_state.history)
        sc1.metric("Total Predictions", total)
        sc2.metric("Diabetes Checks",   sum(1 for h in st.session_state.history if h["Disease"] == "Diabetes"))
        sc3.metric("Heart Checks",      sum(1 for h in st.session_state.history if h["Disease"] == "Heart Disease"))
        sc4.metric("Burnout Checks",    sum(1 for h in st.session_state.history if h["Disease"] == "Burnout"))

# -----------------------------
# Diabetes Prediction Tab
# -----------------------------
with tabs[1]:
    st.markdown("""
    <div class="section-lead">
        <h2>Diabetes Prediction</h2>
        <p>Enter common diagnostic measurements, then run the trained diabetes model.</p>
    </div>
    """, unsafe_allow_html=True)
    # Feature 4: BMI Calculator Helper
    with st.expander("BMI Calculator - calculate BMI before filling the form"):
        bmi_col1, bmi_col2, bmi_col3 = st.columns(3)
        with bmi_col1:
            weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, key="bmi_weight")
        with bmi_col2:
            height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, key="bmi_height")
        with bmi_col3:
            st.write("")
            st.write("")
            if st.button("Calculate BMI"):
                bmi_result = weight_kg / ((height_cm / 100) ** 2)
                if bmi_result < 18.5:
                    category = "Underweight"
                elif bmi_result < 25:
                    category = "Normal"
                elif bmi_result < 30:
                    category = "Overweight"
                else:
                    category = "Obese"
                st.success(f"Your BMI: **{bmi_result:.1f}** - {category}. Copy this value into the BMI field below.")

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

    # Feature 2: Reset button
    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        if st.button('Predict Diabetes'):
            try:
                if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
                    st.error("Please fill in all fields before prediction.")
                else:
                    inputs = [float(Pregnancies), float(Glucose), float(BloodPressure),
                              float(SkinThickness), float(Insulin), float(BMI),
                              float(DiabetesPedigreeFunction), float(Age)]
                    diab_prediction = diabetes_model.predict([inputs])
                    result = 'Diabetic' if diab_prediction[0] == 1 else 'Not Diabetic'
                    if diab_prediction[0] == 1:
                        st.success('Prediction result: diabetic.')
                    else:
                        st.success('Prediction result: not diabetic.')
                    # Feature 1: Log to history
                    st.session_state.history.append({
                        "Disease": "Diabetes", "Result": result,
                        "Inputs": f"Glucose={Glucose}, BMI={BMI}, Age={Age}"
                    })
            except ValueError:
                st.error("Please enter valid numeric values only.")

# -----------------------------
# Heart Disease Prediction Tab
# -----------------------------
with tabs[2]:
    st.markdown("""
    <div class="section-lead">
        <h2>Heart Disease Prediction</h2>
        <p>Use cardiovascular measurements to estimate heart disease risk.</p>
    </div>
    """, unsafe_allow_html=True)
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

    if st.button('Predict Heart Disease'):
        try:
            if not all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
                st.error("Please fill in all fields before prediction.")
            else:
                inputs = [float(age), float(sex), float(cp), float(trestbps), float(chol),
                          float(fbs), float(restecg), float(thalach), float(exang),
                          float(oldpeak), float(slope), float(ca), float(thal)]
                heart_prediction = heart_disease_model.predict([inputs])
                result = 'Has Heart Disease' if heart_prediction[0] == 1 else 'No Heart Disease'
                if heart_prediction[0] == 1:
                    st.success('Prediction result: heart disease indicated.')
                else:
                    st.success('Prediction result: no heart disease indicated.')
                # Feature 1: Log to history
                st.session_state.history.append({
                    "Disease": "Heart Disease", "Result": result,
                    "Inputs": f"Age={age}, Chol={chol}, MaxHR={thalach}"
                })
        except ValueError:
            st.error("Please enter valid numeric values only.")

# -----------------------------
# Parkinson's Prediction Tab
# -----------------------------
with tabs[3]:
    st.markdown("""
    <div class="section-lead">
        <h2>Parkinson's Prediction</h2>
        <p>Provide voice measurement values within the displayed ranges.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo       = st.text_input('MDVP:Fo(Hz) (88–260)')
        RAP      = st.text_input('MDVP:RAP (0.00068–0.02144)')
        APQ3     = st.text_input('Shimmer:APQ3 (0.01026–0.03134)')
        HNR      = st.text_input('HNR (8.44–33.04)')
        D2       = st.text_input('D2 (1.42–3.67)')
    with col2:
        fhi      = st.text_input('MDVP:Fhi(Hz) (102–592)')
        PPQ      = st.text_input('MDVP:PPQ (0.00092–0.01958)')
        APQ5     = st.text_input('Shimmer:APQ5 (0.01161–0.04518)')
        RPDE     = st.text_input('RPDE (0.256–0.685)')
        PPE      = st.text_input('PPE (0.044–0.527)')
    with col3:
        flo      = st.text_input('MDVP:Flo(Hz) (65–239)')
        DDP      = st.text_input('Jitter:DDP (0.002–0.064)')
        APQ      = st.text_input('MDVP:APQ (0.013–0.043)')
        DFA      = st.text_input('DFA (0.574–0.825)')
        spread1  = st.text_input('spread1 (-7.96 – -2.43)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%) (0.0016–0.033)')
        Shimmer        = st.text_input('MDVP:Shimmer (0.009–0.119)')
        DDA            = st.text_input('Shimmer:DDA (0.013–0.169)')
        spread2        = st.text_input('spread2 (0.006–0.450)')
    with col5:
        Jitter_Abs  = st.text_input('MDVP:Jitter(Abs) (0.000007–0.00026)')
        Shimmer_dB  = st.text_input('MDVP:Shimmer(dB) (0.085–1.30)')
        NHR         = st.text_input('NHR (0.0006–0.314)')

    if st.button("Predict Parkinson's Disease"):
        try:
            inputs = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                      Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                      RPDE, DFA, spread1, spread2, D2, PPE]
            if not all(inputs):
                st.error("Please fill in all fields before prediction.")
            else:
                inputs = [float(x) for x in inputs]
                parkinsons_prediction = parkinsons_model.predict([inputs])
                result = "Has Parkinson's" if parkinsons_prediction[0] == 1 else "No Parkinson's"
                if parkinsons_prediction[0] == 1:
                    st.success("Prediction result: Parkinson's disease indicated.")
                else:
                    st.success("Prediction result: no Parkinson's disease indicated.")
                # Feature 1: Log to history
                st.session_state.history.append({
                    "Disease": "Parkinson's", "Result": result,
                    "Inputs": f"Fo={fo}, HNR={HNR}, PPE={PPE}"
                })
        except ValueError:
            st.error("Please enter valid numeric values only.")

# -----------------------------
# Burnout Prediction Tab
# -----------------------------
with tabs[4]:
    st.markdown("""
    <div class="section-lead">
        <h2>Employee Burnout Prediction</h2>
        <p>Predict low, medium, or high burnout risk from workplace and lifestyle signals.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        b_age                 = st.text_input('Age (18–65)', key='b_age')
        b_experience_years    = st.text_input('Experience Years (0–40)', key='b_exp')
        b_work_hours_per_week = st.text_input('Work Hours Per Week (20–80)', key='b_wh')
        b_overtime_hours      = st.text_input('Overtime Hours Per Week (0–30)', key='b_ot')
        b_meetings_per_day    = st.text_input('Meetings Per Day (0–15)', key='b_meet')
        b_deadlines_missed    = st.text_input('Deadlines Missed (Per Month) (0–10)', key='b_dl')
        b_job_satisfaction    = st.text_input('Job Satisfaction (1–10)', key='b_js')
        b_manager_support     = st.text_input('Manager Support Score (1–10)', key='b_ms')
    with col2:
        b_work_life_balance   = st.text_input('Work-Life Balance Score (1–10)', key='b_wlb')
        b_sleep_hours         = st.text_input('Sleep Hours Per Night (3–10)', key='b_sl')
        b_physical_activity   = st.text_input('Physical Activity Days Per Week (0–7)', key='b_pa')
        b_screen_time         = st.text_input('Screen Time Hours Per Day (2–16)', key='b_sc')
        b_caffeine_intake     = st.text_input('Caffeine Intake (cups/day) (0–10)', key='b_caf')
        b_social_support      = st.text_input('Social Support Score (1–10)', key='b_ss')
        b_stress_level        = st.text_input('Stress Level (1–10)', key='b_stl')
        b_anxiety_score       = st.text_input('Anxiety Score (1–10)', key='b_anx')
    with col3:
        b_depression_score    = st.text_input('Depression Score (1–10)', key='b_dep')
        b_gender              = st.selectbox('Gender', ['Male', 'Female', 'Non-binary', 'Prefer not to say'], key='b_gen')
        b_job_role            = st.selectbox('Job Role', [
                                    'Engineer', 'Manager', 'Analyst', 'Designer',
                                    'Developer', 'HR', 'Sales', 'Marketing', 'Other'
                                ], key='b_jr')
        b_company_size        = st.selectbox('Company Size', ['Small', 'Medium', 'Large'], key='b_cs')
        b_work_mode           = st.selectbox('Work Mode', ['Remote', 'Hybrid', 'On-site'], key='b_wm')
        b_has_therapy         = st.selectbox('Currently in Therapy?', ['Yes', 'No'], key='b_ther')

    if st.button('Predict Burnout Level'):
        numeric_fields = {
            'age': b_age, 'experience_years': b_experience_years,
            'work_hours_per_week': b_work_hours_per_week, 'overtime_hours': b_overtime_hours,
            'meetings_per_day': b_meetings_per_day, 'deadlines_missed': b_deadlines_missed,
            'job_satisfaction': b_job_satisfaction, 'manager_support': b_manager_support,
            'work_life_balance': b_work_life_balance, 'sleep_hours': b_sleep_hours,
            'physical_activity_days': b_physical_activity, 'screen_time_hours': b_screen_time,
            'caffeine_intake': b_caffeine_intake, 'social_support_score': b_social_support,
            'stress_level': b_stress_level, 'anxiety_score': b_anxiety_score,
            'depression_score': b_depression_score
        }
        has_therapy_val = '1' if b_has_therapy == 'Yes' else '0'
        numeric_fields['has_therapy'] = has_therapy_val

        if not all(numeric_fields.values()):
            st.error("Please fill in all numeric fields before prediction.")
        else:
            try:
                input_dict: dict[str, float | str] = {k: float(v) for k, v in numeric_fields.items()}
                input_dict['gender']       = b_gender
                input_dict['job_role']     = b_job_role
                input_dict['company_size'] = b_company_size
                input_dict['work_mode']    = b_work_mode

                row = pd.DataFrame([input_dict])

                for col in burnout_preprocessing["categorical_cols"]:
                    le  = burnout_preprocessing["cat_encoders"][col]
                    val = str(row[col].iloc[0]) if col in row.columns else "Unknown"
                    row[col] = le.transform([val])[0] if val in le.classes_ else -1

                for col in burnout_preprocessing["numerical_cols"]:
                    if row[col].isnull().any():
                        row[col] = burnout_preprocessing["num_medians"][col]
                row[burnout_preprocessing["numerical_cols"]] = burnout_preprocessing["scaler"].transform(
                    row[burnout_preprocessing["numerical_cols"]]
                )

                X_inf        = row[burnout_preprocessing["feature_columns"]].values
                pred_encoded = burnout_model.predict(X_inf)[0]
                pred_label   = burnout_label_encoder.inverse_transform([pred_encoded])[0]
                pred_proba   = burnout_model.predict_proba(X_inf)[0]
                proba_dict   = dict(zip(burnout_label_encoder.classes_, pred_proba.round(3)))

                if pred_label == "High":
                    st.error(f"Burnout Level: **{pred_label}** - High risk. Consider seeking professional support.")
                elif pred_label == "Medium":
                    st.warning(f"Burnout Level: **{pred_label}** - Moderate risk. Monitor stress and work-life balance.")
                else:
                    st.success(f"Burnout Level: **{pred_label}** - Low risk. Keep maintaining healthy work habits!")

                # Feature 3: Confidence Gauge using progress bars
                st.markdown("#### Prediction Confidence")
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                low_p    = proba_dict.get('Low', 0)
                medium_p = proba_dict.get('Medium', 0)
                high_p   = proba_dict.get('High', 0)
                with prob_col1:
                    st.metric("Low", f"{low_p*100:.1f}%")
                    st.progress(float(low_p))
                with prob_col2:
                    st.metric("Medium", f"{medium_p*100:.1f}%")
                    st.progress(float(medium_p))
                with prob_col3:
                    st.metric("High", f"{high_p*100:.1f}%")
                    st.progress(float(high_p))

                # Feature 1: Log to history
                st.session_state.history.append({
                    "Disease": "Burnout", "Result": pred_label,
                    "Inputs": f"Stress={b_stress_level}, WorkHrs={b_work_hours_per_week}, Sleep={b_sleep_hours}"
                })

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# Feature 1: Prediction History Tab
# -----------------------------
with tabs[5]:
    st.markdown("""
    <div class="section-lead">
        <h2>Prediction History</h2>
        <p>Review the prediction results created during the current session.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("All predictions made during this session are recorded here.")

    if not st.session_state.history:
        st.info("No predictions made yet. Run a prediction from any disease tab to see it here.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        history_df.index += 1
        history_df.index.name = "#"
        st.dataframe(history_df, use_container_width=True)

        # Summary counts
        st.markdown("#### Summary")
        sum_cols = st.columns(len(history_df["Disease"].unique()))
        for i, disease in enumerate(history_df["Disease"].unique()):
            count = len(history_df[history_df["Disease"] == disease])
            sum_cols[i].metric(disease, f"{count} prediction{'s' if count > 1 else ''}")

        # Feature 2: Clear history button
        st.write("")
        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.rerun()

# -----------------------------
# Blog / Insights Tab
# -----------------------------
with tabs[6]:
    st.markdown("""
    <div class="section-lead">
        <h2>Insights</h2>
        <p>Reserved space for model notes, limitations, and future dataset exploration.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("""
    This section will include:
    - Detailed explanation of each ML model
    - Visualizations of dataset distributions
    - Limitations and future improvements
    - Research references and external resources
    """)

# -----------------------------
# Models Tab
# -----------------------------
with tabs[7]:
    st.markdown("""
    <div class="section-lead">
        <h2>Model Notebook</h2>
        <p>View the Jupyter notebook used for model training.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("This tab displays the Jupyter Notebook containing the code for training the models.")

    notebook_path = "Heart_disease_model.ipynb"
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    html_exporter.exclude_input = False
    html_data, _ = html_exporter.from_notebook_node(notebook_content)
    components.html(html_data, height=800, scrolling=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr style="border: 1px solid #ddd;">
<p style="text-align:center;">Designed and developed by <b>Japanjot Singh</b></p>
""", unsafe_allow_html=True)

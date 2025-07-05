import streamlit as st # web apps
import pandas as pd # data frames
import numpy as np # numbers
import joblib # load model
import warnings # warnings

warnings.filterwarnings('ignore') # ignore warnings

# --- Load Model ---
@st.cache_resource # cache model
def load_model_data():
    try:
        # load model
        mod = joblib.load('logistic_regression_model.pkl')
        # load columns
        cols = joblib.load('training_columns.pkl')

        # categories
        age_cats = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                          '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
        race_cats = ['White', 'Black', 'Asian', 'American Indian/Alaskan Native',
                           'Other', 'Hispanic']
        gen_health_cats = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']

        st.sidebar.write("Model loaded.") # confirm load
        return mod, cols, age_cats, race_cats, gen_health_cats
    except FileNotFoundError:
        st.error("Model files missing! Run 'train_model.py'.") # file error
        st.stop() # stop app
    except Exception as e:
        st.error(f"Load error: {e}") # other error
        st.stop()

# load all
mod, train_cols, age_cats, race_cats, gen_health_cats = load_model_data()

# --- UI Config ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# title
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("""
    This application helps Dr. Mendoza's nursing staff predict whether a patient is at risk of heart disease
    based on their health metrics.
    Please fill in the patient's details below to get an instant prediction.
""")

# sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
    This app uses a Logistic Regression model trained on health data to classify patients
    as 'at risk' or 'not at risk' for heart disease.
    The model is pre-trained and loaded for quick predictions.
""")

st.header("Patient Info") # input header

# two columns
c1, c2 = st.columns(2)

# --- Inputs ---
with c1:
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    ph = st.number_input("Physical Health (Days poor)", min_value=0, max_value=30, value=0, step=1)
    mh = st.number_input("Mental Health (Days poor)", min_value=0, max_value=30, value=0, step=1)
    sl_t = st.number_input("Sleep Time (Avg hours)", min_value=1.0, max_value=24.0, value=7.0, step=0.1)
    sex = st.radio("Sex", ("Female", "Male"))
    age_cat = st.selectbox("Age Category", age_cats)
    race = st.selectbox("Race", race_cats)

with c2:
    smoke = st.radio("Smoker?", ("No", "Yes"))
    alc = st.radio("Heavy Drinker?", ("No", "Yes"))
    stroke = st.radio("Had Stroke?", ("No", "Yes"))
    diff_walk = st.radio("Difficulty Walking?", ("No", "Yes"))
    phys_act = st.radio("Physical Activity?", ("No", "Yes"))
    diab = st.radio("Diabetic?", ("No", "Yes"))
    asthma = st.radio("Has Asthma?", ("No", "Yes"))
    kid_dis = st.radio("Has Kidney Disease?", ("No", "Yes"))
    skin_can = st.radio("Has Skin Cancer?", ("No", "Yes"))
    gen_h = st.selectbox("General Health", gen_health_cats)

st.markdown("---") # separator

# --- Predict Button ---
if st.button("Predict Heart Disease Risk"):
    # 1. Prep Input
    inp_data = { # collect inputs
        'BMI': bmi,
        'Smoking': smoke,
        'AlcoholDrinking': alc,
        'Stroke': stroke,
        'PhysicalHealth': ph,
        'MentalHealth': mh,
        'DiffWalk': diff_walk,
        'Sex': sex,
        'AgeCategory': age_cat,
        'Race': race,
        'Diabetic': diab,
        'PhysicalActivity': phys_act,
        'GenHealth': gen_h,
        'SleepTime': sl_t,
        'Asthma': asthma,
        'KidneyDisease': kid_dis,
        'SkinCancer': skin_can
    }

    inp_df = pd.DataFrame([inp_data]) # to dataframe

    # 2. Preprocess
    bin_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0} # binary map
    for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalk',
                'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex', 'Diabetic']:
        inp_df[col] = inp_df[col].map(bin_map)

    proc_inp = pd.DataFrame(0, index=[0], columns=train_cols) # empty frame

    for col in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']: # fill numeric
        if col in proc_inp.columns:
            proc_inp[col] = inp_df[col].values[0]

    for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalk', # fill binary
                'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex', 'Diabetic']:
        if col in proc_inp.columns:
            proc_inp[col] = inp_df[col].values[0]

    age_col_name = f'AgeCategory_{inp_df["AgeCategory"].values[0]}' # age category
    if age_col_name in proc_inp.columns:
        proc_inp[age_col_name] = 1

    race_col_name = f'Race_{inp_df["Race"].values[0]}' # race category
    if race_col_name in proc_inp.columns:
        proc_inp[race_col_name] = 1

    gen_h_col_name = f'GenHealth_{inp_df["GenHealth"].values[0]}' # health category
    if gen_h_col_name in proc_inp.columns:
        proc_inp[gen_h_col_name] = 1

    proc_inp = proc_inp.apply(pd.to_numeric, errors='coerce') # to numeric
    proc_inp.fillna(0, inplace=True) # fill na

    proc_inp = proc_inp[train_cols] # match columns

    # 3. Predict
    pred = mod.predict(proc_inp)[0] # get prediction
    pred_proba = mod.predict_proba(proc_inp)[0] # get probabilities

    # 4. Show Result
    st.subheader("Prediction Result:")
    if pred == 1:
        st.error("🚨 Patient is **AT RISK** of Heart Disease!") # at risk
        conf = pred_proba[1] * 100 # confidence
    else:
        st.success("✅ Patient is **NOT AT RISK** of Heart Disease.") # not at risk
        conf = pred_proba[0] * 100 # confidence

    st.write(f"Confidence Score: **{conf:.2f}%**")
    st.markdown("""
        *Note: This is a predictive tool and should not replace professional medical advice.*
    """)

st.markdown("---") # footer separator
st.markdown("Developed by Han Bi Kim Budiao for Dr. Mendoza's Community Health Clinic") # footer

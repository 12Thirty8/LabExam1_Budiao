import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore') # ignore

@st.cache_data # cache
def load_data_and_train_model():
    try:
        df = pd.read_csv('heart_2020_uncleaned.csv')

        init_rows = df.shape[0]
        df.dropna(inplace=True)
        st.sidebar.info(f"Dropped {init_rows - df.shape[0]} rows with missing values.")

        df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1, 'No': 0})

        bin_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalk',
                       'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
        for col in bin_cols:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

        df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

        df['Diabetic'] = df['Diabetic'].replace({
            'No, borderline diabetes': 'No',
            'Yes (during pregnancy)': 'Yes'
        }).map({'Yes': 1, 'No': 0})

        cat_cols = ['AgeCategory', 'Race', 'GenHealth']
        df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        X = df_enc.drop('HeartDisease', axis=1)
        y = df_enc['HeartDisease']

        X = X.apply(pd.to_numeric, errors='coerce')
        X.dropna(inplace=True)
        y = y[X.index]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        mod = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
        mod.fit(X_tr, y_tr)

        train_cols = X.columns.tolist()

        age_cats = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                          '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
        race_cats = ['White', 'Black', 'Asian', 'American Indian/Alaskan Native',
                           'Other', 'Hispanic']
        gen_health_cats = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']

        y_pred = mod.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)
        roc_auc = roc_auc_score(y_te, y_pred)

        st.sidebar.subheader("Model Summary:")
        st.sidebar.write(f"Accuracy: {acc:.2f}")
        st.sidebar.write(f"F1-Score: {f1:.2f}")
        st.sidebar.write(f"ROC AUC: {roc_auc:.2f}")
        st.sidebar.write("Model trained.")

        return mod, train_cols, age_cats, race_cats, gen_health_cats, df
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

mod, train_cols, age_cats, race_cats, gen_health_cats, orig_df = load_data_and_train_model()

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Risk Predictor")
st.markdown("""
    This application helps Dr. Mendoza's nursing staff predict whether a patient is at risk of heart disease
    based on their health metrics.
    Please fill in the patient's details below to get an instant prediction.
""")

st.sidebar.header("About")
st.sidebar.markdown("""
    This app uses a Logistic Regression model trained on health data to classify patients
    as 'at risk' or 'not at risk' for heart disease.
    The model and data preprocessing steps are included within the app for demonstration.
""")

st.header("Patient Info")

c1, c2 = st.columns(2)

with c1:
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    ph = st.number_input("Physical Health (Days)", min_value=0, max_value=30, value=0, step=1)
    mh = st.number_input("Mental Health (Days)", min_value=0, max_value=30, value=0, step=1)
    st = st.number_input("Sleep Time (Hours)", min_value=1.0, max_value=24.0, value=7.0, step=0.1)
    sex = st.radio("Sex", ("Female", "Male"))
    age_cat = st.selectbox("Age", age_cats)
    race = st.selectbox("Race", race_cats)

with c2:
    smoke = st.radio("Smoker?", ("No", "Yes"))
    alc = st.radio("Drinker?", ("No", "Yes"))
    stroke = st.radio("Stroke?", ("No", "Yes"))
    diff_walk = st.radio("Walk difficulty?", ("No", "Yes"))
    phys_act = st.radio("Phys Activity?", ("No", "Yes"))
    diab = st.radio("Diabetic?", ("No", "Yes"))
    asthma = st.radio("Asthma?", ("No", "Yes"))
    kid_dis = st.radio("Kidney Disease?", ("No", "Yes"))
    skin_can = st.radio("Skin Cancer?", ("No", "Yes"))
    gen_h = st.selectbox("General Health", gen_health_cats)

st.markdown("---")
if st.button("Predict"):
    inp_data = {
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
        'SleepTime': st,
        'Asthma': asthma,
        'KidneyDisease': kid_dis,
        'SkinCancer': skin_can
    }

    inp_df = pd.DataFrame([inp_data])

    bin_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalk',
                'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex', 'Diabetic']:
        inp_df[col] = inp_df[col].map(bin_map)

    proc_inp = pd.DataFrame(0, index=[0], columns=train_cols)

    for col in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']:
        if col in proc_inp.columns:
            proc_inp[col] = inp_df[col].values[0]

    for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalk',
                'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex', 'Diabetic']:
        if col in proc_inp.columns:
            proc_inp[col] = inp_df[col].values[0]

    age_col_name = f'AgeCategory_{inp_df["AgeCategory"].values[0]}'
    if age_col_name in proc_inp.columns:
        proc_inp[age_col_name] = 1

    race_col_name = f'Race_{inp_df["Race"].values[0]}'
    if race_col_name in proc_inp.columns:
        proc_inp[race_col_name] = 1

    gen_h_col_name = f'GenHealth_{inp_df["GenHealth"].values[0]}'
    if gen_h_col_name in proc_inp.columns:
        proc_inp[gen_h_col_name] = 1

    proc_inp = proc_inp.apply(pd.to_numeric, errors='coerce')
    proc_inp.fillna(0, inplace=True)

    proc_inp = proc_inp[train_cols]

    pred = mod.predict(proc_inp)[0]
    pred_proba = mod.predict_proba(proc_inp)[0]

    st.subheader("Result:")
    if pred == 1:
        st.error("🚨 Patient is **AT RISK**!")
        conf = pred_proba[1] * 100
    else:
        st.success("✅ Patient is **NOT AT RISK**.")
        conf = pred_proba[0] * 100

    st.write(f"Confidence: **{conf:.2f}%**")
    st.markdown("""
        *Note: This is a predictive tool and should not replace professional medical advice.*
    """)

st.markdown("---")
st.markdown("Developed for Dr. Mendoza's Community Health Clinic")
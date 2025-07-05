import streamlit as st # The Streamlit library for creating web apps
import pandas as pd # For data manipulation
import numpy as np # For numerical operations
import joblib # For loading the saved model and other objects
import warnings # To manage warnings

warnings.filterwarnings('ignore') # ignore warnings for cleaner output

# --- Load Model and Data Info (Cached) ---
# The @st.cache_resource decorator tells Streamlit to run this function only once
# when the app starts, and then reuse the results. This makes the app load instantly.
@st.cache_resource # cache model loading
def load_model_and_data_info():
    try:
        # Load the pre-trained model and the list of columns used during training
        mod = joblib.load('logistic_regression_model.pkl')
        train_cols = joblib.load('training_columns.pkl')

        # Define the categories for categorical features.
        # These are hardcoded as they are fixed based on your dataset.
        age_cats = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                          '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
        race_cats = ['White', 'Black', 'Asian', 'American Indian/Alaskan Native',
                           'Other', 'Hispanic']
        gen_health_cats = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']

        st.sidebar.write("Model loaded.") # Confirmation message in the sidebar
        return mod, train_cols, age_cats, race_cats, gen_health_cats
    except FileNotFoundError:
        # If the model files aren't found, inform the user to run the training script first.
        st.error("Model files not found! Please run 'train_model.py' first to generate 'logistic_regression_model.pkl' and 'training_columns.pkl'.")
        st.stop() # Stop the app execution
    except Exception as e:
        # Catch any other errors during loading.
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model and related information when the app starts
mod, train_cols, age_cats, race_cats, gen_health_cats = load_model_and_data_info()

# --- Streamlit UI Configuration ---
# Sets up the basic page configuration for the web app.
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Main title of the application
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("""
    This application helps Dr. Mendoza's nursing staff predict whether a patient is at risk of heart disease
    based on their health metrics.
    Please fill in the patient's details below to get an instant prediction.
""")

# Sidebar information
st.sidebar.header("About")
st.sidebar.markdown("""
    This app uses a Logistic Regression model trained on health data to classify patients
    as 'at risk' or 'not at risk' for heart disease.
    The model is pre-trained and loaded for quick predictions.
""")

st.header("Patient Information") # Section header for input fields

# Create two columns for better layout of input fields
c1, c2 = st.columns(2)

# --- Input Fields ---
# Widgets for users to input patient data
with c1:
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    ph = st.number_input("Physical Health (Days of poor physical health in last 30 days)", min_value=0, max_value=30, value=0, step=1)
    mh = st.number_input("Mental Health (Days of poor mental health in last 30 days)", min_value=0, max_value=30, value=0, step=1)
    sl_t = st.number_input("Sleep Time (Average hours of sleep per 24 hours)", min_value=1.0, max_value=24.0, value=7.0, step=0.1)
    sex = st.radio("Sex", ("Female", "Male"))
    age_cat = st.selectbox("Age Category", age_cats)
    race = st.selectbox("Race", race_cats)

with c2:
    smoke = st.radio("Smoker?", ("No", "Yes"))
    alc = st.radio("Heavy Alcohol Drinker?", ("No", "Yes"))
    stroke = st.radio("Had a Stroke?", ("No", "Yes"))
    diff_walk = st.radio("Difficulty Walking?", ("No", "Yes"))
    phys_act = st.radio("Physical Activity in last 30 days?", ("No", "Yes"))
    diab = st.radio("Diabetic?", ("No", "Yes")) # Simplified for UI
    asthma = st.radio("Has Asthma?", ("No", "Yes"))
    kid_dis = st.radio("Has Kidney Disease?", ("No", "Yes"))
    skin_can = st.radio("Has Skin Cancer?", ("No", "Yes"))
    gen_h = st.selectbox("General Health", gen_health_cats)

st.markdown("---") # Visual separator

# --- Prediction Button ---
if st.button("Predict Heart Disease Risk"):
    # 1. Prepare Input for Prediction
    # Collect all user inputs into a dictionary.
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
        'SleepTime': sl_t,
        'Asthma': asthma,
        'KidneyDisease': kid_dis,
        'SkinCancer': skin_can
    }

    # Convert the input dictionary to a pandas DataFrame.
    inp_df = pd.DataFrame([inp_data])

    # 2. Apply the same preprocessing steps as during training
    # Map binary columns ('Yes'/'No', 'Male'/'Female') to 0/1.
    bin_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalk',
                'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex', 'Diabetic']:
        inp_df[col] = inp_df[col].map(bin_map)

    # Create an empty DataFrame with all possible one-hot encoded columns from training.
    # This is crucial to ensure the input DataFrame has the exact same columns and order
    # as the data the model was trained on, even if a specific category isn't present
    # in the current single input row.
    proc_inp = pd.DataFrame(0, index=[0], columns=train_cols)

    # Fill in numerical values from the user input.
    for col in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']:
        if col in proc_inp.columns: # Check if column exists in the training columns
            proc_inp[col] = inp_df[col].values[0]

    # Fill in binary encoded values from the user input.
    for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalk',
                'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex', 'Diabetic']:
        if col in proc_inp.columns:
            proc_inp[col] = inp_df[col].values[0]

    # Fill in one-hot encoded categorical values.
    # For each categorical feature, set the corresponding one-hot encoded column to 1.
    age_col_name = f'AgeCategory_{inp_df["AgeCategory"].values[0]}'
    if age_col_name in proc_inp.columns:
        proc_inp[age_col_name] = 1

    race_col_name = f'Race_{inp_df["Race"].values[0]}'
    if race_col_name in proc_inp.columns:
        proc_inp[race_col_name] = 1

    gen_h_col_name = f'GenHealth_{inp_df["GenHealth"].values[0]}'
    if gen_h_col_name in proc_inp.columns:
        proc_inp[gen_h_col_name] = 1

    # Ensure all columns are numeric after dummy variable creation.
    proc_inp = proc_inp.apply(pd.to_numeric, errors='coerce')
    proc_inp.fillna(0, inplace=True) # Fill any NaNs that might arise from missing dummy columns with 0

    # Ensure the order of columns matches the training data. This is critical for the model.
    proc_inp = proc_inp[train_cols]

    # 3. Make Prediction
    # Use the loaded model to predict the class (0 or 1) and the probabilities.
    pred = mod.predict(proc_inp)[0] # The predicted class (0 or 1)
    pred_proba = mod.predict_proba(proc_inp)[0] # Probabilities for each class [prob_not_at_risk, prob_at_risk]

    # 4. Display Prediction Result
    st.subheader("Prediction Result:")
    if pred == 1:
        st.error("🚨 Patient is **AT RISK** of Heart Disease!")
        conf = pred_proba[1] * 100 # Confidence for 'at risk' class
    else:
        st.success("✅ Patient is **NOT AT RISK** of Heart Disease.")
        conf = pred_proba[0] * 100 # Confidence for 'not at risk' class

    st.write(f"Confidence Score: **{conf:.2f}%**")
    st.markdown("""
        *Note: This is a predictive tool and should not replace professional medical advice.*
    """)

st.markdown("---") # Footer separator
st.markdown("Developed for Dr. Mendoza's Community Health Clinic") # Footer text

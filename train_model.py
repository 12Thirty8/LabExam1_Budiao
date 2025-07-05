import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split # To split data into training and testing sets
from sklearn.linear_model import LogisticRegression # Our chosen machine learning model
import joblib # A library to efficiently save and load Python objects (like our trained model)

print("Starting model training script...")

try:
    # 1. Load the dataset
    # Reads the CSV file into a pandas DataFrame.
    df = pd.read_csv('heart_2020_uncleaned.csv')
    print(f"Original dataset shape: {df.shape}")

    # 2. Data Cleaning and Preparation
    # Handle missing values: We'll drop any rows that have missing data.
    # In a real-world scenario, you might use more sophisticated imputation techniques.
    initial_rows = df.shape[0]
    df.dropna(inplace=True) # Removes rows with NaN values
    print(f"Dropped {initial_rows - df.shape[0]} rows with missing values.")

    # Convert 'HeartDisease' to binary (0 or 1)
    # Our target variable needs to be numerical for the model.
    df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1, 'No': 0})

    # Convert other binary categorical columns ('Yes'/'No') to 0/1
    # This prepares these features for the numerical model.
    bin_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalk',
                   'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
    for col in bin_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Convert 'Sex' to binary (Male=1, Female=0)
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

    # Handle 'Diabetic' column which has more than two categories initially
    # We simplify it to 'Yes' (1) or 'No' (0).
    df['Diabetic'] = df['Diabetic'].replace({
        'No, borderline diabetes': 'No',
        'Yes (during pregnancy)': 'Yes'
    }).map({'Yes': 1, 'No': 0})

    # One-Hot Encode remaining categorical features
    # 'AgeCategory', 'Race', and 'GenHealth' have multiple categories.
    # One-hot encoding converts each category into a new binary (0/1) column.
    # `drop_first=True` avoids multicollinearity.
    cat_cols = ['AgeCategory', 'Race', 'GenHealth']
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Separate features (X) and target (y)
    # X contains all the input variables, y is what we want to predict.
    X = df_enc.drop('HeartDisease', axis=1)
    y = df_enc['HeartDisease']

    # Ensure all columns are numeric
    # This is a safety step to catch any non-numeric data that might have slipped through.
    X = X.apply(pd.to_numeric, errors='coerce')
    X.dropna(inplace=True) # Drop rows where conversion to numeric failed
    y = y[X.index] # Align the target variable 'y' with the cleaned features 'X'
    print(f"Processed dataset shape for training: {X.shape}")

    # 3. Model Training
    # Split data into training and testing sets
    # We use 80% of the data for training and 20% for testing.
    # `stratify=y` ensures that the proportion of 'HeartDisease' (0s and 1s) is the same
    # in both training and testing sets, which is important for classification.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the Logistic Regression model
    # `solver='liblinear'` is a good choice for smaller datasets or when you want L1/L2 regularization.
    # `max_iter=1000` sets the maximum number of iterations for the solver to converge.
    # `random_state=42` ensures reproducibility of the training process.
    mod = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    mod.fit(X_tr, y_tr) # This is where the model learns from the data
    print("Model trained successfully.")

    # Store the columns used for training
    # This list is crucial to ensure that when new patient data is fed to the model in the app,
    # the columns are in the exact same order and format as during training.
    train_cols = X.columns.tolist()

    # 4. Save Model and Training Columns
    # We use joblib to save the trained model and the list of training columns.
    # These files will be loaded by our Streamlit app.
    joblib.dump(mod, 'logistic_regression_model.pkl') # Saves the trained model
    joblib.dump(train_cols, 'training_columns.pkl') # Saves the list of column names
    print("Model and training columns saved as 'logistic_regression_model.pkl' and 'training_columns.pkl'.")

except Exception as e:
    print(f"Error during model training: {e}")

print("Model training script finished.")

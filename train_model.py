import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

print("Starting model training script...")

try:
    df = pd.read_csv('heart_2020_uncleaned.csv')
    print(f"Original dataset shape: {df.shape}")
    df.columns = df.columns.str.strip()

    # NEW: Standardize all string values to title case
    df = df.apply(lambda x: x.str.strip().str.title() if x.dtype == "object" else x)

    # View unique values for debugging
    print("\n🔍 Unique values per column before mapping:")
    for col in df.columns:
        print(f"{col}: {df[col].unique()[:10]}")  # limit preview to first 10 values

    # Convert numeric columns with coercion
    num_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found. Skipping.")

    # Map binary columns safely
    bin_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking',
                'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
    for col in bin_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        else:
            print(f"Warning: Column '{col}' not found. Skipping.")

    # Map Sex
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

    # Fix Diabetic mappings
    df['Diabetic'] = df['Diabetic'].replace({
        'No, borderline diabetes': 'No',
        'Yes (during pregnancy)': 'Yes'
    }).map({'Yes': 1, 'No': 0})

    # Map HeartDisease
    df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1, 'No': 0})

    # Drop rows where mappings resulted in NaNs
    mapped_cols = ['HeartDisease', 'Sex', 'Diabetic'] + bin_cols
    df = df[df[mapped_cols].notna().all(axis=1)]

    # Show missing values before dropping
    print("\n🧹 Missing values before dropna:")
    print(df.isna().sum())

    # Drop remaining NaNs from numeric columns
    init_rows = df.shape[0]
    df.dropna(inplace=True)
    print(f"\nDropped {init_rows - df.shape[0]} rows with missing values (including from conversion).")

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning. No data to train.")

    # One-hot encode categorical columns
    cat_cols = ['AgeCategory', 'Race', 'GenHealth']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Align indices and check for any NaNs
    X.dropna(inplace=True)
    y = y[X.index]

    print(f"\n✅ Final processed shape: {X.shape}")

    if X.empty:
        raise ValueError("Features DataFrame is empty after final processing. No data to train.")

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("🎉 Model training complete.")

    # Save model and feature names
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(X.columns.tolist(), 'training_columns.pkl')
    print("📦 Model and columns saved.")

except Exception as e:
    print(f"\n❌ Error during training: {e}")

print("\nScript finished.")

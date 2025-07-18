# File: src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df['income'] != '?']
    df = df.dropna()
    return df

def preprocess_data(df):
    selected_features = ['age', 'hours-per-week', 'capital-gain', 'capital-loss', 'education', 'income']
    df = df[selected_features]

    # Label encode 'education'
    le = LabelEncoder()
    df['education'] = le.fit_transform(df['education'])

    # Save the encoder for use in Streamlit
    os.makedirs("models", exist_ok=True)
    with open("models/education_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
    return df, le

def split_data(df):
    X = df.drop('income', axis=1)
    y = df['income']
    return train_test_split(X, y, test_size=0.2, random_state=42)

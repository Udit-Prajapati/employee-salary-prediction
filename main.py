from src.data_preprocessing import load_and_clean_data, preprocess_data, split_data
from src.model_training import train_models
from src.evaluate import evaluate_models
import pickle
import os

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Step 1: Load and clean data
df = load_and_clean_data("data/adult3.csv")

# Step 2: Preprocess and encode
df, encoders = preprocess_data(df)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = split_data(df)

# Step 4: Train models
models = train_models(X_train, y_train)

# Step 5: Evaluate models
evaluate_models(models, X_test, y_test)

# Step 6: Save best model (Random Forest here)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(models["Random Forest"], f)

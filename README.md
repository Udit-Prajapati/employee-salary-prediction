# Employee Salary Prediction

This project predicts if an employee earns >50K using a machine learning classification model trained on the Adult dataset.

## 📂 Folder Structure

employee-salary-prediction/
├── data/ → Contains dataset (adult3.csv)
├── notebooks/ → For optional EDA
├── src/ → Python modules for data & model
├── models/ → Trained model (best_model.pkl)
├── app/ → Streamlit frontend
├── outputs/ → Confusion matrix images


## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

python main.py

streamlit run app/streamlit_app.py


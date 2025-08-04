# Employee Salary Prediction 💼

This project implements a machine learning solution to predict whether an employee's salary will exceed $50,000 annually based on various features such as age, education, work hours, and capital gains/losses. The system uses multiple ML models and provides an interactive web interface for real-time predictions.

## 🎯 Features

- Multiple ML Models Comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Interactive Web Interface
- Real-time Predictions
- Comprehensive Model Evaluation
- Data Visualization

## 📂 Project Structure

```
employee-salary-prediction/
├── app/
│   └── streamlit_app.py      # Web interface for predictions
├── data/
│   └── adult3.csv           # Dataset
├── models/
│   ├── best_model.pkl       # Saved trained model
│   └── education_encoder.pkl # Label encoder for education
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data analysis notebook
├── outputs/
│   ├── confusion_matrix_Decision_Tree.png
│   ├── confusion_matrix_Logistic_Regression.png
│   ├── confusion_matrix_Random_Forest.png
│   └── confusion_matrix_SVM.png
├── src/
│   ├── data_preprocessing.py # Data preprocessing functions
│   ├── evaluate.py          # Model evaluation functions
│   └── model_training.py    # Model training scripts
├── main.py                  # Main execution script
└── requirements.txt         # Project dependencies
```

## 🛠️ Technical Stack

- Python 3.x
- Key Libraries:
  - pandas: Data manipulation and analysis
  - scikit-learn: Machine learning models
  - streamlit: Web interface
  - matplotlib & seaborn: Data visualization
  - pickle: Model serialization

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd employee-salary-prediction
   ```

2. Create and activate a virtual environment (Optional but recommended):
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the Project

1. Train the Models:
   ```bash
   python main.py
   ```
   This will:
   - Process the data
   - Train all models
   - Generate evaluation metrics
   - Save the best model

2. Launch the Web Interface:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Access the application at `http://localhost:8501`

## 🎮 Using the Web Interface

The web application allows you to input:
- Age (17-90 years)
- Education Level (select from available categories)
- Hours Worked per Week (1-99 hours)
- Capital Gain ($)
- Capital Loss ($)

Click "Predict" to get the salary prediction (>50K or ≤50K).

## 📊 Model Performance

Current model performance metrics:

1. Logistic Regression:
   - Accuracy: 81%
   - High recall (97%) for ≤50K predictions

2. Decision Tree:
   - Accuracy: 83%
   - Balanced performance across classes

3. Random Forest:
   - Accuracy: 83%
   - Best overall performance
   - Good balance between precision and recall

4. SVM:
   - Accuracy: 81%
   - Similar performance to Logistic Regression

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Adult Income dataset
- Special thanks to IBM for the internship opportunity and guidance


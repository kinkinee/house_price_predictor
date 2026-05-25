House Price Prediction Web Application

A machine learning-based web application for predicting Bengaluru house prices using XGBoost and Streamlit.

Features
- Data cleaning and preprocessing
- Handling missing values
- Outlier removal using IQR
- One-hot encoding for categorical data
- House price prediction using XGBoost regression
- Interactive web interface using Streamlit

Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

Dataset
Bengaluru Housing Dataset from Kaggle.

Model Performance
- R² Score: 73.13%

Run Locally

```bash
pip install -r requirements.txt
py -m streamlit run hpp.py

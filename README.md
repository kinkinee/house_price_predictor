# House Price Prediction Web Application

A machine learning-based web application for predicting Bengaluru house prices using XGBoost and Streamlit.

## Machine Learning Model Used  
### Regression Model
- XGBoost Regressor

## Input Features
- Area (sq.ft.)
- Size (BHK)
- Bathrooms
- Balconies
- Location

## Target Variable
- House Price

## Data Preprocessing
- Removed duplicate records.
- Handled missing values using median imputation.
- Removed outliers using the IQR method.
- Applied One-Hot Encoding to categorical location features.
- Grouped rare locations into an "Other" category.

## Data Visualization
- Created a log-scale scatter plot to analyze the relationship between area and house price.

## Application Features
- Real-time house price prediction
- Interactive web interface using Streamlit

## Use Cases
- Assists home buyers in estimating property prices based on key housing features.
- Helps real estate professionals analyze property valuations based on existing properties.
- Supports analysis of Bengaluru housing market trends and identification of potentially overpriced or undervalued locations.
- Provides a practical demonstration of machine learning-based price prediction.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

## Dataset
Bengaluru Housing Dataset from Kaggle.

**Dataset Link:**  
https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

## Model Performance
- MAE : 20.6366
- MSE : 1273.2268
- RMSE : 35.6823
- R² Score: 73.13%

## Run Locally

```bash
pip install -r requirements.txt
py -m streamlit run hpp.py

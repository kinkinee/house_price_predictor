import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sb
import streamlit as st

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

# PAGE CONFIGURATION
st.set_page_config(page_title="House Price Prediction")
st.title("Price Predictor")

df = pd.read_csv("Bengaluru_House_Data.csv")
df = df.drop_duplicates()

st.subheader('Dataset Preview')
st.write (df.head())
st.write('Dataset Shape', df.shape)

def clean_area(value) :
    if ('-' in value) :
        parts=value.split('-')
        average=(float(parts[0])+ float(parts[1]))/len(parts)
        return (average)

    else:
        try:
            float(value)
            return float(value)
        
        except:
            return (np.nan)
        
df['area'] = df['area'].apply(clean_area)
df = df[df['area'] >= 300]

def clean_size(room) :
    try:
        if str(room) :
            part=room.split()
            return float(part[0])
        else:
            return float(room)
    except:
        return (np.nan)

df['size'] = df['size'].apply(clean_size)
    
df['location'] = df['location'].fillna(value = "Unknown")

df['area'] = df['area'].fillna(df['area'].median())
df['price'] = df['price'].fillna(df['price'].median())
df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df['size'] = df['size'].fillna(df['size'].median())

df=df.drop('society', axis=1)
df['price_per_sqft'] = df['price']/ df['area']

max_threshold=df['price_per_sqft'].quantile(0.99)
q1=df['price_per_sqft'].quantile(0.25)
q2=df['price_per_sqft'].quantile(0.50)
q3=df['price_per_sqft'].quantile(0.75)
min_threshold=df['price_per_sqft'].quantile(0.01)
iqr=q3-q1
st.write("IQR :", iqr)

lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr

st.write ("Lower Limit : ",lower_limit )
st.write ("Upper Limit : ", upper_limit)

no_outlier_region = df[(df['price_per_sqft']>=lower_limit) & (df['price_per_sqft']< upper_limit)]

#PLOT VISUALIZATION
x=np.log(no_outlier_region['area'])
y=np.log(no_outlier_region['price'])
sb.set()
fig, ax = pyplot.subplots()
ax.scatter(x, y)
ax.set_xlabel("Area in sqft. (log scale)")
ax.set_ylabel("Price in Lakhs (log scale)")
ax.set_title("Log(Area in sqft.) vs Log(Price in Lakhs)")
st.pyplot(fig)

# TEXT VECTORIZATION
location_stats = no_outlier_region['location'].value_counts()
rare_locations = location_stats[location_stats <= 50]
no_outlier_region['location'] = no_outlier_region['location'].apply(lambda x: 'Other' if x in rare_locations else x)
dummies=pd.get_dummies(no_outlier_region['location'], drop_first=True)
merged=pd.concat([dummies, no_outlier_region], axis=1)
merged = merged.drop('location', axis=1)

# FEATURES AND LABELS
x = merged.drop(['price', 'price_per_sqft'],axis=1)
y = merged['price']

# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
     
# TRAIN MODEL
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=100)
model.fit(x_train, y_train)

# MODEL EVALUATION
y_pred = model.predict(x_test)

# EVALUATION METRICS
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# DISPLAY RESULTS
st.subheader("Model Evaluation")
st.write(f"MAE : {mae:.4f}")
st.write(f"MSE : {mse:.4f}")
st.write(f"RMSE : {rmse:.4f}")
st.write(f"R2 Score : {r2*100 :.2f}%")


# PREDICT NEW DATA
st.subheader("Predict New Data : ")
size = st.number_input("Size (BHK)")
area = st.number_input("Area (sq.ft)")
bath = st.number_input("Bathrooms")
balcony = st.number_input("Balcony")
location = st.selectbox("Location", sorted(dummies.columns))

if st.button("Predict Price"):

    input_data = pd.DataFrame(np.zeros((1, len(x.columns))), columns=x.columns)

    input_data.loc[0, 'size'] = size
    input_data.loc[0, 'area'] = area
    input_data.loc[0, 'bath'] = bath
    input_data.loc[0, 'balcony'] = balcony

    if location in input_data.columns:
        input_data.loc[0, location] = 1
    prediction = model.predict(input_data)
    final_price = prediction[0]
    st.write("Predicted House Price :", final_price, "Lakhs")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('TSLA.csv')

# Preprocessing steps
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# Feature engineering
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Define features and target
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Ensure features and target are numpy arrays and writable
features = np.array(features, copy=True)
target = np.array(target, copy=True)

# Scaling features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)

# List of models to train
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

# Train models
def train_models(X_train, Y_train):
    for model in models:
        model.fit(X_train, Y_train)
    return models

models = train_models(X_train, Y_train)

# Streamlit UI
st.title("Tesla Stock Price Prediction")

# CSS for background image with custom size
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://th.bing.com/th/id/OIP.cQ_p_bjrZ9_1hLgblK4XWAHaE8?w=1199&h=800&rs=1&pid=ImgDetMain");
    background-size: 750px 750px; /* Set the width and height of the background image */
    background-repeat: no-repeat; /* Prevent the image from repeating */
    background-position: center center; /* Center the image in the container */
    background-attachment: fixed; /* Make the background image fixed */
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Input features
open_close = st.number_input('Open-Close')
low_high = st.number_input('Low-High')
is_quarter_end = st.selectbox('Is Quarter End?', [0, 1])

# Make predictions
input_data = np.array([[open_close, low_high, is_quarter_end]])
input_data_scaled = scaler.transform(input_data)

selected_model = st.selectbox('Choose Model', ['Logistic Regression', 'SVC', 'XGBoost'])
model_index = {'Logistic Regression': 0, 'SVC': 1, 'XGBoost': 2}
model = models[model_index[selected_model]]

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Predicted stock price movement: {"Up" if prediction[0] == 1 else "Down"}')

# Visualizing model performance (optional)
if st.checkbox('Show Model Performance'):
    st.write('Confusion Matrix:')
    Y_pred = model.predict(X_valid)
    cm = metrics.confusion_matrix(Y_valid, Y_pred)
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)

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

df = pd.read_csv('/content/TSLA.csv')
df.head()

df.shape

df.describe()

df.info()

"""Exploratory Data Analysis"""

plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

df.head()

df[df['Close'] == df['Adj Close']].shape

df = df.drop(['Adj Close'], axis=1)

df.isnull().sum()

# Define the features to visualize
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Plot distribution of each feature
plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)  # Indent this line properly
    sb.histplot(df[col], kde=True)  # Indent this line properly
    plt.title(f'Distribution of {col}')  # Indent this line properly

plt.show()

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract year, month, and day into separate columns
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Display the first few rows to verify the result
df.head()

df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()

# Group the data by year and calculate the mean
data_grouped = df.groupby('year').mean()

# Create subplots for the 'Open', 'High', 'Low', 'Close' columns
plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i+1)  # Create a 2x2 grid of subplots
    data_grouped[col].plot.bar()  # Plot each column as a bar chart
    plt.title(f'Mean {col} per Year')  # Add a title to each subplot

plt.tight_layout()  # Adjust the layout to avoid overlap
plt.show()  # Show the plots after the loop

df.groupby('is_quarter_end').mean()

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

plt.pie(df['target'].value_counts().values,
		labels=[0, 1], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

# List of models to train
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

# Train and evaluate each model
for i, model in enumerate(models):
    model.fit(X_train, Y_train)  # Fit the model

    print(f'{model.__class__.__name__} :')  # Print the model's class name
    print('Training Accuracy : ', metrics.roc_auc_score(
        Y_train, model.predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        Y_valid, model.predict_proba(X_valid)[:,1]))
    print()

from sklearn.metrics import confusion_matrix
import seaborn as sb

# Predict the labels for the validation set
Y_pred = models[0].predict(X_valid)

# Compute the confusion matrix
cm = confusion_matrix(Y_valid, Y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8,6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Class 0', 'Class 1'],
           yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


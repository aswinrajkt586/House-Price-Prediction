# -*- coding: utf-8 -*-
"""House Price Prediction

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_w7dME5weTnSb90ix7mIlYW9FvvvmC_X

# House Price Prediction using Linear Regression
# Project Overview
This project implements a linear regression model to predict house prices based on various features of the properties. The dataset used for this project contains information about different houses, including attributes such as the number of bedrooms, bathrooms, square footage, and more. The goal is to develop a predictive model that can estimate house prices accurately.

# 1. Import Necessary Libraries
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score , mean_squared_error, r2_score

"""# 2. Load and Explore the Dataset
Let’s assume you are using a dataset with various house attributes and their corresponding prices.
"""

data = pd.read_csv('/content/drive/MyDrive/Datasets/data.csv')
data

"""# 3. Data Preprocessing
Clean the dataset (handle missing values, remove outliers, etc.). For instance, if any columns have missing values, you can either fill them or drop those rows.
"""

data.isna().sum()

"""data.types - To check the datatypes of each field"""

data.dtypes

"""data.info() -  To check the information of the data"""

data.info()

from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()

for i in data:
  data[i] = labelencoder.fit_transform(data[i])

data.head()

"""# 4. Visualizing the Data

 In this step, we visualize the data to get an initial understanding of how the features are distributed
"""

df = data[['bedrooms', 'bathrooms', 'sqft_lot', 'price']]
sns.pairplot(df)
plt.show()

"""# 5. Feature Selection
Choose relevant features (independent variables) to predict house prices (target variable).
"""

x = data.drop('price',axis=1)
y = data['price']

"""# 6. Train-Test Split
Split the data into training and testing sets to evaluate the model's performance.
"""

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 1, test_size =.2)

print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

"""# 7. Train the Linear Regression Model
Fit the model on the training data.
"""

model = LinearRegression()
model.fit(x_train,y_train)

"""# 8. Make Predictions
Predict house prices on the test data.
"""

y_pred = model.predict(x_test)

y_pred

"""# 9. Evaluate the Model
Evaluate the performance of the model using metrics like Mean Squared Error (MSE) and R² score.
"""

model.score(x_test,y_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2_score = model.score(x_test, y_test)
print(f'R² Score: {r2_score}')

"""# 10. Visualize the Results
You can plot the predicted vs actual house prices to visualize the model's performance.
"""

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='b')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Predicted vs Actual House Prices')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test - y_pred, alpha=0.5, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Actual House Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.show()
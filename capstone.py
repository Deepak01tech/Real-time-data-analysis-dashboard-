import os
import warnings

import matplotlib
import numpy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import datasets

data = pd.read_csv("Store Data.csv")


data.head()


data.tail()


data.columns


data.isnull().sum()


numeric_variables = ['Age', 'Amount', 'Qty']

summary_stats_numeric = data[numeric_variables].describe()
print(summary_stats_numeric)

import matplotlib.pyplot as plt

for var in numeric_variables:

    plt.figure(figsize=(8, 6))
    sns.histplot(data[var], kde=True)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {var}')
    plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x=var)
plt.xlabel(var)
plt.title(f'Box Plot of {var}')
plt.show()

categorical_variables = ['Gender', 'Age Group', 'Status', 'Channel ', 'Category']


for var in categorical_variables:

    plt.figure(figsize=(8, 6))
    data[var].value_counts().plot(kind='bar')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Frequency Distribution of {var}')
    plt.xticks(rotation=45)
    plt.show()

plt.figure(figsize=(8, 6))
data[var].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title(f'Percentage Distribution of {var}')
plt.show()
numeric_variables = ['Age', 'Amount', 'Qty']
for i in range(len(numeric_variables)):
    for j in range(i+1, len(numeric_variables)):
        var1 = numeric_variables[i]
        var2 = numeric_variables[j]
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x=var1, y=var2)
plt.xlabel(var1)
plt.ylabel(var2)
plt.title(f'Scatter plot of {var1} vs {var2}')
plt.show()
correlation_coefficient = data[var1].corr(data[var2])
print(f"Correlation coefficient between {var1} and {var2}: {correlation_coefficient}\n")
for cat_var in categorical_variables:

    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=cat_var, hue='Age Group')
    plt.xlabel(cat_var)
    plt.ylabel('Count')
    plt.title(f'Grouped bar chart of {cat_var} by Age Group')
    plt.xticks(rotation=45)
    plt.legend(title='Age Group')
    plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

data_numeric = data.select_dtypes(include='number')

plt.figure(figsize=(10, 8))
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

X = data[['Age', 'Amount']]
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Age', y='Amount', hue='Cluster', palette='Set1', legend='full')
plt.title('K-means Clustering')
plt.xlabel('Age')
plt.ylabel('Amount')
plt.show()

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[['Age', 'Amount']])
data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', legend='full')
plt.title('PCA Visualization')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
plt.figure(figsize=(10, 6))
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['Amount'].plot()
plt.title('Time-series Plot of Sales Amount')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()

decomposition = seasonal_decompose(data['Amount'], model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(data['Amount'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
model = ARIMA(data['Amount'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)

plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Amount'], label='Actual')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title('Time-series Forecasting with ARIMA')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()


# %% [markdown]
# # Real-Time Data Analysis Dashboard
# 

# %% [markdown]
# ### A real-time data analysis dashboard is a graphical interface that provides users with up-to-date information and insights derived from analyzing streaming or constantly updating data. These dashboards typically display key metrics, trends, and visualizations that help users monitor, understand, and act on the data in real-time.
# 

# %% [markdown]
# #### Why Exploratory data analysis for this?

# %% [markdown]
# ##### Performing Exploratory Data Analysis (EDA) on your dataset, even if it's in CSV format for a real-time dashboard, can provide valuable insights and help you better understand the characteristics of your data. While real-time dashboards typically focus on monitoring ongoing trends and patterns, conducting EDA can still be beneficial for several reasons:

# %% [markdown]
# ##### 1.Data Quality Assessment: EDA allows you to check the quality and integrity of your data. You can identify missing values, outliers, inconsistencies, or errors that may affect the reliability of your real-time dashboard.

# %% [markdown]
# ##### 2.Understanding Data Distribution: By visualizing the distribution of your data through histograms, box plots, or density plots, you can gain insights into the range, central tendency, and variability of your variables. This understanding can inform your choice of visualization techniques and help you interpret the real-time trends more effectively.

# %% [markdown]
# ##### 3.Identifying Patterns and Relationships: Exploring relationships between different variables using scatter plots, correlation matrices, or heatmaps can reveal patterns and dependencies in your data. Understanding these relationships can guide the selection of relevant metrics and KPIs to monitor in your real-time dashboard.

# %% [markdown]
# ##### Feature Engineering: EDA can inspire feature engineering ideas by highlighting potential transformations, interactions, or combinations of variables that may improve the predictive power or interpretability of your dashboard metrics.
# 
# 

# %% [markdown]
# ##### 5.User Requirements Analysis: EDA can also inform your understanding of user requirements and preferences. By visualizing different aspects of the data, you can identify key insights and trends that are relevant to your users' needs and interests.

# %% [markdown]
# ###### Step 1: Importing Libraries

# %%
import numpy

# %%
import matplotlib

# %%
import pandas as pd

# %%
#Datasets is a lightweight library providing two main features: one-line dataloaders for many public datasets: one-liners to download and pre-process any of the. major public datasets (image datasets, audio datasets, text datasets in 467 languages and dialects, etc.)
from sklearn import datasets
from matplotlib import pyplot

# %%
from pandas.plotting import scatter_matrix

# %%
import warnings

# %%
import seaborn as sns
import os


# %% [markdown]
# ##### Step 2: Reading Dataset

# %%
data = pd.read_csv("Store Data.csv")

# %%
data.head()

# %%
data.tail()

# %%
data.columns

# %%
data.isnull().sum()

# %% [markdown]
# ##### 1. Based on Unserstanding of the data, what kind of business is this comapny in?

# %% [markdown]
# ###### Answer: In-store data presents information about retail businesses' in-store activities and metrics such as footfall traffic (people counting), customer behavior, sales data, customer buying patterns, and product stocks.

# %%
# Univariate Analysis for Numeric Variables
numeric_variables = ['Age', 'Amount', 'Qty']

# %%
# Summary statistics
summary_stats_numeric = data[numeric_variables].describe()
print(summary_stats_numeric)

# %%
# Visualizations for numeric variables
import matplotlib.pyplot as plt
for var in numeric_variables:
    # Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data[var], kde=True)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {var}')
    plt.show()

# %% [markdown]
# ##### Insights:-

# %% [markdown]
# ###### Usually people spend 500 - 1000 for shopping and most of the people purchase only single items.

# %%
# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x=var)
plt.xlabel(var)
plt.title(f'Box Plot of {var}')
plt.show()

# %%
# Univariate Analysis for Categorical Variables
categorical_variables = ['Gender', 'Age Group', 'Status', 'Channel ', 'Category']

# Visualizations for categorical variables
for var in categorical_variables:
    # Frequency distribution (bar chart)
    plt.figure(figsize=(8, 6))
    data[var].value_counts().plot(kind='bar')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Frequency Distribution of {var}')
    plt.xticks(rotation=45)
    plt.show()

# %% [markdown]
# ##### Insights:

# %% [markdown]
# ##### Mostly women do shopping, seems men are half in quantity of women

# %% [markdown]
# ##### From Adult, Teenager and senior, mostly Adult prefer shopping so Markets tarket should be Adult.

# %% [markdown]
# ##### Top e- commerce business are Amazon then Myntra and then Flipkart

# %%
# Frequency distribution (pie chart) 
plt.figure(figsize=(8, 6))
data[var].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title(f'Percentage Distribution of {var}')
plt.show()

# %% [markdown]
# ###### Insights

# %% [markdown]
# ###### Mostly preferabble dresses are set whose percent are 39.9% after that kurta with 33.6%.

# %%
# List of numeric variables
numeric_variables = ['Age', 'Amount', 'Qty']

# Bivariate Analysis for Numeric Variables
# Scatter plots and Correlation Coefficients
for i in range(len(numeric_variables)):
    for j in range(i+1, len(numeric_variables)):
        var1 = numeric_variables[i]
        var2 = numeric_variables[j]
        

# %%
# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x=var1, y=var2)
plt.xlabel(var1)
plt.ylabel(var2)
plt.title(f'Scatter plot of {var1} vs {var2}')
plt.show()

# %%
# Correlation coefficient
correlation_coefficient = data[var1].corr(data[var2])
print(f"Correlation coefficient between {var1} and {var2}: {correlation_coefficient}\n")


# %%
# Bivariate Analysis for Categorical Variables
for cat_var in categorical_variables:
    # Grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=cat_var, hue='Age Group')
    plt.xlabel(cat_var)
    plt.ylabel('Count')
    plt.title(f'Grouped bar chart of {cat_var} by Age Group')
    plt.xticks(rotation=45)
    plt.legend(title='Age Group')
    plt.show()

# %% [markdown]
# ###### From above plot we can usually women perform shopping and most preferabble e-commerce site are Amazon and myntra.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is your DataFrame
# If you have non-numeric columns, you might want to drop them or handle them appropriately

# Drop non-numeric columns if necessary
data_numeric = data.select_dtypes(include='number')

plt.figure(figsize=(10, 8))
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


# %%
# Cluster analysis (K-means clustering)
X = data[['Age', 'Amount']]
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Age', y='Amount', hue='Cluster', palette='Set1', legend='full')
plt.title('K-means Clustering')
plt.xlabel('Age')
plt.ylabel('Amount')
plt.show()

# %% [markdown]
# ###### Here in above code:-X = data[['Age', 'Amount']] This line selects the 'Age' and 'Amount' columns from the DataFrame data and assigns them to the variable X. These columns will be used as features for clustering.

# %%
# Dimensionality reduction (PCA)
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

# %% [markdown]
# ###### Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving most of the variance in the original data. In simpler terms, PCA identifies patterns in data and represents it in a more compact form.

# %% [markdown]
# ###### Dimensionality Reduction: The original dataset may contain multiple features (columns), which can make it difficult to visualize and analyze effectively, especially when dealing with high-dimensional data. PCA reduces the number of features while retaining as much of the variability in the data as possible. In this case, PCA is applied to the 'Age' and 'Amount' features to transform them into two principal components (PCA1 and PCA2), which capture the most significant variability in the data.

# %% [markdown]
# ###### Visualization: PCA allows for the visualization of high-dimensional data in a lower-dimensional space, making it easier to interpret and analyze. By reducing the dimensionality of the data to two dimensions, it becomes possible to create scatter plots or other visualizations that help to understand the relationships and patterns in the data. In the provided code, the transformed data is plotted in a two-dimensional scatter plot, where each point represents a data point, and its position is determined by the values of the principal components (PCA1 and PCA2). Additionally, the points are colored according to the cluster they belong to, providing insights into the clustering structure of the data.

# %% [markdown]
# ###### Clustering Analysis: PCA can be used as a preprocessing step for clustering analysis. By reducing the dimensionality of the data, PCA can improve the performance and interpretability of clustering algorithms. In this case, the PCA-transformed data is visualized with cluster assignments, allowing for a better understanding of the clusters formed by the K-means clustering algorithm

# %%
# Temporal Analysis
# Time-series plot
plt.figure(figsize=(10, 6))
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['Amount'].plot()
plt.title('Time-series Plot of Sales Amount')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()

# %% [markdown]
# ###### Temporal Analysis:
# 
# 

# %% [markdown]
# ###### Analyze temporal patterns using time-series plots to visualize trends over time.
# Decompose time-series data into trend, seasonality, and residual components using techniques like seasonal decomposition.
# Use time-series forecasting models (e.g., ARIMA, exponential smoothing) to predict future values based on historical data.

# %%
# Seasonal decomposition
decomposition = seasonal_decompose(data['Amount'], model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# %% [markdown]
# Seasonal Decomposition: Seasonal decomposition is a technique used to decompose a time series into its constituent components: trend, seasonality, and residual. These components help us understand the underlying patterns and variations in the time series data.
# 
# seasonal_decompose(data['Amount'], model='additive', period=12): This line applies seasonal decomposition to the 'Amount' column of the DataFrame data. The model='additive' argument indicates that the decomposition model assumes that the time series is an additive combination of its components. The period=12 argument specifies the length of the seasonal cycle, assuming that the data has a seasonal pattern repeating every 12 time periods (e.g., months, quarters).
# 
# decomposition.trend: This line extracts the trend component from the seasonal decomposition. The trend represents the underlying long-term pattern or directionality of the time series, capturing gradual changes over time.
# 
# decomposition.seasonal: This line extracts the seasonal component from the seasonal decomposition. The seasonal component represents the periodic fluctuations or seasonal patterns in the data, such as monthly or quarterly variations.
# 
# decomposition.resid: This line extracts the residual component from the seasonal decomposition. The residual component represents the random or unexplained variation in the data after removing the trend and seasonal components.
# 
# By decomposing the time series into these components, seasonal decomposition helps in understanding the various factors contributing to the overall behavior of the data. It can be useful for identifying seasonal patterns, detecting anomalies, and forecasting future values based on historical data.

# %%
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

# %% [markdown]
# ###### The resulting visualization provides a clear view of the original time series data along with its decomposed components: trend, seasonality, and residuals. This breakdown helps in understanding the individual contributions of each component to the overall behavior of the time series. It can aid in identifying patterns, detecting anomalies, and making informed decisions in time series analysis and forecasting.

# %%

# Time-series forecasting (ARIMA)
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

# %% [markdown]
# ###### the overall statement executes the entire process of initializing, fitting, and using the ARIMA model to generate forecasts for a time series dataset. It then visualizes these forecasts alongside the actual data, providing insights into future trends and patterns.

# %% [markdown]
# The ARIMA (AutoRegressive Integrated Moving Average) model is a popular and widely used time series forecasting technique. It's particularly effective for modeling univariate time series data that exhibits a trend or seasonality. Here's a breakdown of how the ARIMA model works and how it's used:
# 
# AutoRegressive (AR) Component:
# 
# The ARIMA model includes an autoregressive component, denoted by the parameter p, which captures the relationship between an observation and a fixed number of lagged observations (previous time steps).
# This component models the linear dependence between an observation at time 't' and its 'p' most recent observations.
# Integrated (I) Component:
# 
# The ARIMA model includes an integrated component, denoted by the parameter d, which represents the degree of differencing applied to the time series data.
# Differencing involves subtracting the current observation from the previous observation to make the data stationary (i.e., remove trend or seasonality).
# The integrated component transforms the original time series into a stationary series, making it suitable for modeling.
# Moving Average (MA) Component:
# 
# The ARIMA model also includes a moving average component, denoted by the parameter q, which captures the relationship between an observation and a residual error from a moving average model applied to lagged observations.
# This component models the dependency between an observation and a residual error from a moving average model.
# Model Identification:
# 
# The parameters p, d, and q of the ARIMA model need to be determined or selected based on the characteristics of the time series data.
# This process often involves visual inspection of the time series plot, autocorrelation function (ACF) plot, and partial autocorrelation function (PACF) plot to identify potential values for p, d, and q.
# Model Fitting:
# 
# Once the parameters are determined, the ARIMA model is fitted to the time series data using historical observations.
# The model estimation involves finding the optimal coefficients for the autoregressive, differencing, and moving average components that minimize the error between the observed and predicted values.
# Forecasting:
# 
# After fitting the ARIMA model, it can be used to generate forecasts for future time steps.
# These forecasts provide estimates of future values based on the historical patterns and relationships captured by the model.
# Model Evaluation:
# 
# The performance of the ARIMA model can be evaluated using various metrics such as mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), etc.
# These metrics quantify the accuracy of the model's forecasts relative to the actual observed values.
# Overall, the ARIMA model is a powerful tool for time series forecasting, particularly when dealing with data that exhibits trends or seasonality. It allows analysts and data scientists to make informed predictions about future values based on historical observations.

# %% [markdown]
# ### Future-Proceeding

# %% [markdown]
# 
# Select a dashboarding tool: There are many dashboarding tools available that can help you create real-time dashboards with ease. Some popular options include Tableau, Power BI, Google Data Studio, and Grafana. Choose a tool that best fits your requirements in terms of features, ease of use, and compatibility with your data sources.
# 
# Connect your data source: Once you have chosen a dashboarding tool, connect it to your data source(s). This typically involves setting up data connections or integrations provided by the dashboarding tool.
# 
# Design your dashboard: Design the layout of your dashboard and decide what visualizations you want to include. Common types of visualizations include charts, graphs, tables, maps, and gauges. Make sure the layout is intuitive and easy to understand for your target audience.
# 
# Implement real-time updates: Configure your dashboard to update automatically as new data comes in. Most dashboarding tools support real-time data updates through features like live data connections or scheduled data refreshes.
# 
# Add interactivity: Enhance the usability of your dashboard by adding interactive elements such as filters, drill-down capabilities, and dynamic parameters. This allows users to explore the data and gain deeper insights.
# 
# Test and iterate: Test your dashboard thoroughly to ensure that it works as expected and provides accurate insights. Gather feedback from users and iterate on the design based on their input.
# 
# Deploy your dashboard: Once you're satisfied with your dashboard, deploy it to your intended audience. This could be through a web server, intranet, or embedding it within other applications.
# 
# Monitor and maintain: Monitor the performance of your dashboard over time and make any necessary updates or optimizations. Regularly review the data sources, visualizations, and user feedback to ensure that the dashboard remains relevant and effective.

# %%


# %%


# %%




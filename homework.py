# import lots of things
import pandas as pd
import pandas_datareader.data as web
from pandas import Series, DataFrame
import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style

import math
import numpy as np

start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2019, 9, 7)

dataset = web.DataReader("NFLX", 'yahoo', start, end)

datasetfreg = dataset.loc[:, ['Adj Close', 'Volume']]
datasetfreg['HL_PCT'] = (dataset['High'] - dataset['Low']
                         ) / dataset['Close'] * 100.0
datasetfreg['PCT_change'] = (
    dataset['Close'] - dataset['Open']) / dataset['Open'] * 100.0

# Drop missing value
datasetfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.02 * len(datasetfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
datasetfreg['label'] = datasetfreg[forecast_col].shift(-forecast_out)
X = np.array(datasetfreg.drop(['label'], 1))


# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(datasetfreg['label'])
y = y[:-forecast_out]

print(datasetfreg.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=0)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test, y_test)
confidencepoly3 = clfpoly3.score(X_test, y_test)
confidenceknn = clfknn.score(X_test, y_test)

print('The linear regression confidence is ', confidencereg)
print('The quadratic regression 2 confidence is', confidencepoly2)
print('The quadratic regression 3 confidence is', confidencepoly3)
print('The knn regression confidence is', confidenceknn)

forecast_set = clfpoly2.predict(X_lately)
regression_set = clfpoly2.predict(X)
datasetfreg['Forecast'] = np.nan


mpl.rc('figure', figsize=(20, 10))
mpl.__version__

last_date = datasetfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)


for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    datasetfreg.loc[next_date] = [
        np.nan for _ in range(len(datasetfreg.columns)-1)]+[i]

plt.plot(datasetfreg['Adj Close'].tail(400))
plt.plot(datasetfreg['Forecast'].tail(400))


plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

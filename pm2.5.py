import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sys

from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Sequential
from keras.layers import Dropout

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

print("Choose the number corresponding to the location site and press enter:")
print("1. Aotizhongxin\n2. Changping\n3. Dingling\n4. Dongsi\n5. Guanyuan\n6. Gucheng\n7. Huairou\n8. Nongzhanguan\n9. Shunyi\n10. Tiantan\n11. Wanliu\n12. Wanshouxigong\n")


location = input()

if int(location)==1:
    data_original = pd.read_csv("PRSA_Data_Aotizhongxin_20130301-20170228.csv")
if int(location)==2:
    data_original = pd.read_csv("PRSA_Data_Changping_20130301-20170228.csv")
if int(location)==3:
    data_original = pd.read_csv("PRSA_Data_Dingling_20130301-20170228.csv")
if int(location)==4:
    data_original = pd.read_csv("PRSA_Data_Dongsi_20130301-20170228.csv")
if int(location)==5:
    data_original = pd.read_csv("PRSA_Data_Guanyuan_20130301-20170228.csv")
if int(location)==6:
    data_original = pd.read_csv("PRSA_Data_Gucheng_20130301-20170228.csv")
if int(location)==7:
    data_original = pd.read_csv("PRSA_Data_Huairou_20130301-20170228.csv")
if int(location)==8:
    data_original = pd.read_csv("PRSA_Data_Nongzhanguan_20130301-20170228.csv")
if int(location)==9:
    data_original = pd.read_csv("PRSA_Data_Shunyi_20130301-20170228.csv")
if int(location)==10:
    data_original = pd.read_csv("PRSA_Data_Tiantan_20130301-20170228.csv")
if int(location)==11:
    data_original = pd.read_csv("PRSA_Data_Wanliu_20130301-20170228.csv")
if int(location)==12:
    data_original = pd.read_csv("PRSA_Data_Wanshouxigong_20130301-20170228.csv")

print(data_original.head())
print(data_original.info())
print(data_original.describe(include='all'))

# checking the categorical attribute "wd"
print("Unique values of wd: ", data_original['wd'].unique())
print("No. of unique values of wd: ", data_original['wd'].nunique())

# Creating a dataframe for time
time = pd.DataFrame([data_original["year"],data_original["month"],data_original["day"],data_original["hour"]])
time = time.T

# Drawing a time series graph of data
x_time = pd.to_datetime(time, format='%Y/%m/%d%H')
y_total = data_original["PM2.5"]  
plt.figure(figsize=(500,20))  
plt.plot_date(x_time, y_total, linestyle='-')
plt.xlabel('Time')
plt.ylabel('PM2.5')
plt.title('Time Series Graph of Original Data')
plt.show()

# Dropping the columns which are not required
data_original = data_original.drop(['No','year','month','day','hour','station'], axis=1)

# On hot encoding using pd.get_dummies() for wind direction
data_original = pd.get_dummies(data_original, columns = ["wd"])

# Checking null values again
print(data_original.isnull().sum())
# #Interpolation 
# df['PM2.5'].interpolate(inplace=True)   
# df['PM10'].interpolate(inplace=True)
# df['SO2'].interpolate(inplace=True)
# df['NO2'].interpolate(inplace=True)
# df['CO'].interpolate(inplace=True)
# df['O3'].interpolate(inplace=True)
# df['TEMP'].interpolate(inplace=True)
# df['PRES'].interpolate(inplace=True)
# df['DEWP'].interpolate(inplace=True)
# df['RAIN'].interpolate(inplace=True)
#### correlation check after interpolation 
# corr = df.corr()
# fig, ax = plt.subplots(figsize=(20,20))   # Sample figsize in inches
# sns.set(font_scale = 1.60)
# sns.heatmap(corr, cmap="Blues", annot=True, linewidths=.5, ax=ax) 

# Putting 0 as special values for NaN (missing) data
data_original = data_original.fillna(0)

# Checking data description again
pd.set_option('display.max_columns', None)
print(data_original.describe())

# Plotting the historgam of attributes to check for outliers
data_original.hist(bins=50, figsize=(20, 15))
plt.show()

# Plotting the correlation matrix of attributes
plt.figure(figsize=(15,15))
sns.heatmap(data_original.corr(), annot=True, cmap="coolwarm").set_title('Heat Map of correlation matrix')
plt.show()

# ========== ARIMAX model ========== 
print('========================================')
print('ARIMAX model')
# scaling the data due to difference in range of values of attributes
data_arimax = data_original.copy()
scaler = preprocessing.StandardScaler().fit(data_arimax)
data_arimax = pd.DataFrame(scaler.transform(data_arimax))
print(data_arimax.describe())

# Drawing a time series graph to start with
plt.figure(figsize=(100,20))  
plt.plot_date(x_time, data_arimax.iloc[:,0], linestyle='-')
plt.show()

# Checking for trend and seasonality
decompose = seasonal_decompose(data_arimax.iloc[:,0], model='additive', period=int(365*24/2))
fig = plt.figure()  
fig = decompose.plot()  
fig.set_size_inches(15, 15)

# removing seasonality
for i in range(27):
   tmp = seasonal_decompose(data_arimax.iloc[:,i], model='additive', period=int(365*24/2))
   data_arimax.iloc[:,i] = data_arimax.iloc[:,i] - tmp.seasonal

# Checking for trend and seasonality again
decompose = seasonal_decompose(data_arimax.iloc[:,0], model='additive', period=int(365*24/2))
fig = plt.figure()  
fig = decompose.plot()  
fig.set_size_inches(15, 15)

# finding the optimal lag
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(data_arimax.iloc[:,0],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(data_arimax.iloc[:,0],lags=40,ax=ax2)

lag = 1

# checking whether data is stationary or not
for _,i in data_arimax.iteritems():
  result = adfuller(i, autolag='AIC',maxlag=1)
  print(result)

# data is stationary, no differencing required

y_arimax = data_arimax.iloc[:,0]
x_arimax = data_arimax.iloc[:,1:]

# Dividing the data into train and test
split_train_test = int(len(data_arimax) * 0.8)
x_train, y_train = x_arimax[:split_train_test],y_arimax[:split_train_test]
x_test, y_test =  x_arimax[split_train_test-1:],y_arimax[split_train_test-1:]

# Creating and training an ARIMAX model
model_arimax=ARIMA(endog=y_train,exog=x_train,order=[1,0,1])
model_arimax=model_arimax.fit()

print(model_arimax.summary())

# from z-test, we remove attributes with coeff having p-value > 0.05.
# These attributes do not impact the target variable PM2.5
x_arimax = x_arimax.drop(x_arimax.columns[[6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]], axis=1)

# Dividing the data into train and test again
split_train_test = int(len(data_arimax) * 0.8)
x_train = x_arimax[:split_train_test]
x_test =  x_arimax[split_train_test-1:]

# Creating and training an ARIMAX model
model_arimax=ARIMA(endog=y_train,exog=x_train,order=[1,0,1])
model_arimax=model_arimax.fit()
print(model_arimax.summary())

# predicting values
predict_train = model_arimax.predict(exog=x_train, dynamic=True)
predict_test = model_arimax.predict(start=len(predict_train)-1,end=len(predict_train)+len(x_test)-1,exog=x_test, dynamic=True)

# plot the result
plt.figure(figsize=(500, 20))
plt.plot(x_time[:], y_arimax, color="b", label="actual value")
plt.plot(x_time[lag:split_train_test], predict_train, color="r", linestyle="dashed", label="train data prediction")
plt.plot(x_time[split_train_test-1-lag:], predict_test, color="g", linestyle="dashed", label="test data prediction")
plt.legend()
plt.show()

# precision check (error check)
print(mean_squared_error(predict_test, y_arimax.iloc[split_train_test-1-lag:,]))

# ========== LSTM with RNN model ========== 
print('========================================')
print('LSTM with RNN')
# scaling the data for RNN
data_rnn = data_original.copy()

scaler = preprocessing.StandardScaler().fit(data_rnn)
data_rnn = pd.DataFrame(scaler.transform(data_rnn))
print(data_rnn)

# the optimum length of lag found earlier
lag = 1

# defining a function to create a dataset for RNN
def make_dataset(data, duration):
    X = []
    y = []

    for i in range(len(data) - duration):
        X.append(data[i : i + duration])
        y.append(data[i + duration])
    X = np.array(X).reshape(len(X), duration, 1)
    y = np.array(y).reshape(len(y), 1)

    return X, y

# transfering the format of the dataset by using the funciton 
x = -1
y = -1
isFirst = True

# the first row of the dataset is the target, so it is stored as y value
for i in range(27):
  xx, yy = make_dataset(data_rnn.iloc[:,i], lag)
  if isFirst:
    x = xx
    y = yy
    isFirst = False
  else:
    x = np.concatenate([x,xx], axis=2)

# total size of the data
total_size = len(data_rnn) - lag
# defining the train size
train_size = int(total_size*0.8)

# splitting the dataset into train, validation and test
x_train, y_train = x[:train_size],y[:train_size]
x_test, y_test = x[train_size-1:],y[train_size-1:]

test_size = int((len(x_test)-lag)*0.5)

x_valid, y_valid = x_test[:test_size], y_test[:test_size]
x_test, y_test = x_test[test_size-1:], y_test[test_size-1:]

#checking the size of train, validation and test data
print("Size of y_train: ", y_train.shape)
print("Size of x_train: ", x_train.shape)
print("Size of y_valid: ", y_valid.shape)
print("Size of x_valid: ", x_valid.shape)
print("Size of y_test: ", y_test.shape)
print("Size of x_test: ", x_test.shape)

# creating LSTM with RNN model and training it
model_rnn = Sequential()
model_rnn.add(LSTM(16,dropout=0.2, return_sequences=True, input_shape=((np.array(x_train).shape[1], np.array(x_train).shape[2]))))
#model_rnn.add(LSTM(16,dropout=0.2, return_sequences=True))
model_rnn.add(LSTM(16,dropout=0.2))
model_rnn.add(Dense(1))

model_rnn.summary()

model_rnn.compile(loss='mean_squared_error', optimizer='adam')
history = model_rnn.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1, validation_data=(x_valid,y_valid), shuffle=False)

# #LSTM - Trail
# model_rnn = Sequential() 
# model_rnn.add(LSTM(units = 320, activation = 'tanh', return_sequences = True, input_shape = ((np.array(x_train).shape[1], np.array(x_train).shape[2]))))
# model_rnn.add(Dropout(0.2)) 
# model_rnn.add(LSTM(units = 240, activation = 'tanh'))
# model_rnn.add(Dropout(0.2)) 
# model_rnn.add(Dense(units = 3))
# print(model_rnn.summary()) 
# model_rnn.compile(optimizer = 'adam', loss = 'mean_squared_error',)
# history = model_rnn.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1, validation_data=(x_valid,y_valid), shuffle=False)

# making predictions
used_predictions = model_rnn.predict(np.concatenate([x_train,x_valid]))
test_predictions = model_rnn.predict(x_test)

plt.figure(figsize=(500, 20))
plt.plot(x_time[:], data_rnn.iloc[:,0], color="b", label="actual value")
plt.plot(x_time[lag:lag + train_size+test_size], used_predictions, color="r", linestyle="dashed", label="train & valid data prediction")
plt.plot(x_time[lag + train_size+test_size-2:], test_predictions, color="g", linestyle="dashed", label="test data prediction")
plt.legend()
plt.show()

# checking the prediction score
scores = model_rnn.evaluate(x_test, y_test)
print(scores)



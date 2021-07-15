
from pandas import read_csv, concat
from matplotlib import pyplot

import matplotlib.pyplot as plt

# lstm autoencoder to recreate a timeseries
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


# Load data using read_csv
# write the path and column names of the CSV file
df = read_csv('data/dataset.csv',encoding="utf-8")
# df = read_csv('data/data2.csv',encoding="utf-8")

print(df.head())

# total power plots
pyplot.figure(figsize=(18,5))
pyplot.plot(df["total power"])
pyplot.title("the dataset (total power)")

# C electricity plots
pyplot.figure(figsize=(18,5))
pyplot.plot(df["C electricity"],"g")
pyplot.title("the dataset (C electricity)")

pyplot.show()

# printing Stats
print("\nThe table below shows the summary caracteristics of the dataset :\n")
print(df.describe(),"\n\n")

'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''
def temporalize(X, lookback):
    output_X = []
    for i in range(len(X)-lookback+1):
        t = []
        for j in range(0,lookback):
            # Gather past records upto the lookback period
            t.append(X[[(i+j)], :])
        output_X.append(t)
    return output_X

power_data = df["total power"]
electricitye_data = df["C electricity"]

# define input timeseries
timeseries = np.array([power_data,electricitye_data]).transpose()
print(timeseries.shape)
timeseries[:10]

# timesteps = 100
timesteps = 1

n_features = 2
# X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)
X= temporalize(X = timeseries, lookback = timesteps)

X = np.array(X)
X = X.reshape(X.shape[0], timesteps, n_features)

print(X.shape)

# define model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(RepeatVector(timesteps))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='rmsprop', loss='mae', metrics=["accuracy"])
model.summary()

# fit model
history = model.fit(X, X, epochs=20, batch_size=5, verbose=1, validation_split=0.2)
# demonstrate reconstruction
yhat = model.predict(X, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()

a=len(X)//timesteps
X1 = X[[i*timesteps for i in range(a)],:,0]
X1=X1.reshape(X1.shape[0]*X1.shape[1])
X1=list(X1)
yhat1 = yhat[[i*timesteps for i in range(a)],:,0]
yhat1=yhat1.reshape(yhat1.shape[0]*yhat1.shape[1])
yhat1=list(yhat1)

X2 = X[[i*timesteps for i in range(a)],:,1]
X2=X2.reshape(X2.shape[0]*X2.shape[1])
X2=list(X2)
yhat2 = yhat[[i*timesteps for i in range(a)],:,1]
yhat2=yhat2.reshape(yhat2.shape[0]*yhat2.shape[1])
yhat2=list(yhat2)

l1=list(range(len(X1)))

plt.figure(figsize=(20,5))
plt.plot(l1, X1, label='Actual data')
plt.plot(l1, yhat1, 'r', label='Predicted reconstruction data')
plt.title('Reconstruction of the total power measurements')
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(20,5))
plt.plot(l1, X2, label='Actual data')
plt.plot(l1, yhat2, 'r', label='Predicted reconstruction data')
plt.title('Reconstruction of the C electricity measurements')
plt.legend(loc='upper left')
plt.show()

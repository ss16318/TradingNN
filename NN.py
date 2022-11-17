import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Read in data
d = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)

#Extract Open Data
df = d[["Open"]]

#Plot data
df.plot()

#Split training and testing data
train = df.iloc[:240]
test = df.iloc[240:]

#Rescale data 
scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


n_input = 3
n_features = 1

# Divides data in batches for training
# TimeseriesGenerator(scaled_train, #data
#                     scaled_train, #target values
#                     length=,      #length of input
#                     batch_size=)  #length of output

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)



# Defining model

#Each layer has one input and one output
model = Sequential()

model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(generator,epochs=50)


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test.plot(figsize=(14,5))


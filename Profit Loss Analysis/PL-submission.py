from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load test data and model
xtest = np.load('xtest.npy')
ytest = np.load('ytest.npy')
model = keras.models.load_model('TradingNN.h5')

# Use model for predictions
y_pred = model.predict(xtest)
print(y_pred)
y_pred_one_hot = (y_pred > 0.5).astype(int) 


# Load raw data during test period 
data = pd.read_csv('spy_vix.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.loc[(data['Date'] > '2022')]
data = data.reset_index(drop=True)

#Find the open-close % difference at the end of the horizon
change = np.array([])

for x in range (3, len(data)-2):    #Horizon is 2 hence the -2
    
    # If zero we don't have a stop loss (otherwise we do)
    low1 = 0 #100 * (data.loc[x][['Low']].values - data.loc[x][['Open']].values) / data.loc[x][['Open']].values
    low2 = 0 #100 * (data.loc[x+1][['Low']].values - data.loc[x][['Open']].values) / data.loc[x][['Open']].values
    
    # We set a stop loss at -2%
    if (low1 < -2) or (low2 < -2):
        change = np.append(change,-2)
    
    # Otherwise the position expires at close of day 2
    else:
        diff = 100 * (data.loc[x+2][['Close']].values - data.loc[x][['Open']].values) / data.loc[x][['Open']].values
        change = np.append(change,diff)

#Record the profit loss during the test period
PL = np.array([0])

for x in range (len(y_pred_one_hot)):
    
    #If no trade, don't change P/L
    if y_pred_one_hot[x] == 0: 
        PL = np.append(PL, 0 + PL[x])
    
    #If trade and hit take profit increase 0.5%
    elif y_pred_one_hot[x] == 1 and ytest[x] == 1:
        PL = np.append(PL, 0.5 + PL[x])
    
    #If trade and don't hit take profit (take change which is either stop loss or close of day 2)
    else:
        PL = np.append(PL, change[x] + PL[x])


        
plt.figure()
plt.plot(PL)
plt.title("Neural Network Cumulative Profit/Loss % During 2022 (without Stop Loss)")
plt.xlabel("Day")
plt.ylabel("Profit/Loss %")
        


    
    
    
    

           
    


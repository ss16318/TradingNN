# %%
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import math

def main(trading_horizon, days_between, pct_change, close_high_comparison):
    # %% User-defined parameters

    #choose number of days for the forecast
    horizon = trading_horizon 

    #analyse days between horizon
    between = days_between   

    #choose % increase in stock price
    change = pct_change

    #choose % change comparison: Close (open-close) or High (open-high)
    comparison = close_high_comparison


    # %% Feature Preparation


    #Load file
    df = pd.read_csv('spy_vix.csv') 

    #Add date features
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day_of_year'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.weekday

    #drop the actual date so that its not a feature
    df.drop('Date', axis=1, inplace=True)

    #get data before 2021 to train with
    data_less_2021 = df.loc[(df['year'] < 2021)]
    data_less_2021.to_pickle("training_data_before_2021.pkl")
    #get data in 2021 to validate
    data2021 = df.loc[((df['year'] >= 2021) & (df['year'] < 2022))]
    data2021.to_pickle("validation_data_2021.pkl")
    #get data in 2022 to test with
    data2022 = df.loc[((df['year'] >= 2022) & (df['year'] < 2023))]
    data2022.to_pickle("testing_data_2022.pkl")

    # %% Input-Output preparation 

    xtrain, ytrain = label_data(data_less_2021, horizon, between, change, comparison)
    xval, yval = label_data(data2021, horizon, between, change, comparison)
    xtest, ytest = label_data(data2022, horizon, between, change, comparison)

    #Scales training data and then saves the scaller to transform validation & testing data
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xval = scaler.transform(xval)
    xtest = scaler.transform(xtest)

    # save test data
    np.save('xtest',xtest)
    np.save('ytest',ytest)

    # %% Artificial Neural Network
    model = keras.Sequential([
        
        #Input layer
        keras.layers.Dense(int(3*len(df.columns)), input_shape=(np.size(xtrain,1),)),

        keras.layers.Flatten(), #flatten to ensure its 1 array (it should be)
        keras.layers.BatchNormalization(), #added normalization

        #Hidden Layer 1
        keras.layers.Dense(int(3*len(df.columns)), activation='relu'),
       
        #Output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile and changing learning rate to smooth out validation loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())
    K.set_value(model.optimizer.learning_rate, 0.00002) #change learning rate to be lower so that it doesnt just return the same value for all predictions
    print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())

    print('------------------------------------------------------------------------')
    print(f'Training for Trading Horizon {horizon}, days between {between}, pct change {change}, comparison {comparison} ...')

    #Change epochs and load validation data
    history = model.fit(xtrain, ytrain, epochs=100, batch_size=10, verbose = 0, validation_data=(xval,yval))
    
    y_pred = model.predict(xtest)
    y_pred_one_hot = (y_pred > 0.5).astype(int) #sigmoid function returns bounded float between 0 and 1. any value >0.5 is considered class 1

    #Save model to be tested later
    model.save("TradingNN.h5")
# %% Visualising results
    confusion_matrix = metrics.confusion_matrix(ytest, y_pred_one_hot)
    
    accuracy = int( 100 * (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1]))
    TPR = 100 * confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    
    if math.isnan(TPR) == 1:
        TPR = "N/A" #avoid div0 case
    else:
        TPR = int(TPR)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["No", "Yes"])
    cm_display.plot(colorbar=0)
    plt.title("Trading Horizon " + str(horizon) + " – Will stock increase by " + str(change) +"%? by comparing " + comparison + " looking at " + str(between) + " days between \n \n  Accuracy: " + str(accuracy) + "% & True Positive Rate: " + str(TPR) + "%" )
    #plt.savefig(f'horizon{horizon}_between{between}_change{change}_comparison{comparison}_cm.png')
    #plt.show(block=False)

    #Save training and validation loss & accuracy to determine over/under fit
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title("Trading Horizon  " + str(horizon) + " – pct increase " + str(change) + ' Training and validation accuracy')
    plt.legend()
    #plt.savefig(f'horizon{horizon}_between{between}_change{change}_comparison{comparison}_accuracy.png')

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title("Trading Horizon  " + str(horizon) + " – pct increase " + str(change) + ' Training and validation loss')
    plt.legend()
    #plt.savefig(f'horizon{horizon}_between{between}_change{change}_comparison{comparison}_loss.png')
    plt.show()

#function to label and normalize data
def label_data(df, horizon, between, change, comparison):
    df.index = range(len(df))
    features = 3*len(df.columns)         #choose number of features for datapoint

    data = np.zeros((1, features +1))    #create array to save input and output data

    for x in range (len(df)-(3+horizon)):                      #loop through each day
        
        day1 = df.loc[x]             #day 1 data
        day2 = df.loc[x+1]           #day 2 data
        day3 = df.loc[x+2]           #day 3 data
        
        #if we only want to consider data on the last day
        if between == 0:
            diff = (df.loc[x+3+horizon-1][[comparison]].values - df.loc[x+3][['Open']].values) / df.loc[x+3][['Open']].values       
        
        #otherwise look at days during the horizon
        else:
            
            diff = 0 #initialize the diff 
            
            #loop through each day in the horizon
            for y in range(0,horizon):
                #get the change compared to the open of day4
                tempDiff = (df.loc[x+3+y][[comparison]].values - df.loc[x+3][['Open']].values) / df.loc[x+3][['Open']].values
                
                #update to largest difference
                if tempDiff > diff:
                    diff = tempDiff
        
        #if difference exceeds threshold set label to 1
        if diff > change/100:
            label = pd.Series(1)
            
        else:
            label = pd.Series(0)
            
        
        #concatenate all inputs and outputs into 1 datapoint
        dp = pd.concat([day1,day2,day3,label]).to_numpy()
        #add datapoint to data matrix
        data = np.vstack([data, dp] )

    data = np.delete(data, (0), axis=0) 

    #Rescale input data
    x = data[:,0:features]

    #Output data
    y = data[:,features]

    return x, y 

if __name__ == '__main__':
    trading_horizon = [2]
    days_between = [1]
    pct_change = [0.5]
    close_high_comparison = ['High']

    for trading_horizon, days_between, pct_change, close_high_comparison in product(trading_horizon, days_between, pct_change, close_high_comparison):
        main(trading_horizon, days_between, pct_change, close_high_comparison)
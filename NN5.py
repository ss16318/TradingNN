# %%
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import math


# %% User-defined parameters

#choose number of days for the forecast
horizon = 3  

#analyse days between horizon
between = 1   

#choose % increase in stock price
change = 3

#choose % change comparison: Close (open-close) or High (open-high)
comparison = 'High'


# %% Feature Preparation


#Load file
df = pd.read_csv('spy_vix_yieldcurve.csv') 

#Add date features
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day_of_year'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.weekday

df.drop('Date', axis=1, inplace=True)

df = df.loc[(df['year'] >= 2020)]


# %% Input-Output preparation 

df.index = range(len(df))
features = 3*len(df.columns)         #choose number of features for datapoint

data = np.zeros((1, features +1))    #create array to save input and output data

for x in range (len(df)-(3+horizon)):                      #loop through each day
    
    day1 = df.loc[x]             #day 1 data
    day2 = df.loc[x+1]           #day 2 data
    day3 = df.loc[x+2]           #day 3 data
    
    #if we only want to consider data on the last day
    if between == 0:
        diff = (df.loc[x+3+horizon][[comparison]].values - df.loc[x+3][['Open']].values) / df.loc[x+3][['Open']].values       
    
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
scaler = MinMaxScaler()
scaler.fit(data[:,0:features-1])
x = scaler.transform(data[:,0:features-1])

#Output data
y = data[:,features]


# %% Creating a train and test 

#split into 10 folds
kfold = KFold(n_splits=10, shuffle=True)

fold_no = 1 #fold counter

# %% Artificial Neural Network


for train, test in kfold.split(x, y):
    model = keras.Sequential([
        
        #Input layer
        #keras.layers.BatchNormalization(), #added normalization
        keras.layers.Dense(71, input_shape=(np.size(x,1),)),
        keras.layers.Flatten(), #flatten to ensure its 1 array (it should be)
        #Hidden Layer 1
        keras.layers.Dense(4, activation='relu'), #kernel_regularizer=keras.regularizers.l2(l=0.1)), #added regularization
        #Output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())
    #K.set_value(model.optimizer.learning_rate, 0.001) #change learning rate to be lower so that it doesnt just return the same value for all predictions
    print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    history = model.fit(x[train], y[train], epochs=50, batch_size=10, verbose = 0)
    
    y_pred = model.predict(x[test])
    print(y_pred)
    y_pred_one_hot = (y_pred > 0.5).astype(int) #sigmoid function returns bounded float between 0 and 1. any value >0.5 is considered class 1

# %% Visualising results


    confusion_matrix = metrics.confusion_matrix(y[test], y_pred_one_hot)
    
    accuracy = int( 100 * (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1]))
    TPR = 100 * confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    
    if math.isnan(TPR) == 1:
        TPR = "N/A"
    else:
        TPR = int(TPR)

    # Plotting confusion matrix
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["No", "Yes"])
    cm_display.plot(colorbar=0)
    plt.title("Iteration " + str(fold_no) + ": Will stock increase by " + str(change) +"%? \n \n Accuracy: " + str(accuracy) + "% & True Positive Rate: " + str(TPR) + "%" )
    plt.show(block=False)
    
    
    fold_no = fold_no + 1
    
plt.show()



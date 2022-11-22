import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
import matplotlib.pyplot as plt

#Choose stock increase %
increase = 0.5

#Read in data
df = pd.read_csv('SPY.csv')


#Add Date Info to dataframe
df['Date'] = pd.to_datetime(df['Date'])

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day_of_year'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.weekday

#Delete merged date column
df.drop('Date', axis=1, inplace=True)



#Preparing data as input and ouput

features = 3*len(df.columns)    #choose number of features for datapoint

data = np.zeros((1, features +1))    #create array to save input and output data


for x in range (len(df)-3):     #loop through each day
    
    # if Monday or Tuesday we create a datapoint (otherwise we are modelling over
    # a weekend)
    if (df.loc[x].at["day_of_week"] == 0) or (df.loc[x].at["day_of_week"] == 1):
        
        day1 = df.loc[x]                #day 1 data
        day2 = df.loc[x+1]              #day 2 data
        day3 = df.loc[x+2]              #day 3 data
        
        #join 3 days of stock data and convert it np array
        dp = pd.concat([day1,day2,day3]).to_numpy()
        
        #get open and close on day 4
        day4close = df.loc[x+3].at["Close"]
        day4open  = df.loc[x+3].at["Open"]
        
        #calculate reutrn on day4
        change = 100 * (day4close-day4open) / day4open
        
        if change >= increase:  #if return exceeds desired return
            
            #add 1 output to datapoint
            dp = np.append(dp, 1)          

            
        else:
            
            #add 0 output to datapoint
            dp = np.append(dp, 0)
            


        dp.resize(1,len(dp))
        
        #add input/output datapoint to data array
        data = np.vstack([data, dp] )

# delete first datapoint initialized to 0
data = np.delete(data, (0), axis=0)


# Splitting data into training, validating and testing (0.7, 0.15, 0.15 respecitvely)
x_train, x_test, y_train, y_test = train_test_split(data[:,0:features-1], data[:,features], test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.8)



# Defining the neural network model

model = keras.Sequential([
    
    #Input layer
    keras.layers.Dense(32, input_shape=(np.size(x_train,1),), activation='relu'),
    #Hidden Layer 1
    keras.layers.Dense(16, activation='relu'),
    #Hidden Layer 2
    keras.layers.Dense(16, activation='relu'),
    #Hidden Layer 3
    keras.layers.Dense(12, activation='relu'),
    #Output layer
    keras.layers.Dense(1, activation='sigmoid')
])
 
# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
              
 
# fitting the model
model.fit(x_train, y_train, epochs=50, batch_size=30)

# Predicting validation outputs
y_pred = model.predict(x_val)

#Generating a confusion matrix for actual vs predictions
confusion_matrix = metrics.confusion_matrix(y_val, y_pred)

# Plotting confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()







        
        
        
        



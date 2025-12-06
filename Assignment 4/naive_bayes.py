#-------------------------------------------------------------------------
# AUTHOR: Kaitlin Yen
# FILENAME: naive_bayes.py
# SPECIFICATION: Use Naive Bayes to take in weather data and classify
# weather_test.csv
# FOR: CS 4440- Assignment #4
# TIME SPENT: 8 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
df = pd.read_csv('weather_training.csv')

#update the training class values according to the discretization (11 values only)
X_training = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Using numpy cut to bin classes into 11 bins
y_training, bins = pd.cut(y, bins=11, retbins=True, labels=False)

#reading the test data
df_test = pd.read_csv('weather_test.csv')

#update the test class values according to the discretization (11 values only)
X_test = df_test.iloc[:, :-1].values
y_test_continuous = df_test.iloc[:, -1].values
y_test = np.digitize(y_test_continuous, bins)

#loop over the hyperparameter value (s)
for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    #--> add your Python code here

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here

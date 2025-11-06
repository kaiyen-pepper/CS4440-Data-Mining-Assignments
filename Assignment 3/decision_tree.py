# -------------------------------------------------------------------------
# AUTHOR: Kaitlin Yen
# FILENAME: decision_tree.py
# SPECIFICATION: Read files from cheat_training_1.csv, cheat_training_2.csv, 
# cheat_training_3.csv to build decision trees
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: 8 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []
    average = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must be converted to a float.
    data_training[:, 2] = [float(s[:-1]) for s in data_training[:, 2]] # Convert Taxable Income to float
    
    encoder = OneHotEncoder(sparse_output=False) # One-Hot Encode Marital Status
    marital_status_encoded = encoder.fit_transform(data_training[:, 1].reshape(-1, 1))
    
    # Encode Categorical Features
    refund = df["Refund"].str.strip().replace({"No": 0, "Yes": 1}).astype(int) # Refund
    X = np.array(refund).reshape(-1, 1)
    X = np.hstack((X, marital_status_encoded))  # Combine with one-hot encoded Marital Status
    X = np.hstack((X, np.array(data_training[:, [2]], dtype=float)))  # Combine with Taxable Income

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    Y = df["Cheat"].str.strip().replace({"No": 0, "Yes": 1}).astype(int)

    #loop your training and test tasks 10 times here
    for i in range (10):
        true_positive = 0

        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        #read the test data and add this data to data_test NumPy
        data_test = pd.read_csv('cheat_test.csv', sep=',', header=0).to_numpy()

        for data in data_test:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            taxable_income = float(data[3][:-1]) # Convert Taxable Income to float
            refund = 1 if data[1].strip() == "Yes" else 0 # Refund
            marital_status = encoder.transform([[data[2].strip()]])[0] # One-Hot Encode Marital Status
            test_instance = [refund] + list(marital_status) + [taxable_income] # Combine all features

            class_predicted = clf.predict([test_instance])[0]

            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            if class_predicted == (1 if data[4].strip() == "Yes" else 0):
               true_positive += 1

        #find the average accuracy of this model during the 10 runs (training and test set)
        average.append(true_positive/len(data_test))

    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    average_accuracy = sum(average)/len(average)
    print("Final accuracy when training on " + ds + ": " + str(average_accuracy))
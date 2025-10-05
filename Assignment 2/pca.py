# -------------------------------------------------------------------------
# AUTHOR: Kaitlin Yen
# FILENAME: pca.py
# SPECIFICATION: Apply PCA on heart disease dataset multiple times, removing one feature at a time. 
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: 4 hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
df = pd.read_csv('heart_disease_dataset.csv')
print(df.head())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Get the number of features
num_features = np.array(df.shape[1])

#Define max_pc1 to store the maximum variance of PC1
max_pc1 = 0

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    pca = PCA(n_components=2)
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    pc1 = pca.explained_variance_ratio_[0]
    print("Variance of PC1 after removing feature", df.columns[i], ":", pc1)

    # Find the maximum PC1 variance
    if pc1 > max_pc1:
        max_pc1 = pc1
        feature_removed = df.columns[i]

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print("Highest PC1 variance found:", max_pc1, "when removing", feature_removed)

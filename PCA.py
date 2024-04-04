"""
How to apply PCA to a dataset
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set the base directory
basedir = os.getcwd()

# First read the data. We will use the MNIST dataset
df = pd.read_csv(os.path.join(basedir, "Data/mnist_train.csv"))

X = df.iloc[:, 1:]
Y = df.iloc[:, 0]


########## MUST SCALE THE DATA!!!!! ############
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Now can do PCA
pca_fit1 = PCA(n_components=10)
X_pca1 = pca_fit1.fit_transform(X_scaled)

pca_fit2 = PCA(n_components=0.95)
X_pca2 = pca_fit2.fit_transform(X_scaled)

print("Explained Variance Ratio PCA1 (10 Comp)", pca_fit1.explained_variance_ratio_.sum())
print("Explained Variance Ratio PCA2 (0.95)", pca_fit2.explained_variance_ratio_.sum())









fin = 1
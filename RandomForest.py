import os
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the base directory
basedir = os.getcwd()

# First read the data. We will use the MNIST dataset
df = pd.read_csv(os.path.join(basedir, "Data/mnist_train.csv"))

# Get train and test X and Y
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

# Get train and test data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8, random_state=7)

# Scale the data and apply PCA
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


pca = PCA(n_components=0.9)
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Fit the Random Forest
rand_for1 = RandomForestClassifier()
rand_for2 = RandomForestClassifier(n_estimators=30, max_depth=100)

rand_for1.fit(X_train_pca, Y_train)
rand_for2.fit(X_train_pca, Y_train)

Y1_pred = rand_for1.predict(X_test_pca)
Y2_pred = rand_for2.predict(X_test_pca)

accuracy1 = accuracy_score(Y_test, Y1_pred)
accuracy2 = accuracy_score(Y_test, Y2_pred)
print(f"Accuracy Default Model: {accuracy1*100}%")
print(f"Accuracy Specific Model: {accuracy2*100}%")


fin=1
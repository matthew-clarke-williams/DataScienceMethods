import os
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Fit PCA model
pca = PCA(n_components=0.9)
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model = KNeighborsClassifier(n_neighbors=5)     # n_neighbours - the number of neighbours checked to classify a point
model.fit(X_train_pca, Y_train)     # Leving n_neighbours default is best in this case

# Predict the test data
Y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy (with PCA): {accuracy*100}%")

# Classification report
print(classification_report(Y_test, Y_pred, digits=3))

# model_raw = KNeighborsClassifier(n_neighbors=10)
# model_raw.fit(X_train, Y_train)
# Y_pred_raw = model_raw.predict(X_test)
# print(f"Accuracy (No PCA): {accuracy_score(Y_test, Y_pred_raw)*100}%")
#
# model_default = KNeighborsClassifier()
# model_default.fit(X_train, Y_train)
# Y_pred_default = model_default.predict(X_test)
# print(f"Accuracy (Default No PCA): {accuracy_score(Y_test, Y_pred_default)*100}%")

fin=1
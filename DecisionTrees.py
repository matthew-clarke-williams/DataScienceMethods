import os
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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


pca = PCA(n_components=0.9)
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# Fit the classifier
classifier = DecisionTreeClassifier(max_depth=20, criterion='gini', min_samples_split=100, max_features=200)
classifier.fit(X_train_pca, Y_train)
Y_pred = classifier.predict(X_test_pca)

# Compare predicted Y and actual Y
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy Specific Model: {accuracy*100}%")

default_classifier = DecisionTreeClassifier()
default_classifier.fit(X_train_pca, Y_train)
Y_pred2 = default_classifier.predict(X_test_pca)

# Compare predicted Y and actual Y
accuracy2 = accuracy_score(Y_test, Y_pred2)
print(f"Accuracy Default Model: {accuracy2*100}%")

# Classification report
print(classification_report(Y_test, Y_pred, digits=3))


fin=1
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()

X = data["data"]
Y = data["target"]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8, random_state=29)

# Apply scaling and PCA
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.9)
pca.fit(X_train_scaled)
print(f"Number of components at 90% PCA level: {pca.n_components_}")
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Now fit the model
model = LogisticRegression()
model.fit(X_train_pca, Y_train)
Y_pred = model.predict(X_test_pca)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy*100}%")

# Classification report
print(classification_report(Y_test, Y_pred, digits=3))

fin = 1
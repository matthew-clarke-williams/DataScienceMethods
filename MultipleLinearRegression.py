import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = load_diabetes()

X = data["data"]
Y = data["target"]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8, random_state=29)

# Let's scale and apply PCA. Will compare models with and without PCA
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit PCA model
pca = PCA(n_components=0.9)
pca.fit(X_train_scaled)
print(f"{pca.n_features_in_} Features Before PCA. {pca.n_components_} After PCA.")
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Now fit the regression model
model_pca = LinearRegression()
model_no_pca = LinearRegression()
model_pca.fit(X_train_pca, Y_train)
model_no_pca.fit(X_train, Y_train)

# Predict Y for test data
Y_pred_pca = model_pca.predict(X_test_pca)
Y_pred_no_pca = model_no_pca.predict(X_test)

MSE_pca = mean_squared_error(Y_test, Y_pred_pca)
MSE_no_pca = mean_squared_error(Y_test, Y_pred_no_pca)

print(f"MSE of PCA Regression Model: {MSE_pca}%")
print(f"MSE of Regression Model Without PCA: {MSE_no_pca}%")

print(f"Score PCA {model_pca.score(X_test_pca, Y_test)}")
print(f"Score No PCA {model_no_pca.score(X_test, Y_test)}")

fin=1
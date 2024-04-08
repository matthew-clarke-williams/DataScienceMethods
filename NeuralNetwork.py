import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
# pca = PCA(n_components=0.9)
pca = PCA(n_components=2)
pca.fit(X_train_scaled)
print(f"{pca.n_features_in_} Features Before PCA. {pca.n_components_} After PCA.")
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Now fit the Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_pca.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer with one neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_pca, Y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# Evaluate the model on the testing data
Y_pred = model.predict(X_test_pca)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)



# Plot the results 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(X_test_pca[:,0], X_test_pca[:,1], Y_test, c='r', marker='o', label="Real")
ax.scatter(X_test_pca[:,0], X_test_pca[:,1], Y_pred, c='b', marker='o', label="Predicted")

# Labeling axes
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target Variable (y)')

plt.title('3D Plot of Features against Target Variable')

plt.show()



fin = 1

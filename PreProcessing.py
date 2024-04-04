"""
Script to demonstrate some of the useful preprocessing techniques of data prior to Analysis

Includes:
    - Reading Data
    - Randomising Data
    - Sampling Data
    - Test/Train Split
    - Removing Columns/Rows
    - Filling Empty values
"""
import os
import numpy as np
import pandas as pd
from sklearn import model_selection


# Set the base directory
basedir = os.getcwd()

# First read the data. We will use the MNIST dataset
mnist_train = pd.read_csv(os.path.join(basedir, "Data/mnist_train.csv"))
mnist_test = pd.read_csv(os.path.join(basedir, "Data/mnist_test.csv"))
mnist_all = pd.concat([mnist_train, mnist_test], ignore_index=True)        # Joining the two so that we can split them

########## RANDOMISE THE DATA ORDER AND SAMPLING ##########
df_shuffle = mnist_all.sample(frac=1)   # Randomise the order
df_sample = mnist_all.sample(n=1000, random_state=99)    # Random sample of 1000



########## TEST TRAIN SPLIT ##########
X = df_shuffle.iloc[:, 1:]
Y = df_shuffle.iloc[:,0]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8, random_state=29)



########## EMPTY VALUES ##########
empty_df = pd.DataFrame(X_train.copy())
empty_df.iloc[100:400] = np.nan
empty_df.iloc[:, -1] = np.nan

# Fill empty values in rows 100-200 with the mean
mean_vals = empty_df.mean()
empty_df.iloc[100:201] = empty_df.iloc[100:201].fillna(mean_vals)

# Remove any columns that contain empty values
empty_df = empty_df.dropna(axis=1, how="all")     # Can also be "any"

# Remove any rows that contain empty values
empty_df = empty_df.dropna()


fin = 1
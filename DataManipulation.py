"""
Group by's
Merges pd.merge
Concat Dataframes
Joins  (same as merge)
Graphs and plots
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt

data = load_diabetes()

X = data["data"]
Y = data["target"]

df = pd.DataFrame(data=X, columns=data["feature_names"])
df["Y"] = Y

# Group By
gb_df = df["age", "sex", ""].groupby(by=["age"], axis=0, )

gb_df = df.groupby('age').agg({'bmi': 'mean', 'bp': 'mean'}).reset_index()
gb_df.columns = ["Age", "Avg BMI", "Avg BP"]    # Rename the columns


# Plot some data
plt.figure()
plt.scatter(gb_df["Age"], gb_df["Avg BMI"], color="b", label="Average BMI")
plt.scatter(gb_df["Age"], gb_df["Avg BP"], color="r", label="Average Blood Pressure")
plt.xlabel("Age")
plt.ylabel("Average BMI")
plt.show()

fin = 1
# Step 1: Import Libraries

import pandas as pd
import matplotlib.pyplot as plt


# Step 2: Load the CSV File

df = pd.read_csv("https://raw.githubusercontent.com/cybervaults/Bi/refs/heads/main/practical8.csv")
print(df.head())


# Step 3: Check Dataset Shape

print(df.shape)


# Step 4: Display Column Names

print(df.columns)


# Step 5: Check Missing Values

print(df.isnull().sum())


# Step 6: Count Records by Species

print(df["species"].value_counts())


# Step 7: Visualize Species Distribution

df["species"].value_counts().plot(kind="bar")
plt.title("Number of Flowers in Each Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")
plt.show()
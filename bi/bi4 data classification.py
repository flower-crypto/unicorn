# Step 1: Import Libraries and Load the Dataset

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
   accuracy_score, confusion_matrix, classification_report,
   ConfusionMatrixDisplay
)

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print(df.head())
print(df["species"].value_counts())


# Step 2: Prepare Features and Split the Data

X = df.drop(columns="species")
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set:", X_train.shape, " Testing set:", X_test.shape)


# Step 3: Decision Tree Classifier

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, pred_tree)
print("Decision Tree accuracy:", round(acc_tree, 3))

plt.figure(figsize=(11, 6))
plot_tree(
   tree, feature_names=X.columns,
   class_names=list(tree.classes_), filled=True, fontsize=8
)
plt.show()


# Step 4: K-Nearest Neighbours (KNN) Classifier

scaler = StandardScaler().fit(X_train)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(scaler.transform(X_train), y_train)
pred_knn = knn.predict(scaler.transform(X_test))
acc_knn = accuracy_score(y_test, pred_knn)
print("KNN accuracy:", round(acc_knn, 3))


# Step 5: Naive Bayes Classifier

nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, pred_nb)
print("Naive Bayes accuracy:", round(acc_nb, 3))


# Step 6: Compare the Classifiers

results = pd.DataFrame({
   "Model": ["Decision Tree", "KNN", "Naive Bayes"],
   "Accuracy": [acc_tree, acc_knn, acc_nb]
}).round(3)
print(results)

results.plot(
   x="Model", y="Accuracy", kind="bar",
   ylim=(0.8, 1.02), legend=False,
   figsize=(6, 3.5), rot=0
)
plt.title("Accuracy of the classifiers")
plt.show()


# Step 7: Confusion Matrix and Classification Report

print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
ConfusionMatrixDisplay.from_predictions(y_test, pred_knn, cmap="Blues")
plt.title("Confusion matrix - KNN")
plt.show()


# Step 8: Predict a New Record

new = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]], columns=X.columns)
print("Predicted species:", knn.predict(scaler.transform(new))[0])

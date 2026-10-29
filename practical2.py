import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score, classification_report


# Step 2: Generate the Dataset

np.random.seed(42)

n = 20

data = pd.DataFrame({
    'file_size': np.random.randint(10, 1000, n),
    'suspicious_calls': np.random.randint(0, 30, n),
    'entropy': np.random.uniform(2, 8, n)
})

# Create labels: 0 = Benign, 1 = Malware

data['malware'] = (
    (data['suspicious_calls'] > 15) |
    (data['entropy'] > 5.5)
).astype(int)

print("Dataset:")
print(data)


# Step 3: Separate Features and Target

X = data[['file_size', 'suspicious_calls', 'entropy']]

y = data['malware']


# Step 4: Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# Step 5: Create and Train the Decision Tree

model = DecisionTreeClassifier(max_depth=4, random_state=42)

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Step 6: Predict and Evaluate the Model

y_pred = model.predict(X_test)

print("\nPredicted values:", y_pred)

print("Actual values:", y_test.values)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=['Benign', 'Malware'],
    zero_division=0
))


# Step 7: Predict a New File

new_file = pd.DataFrame(
    [[750, 20, 7.0]],
    columns=['file_size', 'suspicious_calls', 'entropy']
)

prediction = model.predict(new_file)

if prediction[0] == 1:
    print("\nNew File Prediction: Malware")
else:
    print("\nNew File Prediction: Benign")


# Step 8: Visualize the Decision Tree

plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Benign', 'Malware'],
    filled=True,
    rounded=True
)

plt.title("Decision Tree for Malware Detection")

plt.show()
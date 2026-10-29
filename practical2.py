# Using Decision Tree and Isolation Forest

# Step 1: Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Create the Sample Network Traffic Data
np.random.seed(42)

# 50 normal network traffic records
normal = pd.DataFrame({
    'duration': np.random.normal(5, 1.5, 50),
    'bytes_sent': np.random.normal(500, 100, 50),
    'connections': np.random.normal(10, 3, 50),
    'label': 0
})

# 25 attack network traffic records
attack = pd.DataFrame({
    'duration': np.random.normal(30, 5, 25),
    'bytes_sent': np.random.normal(5000, 800, 25),
    'connections': np.random.normal(120, 20, 25),
    'label': 1
})

# Combine normal and attack records
data = pd.concat([normal, attack], ignore_index=True)

print("Sample Network Traffic Data:")
print(data.head())

print("\nLabel Counts:")
print(data['label'].value_counts())

# Step 3: Visualize the Data
colors = data['label'].map({0: 'blue', 1: 'red'})

plt.scatter(
    data['bytes_sent'],
    data['connections'],
    c=colors,
    alpha=0.8
)

plt.xlabel('Bytes Sent')
plt.ylabel('Connections')
plt.title('Blue = Normal, Red = Attack')
plt.show()

# Step 4: Split the Data
X = data[['duration', 'bytes_sent', 'connections']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Step 5: Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nDecision Tree Classification Report:")
print(classification_report(
    y_test,
    pred,
    labels=[0, 1],
    target_names=['Normal', 'Attack'],
    zero_division=0
))

# Step 6: Isolation Forest
iso = IsolationForest(
    n_estimators=100,
    contamination=0.33,
    random_state=42
)

# Train without using labels
iso.fit(X)

# Isolation Forest: 1 = Normal, -1 = Anomaly
data['anomaly'] = iso.predict(X)

# Convert predictions to 0 = Normal, 1 = Anomaly
data['anomaly'] = data['anomaly'].map({
    1: 0,
    -1: 1
})

print("\nIsolation Forest Confusion Matrix:")
print(confusion_matrix(data['label'], data['anomaly']))

print("\nIsolation Forest Classification Report:")
print(classification_report(
    data['label'],
    data['anomaly'],
    labels=[0, 1],
    target_names=['Normal', 'Attack'],
    zero_division=0
))

# Step 7: Plot Detected Anomalies
colors = data['anomaly'].map({0: 'blue', 1: 'red'})

plt.scatter(
    data['bytes_sent'],
    data['connections'],
    c=colors,
    alpha=0.8
)

plt.xlabel('Bytes Sent')
plt.ylabel('Connections')
plt.title('Isolation Forest: Red = Detected Anomaly')
plt.show()

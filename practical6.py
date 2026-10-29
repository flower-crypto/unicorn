import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


np.random.seed(42)

genuine = pd.DataFrame({

    'typing_speed': np.random.normal(60, 5, 200),

    'key_hold_time': np.random.normal(0.15, 0.02, 200),

    'key_gap_time': np.random.normal(0.30, 0.04, 200),

    'label': 1})

imposter = pd.DataFrame({

    'typing_speed': np.random.normal(45, 8, 200),

    'key_hold_time': np.random.normal(0.22, 0.04, 200),

    'key_gap_time': np.random.normal(0.45, 0.08, 200),

    'label': 0})

data = pd.concat([genuine, imposter]).reset_index(drop=True)

print(data.head())


X = data.drop('label', axis=1)

y = data['label']

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

new_attempt = pd.DataFrame([[58, 0.16, 0.31]], columns=X.columns)

result = knn.predict(scaler.transform(new_attempt))

print("Access Granted" if result[0] == 1 else "Access Denied")
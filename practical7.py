import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Generate Sample Dataset
np.random.seed(42)

n_samples = 2000

data = pd.DataFrame({
    'amount': np.random.uniform(10, 10000, n_samples),
    'transaction_hour': np.random.randint(0, 24, n_samples),
    'location_mismatch': np.random.randint(0, 2, n_samples),
    'international': np.random.randint(0, 2, n_samples),
    'previous_fraud': np.random.randint(0, 2, n_samples),
    'transaction_frequency': np.random.randint(1, 20, n_samples)
})

# Create fraud labels
fraud_probability = (
    0.02
    + 0.15 * data['location_mismatch']
    + 0.20 * data['international']
    + 0.30 * data['previous_fraud']
    + 0.01 * (data['transaction_frequency'] > 15)
)

fraud_probability = np.clip(fraud_probability, 0, 0.90)

data['fraud'] = (
    np.random.random(n_samples) < fraud_probability
).astype(int)

print("Dataset Sample:")
print(data.head())

print("\nClass Distribution:")
print(data['fraud'].value_counts())

# Step 3: Separate Features and Target
X = data.drop('fraud', axis=1)
y = data['fraud']

# Step 4: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 5: Feature Scaling for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression Model
lr_model = LogisticRegression(class_weight='balanced')

lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, lr_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    lr_predictions,
    zero_division=0
))

# Step 7: Train Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, rf_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    zero_division=0
))

# Step 8: Compare Both Models
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, lr_predictions),
        accuracy_score(y_test, rf_predictions)
    ]
})

print("\n--- Model Comparison ---")
print(comparison)

# Step 9: Result
print("\nResult:")
print("Fraud detection models were trained and evaluated successfully.")
print("Both models were compared using accuracy and classification reports.")

# Step 10: Conclusion
print("\nConclusion:")
print("Machine learning can help identify potentially fraudulent transactions.")
print("The model's performance should be evaluated using fraud precision,")
print("recall, F1-score, and the confusion matrix, not accuracy alone.")
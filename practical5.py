
# PRACTICAL 4: Face-Lock System using PCA and KNN

# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load the Face Dataset
faces = fetch_olivetti_faces(shuffle=False)

X = faces.data
y = faces.target

print("Dataset loaded successfully")
print("Total face images:", X.shape[0])
print("Number of people:", len(np.unique(y)))

# Step 3: Enroll Authorized Users
# People 0-29 are authorized.
# People 30-39 are treated as impostors.

authorized_ids = np.arange(0, 30)
impostor_ids = np.arange(30, 40)

# Use first 5 images per authorized person for enrollment
train_indices = []
calibration_indices = []
test_indices = []
impostor_indices = []

for person in authorized_ids:
    person_indices = np.where(y == person)[0]

    train_indices.extend(person_indices[:5])
    calibration_indices.extend(person_indices[5:7])
    test_indices.extend(person_indices[7:])

for person in impostor_ids:
    person_indices = np.where(y == person)[0]
    impostor_indices.extend(person_indices)

X_train = X[train_indices]
y_train = y[train_indices]

X_cal = X[calibration_indices]
y_cal = y[calibration_indices]

X_test = X[test_indices]
y_test = y[test_indices]

X_impostor = X[impostor_indices]

print("\nAuthorized users enrolled:", len(authorized_ids))
print("Enrollment images:", len(X_train))
print("Calibration images:", len(X_cal))
print("Authorized test images:", len(X_test))
print("Impostor test images:", len(X_impostor))

# Step 4: PCA + KNN
# PCA reduces the number of facial features.
pca = PCA(n_components=100, whiten=True, random_state=42)

X_train_pca = pca.fit_transform(X_train)
X_cal_pca = pca.transform(X_cal)
X_test_pca = pca.transform(X_test)
X_impostor_pca = pca.transform(X_impostor)

# KNN finds the nearest enrolled face.
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train_pca, y_train)

print("\nPCA and KNN models trained successfully")

# Step 5: Set the Acceptance Threshold
# Calibrate the threshold using genuine authorized faces.
cal_distances, cal_neighbors = knn.kneighbors(X_cal_pca)

threshold = np.percentile(cal_distances[:, 0], 95)

print("\nAcceptance Threshold:", round(threshold, 3))

# Step 6: Test FAR and FRR
test_distances, test_neighbors = knn.kneighbors(X_test_pca)
impostor_distances, impostor_neighbors = knn.kneighbors(
    X_impostor_pca
)

# Accept only when the closest distance is within the threshold.
authorized_accepted = test_distances[:, 0] <= threshold
impostor_accepted = impostor_distances[:, 0] <= threshold

# FRR: Authorized faces incorrectly rejected.
FRR = np.mean(~authorized_accepted) * 100

# FAR: Impostor faces incorrectly accepted.
FAR = np.mean(impostor_accepted) * 100

print("\nFalse Acceptance Rate (FAR):", round(FAR, 2), "%")
print("False Rejection Rate (FRR):", round(FRR, 2), "%")

# Optional: Identity classification accuracy
predicted_ids = knn.predict(X_test_pca)
print(
    "\nNearest-neighbour identity accuracy:",
    round(accuracy_score(y_test, predicted_ids) * 100, 2),
    "%"
)

# Step 7: Authenticate an Authorized User and an Impostor
def authenticate(face_image, description):
    face_pca = pca.transform(face_image.reshape(1, -1))

    distances, neighbors = knn.kneighbors(face_pca)
    distance = distances[0, 0]
    matched_id = knn.predict(face_pca)[0]

    print("\nTesting:", description)
    print("Nearest enrolled user ID:", matched_id)
    print("Distance:", round(distance, 3))

    if distance <= threshold:
        print("Access Granted")
    else:
        print("Access Denied")

# Test an authorized user's held-out image
authenticate(X_test[0], "Authorized User")

# Test an impostor's image
authenticate(X_impostor[0], "Impostor")

# Display sample face images
plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.imshow(X_test[0].reshape(64, 64), cmap='gray')
plt.title("Authorized User")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(X_impostor[0].reshape(64, 64), cmap='gray')
plt.title("Impostor")
plt.axis('off')

plt.tight_layout()
plt.show()

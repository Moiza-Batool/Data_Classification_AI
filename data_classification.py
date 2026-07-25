# Data Classification Using AI
# Author: Moiza Batool
# Language: Python 3.12

# Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# INPUT
# Load the Iris Dataset
iris = load_iris()
X = iris.data          # Features
y = iris.target        # Labels

print("        DATA CLASSIFICATION USING AI")

print(f"\nDataset Name : Iris")
print(f"Samples      : {len(X)}")
print(f"Features     : {len(iris.feature_names)}")
print(f"Classes      : {len(iris.target_names)}")

print("\nFeature Names:")
for feature in iris.feature_names:
    print(f"• {feature}")

print("\nClass Names:")
for flower in iris.target_names:
    print(f"• {flower}")

# PROCESS
# Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures successfully standardized.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# KNN Classification Model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully.")

# Predictions
predictions = model.predict(X_test)

# OUTPUT
# Model Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\n")
print("MODEL PERFORMANCE")

print(f"\nAccuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)

print("=" * 60)
print("Classification completed successfully.")
print("=" * 60)
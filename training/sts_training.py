import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# Directory containing the CSV files
data_directory = "../dataset/sts_parkinson/"

# Initialize lists for features and labels
features = []
labels = []

# Function to extract enhanced features
def extract_features(csv_file):
    data = pd.read_csv(csv_file)
    joint_data = data.iloc[:, 2:].values
    # Extract enhanced features
    mean_values = np.mean(joint_data, axis=0)
    std_values = np.std(joint_data, axis=0)
    min_values = np.min(joint_data, axis=0)
    max_values = np.max(joint_data, axis=0)
    return np.hstack([mean_values, std_values, min_values, max_values])

# Loop through all CSV files in the directory with progress bar
print("Extracting features from CSV files...")
for file_name in tqdm(os.listdir(data_directory), desc="Processing CSV files"):
    if file_name.endswith(".csv"):
        file_path = os.path.join(data_directory, file_name)
        features.append(extract_features(file_path))
        labels.append(1 if "PD" in file_name else 0)

# Convert lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)

print(f"Feature extraction complete. Total samples: {len(labels)}")

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Optimize and train Random Forest classifier
print("Optimizing and training the Random Forest classifier...")
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=1)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and scaler
print("Saving the trained model and scaler...")
os.makedirs("../models", exist_ok=True)
joblib.dump(clf.best_estimator_, "../models/optimized_parkinson_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
print("Model and scaler saved.")

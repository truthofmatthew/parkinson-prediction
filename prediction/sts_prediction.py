import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def load_model(model_path, scaler_path):
    """Load the pre-trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def extract_enhanced_features(file_path):
    """Extract enhanced features from a CSV file."""
    data = pd.read_csv(file_path)
    joint_data = data.iloc[:, 2:].values  # Assuming joint data starts from the 3rd column

    # Extract enhanced features
    mean_values = np.mean(joint_data, axis=0)
    std_values = np.std(joint_data, axis=0)
    min_values = np.min(joint_data, axis=0)
    max_values = np.max(joint_data, axis=0)

    return np.hstack([mean_values, std_values, min_values, max_values])

def predict_single_file(file_path, model, scaler):
    """Predict Parkinson's Disease for a single file."""
    try:
        # Extract features
        features = extract_enhanced_features(file_path)

        # Scale features
        scaled_features = scaler.transform([features])

        # Predict
        prediction = model.predict(scaled_features)[0]

        return 'Parkinson' if prediction == 1 else 'Healthy'

    except Exception as e:
        return f"Error: {e}"

def predict_directory_summary(directory_path, model_path, scaler_path):
    """Predict and calculate Parkinson/Healthy percentages for all files in the directory."""
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}Starting Prediction for Files in Directory: {directory_path}")
    print(f"{Fore.CYAN}{'='*60}")

    # Load the pre-trained model and scaler
    model, scaler = load_model(model_path, scaler_path)

    parkinson_count = 0
    healthy_count = 0
    total_count = 0

    files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
    results = []

    for file_name in tqdm(files, desc="Processing files", bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}"):
        file_path = os.path.join(directory_path, file_name)

        prediction = predict_single_file(file_path, model, scaler)
        if prediction.startswith("Error"):
            results.append(f"{Fore.RED}File: {file_name} | {prediction}")
        else:
            total_count += 1
            if prediction == 'Parkinson':
                parkinson_count += 1
            else:
                healthy_count += 1

            # Append prediction result
            if prediction == 'Parkinson':
                results.append(f"{Fore.RED}File: {file_name} | Prediction: {prediction}")
            else:
                results.append(f"{Fore.GREEN}File: {file_name} | Prediction: {prediction}")

    # Print all results after tqdm finishes
    for result in results:
        print(result)

    # Calculate percentages
    parkinson_percent = (parkinson_count / total_count) * 100 if total_count > 0 else 0
    healthy_percent = (healthy_count / total_count) * 100 if total_count > 0 else 0

    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}Prediction Summary:")
    print(f"{Fore.YELLOW}Parkinson Percentage: {parkinson_percent:.2f}%")
    print(f"{Fore.YELLOW}Healthy Percentage: {healthy_percent:.2f}%")
    print(f"{Fore.CYAN}{'='*60}")

    return parkinson_percent, healthy_percent

# Example usage
directory_path = "../dataset/sts_parkinson/"  # Replace with the directory containing test CSV files
model_path = "../models/optimized_parkinson_model.pkl"  # Path to the saved optimized model
scaler_path = "../models/scaler.pkl"  # Path to the saved scaler

# Get predictions for all files and a summary
parkinson_percent, healthy_percent = predict_directory_summary(directory_path, model_path, scaler_path)

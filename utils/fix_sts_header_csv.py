import os
import pandas as pd

def fix_csv_header(csv_file):
    """Fix the header of the CSV file dynamically and save the fixed file."""
    # Define correct headers for BODY_25
    correct_headers = ['frame number', 'time (s)']
    for i in range(25):  # BODY_25 has 25 joints
        correct_headers.append(f'x{i}')
        correct_headers.append(f'y{i}')

    # Read the file, skipping the incorrect header row
    data = pd.read_csv(csv_file, skiprows=1, header=None)  # Skip the incorrect header
    data.columns = correct_headers  # Assign corrected headers

    # Save the fixed file with a "_fix" suffix
    fixed_file = csv_file.replace('.csv', '_fix.csv')
    data.to_csv(fixed_file, index=False)
    print(f"Fixed CSV saved to: {fixed_file}")
    return fixed_file

def fix_csv_headers_in_directory(directory):
    """Fix headers for all CSV files in the given directory."""
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            csv_file = os.path.join(directory, file_name)
            try:
                fix_csv_header(csv_file)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

# Directory containing your CSV files
directory_path = "dataset/SitToStand/Data/"  # Replace with your directory path

try:
    # Fix headers for all CSVs in the directory
    fix_csv_headers_in_directory(directory_path)
except Exception as e:
    print("Error:", e)

import numpy as np
import csv
import os

def npy_to_csv(input_folder, output_folder):
    # List all .npy files in the input folder
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    # Check if there are any .npy files in the folder
    if len(npy_files) == 0:
        print("No .npy files found in the input folder.")
        return
    
    # Loop through each .npy file
    for npy_file in npy_files:
        # Check if corresponding CSV file already exists
        csv_file = os.path.splitext(npy_file)[0] + '.csv'
        csv_path = os.path.join(output_folder, csv_file)
        if os.path.exists(csv_path):
            print(f"CSV file '{csv_file}' already exists. Skipping...")
            continue
        
        # Load the .npy file
        data = np.load(os.path.join(input_folder, npy_file))
        
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Write the data to the CSV file
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Data'])
            csv_writer.writerow(data)

        print(f"CSV file '{csv_path}' created successfully.")

# Example usage:
input_folder = '/Users/camillaucheomaenwereuzor/Desktop/RA IBL task/mouse version prova/pregenerated_sequences/raws'
output_folder = '/Users/camillaucheomaenwereuzor/Desktop/RA IBL task/mouse version prova/pregenerated_sequences'

npy_to_csv(input_folder, output_folder)

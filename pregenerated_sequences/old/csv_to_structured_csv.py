import csv
import os

def process_original_csv(original_csv, output_csv):
    # Check if the original CSV file exists
    if not os.path.isfile(original_csv):
        print(f"Original CSV file '{original_csv}' not found.")
        return
    
    # Read data from the original CSV file
    with open(original_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header row
        data_row = next(reader) # Read the data row
    
    # Split each value in the data row and extract eccentricity and contrast
    processed_data = []
    for value in data_row:
        # Remove brackets and split the value
        value = value.strip('[]').split()
        eccentricity = value[0] if len(value) >= 1 else ''
        contrast = value[1] if len(value) >= 2 else ''
        q = value[2] if len(value) >= 3 else ''
        processed_data.append([eccentricity, contrast, q])
    
    # Write the processed data to a new CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eccentricity', 'contrast', 'q'])  # Write header row
        writer.writerows(processed_data)               # Write processed data
    
    print(f"New CSV file '{output_csv}' created successfully.")

# Example usage:
original_csv = '/Users/camillaucheomaenwereuzor/Desktop/RA IBL task/mouse version prova/pregenerated_sequences/session_0_ephys_pcqs.csv'
output_csv = '/Users/camillaucheomaenwereuzor/Desktop/RA IBL task/mouse version prova/pregenerated_sequences/structured csvs/session_0.csv'

process_original_csv(original_csv, output_csv)


import os
import csv

# Define the path to the directory containing the subdirectories with the csv files
root_dir = 'week3\\data\\trackers\\mot_challenge\\parabellum-train\\'

# Define the name of the csv file to save the results to
result_filename = 'results.csv'

# Define the header for the results csv file
header = ['method', 'model', 'miniou', 'max_age', 'HOTA', 'IDF1']

# Initialize the list of results
results = []

# Loop through the subdirectories and extract the metrics from the csv files
for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        # Check if the file is a csv file
        if filename.endswith('.csv'):
            # Extract the parameters from the folder name
            foldername = subdir.split('\\')[-1]
            params = foldername.split('_')
            method = params[0]
            model = params[1]
            # get thr index
            miniou_index = params.index('miniou')
            miniou = float(params[miniou_index+1]) / 100
            # get maxage index
            maxage_index = params.index('maxage')
            max_age = int(params[maxage_index+1])

            # Read the metrics from the csv file
            with open(os.path.join(subdir, filename), 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                row = next(reader)
                hota = row['HOTA___AUC']
                idf1 = row['IDF1']

            # Add the results to the list
            results.append([method, model, miniou, max_age, hota, idf1])

# Write the results to the csv file
with open(result_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in results:
        writer.writerow(row)

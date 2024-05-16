import os
import pandas as pd
from collections import defaultdict

def is_incremental_integer_column(series):
    # Check if the column values are integers and increment by 1 in consecutive rows
    if series.dtype != 'int64':
        return False
    
    # Find the first non-null value
    first_value = series.dropna().iloc[0]
    if first_value != 1:
        return False

    # Check if the values increment by 1 after a few rows
    prev_value = first_value
    for value in series.dropna():
        if value != prev_value:
            if value != prev_value + 1:
                return False
            prev_value = value
    return True


def split_datasets_random(dataset_folder, num_parts):
    data_folder = "Data"
    os.makedirs(data_folder, exist_ok=True)  # Create the data folder if it doesn't exist

    # Iterate over each file in the dataset folder
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(dataset_folder, filename)
            
            # Check if 5 copies of the file already exist
            num_existing_copies = sum(1 for name in os.listdir(data_folder) if name.startswith(os.path.splitext(filename)[0]))
            if num_existing_copies >= num_parts:
                print(f"Already have 5 copies of {filename}, skipping...")
                continue
            
            df = pd.read_csv(filepath)

            # Check if the first column represents an incremental integer index
            first_column = df.iloc[:, 0]
            if is_incremental_integer_column(first_column):
                df.drop(columns=[df.columns[0]], inplace=True)

            # Shuffle the dataset
            df_shuffled = df.sample(frac=1).reset_index(drop=True)

            # Identify the class column
            class_column = df_shuffled.columns[-1]  # Assume the class label is in the last column

            # Split the dataset into two DataFrames based on class
            class_1_df = df_shuffled[df_shuffled[class_column] == 0]
            class_2_df = df_shuffled[df_shuffled[class_column] == 1]

            # Split each class DataFrame into parts
            class_1_parts = [class_1_df.iloc[i::num_parts] for i in range(num_parts)]
            class_2_parts = [class_2_df.iloc[i::num_parts] for i in range(num_parts)]

            # Concatenate corresponding parts from each class and save to CSV
            for part_index, (class_1_part, class_2_part) in enumerate(zip(class_1_parts, class_2_parts)):
                part_df = pd.concat([class_1_part, class_2_part])
                part_filename = f"{os.path.splitext(filename)[0]}_{part_index+1}.csv"
                part_filepath = os.path.join(data_folder, part_filename)
                
                # Save the part to CSV
                part_df.to_csv(part_filepath, index=False)
                print(part_filename, "generated...")

           




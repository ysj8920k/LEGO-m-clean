import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = 'KPI_data'

# List to hold dataframes
df_list = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a dataframe
        df = pd.read_csv(file_path)
        
        # Add a column with the filename
        df['source_file'] = filename
        df['pickup_start']=pd.to_datetime(df['pickup_start'], format='%Y%m%d_%H%M%S', errors='coerce')
        df['pickup_end']=pd.to_datetime(df['pickup_end'], format='%Y%m%d_%H%M%S', errors='coerce')
        df['place_end']=pd.to_datetime(df['place_end'], format='%Y%m%d_%H%M%S', errors='coerce')
        # Append the dataframe to the list
        df_list.append(df)

# Concatenate all dataframes in the list
merged_df = pd.concat(df_list, ignore_index=True)

print("Current column headers:", merged_df.columns)
#merged_df['pickup_start']=pd.to_datetime(df['pickup_start'], format='%Y%m%d_%H%M%S')
#merged_df['pickup_end']=pd.to_datetime(df['pickup_end'], format='%Y%m%d_%H%M%S')
#merged_df['place_end']=pd.to_datetime(df['place_end'], format='%Y%m%d_%H%M%S')
#merged_df['sorting_end']=pd.to_datetime(df['sorting_end'], format='%Y%m%d_%H%M%S')

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_after.csv', index=False)

print("CSV files have been merged successfully into 'merged_output.csv'.")

import json
import os
import pandas as pd

def save_to_jsonl(df, file_path):
    """Save a DataFrame to a JSONL file."""
    df.to_json(file_path, orient='records', lines=True)

input_folder_path = "./datasets/"
file1_path = os.path.join(input_folder_path, "1.parquet")
file2_path = os.path.join(input_folder_path, "2.parquet")

df1 = pd.read_parquet(file1_path)
df2 = pd.read_parquet(file2_path)

merged_df = pd.concat([df1, df2], ignore_index=True)

output_folder_path = "./output/"
metadata_file_path = os.path.join(output_folder_path, "metadata.jsonl")

# Ensure the output directory exists
os.makedirs(output_folder_path, exist_ok=True)

metadata = []
for idx, row in merged_df.iterrows():
    image_file_name = f"{idx:04d}.png"
    image_path = os.path.join(output_folder_path, image_file_name)
    
    image_bytes = bytearray(row['image']['bytes'])
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    metadata_entry = {"file_name": image_file_name, "text": row["text"]}
    metadata.append(metadata_entry)

    if idx == 9:
        break

metadata_df = pd.DataFrame(metadata)
save_to_jsonl(metadata_df, metadata_file_path)

import os
import pandas as pd

def save_to_jsonl(df, file_path):
    """Save a DataFrame to a JSONL file."""
    df.to_json(file_path, orient='records', lines=True)

this_dir = os.path.dirname(__file__)

datasets_dir = os.path.join(this_dir, "../data/datasets/")
file1_path = os.path.join(datasets_dir, "1.parquet")
file2_path = os.path.join(datasets_dir, "2.parquet")

df1 = pd.read_parquet(file1_path)
df2 = pd.read_parquet(file2_path)

output_folder_path = os.path.join(this_dir, "../data/train/")
metadata_file_path = os.path.join(output_folder_path, "metadata.jsonl")

# Merge the two dataframes and save the as HF imagefolder - in fine-tuning, only one text column is needed, but we will save also the text from recaptioning
merged_df = pd.concat([df1, df2], ignore_index=True)

metadata = []
for idx, row in merged_df.iterrows():
    image_file_name = f"{idx:04d}.png"
    image_path = os.path.join(output_folder_path, image_file_name)
    
    image_bytes = bytearray(row['image']['bytes'])
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    metadata_entry = {"file_name": image_file_name, "orig_text": row["text"]}
    metadata.append(metadata_entry)

metadata_df = pd.DataFrame(metadata)
save_to_jsonl(metadata_df, metadata_file_path)

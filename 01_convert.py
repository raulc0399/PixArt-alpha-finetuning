import os
import pandas as pd

import paths

file1_path, file2_path = paths.get_input_files_paths()
output_folder = paths.get_train_folder()
metadata_file_path = paths.get_metadata_file_path()

# df1 = pd.read_parquet(file1_path)
df2 = pd.read_parquet(file2_path)

# Merge the two dataframes and save the as HF imagefolder - in fine-tuning, only one text column is needed, but we will save also the text from recaptioning
merged_df = df2 # pd.concat([df1, df2], ignore_index=True)

metadata = []
for idx, row in merged_df.iterrows():
    image_file_name = f"{idx:04d}.png"
    image_path = os.path.join(output_folder, image_file_name)
    
    image_bytes = bytearray(row['image']['bytes'])
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    metadata_entry = {"file_name": image_file_name, "orig_text": row["text"].removesuffix(", The Simpsons")}
    metadata.append(metadata_entry)

metadata_df = pd.DataFrame(metadata)
metadata_df.to_json(metadata_file_path, orient='records', lines=True)
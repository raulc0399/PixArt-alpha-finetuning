import os

this_dir = os.path.dirname(__file__)
datasets_dir = os.path.join(this_dir, "../data/input/")

def get_input_files_paths():
    file1_path = os.path.join(datasets_dir, "1.parquet")
    file2_path = os.path.join(datasets_dir, "2.parquet")
    
    return file1_path, file2_path

def get_output_folder():
    output_folder = os.path.join(this_dir, "../data/train/")
    
    return output_folder

def get_metadata_file_path(output_folder):
    metadata_file_path = os.path.join(output_folder, "metadata.jsonl")
    
    return metadata_file_path
import os

this_dir = os.path.dirname(__file__)
datasets_dir = os.path.join(this_dir, "../data/input/")
train_folder = os.path.join(this_dir, "../data/train/")

def get_input_files_paths():
    file1_path = os.path.join(datasets_dir, "1.parquet")
    file2_path = os.path.join(datasets_dir, "2.parquet")
    
    return file1_path, file2_path

def get_metadata_file_path():
    train_folder = get_train_folder()
    metadata_file_path = os.path.join(train_folder, "metadata.jsonl")
    
    return metadata_file_path

def get_train_folder():
    return train_folder

def get_transformer_peft_folder():
    peft_folder = os.path.join(this_dir, "pixart-simpsons-model/transformer")
    
    return peft_folder
import os
import shutil
import random
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
full_data = os.getenv("LOG_DATA_PATH")
if not full_data:
    print("WARNING!: LOG_DATA_PATH is not set. This will most likely fail")
# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def copy_portion_of_folders(source_folder, destination_folder, portion=0.5):
    """
    Copy a certain portion of top-level folders from the source folder to the destination folder.

    Parameters:
    - source_folder (str): Path to the source directory containing folders to copy.
    - destination_folder (str): Path to the destination directory where folders will be copied.
    - portion (float): A fraction (0 < portion <= 1) representing the portion of folders to copy.
    """
    # Check if the source directory exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Ensure the destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    # List all top-level folders in the source directory
    all_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    # Calculate the number of folders to copy based on the portion
    number_of_folders_to_copy = int(len(all_folders) * portion)

    # Select a random subset of folders to copy
    folders_to_copy = random.sample(all_folders, number_of_folders_to_copy)

    # Copy each selected folder to the destination directory
    for folder_name in folders_to_copy:
        src_path = os.path.join(source_folder, folder_name)
        dst_path = os.path.join(destination_folder, folder_name)
        shutil.copytree(src_path, dst_path)
        print(f"Copied '{src_path}' to '{dst_path}'")

    print(f"Finished copying {number_of_folders_to_copy} folders to '{destination_folder}'.")

source_folder = os.path.join(full_data, "comp_ws", 
                    "all_data")
destination_folder = os.path.join(full_data, "comp_ws", 
                    "all_data_10_percent")
portion_to_copy = 0.1  # Change this to the portion of folders you want to copy

copy_portion_of_folders(source_folder, destination_folder, portion_to_copy)
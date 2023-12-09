# for file read
import os
import sys


def dir_multi_file(folder_path, file_extension):
    """
    input: directory ptah
    :return: file list
    """
    file_paths = []
    # Ensure the folder path is valid
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return file_paths

    # Iterate through files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has the specified extension
            if file.endswith(f".{file_extension}"):
                file_path = os.path.abspath(os.path.join(root, file))  # file = 0.txt  root=folder_path
                file_paths.append(file_path)

    return file_paths


if __name__ == "__main__":
    a = dir_multi_file("/media/zlh/zhang/earth_rosbag/data/test4/pixel4", "txt")
    print(a)
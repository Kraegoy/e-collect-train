import os

def rename_files(folder_path, start_index):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Loop through files in the folder
    for i, filename in enumerate(files, start=start_index):
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, f"{i}")
        try:
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{old_file_path}' to '{new_file_path}'")
        except FileNotFoundError:
            print(f"File '{old_file_path}' not found.")
        except FileExistsError:
            print(f"File '{new_file_path}' already exists.")

folder_path = r"C:\Users\christine\awa\MP_DATA_latest\HELLO\not hello"
start_index = 0

rename_files(folder_path, start_index)

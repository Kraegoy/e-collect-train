import os

def check_files_in_folders(root_folder):
    # Iterate over each folder in the root folder
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            missing_files = []
            # Check for each expected file
            for i in range(23):  # Check for files 0.npy to 22.npy
                file_path = os.path.join(folder_path, f"{i}.npy")
                if not os.path.exists(file_path):
                    missing_files.append(f"{i}.npy")
            if missing_files:
                print(f"The following files are missing in folder '{folder}':")
                for file_name in missing_files:
                    print(file_name)
                print()  # Add a blank line for readability
                break  # Break out of the loop if missing files are found
            else:
                print(folder)

root_folder = r"C:\Users\christine\awa\MP_DATA\when"
check_files_in_folders(root_folder)

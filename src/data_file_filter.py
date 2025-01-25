import os

folderList = ["data/temperature", "data/biosignals_filtered"]
search_strings = ["-PA1-", "-PA2-", "-PA3-"]

def delete_matching_excel_files(folder_path, search_strings):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.csv')) and any(s in file for s in search_strings):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Fehler beim Löschen von {file_path}: {e}")

for folder in folderList:
    main_folder_path = folder
    print("Current path: " + os.path.abspath(main_folder_path))
    delete_matching_excel_files(main_folder_path, search_strings)



import os
import json

class JSONFileReader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def list_json_files(self):
        json_files = []
        for filename in os.listdir(self.folder_path):
            json_files.append(filename)
        return json_files

    def read_json_file(self, file_name):
        file_path = os.path.join(self.folder_path, file_name)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(e)
            print(f"Error decoding JSON in file: {file_path}")
            return None

    def read_all_json_files(self):
        data = {}
        json_files = self.list_json_files()
        for file_name in json_files:
            file_data = self.read_json_file(file_name)
            if file_data is not None:
                data[file_name] = file_data
        return data

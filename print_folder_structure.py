import os

def print_directory_contents(path, file):
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            file.write(f"directory: {full_path}\n")
            print_directory_contents(full_path, file)  # Recursive call
        else:
            file.write(f"file: {full_path}\n")

with open('repo_structure.txt', 'w') as f:
    print_directory_contents('.', f)

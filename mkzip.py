import os
import shutil


def rename(start_directory, old_filename, new_filename):
    for root, dirs, files in os.walk(start_directory):
        for file in files:
            if file == old_filename:
                # Construct the full path of the file
                old_file_path = os.path.join(root, file)

                # Create the new file path by replacing 'init.py' with '__init__.py'
                new_file_path = os.path.join(root, file.replace(old_filename, new_filename))

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')


if __name__ == "__main__":
    rename('./src', '__init__.py', 'init.py')

    shutil.copytree('./src', './ziptemp/src')
    shutil.make_archive('./src', 'zip', './ziptemp')
    shutil.rmtree('./ziptemp')

    rename('./src', 'init.py', '__init__.py')

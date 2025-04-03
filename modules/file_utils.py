# file_utils
import os
import shutil
from pathlib import Path

def get_file_parts(file_path: str):
    # Split the path into directory and filename
    directory, filename = os.path.split(file_path)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Convert the extension to lowercase
    new_ext = ext.lower()
    return directory, filename, name, ext, new_ext

def rename_file_to_lowercase_extension(file_path: str) -> str:
    """
    Renames a file's extension to lowercase in place.

    Parameters:
        file_path (str): The original file path.

    Returns:
        str: The new file path with the lowercase extension.

    Raises:
        OSError: If there is an error renaming the file (e.g., file not found, permissions issue).
    """
    directory, filename, name, ext, new_ext = get_file_parts(file_path)
    # If the extension changes, rename the file
    if ext != new_ext:
        new_filename = name + new_ext
        new_file_path = os.path.join(directory, new_filename)
        try:
            os.rename(file_path, new_file_path)
            print(f"Rename {file_path} to {new_file_path}\n")
        except Exception as e:
            print(f"os.rename failed: {e}. Falling back to binary copy operation.")
            try:
                # Read the file in binary mode and write it to new_file_path
                with open(file_path, 'rb') as f:
                    data = f.read()
                with open(new_file_path, 'wb') as f:
                    f.write(data)
                    print(f"Copied {file_path} to {new_file_path}\n")
                # Optionally, remove the original file after copying
                #os.remove(file_path)
            except Exception as inner_e:
                print(f"Failed to copy file from {file_path} to {new_file_path}: {inner_e}")
                raise inner_e
        return new_file_path
    else:
        return file_path

def get_filename(file):
    # extract filename from file object
    filename = None
    if file is not None:
        filename = file.name
    return filename

def convert_title_to_filename(title):
    # convert title to filename
    filename = title.lower().replace(" ", "_").replace("/", "_")
    return filename

def get_filename_from_filepath(filepath):
    file_name = os.path.basename(filepath)
    file_base, file_extension = os.path.splitext(file_name)
    return file_base, file_extension

def delete_file(file_path: str) -> None:
    """
    Deletes the specified file.

    Parameters:
        file_path (str): The path to thefile to delete.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If there is an error deleting the file.
    """
    try:
        path = Path(file_path)
        path.unlink()
        print(f"Deleted original file: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")
# pylint: disable=unspecified-encoding,too-many-statements
"""
Listing for practice with pathlib module
"""

import shutil
from pathlib import Path


def main() -> None:
    """
    Entrypoint for a seminar's listing
    """
    # 1. Filesystem basics
    # 1.1. Creating a path
    current_path = Path('.')
    print(current_path)
    print(current_path.resolve())

    # 1.2. Checking a path exists
    print(f'Path {current_path} exists: {current_path.exists()}')

    # 1.3. Checking a path is a file
    if current_path.is_file():
        print(f'{current_path} is a file')
    else:
        print(f'{current_path} is not a file')

    # 1.3. Checking a path is a directory
    if current_path.is_dir():
        print(f'{current_path} is a directory')
    else:
        print(f'{current_path} is not a directory')

    # 1.4. Better way to build scalable paths - is to start from the current Python module
    current_path = Path(__file__)
    print(current_path)

    current_directory_path = current_path.parent
    print(current_directory_path)

    # 1.5. Building paths with a slash
    target_score_path = current_path.parent.parent.parent / 'config' / 'target_score.txt'

    if target_score_path.exists():
        with open(target_score_path) as f:
            print(f.read())
    else:
        print('No such file!')

    # 1.6. Find files by extension
    target_score_path = current_path.parent.parent.parent / 'config'
    for python_file in target_score_path.glob('**/*.py'):
        print(python_file.name)  # no parents
        print(python_file.stem)  # no extension
        print(python_file.suffix)  # just extension

    # 2. Creating directories
    # 2.1. Creating shallow directories
    new_folder_path = Path(__file__).parent / 'new_folder'
    print(f'{new_folder_path} exists: {new_folder_path.exists()}')
    try:
        new_folder_path.mkdir()  # throws an error if it exists already
    except FileExistsError:
        pass
    print(f'{new_folder_path} exists: {new_folder_path.exists()}')

    new_folder_path.mkdir(exist_ok=True)
    new_folder_path.mkdir(exist_ok=True)
    new_folder_path.mkdir(exist_ok=True)

    # 2.2. Creating nested directories
    another_folder_path = Path(__file__).parent / 'new_folder2' / 'another_folder'
    try:
        another_folder_path.mkdir(exist_ok=True)
    except FileNotFoundError:
        print('Unable to create nested directories')

    another_folder_path.mkdir(exist_ok=True, parents=True)

    # 2.3. Creating a file in a created directory
    new_file_path = another_folder_path / 'test.txt'
    with open(new_file_path, 'w') as f:
        f.write('Hello, World!')

    # 2.4 Removing a just created folder
    try:
        another_folder_path.rmdir()
    except OSError:
        print('Unable to remove non-empty folder')

    # 2.4.1 Option 1: remove all files and all its contents
    new_file_path.unlink()
    another_folder_path.rmdir()

    # 2.4.2. Option 2. remove a folder completely
    another_three_folder_path = Path(__file__).parent / 'new_folder3' / 'another_folder'
    another_three_folder_path.mkdir(exist_ok=True, parents=True)
    with open(another_three_folder_path / 'a.txt', 'w') as f:
        f.write('See you')
    shutil.rmtree(another_three_folder_path.parent)

    # cleaning up all previously created artifacts
    shutil.rmtree(another_folder_path.parent)
    shutil.rmtree(new_folder_path)

    # Task 1. Find number of directories in a given folder 'config'
    # Task 1. Find number of Python files in a given folder 'config'
    # Task 1. Calculate number of lines in Python code in a given folder 'config'
    # Task 1. Print the longest name in a given folder 'config'


if __name__ == '__main__':
    main()

# Copyright (c) IFM Lab. All rights reserved.

import subprocess, os

def add_git_submodule(repo_url, submodule_path):
    try:
        subprocess.run(['git', 'submodule', 'add', repo_url, submodule_path], check=True)
        subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], check=True)
        print(f'Successfully added submodule {repo_url} at {submodule_path}')
    except subprocess.CalledProcessError as e:
        print(f'An error occured while adding the submodule {repo_url} at {submodule_path}: {e}')

def submodule_exists(submodule_path):
    if os.path.isfile('.gitmodules'):
        with open('.gitmodules', 'r') as file:
            if submodule_path in file.read():
                return True
    return False
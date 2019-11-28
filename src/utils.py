import os

import tqdm


def create_dir_if_not_found(directory_path):
    """
    Create the given directory if it does not already exist

    Args:
        directory_path: the path to the directory
    """
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

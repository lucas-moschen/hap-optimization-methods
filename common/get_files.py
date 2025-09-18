# -*- coding: utf-8 -*-
"""
Adapted from code originally written by Kendra M. Reiter.
Original source: https://github.com/ReiterKM/msc-thesis-microsimulations
Modifications and further development by Lucas Moschen.

This script contains three functions related to creating and obtaining paths.
"""

import os

def get_file_path(sub_dir: str, 
                  file_words: list, 
                  ignore_file_words: list = None,
                  num_files: int=1):
    """
    Get files containing all sub-strings in file_words, which do not contain any sub-strings of ignore_file_words 
    and is located in sub_dir. Returns a list of file paths.

    Parameters
    ----------
    sub_dir : string
        Sub-directory where the files can be found. It should start one directory level above the current file location.
    file_words : list
        List of sub-strings used to find the files.
    ignore_file_words : list, optional
        List of sub-strings used to ignore files. 
        The default is None.
    num_files : int, optional
        Number of files to find. 
        The default is 1.

    Raises
    ------
    Exception
        If too many / too little / no files are found.

    Returns
    -------
    file_path : list
        List of paths to files found. Length corresponds to num_files.

    """

    # Get path to sub-directory
    path = get_path_to_folder(sub_dir)
    
    # Find the files containing all the sub-strings from file_words in their title
    files = [f for f in os.listdir(path) if all(
        comm in f for comm in file_words)]
    file_paths = [os.path.join(path, f) for f in files]

    # If any sub-string are given in ignore_file_words then removes files containing these sub-strings
    if ignore_file_words is not None:
        file_paths = [f for f in file_paths if
                      all(comm not in f for comm in ignore_file_words)]

    if num_files is None:
        return file_paths

    if len(file_paths) > num_files:
        print(files)
        raise Exception("Too many files found!")
    elif len(file_paths) < num_files:
        raise Exception("Not enough files found!")
    elif len(file_paths) == 0:
        raise Exception("No files found!")

    return file_paths


def get_path_to_folder(sub_dir: str = "data"):
    """
    This function takes the sub-directory of a folder and returns its full path. 

    Parameters
    ----------
    sub_dir : string, optional
        Sub-directory. It should start one directory level above the current file location.
        The default is "data".

    Returns
    -------
    path_to_folder : string
        Full path to the folder, i.e., to the desired sub-directory.
    """

    # Divides sub_dir by "/"
    sub_dir = sub_dir.strip("/")

    # Get the path to current file
    file_path = os.path.dirname(os.path.realpath(__file__))

    # Go up one directory
    dir_path = os.path.dirname(file_path)
    
    # Get directory
    path_to_folder = os.path.join(dir_path, sub_dir)

    if not os.path.exists(path_to_folder):
        raise Exception("The path: " + path_to_folder + " does not exist.")

    return path_to_folder

def create_paths(save_str: str,
                 w_path: str,
                 method_str: str):

    """
    This function creates paths to the final solution of the solution process in relation to households
    and to dwellings.

    Parameters
    ----------
    save_str : str
        String carrying information about the instance considered.
    w_path : str
        Path of the matrix of assignment weights.
    method_str : str
        String containing solution method used along with its parameters.

    Returns
    -------
    solution_path_hhd : str
        Path to the file containing the final solution in relation to households.
    solution_path_dwe : str
        Path to the file containing the final solution in relation to dwellings.
    """

    # Get name of the file containing the matrix of assignment weights
    w_name_with_format = w_path.split("/")[-1]
    w_name = w_name_with_format.replace("." + w_path.split(".")[-1], "")
    
    # Add w_name in save_str
    save_str = save_str + "_" + w_name

    # Add information on the solving method and its parameters
    save_str = save_str + method_str
    
    # Define directory for final solution in relation to households
    solution_path_hhd = get_path_to_folder("data/matching_solutions_households")
    solution_path_hhd = os.path.join(solution_path_hhd, save_str)

    # Define directory for final solution in relation to dwelllings
    solution_path_dwe = get_path_to_folder("data/matching_solutions_dwellings")
    solution_path_dwe = os.path.join(solution_path_dwe, save_str)

    return solution_path_hhd, solution_path_dwe 
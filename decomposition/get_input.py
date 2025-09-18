# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script contains functions to prepare the input data to pass onto the solution process.
The main function is get_input, which is the function called by the module decomposition.py.

    Parameters
    ----------
    data_words : list
        List of substrings used to find the data set files.
    ignore_data_words : list, optional
        The data sets containing any of these sub-strings will be ignored.
        The default is None.
    sub_dir : str, optional
        Optional sub-directory where the data is located. 
        The default is data/datasets.
    specific_coefficient_for_B_k_per : dict, optional
        Each key of this dictionary must be the ID of a grid cell associated to some of Constraints (5.3e) of the MILP formulation
        presented in the thesis, and its respective value must be the proportion of the total dwelling capacity of this grid cell 
        that corresponds to the value B_{k}^{per}.
        The default is None.
    specific_coefficient_for_B_k_hhd : dict, optional
        Each key of this dictionary must be the ID of a grid cell associated to some of Constraints (5.3d) of the MILP formulation
        presented in the thesis, and each respective value must be the proportion of the number of residential dwellings of this 
        grid cell that corresponds to the value B_{k}^{hhd}.
        The default is None.

    Returns
    -------
    household_df : DataFrame
        Household dataframe processed by this script.
    dwelling_df : DataFrame
        Dwelling dataframe processed by this script.
    B_k : dict
        Dictionary containing two keys: "hhd" and "per". The value associated to "hhd" is a dictionary where the keys are 
        indices of grid cells associated to the Constraints (5.3d) and the values are their respective B_{k}^{hhd} values.
        Analogously, the value associated to "per" is a dictionary where the keys are indices of grid cells associated to 
        the Constraints (5.3e) and the values are their espective B_{k}^{per} values. 
    save_comment : str
        String containing information about the name of the municipality or district considered and about the grid cell constraints.
    len_H : int
        Amount of households in the original data set.
    len_D : int
        Amount of dwellings in the original data set.
    idx_to_ids : dict
        Dictionary containing information about all the indexations made in this script.
    d_restri_cell : list
        List of dwelling indices for which the corresponding Gurobi variable must be integer, i.e., indices of dwellings located 
        in constrained grid cells.
"""

from common.get_files import get_path_to_folder, get_file_path
import os
import json
import pandas as pd
import numpy as np
import math 

def create_indices(dataset: pd.DataFrame,
                   column_name: str,
                   elements_be_unique: bool = False):
    
    """
    This function creates indices for the elements contained on a specific column of a dataset.
    It returns the list of indices, a dictionary showing the indexing and a dictionary showing 
    the inverse indexing. 
    """

    if elements_be_unique == True:
        elements_list = list(dataset[column_name].unique())
        indices_list = list(range(len(elements_list)))
        indexing = dict(zip(indices_list, elements_list))
        inverse_indexing = {v: k for k, v in indexing.items()}
    else:
        elements_list = list(dataset[column_name])
        indices_list = list(range(len(elements_list)))
        indexing = dict(zip(indices_list, elements_list))
        inverse_indexing = {v: k for k, v in indexing.items()}

    return indices_list, indexing, inverse_indexing

def create_save_B_k_dicts(df: pd.DataFrame,
                        inverse_indexing_of_grid_cells: dict,
                        name_of_file: str,  
                        specific_coefficient_for_B_k_per: dict = None,
                        specific_coefficient_for_B_k_hhd: dict = None):
    
    """
    This function creates the grid cell constraints as dictionaries {index of grid cell: upper bound on amount of households} and
    {index of grid cell: upper bound on amount of people}. It also creates the string save_comment that contains information about 
    the corresponding numerical experiment and the list d_restri_cell of dwellings located in constrained grid cells.
    """

    # Create B_hhd dict with keys being grid cell IDs
    B_hhd = {}
    if specific_coefficient_for_B_k_hhd is not None:
        for cell in specific_coefficient_for_B_k_hhd.keys():
            B_hhd[cell] = 0
    
    # Create B_per dict with keys being grid cell IDs
    B_per = {}
    if specific_coefficient_for_B_k_per is not None:
        for cell in specific_coefficient_for_B_k_per.keys():
            B_per[cell] = 0

    # Add households capacity for grid cells in B_hhd and people capacity for grid cells in B_per.
    # Also creates list of dwellings so that the corresponding variable must be integer
    d_restri_cell = []
    for d in df.index:

        apply = False 
        grid_cell = df.at[d, "Gitter_ID_100m"]

        if grid_cell in B_hhd:
            B_hhd[grid_cell] += 1
            apply = True 
        if grid_cell in B_per:
            B_per[grid_cell] += df.at[d, "capacity"]
            apply = True 

        if apply == True:
            d_restri_cell.append(d)

    # Create the upper bounds for grid cell constraints considering the reduction coefficients introduced
    if specific_coefficient_for_B_k_hhd is not None:
        for name, value in specific_coefficient_for_B_k_hhd.items():
            B_hhd[name] = int(math.ceil(value * B_hhd[name]))
        
    if specific_coefficient_for_B_k_per is not None:
        for name, value in specific_coefficient_for_B_k_per.items():
            B_per[name] = int(math.ceil(value * B_per[name]))

    # Takes the name of the municipality or district       
    save_comment = name_of_file.split("_")[-1].split(".")[0]    
    
    # Create path to save B_per dictionary
    save_path = get_path_to_folder("data/B_per_files")

    # Add in save_comment information about existence of hhd grid cell constraints
    save_comment = save_comment + "_" + str(len(B_hhd)) + "_hhd_cell_constrs"
    if specific_coefficient_for_B_k_hhd is not None:
        save_comment = save_comment + "_proportion_" + str(next(iter(specific_coefficient_for_B_k_hhd.values())))

    # Add in save_comment information about existence of per grid cell constraints
    save_comment = save_comment + "_" + str(len(B_per)) + "_per_cell_constrs"
    if specific_coefficient_for_B_k_per is not None:
        save_comment = save_comment + "_proportion_" + str(next(iter(specific_coefficient_for_B_k_per.values())))

    # Create path to save B_per dictionary
    save_path = os.path.join(save_path, "B_per_" + save_comment + ".json")

    # Save B_per dictionary
    with open(save_path, "w") as B_per_file:
        json.dump(B_per, B_per_file, default=convert)
    
    # Create path to save B_hhd dictionary
    save_path = get_path_to_folder("data/B_hhd_files")

    # Save B_hhd dictionary
    save_path = os.path.join(save_path, "B_hhd_" + save_comment + ".json")
    with open(save_path, "w") as B_hhd_file:
        json.dump(B_hhd, B_hhd_file, default=convert)

    # Update B_hhd dict to have indices instead of IDs in the keys
    list_keys_B_hhd = list(B_hhd.keys())
    for label in list_keys_B_hhd:
        B_hhd[inverse_indexing_of_grid_cells[label]] = B_hhd[label] 
        del B_hhd[label] 

    # Update B_per dict to have indices instead of IDs in the keys
    list_keys_B_per = list(B_per.keys())
    for label in list_keys_B_per:
        B_per[inverse_indexing_of_grid_cells[label]] = B_per[label] 
        del B_per[label] 

    # Create B_k dict
    B_k = {}
    B_k["hhd"] = B_hhd
    B_k["per"] = B_per

    return B_k, save_comment, d_restri_cell

def create_grid_cell_index_column(df: pd.DataFrame, inverse_indexing_of_grid_cells: dict):

    """This function creates a grid cell index column for the dataframe df."""

    df["grid_cell"] = None 
    for d in df.index:
        df.at[d, "grid_cell"] = inverse_indexing_of_grid_cells[df.at[d, "Gitter_ID_100m"]]

    return df 

def convert(a):

    """ convert to Int64 """

    if isinstance(a, np.int64):
        return int(a)
    raise TypeError

def get_input(data_words: list, 
              ignore_data_words: list = None,
              sub_dir: str = "data/datasets",
              specific_coefficient_for_B_k_per: dict = None,
              specific_coefficient_for_B_k_hhd: dict = None):
    
    """ This function takes the datasets and prepares all the data to be inserted in the solution process. """

    # Initialize a dictionary that will store paths to the data sets
    data_paths_dict = {}
    
    path_to_files = get_file_path(sub_dir,
                                    file_words=data_words,
                                    ignore_file_words=ignore_data_words,
                                    num_files=2)

    # Store paths to household and dwelling data sets
    for path in path_to_files:
        
        name_of_file = path.split("/")[-1].split(".")[0].lower()
        
        type_of_file = ["h" if "hold" in name_of_file
                        else "d" if "house" in name_of_file else None][0]
        if type_of_file:
            data_paths_dict[type_of_file] = path

    # Initialize dictionary idx_to_ids. It will store a match between index (idx) and ID for the dwellings, households and 
    # grid cells. Both a forward (idx : id) and a backward (id : idx) match is saved
    idx_to_ids = {}

    # Get dwelling data
    dwelling_df = pd.read_csv(data_paths_dict["d"])
    dwelling_df = dwelling_df.reset_index(drop=True)

    # Get amount of dwellings in the original data set
    len_D = len(dwelling_df)

    # Indexing the dwellings
    _, idx_to_ids["D"], idx_to_ids["D_inv"] = create_indices(dataset = dwelling_df, column_name = "ID")

    # Indexing the grid cells
    K, idx_to_ids["K"], idx_to_ids["K_inv"] = create_indices(dataset = dwelling_df,
                                                             column_name = "Gitter_ID_100m",
                                                             elements_be_unique = True) 
    
    # Create new grid cell index column in the dwellings dataframe
    dwelling_df = create_grid_cell_index_column(df = dwelling_df, inverse_indexing_of_grid_cells = idx_to_ids["K_inv"])

    # Remove dwellings with null capacity
    dwelling_df.drop(dwelling_df.loc[dwelling_df["capacity"]==0].index, inplace = True)

    # Create the dictionary containing the B_k values, a string carrying imporant information about the experiment and
    # the list of dwelling indices for which the corresponding MILP variables must be integer
    B_k, save_comment, d_restri_cell = create_save_B_k_dicts(df = dwelling_df,
                                                                inverse_indexing_of_grid_cells = idx_to_ids["K_inv"], 
                                                                specific_coefficient_for_B_k_per = specific_coefficient_for_B_k_per,
                                                                specific_coefficient_for_B_k_hhd = specific_coefficient_for_B_k_hhd,
                                                                name_of_file = name_of_file)
    
    # Give name to the dwelling dataframe index column
    dwelling_df.index.name = "D"

    # Get household data
    household_df = pd.read_csv(data_paths_dict["h"])
    household_df = household_df.astype({"size": int})

    # Get amount of households in the original data set
    len_H = len(household_df)

    # Creates indices for households
    _, idx_to_ids["H"], idx_to_ids["H_inv"] = create_indices(dataset = household_df, column_name = "ID")

    # Give name to the household dataframe index column
    household_df.index.name = "H" 

    return household_df, dwelling_df, B_k, save_comment, len_H, len_D, idx_to_ids, d_restri_cell
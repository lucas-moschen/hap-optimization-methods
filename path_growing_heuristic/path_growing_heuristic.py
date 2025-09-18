# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script imports the information about the considered instance of the problem and calls the path-growing heuristic.

    Parameters
    ----------
    data_words : list
        The names of the files that store the household and dwelling data sets must contain all the strings of this list. 
    path_to_w : str
        Directory to the matrix of assignment weights.
    ignore_data_words : list, optional
        List of sub-strings used to ignore household and dwelling data set files.
        The default is None.
    save_path : str, optional
        Path to save the final solution. 
        The default is None.
    sub_dir_data : str, optional
        Sub-directory to the household and dwelling data sets.
        The default is "data/datasets".         
    specific_coefficient_for_B_k_per : dict, optional
        Each key of this dictionary must be the ID of a grid cell associated to some of Constraints (5.2e) of the formulation
        presented in the thesis, and its respective value must be the proportion of the total dwelling capacity of this grid cell 
        that corresponds to the value B_{k}^{per}. As an example, if the value corresponding to a grid cell ID is 0.5, it means that 
        the upper bound for the amount of people that can be allocated into this grid cell is equal to 50% of its total dwelling 
        capacity, i.e., 50% of the sum of capacities of the dwellings located in this grid cell. 
        The default is None.
    specific_coefficient_for_B_k_hhd : dict, optional
        Each key of this dictionary must be the ID of a grid cell associated to some of Constraints (5.2d) of the formulation
        presented in the thesis, and each respective value must be the proportion of the number of residential dwellings of this 
        grid cell that corresponds to the value B_{k}^{hhd}. As an example, if the value corresponding to a grid cell ID is 0.5, it 
        means that the upper bound for the amount of households that can be allocated into this grid cell is equal to 50% of the 
        number of dwellings with non-null capacity located in this grid cell.
        The default is None. 
    verbose : bool, optional
        If True, detailed information about the operation of the path-growing heuristic is provided.
        The default is False

    Returns
    ----------
    None.

"""

import os
import time 
import json
from path_growing_heuristic.get_input import get_input 
from common.get_files import get_file_path, get_path_to_folder
from path_growing_heuristic.path_growing_algorithm import path_growing_algorithm
import numpy as np
import sys 
import psutil

class PATH_GROWING_HEURISTIC:

    def __init__(self,
                    data_words: list,
                    path_to_w: str,
                    ignore_data_words: list = None,
                    save_path: str = None,
                    sub_dir_data = "data/datasets",
                    specific_coefficient_for_B_k_per = None,
                    specific_coefficient_for_B_k_hhd = None,
                    verbose = False):
        
        self.begin = time.time() 

        print("The version of Python used is")
        print(sys.version)
        
        print("\nStep 1: Data input")
        self.household_df, self.dwelling_df, self.B_k, self.file_title, self.len_H, self.len_D, self.idx_to_ids = get_input(data_words = data_words,
                                                                                            ignore_data_words = ignore_data_words,
                                                                                            sub_dir = sub_dir_data,
                                                                                            specific_coefficient_for_B_k_per = specific_coefficient_for_B_k_per,
                                                                                            specific_coefficient_for_B_k_hhd = specific_coefficient_for_B_k_hhd)
        
        self.H = self.household_df.index.tolist()
        self.D = self.dwelling_df.index.tolist()
        self.p_h = self.household_df["size"].values
        print("Done.")

        print("\nStep 2: Information about experiment:")

        print("Number of Dwellings:", self.len_D)
        print("Number of Dwellings with non-null capacity:", len(self.D))
        print("Number of Households:", self.len_H)

        print("\nB_hhd:")
        print(specific_coefficient_for_B_k_hhd)
        print("\nB_per:")
        print(specific_coefficient_for_B_k_per)

        print("\nStep 3: Input the weight matrix")
        # Load the W matrix
        self.begin_w_input = time.time()
        self.w = np.loadtxt(path_to_w, delimiter = ",")
        self.end_w_input = time.time()
        print("Done.")

        # Compute memory used to input W
        mem = psutil.virtual_memory()
        print(f"RAM usage after import weight matrix: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
        
        # Initialize classes
        self.path_to_w = path_to_w
        self.save_path = save_path
        self.verbose = verbose

    def results(self):

        print("\nNumber of households assigned:", self.hhds_assigned_count, "/", self.len_H)

        if self.hhds_assigned_count != self.len_H:
            print('\nThere are unassigned households!')
            print('Number:', self.len_H - self.hhds_assigned_count)
            
        if self.hhds_assigned_count != len(self.D):
            print('\nThere are unassigned dwellings with non-null capacity!')
            print('Number:', len(self.D) - self.hhds_assigned_count)

        print("\nWeight of final matching:", round(self.weight, 4)) 

        print("\nThe time to import the W matrix is", round(self.end_w_input - self.begin_w_input, 2), "seconds.")

    def label_final_solution(self):

        """
        This function creates two dictionaries of final solutions: one in the form 
        {ID of household: [ID of dwelling assigned, weight of assignment]} and the other in the form 
        {ID of dwelling: [ID of household assigned, weight of assignment]}.
        """

        final_solution_hhd = {}
        final_solution_index_dwe = {}
        final_solution_dwe = {}

        # Give the IDs to the final solution and initialize final solution for dwellings
        list_final_sol_keys = list(self.final_solution.keys())
        for h in list_final_sol_keys:
            id_h = self.idx_to_ids["H"][h]
            if self.final_solution[h][0] != "non-assigned":

                # Add element in final_solution_hhd
                d = int(self.final_solution[h][0])
                id_d = self.idx_to_ids["D"][d]
                weight = self.final_solution[h][1]
                final_solution_hhd[id_h] = [id_d, weight]

                # Add element in final_solution_index_dwe
                final_solution_index_dwe[d] = [id_h, weight]
            else:
                # Add element in final_solution_hhd
                final_solution_hhd[id_h] = self.final_solution[h]

            del self.final_solution[h]

        # Finish final_solution_index_dwe by adding the indices of non-assigned dwellings
        dwes_to_add_in_sol = list(set(range(self.len_D)) - set(final_solution_index_dwe.keys()))
        for d in dwes_to_add_in_sol:
            final_solution_index_dwe[d] = ["non-assigned", 0.0]

        # Creates final_solution_dwe which has keys as being dwelling IDs
        list_final_sol_index_dwe_keys = list(final_solution_index_dwe.keys())
        for d in list_final_sol_index_dwe_keys:
            final_solution_dwe[self.idx_to_ids["D"][d]] = final_solution_index_dwe[d]
            del final_solution_index_dwe[d]

        return final_solution_hhd, final_solution_dwe

    def create_save_path(self, dataset_unit: str):

        # Get W name
        w_name_with_format = self.path_to_w.split("/")[-1]
        w_name = w_name_with_format.replace("." + self.path_to_w.split(".")[-1], "")

        self.save_path = get_path_to_folder("data/matching_solutions_" + dataset_unit)
        description = self.file_title + "_" + w_name + "_path_growing_heuristic"

        self.save_path = os.path.join(self.save_path, description)
        self.save_path = self.save_path + ".json"

        return None 
    
    def save(self, dataset_unit: str):

        if dataset_unit == "households":
            final_solution = self.final_solution_hhd
        elif dataset_unit == "dwellings":
            final_solution = self.final_solution_dwe
        
        # Save final solution in a JSON file.
        with open(self.save_path, "w") as f:
            json.dump(final_solution, f)
        print("\nSaved the " + dataset_unit + " final solution at:", self.save_path) 

        return None 

    def optimize(self):

        # Run path-growing heuristic
        self.final_solution, self.weight, self.hhds_assigned_count = path_growing_algorithm(self.w,
                                                                            self.B_k,
                                                                            self.H,
                                                                            self.D,
                                                                            self.p_h,
                                                                            self.dwelling_df, 
                                                                            verbose = self.verbose)

        # Compute and print results
        self.results()

        # Get final solution with household and dwelling IDs  
        self.final_solution_hhd, self.final_solution_dwe = self.label_final_solution()

        # If there is no save_path then creates one automatic directory    
        if self.save_path:

            # Save final solution households in a JSON file
            self.save("households")

            # Save final solution dwellings in a JSON file
            self.save("dwellings")
        else:

            # Save household final solution
            self.create_save_path("households")
            self.save("households")

            # Save dwelling final solution
            self.create_save_path("dwellings")
            self.save("dwellings")  

        end = time.time()

        print("\nOverall run time:", round(end-self.begin, 2), "seconds.") 

        return None 
# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script runs the Lagrangian-relaxation-based approximation method (LRBAM) described in chapter 7 of the thesis.

    Parameters
    ----------
    data_words : list
        The names of the files that store the household and dwelling data sets must contain all the strings of this list. 
    path_to_w : str
        Directory to the matrix of assignment weights.
    ignore_data_words : list, optional
        List of sub-strings used to ignore household and dwelling data set files.
        The default is None.
    initial_lambdas_hhd : float, optional
        Initial value of \lambda_{k}^{hhd} for all k in K^{hhd}.
        The default is 0.075. 
    initial_lambdas_per : float, optional
        Initial value of \lambda_{k}^{per} for all k in K^{per}.
        The default is 0.01.
    increase : float, optional
        Factor used to update the lambda values.
        The default is 1.01.
    postprocessing : str, optional
        Parameter to select the type of post-processing according to Section 7.4.3.
        It can be either a iteration-wise post-processing ("iwpp") or a single post-processing ("spp").
        The default is "spp".
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
        If True, detailed information about the operation of the path-growing algorithm is provided.
        The default is False.

    Returns
    ----------
    None.
"""

import os
import time 
import json
from lrbam.get_input import get_input 
from common.get_files import get_file_path, get_path_to_folder 
from lrbam.path_growing_algorithm import path_growing_algorithm
import random 
import numpy as np
import sys 
import psutil

class LRBAM:

    def __init__(self,
                    data_words: list,
                    path_to_w: str,
                    ignore_data_words: list = None,
                    initial_lambdas_hhd: float = 0.075,
                    initial_lambdas_per: float = 0.01,
                    increase = 1.01,
                    postprocessing: str = "spp",
                    save_path: str = None,
                    sub_dir_data = "data/datasets",
                    specific_coefficient_for_B_k_per = None,
                    specific_coefficient_for_B_k_hhd = None,
                    verbose = False):
        
        self.begin = time.time() 

        print("The version of Python used is")
        print(sys.version)

        # Se random seed to shuffle dwellings in post-processing if LRBAM-SPP is being executed
        if postprocessing == "spp":
            random.seed(42)
        
        print("\nStep 1: Data input")
        self.household_df, self.dwelling_df, self.B_k, self.file_title, self.len_H, self.len_D, self.idx_to_ids, self.s = get_input(data_words = data_words,
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
        self.kappa = np.loadtxt(path_to_w, delimiter = ",")
        self.end_w_input = time.time()
        print("Done.")

        # Compute memory used to input W
        mem = psutil.virtual_memory()
        print(f"RAM usage after import W matrix: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")

        # ------- Defining lambdas -------
        print("\nStep 4: Define lambdas")

        self.lambdas_hhd = {}
        self.lambdas_per = {}

        # Initialize Lagrangian multipliers
        if specific_coefficient_for_B_k_hhd is not None:
            for cell in specific_coefficient_for_B_k_hhd.keys():
                self.lambdas_hhd[self.idx_to_ids["K_inv"][cell]] = [0, initial_lambdas_hhd]

        if specific_coefficient_for_B_k_per is not None:
            for cell in specific_coefficient_for_B_k_per.keys():
                self.lambdas_per[self.idx_to_ids["K_inv"][cell]] = [0, initial_lambdas_per] 
        
        # Initialize classes
        self.path_to_w = path_to_w
        self.initial_lambdas_hhd = initial_lambdas_hhd
        self.initial_lambdas_per = initial_lambdas_per
        self.increase = increase
        self.postprocessing = postprocessing
        self.save_path = save_path
        self.verbose = verbose
        self.algo_history_hhd = []
        self.algo_history_dwe = [] 
        self.violated_cells_hhd = list(self.lambdas_hhd.keys())
        self.violated_cells_per = list(self.lambdas_per.keys())
        self.path_gr_time_total = 0

    def update_kappa(self):

        """This function updates the necessary kappa values for the next Lagrangian relaxation."""

        if self.postprocessing == "spp":

            # Verify if we should change the weight of each edge according to the lambda values
            for k in self.lambdas_hhd.keys():
                if k in self.violated_cells_hhd:

                    # Get slice of self.kappa coresponding to dwellings from this grid cell
                    slice_kappa = self.kappa[:, self.s[k]]

                    # Identify feasible assignments
                    non_null = slice_kappa != 0

                    # Update weights of feasible assignments
                    slice_kappa[non_null] += -(self.lambdas_hhd[k][1] - self.lambdas_hhd[k][0])

                    # Identify if there is a feasible assignment that after the update becomes exactly zero
                    # If it is the case, then set a simbolic value to this weight so that the null weights keep 
                    # representing only the infeasible assignments
                    null_after_update = (slice_kappa == 0) & non_null
                    slice_kappa[null_after_update] = -0.0000001

                    # Update self.kappa
                    self.kappa[:, self.s[k]] = slice_kappa

            for k in self.lambdas_per.keys():
                if k in self.violated_cells_per:

                    # Get slice of self.kappa coresponding to dwellings from this grid cell
                    slice_kappa = self.kappa[:, self.s[k]]

                    # Identify feasible assignments
                    non_null = slice_kappa != 0

                    # Update weights of feasible assignments
                    for h in self.H:
                        slice_kappa[h, non_null[h]] += -((self.lambdas_per[k][1] - self.lambdas_per[k][0]) * self.p_h[h])

                    # Identify if there is a feasible assignment that after the update becomes exactly zero
                    # If it is the case, then set a simbolic value to this weight so that the null weights keep 
                    # representing only the infeasible assignments
                    null_after_update = (slice_kappa == 0) & non_null
                    slice_kappa[null_after_update] = -0.0000001

                    # Update self.kappa
                    self.kappa[:, self.s[k]] = slice_kappa
        elif self.postprocessing == "iwpp":

            # Verify if we should change the weight of each edge according to the lambda values
            for k in self.lambdas_hhd.keys():
                if k in self.violated_cells_hhd:

                    # Get slice of self.kappa coresponding to dwellings from this grid cell
                    slice_kappa = self.kappa[:, self.s[k]]

                    # Identify feasible assignments that still having positive weight
                    positive = slice_kappa > 0

                    # Update weights of feasible assignments that still having positive weight
                    slice_kappa[positive] += -(self.lambdas_hhd[k][1] - self.lambdas_hhd[k][0])

                    # Update self.kappa
                    self.kappa[:, self.s[k]] = slice_kappa

            for k in self.lambdas_per.keys():
                if k in self.violated_cells_per:

                    # Get slice of self.kappa coresponding to dwellings from this grid cell
                    slice_kappa = self.kappa[:, self.s[k]]

                    # Identify feasible assignments that still having positive weight
                    positive = slice_kappa > 0

                    # Update weights of feasible assignments that still having positive weight
                    for h in self.H:
                        slice_kappa[h, positive[h]] += -((self.lambdas_per[k][1] - self.lambdas_per[k][0]) * self.p_h[h])

                    # Update self.kappa
                    self.kappa[:, self.s[k]] = slice_kappa

    def verifying_cell_constrs(self):

        """This function verifies violations in the side constraints and updates the lambda values accordingly."""

        self.dwes_violated_cells = []
        self.violated_cells_hhd = []
        self.violated_cells_per = []

        for k in self.lambdas_hhd:

            # Compute number of households allocated into grid cell k
            expr_B_hh = self.matching[:, self.s[k]].sum()

            if self.B_k["hhd"][k] < expr_B_hh:
                self.violated_cells_hhd.append(k)
                self.violation = True
                print("The amount of households allocated on the grid cell", k, "is greater than the B_k bound. The amount is", expr_B_hh, "while the value of B_k is", self.B_k["hhd"][k] ) 
                self.lambdas_hhd[k][0] = self.lambdas_hhd[k][1]
                self.lambdas_hhd[k][1] = self.increase * self.lambdas_hhd[k][1]
                self.dwes_violated_cells += self.s[k]

        for k in self.lambdas_per:

            # Compute number of people allocated into grid cell k
            expr_B_per = (self.p_h @ self.matching[:, self.s[k]]).sum()
        
            if self.B_k["per"][k] < expr_B_per:
                self.violated_cells_per.append(k)
                self.violation = True 
                print("The amount of population allocated on the grid cell", k, "is greater than the B_k bound. The amount is", expr_B_per, "while the value of B_k is", self.B_k["per"][k] )
                self.lambdas_per[k][0] = self.lambdas_per[k][1]
                self.lambdas_per[k][1] = self.increase * self.lambdas_per[k][1]
                self.dwes_violated_cells += self.s[k]

        self.dwes_violated_cells = list(set(self.dwes_violated_cells))

    def get_warm_start(self):

        """This function gets warm start for the next Lagrangian relaxation."""

        # Get position of last dwelling not contained in a violated grid cell
        position_last_d_not_in_violated_cell = -1
        for d in self.algo_history_dwe:
            if d in self.dwes_violated_cells:
                break
            position_last_d_not_in_violated_cell += 1

        # Cut history of dwellings processed by the path-growing algorithm
        if position_last_d_not_in_violated_cell == -1:
            self.algo_history_dwe = []
        else:
            self.algo_history_dwe = self.algo_history_dwe[:position_last_d_not_in_violated_cell + 1]

        # Cut history of households processed by the path-growing algorithm
        length_algo_history_dwe = len(self.algo_history_dwe)
        if length_algo_history_dwe == 0:
            self.algo_history_hhd = []
        else:
            self.algo_history_hhd = self.algo_history_hhd[:length_algo_history_dwe + 1]

    def compute_final_solution(self):

        self.weight = 0
        self.final_solution  = {} 

        # Get list of assigned households and assigned dwellings and loop over it
        hhd_assigned, dwe_assigned = self.matching.nonzero()
        for h, d in zip(hhd_assigned, dwe_assigned):

            # Get grid cell index
            k = self.dwelling_df.at[d, "grid_cell"]

            # Get original weight value
            w_val = self.kappa[h,d]
            if k in self.lambdas_hhd.keys():
                w_val = w_val + self.lambdas_hhd[k][1]
            if k in self.lambdas_per.keys():
                w_val = w_val + (self.lambdas_per[k][1] * self.household_df.at[h, "size"])

            # Update final solution and weight
            w_val = round(w_val, 4) 
            self.final_solution[h] = [str(d), w_val]
            self.weight += w_val 

            # If LRBAM-SPP is running, update the grid cell upper bounds if some of the grid cell constraints is activated 
            # for this grid cell.
            if self.postprocessing == "spp":
                if k in self.B_k["hhd"].keys():
                    self.B_k["hhd"][k] = self.B_k["hhd"][k] - 1
                if k in self.B_k["per"].keys():
                    self.B_k["per"][k] = self.B_k["per"][k] - self.household_df.at[h, "size"]

        # Put into solution the information about non-assigned households
        for h in list(set(self.H) - set(hhd_assigned)):
            self.final_solution[h] = ["non-assigned", 0.0]

    def get_scm_matching(self):

        """This function runs the post-processing for the LRBAM-SPP (see Section 7.4.3 of the thesis)."""

        # Get list of non-assigned households and dwellings
        free_hhd = list(set(self.H) - set(self.matching.nonzero()[0]))
        free_dwe = list(set(self.D) - set(self.matching.nonzero()[1]))

        # Sort free_hhd in decreasing household size order
        free_hhd = sorted(free_hhd, key=lambda x: self.household_df.loc[x, 'size'], reverse = True)

        # Shuffle dwellings to avoid direct address filling
        random.shuffle(free_dwe)

        # Get back the original weight values
        for k in list(self.lambdas_hhd.keys()):

            # Get free dwellings in this grid cell
            free_dwe_cell = list(set(self.s[k]) & set(free_dwe))

            # Get slice of self.kappa coresponding to free households and the free dwellings from this grid cell
            slice_kappa = self.kappa[np.ix_(free_hhd, free_dwe_cell)]

            # Identify feasible assignments
            non_null = slice_kappa != 0

            # Update weights of feasible assignments
            slice_kappa[non_null] += self.lambdas_hhd[k][1]

            # Identify if there is a feasible assignment that after the update becomes exactly zero
            # If it is the case, then set a simbolic value to this weight so that the null weights keep 
            # representing only the infeasible assignments
            null_after_update = (slice_kappa == 0) & non_null
            slice_kappa[null_after_update] = 0.0000001

            # Update self.kappa
            self.kappa[np.ix_(free_hhd, free_dwe_cell)] = slice_kappa

            del slice_kappa

        for k in list(self.lambdas_per.keys()):

            # Get free dwellings in this grid cell
            free_dwe_cell = list(set(self.s[k]) & set(free_dwe))

            # Get slice of self.kappa coresponding to free households and the free dwellings from this grid cell
            slice_kappa = self.kappa[np.ix_(free_hhd, free_dwe_cell)]

            # Identify feasible assignments
            non_null = slice_kappa != 0

            for i in range(slice_kappa.shape[0]):
                slice_kappa[i, non_null[i]] += (self.lambdas_per[k][1] * self.p_h[free_hhd[i]])

            # Identify if there is a feasible assignment that after the update becomes exactly zero
            # If it is the case, then set a simbolic value to this weight so that the null weights keep 
            # representing only the infeasible assignments
            null_after_update = (slice_kappa == 0) & non_null
            slice_kappa[null_after_update] = 0.0000001

            # Update self.kappa
            self.kappa[np.ix_(free_hhd, free_dwe_cell)] = slice_kappa

            del slice_kappa 

        # Make assignments
        for h in free_hhd:

            best_dwe = None 
            best_weight = 0

            for d in free_dwe:

                # Get grid cell index of this dwelling
                k = self.dwelling_df.at[d, "grid_cell"]

                # Check if this grid cell has constraints. If so, verify if they hold
                if k in self.B_k["hhd"].keys():
                    if self.B_k["hhd"][k] < 1:
                        continue 
                if k in self.B_k["per"].keys():
                    if self.B_k["per"][k] < self.household_df.at[h, "size"]:
                        continue

                if self.kappa[h, d] > best_weight:
                    best_dwe = d 
                    best_weight = self.kappa[h, d]
            
            # Update final solution and final weight
            if best_dwe is not None:
                
                assignmt_weight = round(self.kappa[h, best_dwe], 4)
                self.final_solution[h] = [str(best_dwe), assignmt_weight]
                self.weight += assignmt_weight
                self.hhds_assigned_count += 1
                free_dwe.remove(best_dwe)
                
                # Update grid cell constraints if necessary
                k = self.dwelling_df.at[best_dwe, "grid_cell"]
                if k in self.B_k["hhd"].keys():
                    self.B_k["hhd"][k] -= 1 
                if k in self.B_k["per"].keys():
                    self.B_k["per"][k] -= self.household_df.at[h, "size"]

    def results(self):

        print("\nNumber of households assigned:", self.hhds_assigned_count, "/", self.len_H)

        if self.hhds_assigned_count != self.len_H:
            print('\nThere are non-assigned households!')
            print('Number:', self.len_H - self.hhds_assigned_count)
            
        if self.hhds_assigned_count != len(self.D):
            print('\nThere are non-assigned dwellings with non-null capacity!')
            print('Number:', len(self.D) - self.hhds_assigned_count)

        print("\nWeight of final matching:", round(self.weight, 4)) 

        print("\nThe time to import the W matrix is", round(self.end_w_input - self.begin_w_input, 2), "seconds.")
        print("The total time used by the path-growing algorithm is", round(self.path_gr_time_total, 2), "seconds.")
        print("Total time of graph solving processes:", round(self.opt_time, 2), "seconds.")
        print("The number of problems solved is", self.number_problems) 

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
        description = self.file_title + "_" + w_name + "_lrbam_" + self.postprocessing
        description = description + "_lambdas_hhd_initial_" + str(self.initial_lambdas_hhd)
        description = description + "_lambdas_per_initial_" + str(self.initial_lambdas_per)
        description = description + "_increasing_" + str(self.increase)

        self.save_path = os.path.join(self.save_path, description)
        self.save_path = self.save_path + ".json"

        return None 
    
    def save(self, dataset_unit: str):

        if dataset_unit == "households":
            final_solution = self.final_solution_hhd
        elif dataset_unit == "dwellings":
            final_solution = self.final_solution_dwe
        
        # Save final solution in a JSON file
        with open(self.save_path, "w") as f:
            json.dump(final_solution, f)
        print("\nSaved the " + dataset_unit + " final solution at:", self.save_path) 

        return None 

    def optimize(self):

        """This function runs the LRBAM-SPP and LRBAM-IWPP (see Section 7.4.3 of the thesis)."""

        # Initialize number of Lagrangian relaxations tackled and optimizing time
        self.number_problems = 1 
        self.opt_time = 0

        # Initialize loop of the algorithm
        while True:

            # Initialize iteration time
            it_time_begin = time.time()

            print("\nModel number", self.number_problems) 

            # Set flag for violation of the side constraints
            self.violation = False  

            print("\nThe lambda_hhd values are", self.lambdas_hhd)
            print("The lambda_per values are", self.lambdas_per)

            print("\nStep 5: Update kappa matrix for the next graph")
            self.update_kappa() 

            print("\nStep 6: Obtain a matching in the graph")
            begin_opt = time.time()
            self.matching, self.weight_penal, self.algo_history_hhd, self.algo_history_dwe, path_gr_time, self.hhds_assigned_count = path_growing_algorithm(postprocessing = self.postprocessing,
                                                                                                                                                            H = self.H,
                                                                                                                                                            D = self.D,
                                                                                                                                                            kappa = self.kappa, 
                                                                                                                                                            verbose = self.verbose,
                                                                                                                                                            warm_start_hhd = self.algo_history_hhd,
                                                                                                                                                            warm_start_dwe = self.algo_history_dwe)
            end_opt = time.time() 

            # Update times
            self.opt_time += end_opt-begin_opt 
            self.path_gr_time_total += path_gr_time 
            print("Time used to find a matching in this model:", round(end_opt-begin_opt, 2), "seconds.") 

            print("\nStep 7: Check violations in grid cell constraints") 
            self.verifying_cell_constrs()

            if self.violation == False:
                print("There is no violation!")

                it_time_end = time.time()
                print("\nThe time used by this iteration is", round(it_time_end - it_time_begin, 2), "seconds.")

                break 
            else:
                print("\nStep 8: Get warm start for next iteration")
                self.get_warm_start()
                self.number_problems += 1

                it_time_end = time.time()
                print("\nThe time used by this iteration is", round(it_time_end - it_time_begin, 2), "seconds.")

        print("\nComputing final solution...")
        self.compute_final_solution()

        if self.postprocessing == "spp":
            print("\nExecuting postprocessing for an SCM matching...")
            begin_scm_time = time.time()
            self.get_scm_matching() 
            end_scm_time = time.time()
            print("\nThe time used by the postprocessing for SCM matching is", round(end_scm_time - begin_scm_time, 2), "seconds.")

        # Compute and print results
        self.results()

        # Get final solution with IDs in relation to households and to dwellings  
        self.final_solution_hhd, self.final_solution_dwe = self.label_final_solution()

        # If there is no save_path then creates one automatic directory    
        if self.save_path:

            # Save final solution households in a JSON file
            self.save("households")

            # Save final solution dwellings in a JSON file
            self.save("dwellings")
        else:

            # Save households final solution
            self.create_save_path("households")
            self.save("households")

            # Save dwellings final solution
            self.create_save_path("dwellings")
            self.save("dwellings")  

        end = time.time()

        print("\nOverall run time:", round(end-self.begin, 2), "seconds.") 

        return None 
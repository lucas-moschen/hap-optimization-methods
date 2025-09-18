# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script runs the decomposition approach explained in Chapter 6 of the thesis.

    Parameters
    ----------
    data_words : list
      The names of the files that store the household and dwelling data sets must contain all the strings of this list. 
    path_to_w : str
      Directory to the matrix of assignment weights.
    ignore_data_words : list, optional
      List of sub-strings used to ignore household and dwelling data set files.
      The default is None.
    sub_dir_data : str, optional
      Sub-directory to the household and dwelling data sets.
      The default is "data/datasets".         
    specific_coefficient_for_B_k_per : dict, optional
      Each key of this dictionary must be the ID of a grid cell associated to some of Constraints (5.3e) of the MILP formulation
      presented in the thesis, and its respective value must be the proportion of the total dwelling capacity of this grid cell 
      that corresponds to the value B_{k}^{per}. As an example, if the value corresponding to a grid cell ID is 0.5, it means that 
      the upper bound for the amount of people that can be allocated into this grid cell is equal to 50% of its total dwelling 
      capacity, i.e., 50% of the sum of capacities of the dwellings located in this grid cell. 
      The default is None.
    specific_coefficient_for_B_k_hhd : dict, optional
      Each key of this dictionary must be the ID of a grid cell associated to some of Constraints (5.3d) of the MILP formulation
      presented in the thesis, and each respective value must be the proportion of the number of residential dwellings of this 
      grid cell that corresponds to the value B_{k}^{hhd}. As an example, if the value corresponding to a grid cell ID is 0.5, it 
      means that the upper bound for the amount of households that can be allocated into this grid cell is equal to 50% of the 
      number of dwellings with non-null capacity located in this grid cell.
      The default is None. 
    gurobi_verbose : bool, optional
      If True, then Gurobi prints information about the solution process of the MILP problems.
      The default is False 
    gurobi_parameters : dict, optional
      Additional parameters to be passed to the Gurobi model as {name : value}. 
      The default is {"MIPGap": 0.005}
    hhd_size_decomposition : bool, optional
      If true, the decomposition by household size (Algorithm 1 of the thesis) is executed if needed.
      The default is True.   
    amount_to_decompose : int, optional
      If the amount of dwellings is greater than this parameter, the regional decomposition (Algorithm 2 of the thesis) 
      must occurs. 
      The default is 2500.
    alpha : float, optional
      If an assignment made in the regional decomposition has weight greater than or equal to this parameter, then it is accepted 
      for the final solution.
      The default is 0.97.
    beta : float, optional
      In the regional decomposition, if the number of assignments with weight sufficiently high is not bigger than assignments_tol, 
      then alpha is reduced by this value.
      The default is 0.05. 
    gamma : float, optional
      In the regional decomposition, if alpha is not bigger than this value then it is set to zero along with assignments_tol.
      The default is 0.3.  
    assignments_tol : int, optional
      In the regional decomposition, if the amount of assignments accepted to the final solution is smaller than or equal 
      to this value, then alpha is reduced.
      The default is 100.

    Returns
    ----------
    None.
"""

from decomposition.milp_formulation import MILPFormulation 
from decomposition.ilp_formulation import PostprocessingNonInteger
from common.get_files import get_file_path, create_paths  
from decomposition.get_input import get_input
from decomposition.milp_postprocessing import milp_postprocessing
from decomposition.divide_region import DivideRegion 
from decomposition.divide_households import divide_hhds
import time 
import json
import copy 
import pandas as pd
import numpy as np
import sys
import psutil 

class Test: 
  
  def __init__(self,
                data_words: list,
                path_to_w: str, 
                ignore_data_words: list = None,
                sub_dir_data: str = "data/datasets",                                                                
                specific_coefficient_for_B_k_per: dict = None,
                specific_coefficient_for_B_k_hhd: dict = None, 
                gurobi_verbose: bool = False,                                                                                                                                                                                                                                                          
                gurobi_parameters: dict = {"MIPGap": 0.005},
                hhd_size_decomposition: bool = True,                              
                amount_to_decompose: int = 2500,                                 
                alpha: float = 0.97,
                beta: float = 0.05,
                gamma: float = 0.3,
                assignments_tol: int = 100):                                     
               
    begin4 = time.time()

    print("The version of Python used is")
    print(sys.version)
    
    # Get input data
    self.household_df, self.dwelling_df, self.B_k, save_str, self.len_H, self.len_D, self.idx_to_ids, self.d_restri_cell = get_input(data_words,
                                                                                                                            ignore_data_words,
                                                                                                                            sub_dir_data,
                                                                                                                            specific_coefficient_for_B_k_per,
                                                                                                                            specific_coefficient_for_B_k_hhd)
    
    # Set method_str
    if len(self.dwelling_df) <= amount_to_decompose:
      method_str = "_exact_solution"
    elif hhd_size_decomposition:
      method_str = "_hhd_size_decomp_alpha_" + str(alpha) + "_beta_" + str(beta) + "_gamma_" + str(gamma) + "_Dbar_" + str(amount_to_decompose)
    else:
      method_str = "_regional_decomp_alpha_" + str(alpha) + "_beta_" + str(beta) + "_gamma_" + str(gamma) + "_Dbar_" + str(amount_to_decompose) + "_Amin_" + str(assignments_tol)
    
    # Load the W matrix
    print("\nLoading the W matrix...")
    self.begin_w_input = time.time()
    self.w = np.loadtxt(path_to_w, delimiter = ",")
    self.end_w_input = time.time()
    print("The W matrix was loaded.")

    # Compute memory used to input W
    mem = psutil.virtual_memory()
    print(f"RAM usage after import W matrix: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
    
    # Generate paths to the final solutions
    self.solution_path_hhd, self.solution_path_dwe = create_paths(save_str = save_str,
                                                                  w_path = path_to_w,
                                                                  method_str = method_str)
    
    # The application of gurobi_verbose
    if gurobi_verbose == False:
      self.gurobi_parameters = gurobi_parameters
      self.gurobi_parameters["OutputFlag"] = 0
    else:
      self.gurobi_parameters = gurobi_parameters
    
    # Initialize the classes
    self.hhd_size_decomposition = hhd_size_decomposition
    self.final_solution = {}
    self.build_time_final = 0
    self.optimizing_time_final = 0
    self.amount_to_decompose = amount_to_decompose
    self.alpha = alpha
    self.alpha_init = alpha
    self.beta = beta
    self.gamma = gamma
    self.assignments_tol = assignments_tol
    self.assignments_tol_init = assignments_tol

    # Measure time spent in this function   
    end4 = time.time()
    self.extra_time = end4 - begin4

  def define_input_to_be_used(self):

    """
    If decomposition by household size is being done, this function defines the input dataframes for the MILP formulation 
    based on the current household size being used. To do so, it separates the current main input dataframe into two: the input 
    dataframe that will be used in the current decomposition iteration and the remaining input dataframe.
    """

    # Create the household dataframe of the round
    self.round_household_df = self.household_df.loc[self.household_df["size"] == self.hhd_size].copy()

    # Get the household indices of the round
    round_hhd_indices = self.round_household_df.index.values

    # Get feasible dwellings
    feasible_dwes = np.any(self.w[round_hhd_indices, :] != 0, axis=0)
    feasible_dwes_indices = np.where(feasible_dwes)[0]
    feasible_dwes_indices = feasible_dwes_indices.tolist()

    # Create the dwelling dataframe of the round
    self.round_dwelling_df = self.dwelling_df.loc[self.dwelling_df.index.isin(feasible_dwes_indices)].copy()

    # Remove the dataframes for the decomposition from the main ones
    self.household_df.drop(self.round_household_df.index.tolist(), inplace = True) 
    self.dwelling_df.drop(self.round_dwelling_df.index.tolist(), inplace = True)      
  
  def optimize_milp(self, 
                   dwe_df: pd.DataFrame,
                   hhd_df: pd.DataFrame): 
    
    """
    This function calls the module milp_formulation.py to build and optimizes the MILP Formulation. If the optimal solution 
    found by Gurobi is non-integer, then the module milp_postprocessing.py is called to round the solution to an optimal integer 
    one (which exists due to Theorem 5 of the thesis).
    If this postprocessing exceeds 100 iterations then we solve an integer Gurobi model corresponding to the part of the problem 
    which generated non-integer values in the solution to finally obtain a full integer solution (see Section 6.4 of the 
    thesis). 
    """

    # Check if dataframes are non-empty
    if (not hhd_df.empty) and (not dwe_df.empty):

      # Create instance of MILP problem
      formulation = MILPFormulation(dwe_df = dwe_df,
                                    hhd_df = hhd_df,
                                    B_k = self.B_k,
                                    gurobi_parameters = self.gurobi_parameters,
                                    d_restri_cell = self.d_restri_cell)

      # Optimize MILP problem
      self.x, build_time, optimizing_time, non_int_assignments, non_int_dwes, non_int_obj_val = formulation.optimize(w = self.w)

      # Check if solution from MILP is integer. If not, then apply MILP postprocessing step
      if len(non_int_dwes) > 0:

        print("\nThe MILP formulation did not returns an integer solution. A postprocessing in the optimal facet is necessary.")

        # Run MILP postprocessing
        self.x, optimizing_time, opt_not_found = milp_postprocessing(self.w,
                                                                          self.x, 
                                                                          non_int_assignments,
                                                                          optimizing_time,
                                                                          non_int_dwes,
                                                                          non_int_obj_val,
                                                                          len(hhd_df),
                                                                          len(dwe_df),
                                                                          100)  

        # If the above MILP postprocessing exceeded the limit of iterations then transform the remaining problem into a 
        # Gurobi integer model to get a full integer solution  
        if opt_not_found:

          print("\nThe non-integer values of the solution will be transformed into integer through the solution of an integer")
          print("Gurobi model corresponding to the non-integer part of the current solution.")

          # Creates instance of ILP problem.
          formulation = PostprocessingNonInteger(self.gurobi_parameters,
                                                build_time, 
                                                optimizing_time)
        
          # Update solution and times after the solution of integer Gurobi model
          self.x, build_time, optimizing_time = formulation.postprocessing_non_int(self.w,
                                                                                  self.x, 
                                                                                  non_int_assignments,
                                                                                  len(dwe_df))    
    
      # Register build and optimizing time
      self.build_time_final += build_time 
      self.optimizing_time_final += optimizing_time
    elif (not hhd_df.empty) and (dwe_df.empty):

      # In the case where there is no dwelling but there are households, pass all households in hhd_df as 
      # non-assigned in solution self.x
      self.x = {}
      for h in hhd_df.index.tolist():

        self.x[h] = ["non-assigned", 0.0]
    else:

      # In the case where there is no household then the solution dictionary is empty
      self.x = {}

  def create_sub_problems(self):

    """This function creates the sub-problems if regional decomposition is needed.""" 

    # Initialize the list of input dataframes for each sub-problem  
    amount_dwe_in_round = len(self.round_dwelling_df)  
    list_dwellings_dataframes = [self.round_dwelling_df] 

    # Use the module divide_region.py to extend list_dwellings_dataframes until the point where each dataframe has an amount 
    # of dwellings lower than amount_to_decompose   
    while any(len(input_dataframe) > self.amount_to_decompose for input_dataframe in list_dwellings_dataframes):

      # Create indices list of dataframes that can be removed from list_dwellings_dataframes because they were divided
      index_to_remove_list = []
      index = 0

      for input_dataframe in list_dwellings_dataframes:

        # Verify if we need to divide the region represented by this dataframe
        if len(input_dataframe) > self.amount_to_decompose:
            
            divide = DivideRegion(dict_indices_K = self.idx_to_ids["K"], 
                                  dataframe_to_be_divided = input_dataframe)
            dataframe_1, dataframe_2 = divide.run() #divide_region 

            # Add the 2 new dataframes to the list
            list_dwellings_dataframes.append(dataframe_1) 
            list_dwellings_dataframes.append(dataframe_2) 

            # Update list of too big dataframes that will be removed after the loop
            index_to_remove_list.append(index)

        index += 1

      # Remove the too big regions that were already divided
      # The reverse parameter here is important due to the dynamic modification of indices during the loop 
      for i in sorted(index_to_remove_list, reverse = True):
        list_dwellings_dataframes.pop(i)

    # Distribution of households among the sub-regions
    list_households_dataframes, build_time, opt_time  = divide_hhds(hhd_df = self.round_household_df,
                                                                    list_of_subregions = list_dwellings_dataframes,
                                                                    w = self.w,
                                                                    amount_dwe_in_round = amount_dwe_in_round) #divide_households

    self.build_time_final += build_time 
    self.optimizing_time_final += opt_time 

    return list_households_dataframes, list_dwellings_dataframes
     
    
  def assignments_quality_verification(self):

    """
    This function checks if the weight of each assignment made in the regional decomposition approach is sufficiently large 
    to be accepted in the final solution.
    For each assignment accepted, the final solution and the upper bounds B_k are updated.
    """

    amount_assignments_made = 0

    for h, val in self.x.items():
      if val[0] != "non-assigned":
        if val[1] >= self.alpha: 

          # Update final solution
          self.final_solution[h] = copy.deepcopy(val) 

          amount_assignments_made += 1 
                      
          # Find the grid cell of this dwelling
          k = self.round_dwelling_df.at[int(val[0]), "grid_cell"]
                      
          # Update the grid cell upper bounds if some of the grid cell constraints is activated for this grid cell
          if k in self.B_k["hhd"].keys():
            self.B_k["hhd"][k] = self.B_k["hhd"][k] - 1
          if k in self.B_k["per"].keys():
            self.B_k["per"][k] = self.B_k["per"][k] - self.round_household_df.at[h, "size"]
                      
          # Update the dataframes of the iteration
          self.round_dwelling_df.drop([int(val[0])], inplace = True)
          self.round_household_df.drop([h], inplace = True)

    print("\nAmount of household assignments accepted in the quality verification step: " + str(amount_assignments_made))

    return amount_assignments_made

  def update_sol_and_Bk(self, household_df, dwelling_df):

    """
    This function is used to update the solution and the upper bounds B_k when assignments_quality_verification 
    is not required.
    """

    for h, val in self.x.items():
      if val[0] == "non-assigned":
         
         # Update final solution
         self.final_solution[h] = copy.deepcopy(val)
      else:    

        # Update final solution
        self.final_solution[h] = copy.deepcopy(val)  
                      
        # Find the grid cell of this dwelling
        k = dwelling_df.at[int(val[0]), "grid_cell"]
                    
        # Update the grid cell upper bounds if some of the grid cell constraints is activated for this grid cell
        if k in self.B_k["hhd"].keys():
          self.B_k["hhd"][k] = self.B_k["hhd"][k] - 1
        if k in self.B_k["per"].keys():
          self.B_k["per"][k] = self.B_k["per"][k] - household_df.at[h, "size"]

        # Update the round dataframes
        dwelling_df.drop(int(val[0]), inplace = True)
        household_df.drop(h, inplace = True)
     
  def update_main_dwelling_dataframe(self):

    """This function updates the main dwelling dataframe after each round of the decomposition by household size."""

    self.dwelling_df = pd.merge(left = self.dwelling_df, right = self.round_dwelling_df, how="outer",
                                on = self.dwelling_df.index.names + self.dwelling_df.columns.tolist(), sort = True)
          
  def save(self):

    """This function saves the two final solutions."""

    # Households final solution
    self.solution_path_hhd = self.solution_path_hhd + ".json"   
    with open(self.solution_path_hhd, "w") as f:
      json.dump(self.final_solution_hhd, f)

    print("\nHouseholds final solution saved at:", self.solution_path_hhd)

    # Dwellings final solution
    self.solution_path_dwe = self.solution_path_dwe + ".json"   
    with open(self.solution_path_dwe, "w") as f:
      json.dump(self.final_solution_dwe, f)

    print("\nDwellings final solution saved at:", self.solution_path_dwe) 
    
  def objective_value(self):

    """This function computes the final objective value."""

    self.amount_of_matched_households = 0
    self.obj_func = 0
    for val in self.final_solution.values():
      if val[0] != "non-assigned":
        self.obj_func += val[1]
        self.amount_of_matched_households += 1
    
  def quality_of_the_solution(self):

    """This function prints information about the quality of the results."""

    print("\n______________________________________________________________________________________________________________________")
    print("Final results:")
            
    print("\nAmount of assigned households in the final solution:", self.amount_of_matched_households, "/", self.len_H)
    print("Amount of non-assigned dwellings in the final solution:", self.len_D - self.amount_of_matched_households) 
        
    print('\nWeight of the final allocation:', round(self.obj_func, 4))

    print("\nThe total build time is", round(self.build_time_final, 2), "seconds.")
    
    print("The total optimizing time is", round(self.optimizing_time_final, 2), "seconds.") 

    print("The time to import the W matrix is", round(self.end_w_input - self.begin_w_input, 2), "seconds.")

  def label_final_solution(self):

    """
    This function creates two dictionaries of final solutions: one in the form 
    {ID of household: [ID of dwelling assigned, weight of assignment]} and the other in the form 
    {ID of dwelling: [ID of household assigned, weight of assignment]}.
    """

    self.final_solution_hhd = {}
    final_solution_index_dwe = {}
    self.final_solution_dwe = {}

    # Give the IDs to the final solution and initialize final solution for dwellings
    list_final_sol_keys = list(self.final_solution.keys())
    for h in list_final_sol_keys:
      id_h = self.idx_to_ids["H"][h]
      if self.final_solution[h][0] != "non-assigned":

        # Add element in final_solution_hhd
        d = int(self.final_solution[h][0])
        id_d = self.idx_to_ids["D"][d]
        weight = self.final_solution[h][1]
        self.final_solution_hhd[id_h] = [id_d, weight]

        # Add element in final_solution_index_dwe
        final_solution_index_dwe[d] = [id_h, weight]
      else:
        # Add element in final_solution_hhd
        self.final_solution_hhd[id_h] = self.final_solution[h]

      del self.final_solution[h]

    # Finish final_solution_index_dwe by adding the indices of non-assigned dwellings
    dwes_to_add_in_sol = list(set(range(self.len_D)) - set(final_solution_index_dwe.keys()))
    for d in dwes_to_add_in_sol:
      final_solution_index_dwe[d] = ["non-assigned", 0.0]

    # Creates final_solution_dwe which has keys as being dwelling IDs
    list_final_sol_index_dwe_keys = list(final_solution_index_dwe.keys())
    for d in list_final_sol_index_dwe_keys:
      self.final_solution_dwe[self.idx_to_ids["D"][d]] = final_solution_index_dwe[d]
      del final_solution_index_dwe[d]

  def regional_decomposition(self):

    """This function runs the regional decomposition, which can be executed either inside the decomposition by household size or 
    as an independent method."""

    # Counts the number of regional decompositions needed
    n = 1

    # Initialize bool variable to solve or not remaining problem if regional decomposition is needed
    self.need_remaining_problem = True

    while len(self.round_dwelling_df) > self.amount_to_decompose and len(self.round_household_df) > 0:

      reg_decomp_round_time_begin = time.time()

      # Initialize variable which computes the amount of assignments made in this regional decomposition round
      amount_assignments_made_in_round = 0

      print("\nThe decomposition number " + str(n) + " of the region is executed.")
      print("A further decomposition of the region may be required if not enough allocations have been made at the quality level required by the") 
      print("alpha value.")
      
      # Generate the list of input dataframes corresponding to each sub-region
      list_households_dataframes, list_dwellings_dataframes = self.create_sub_problems() #regional_decomposition  
      
      # Round of regional decomposition
      for j in range(len(list_dwellings_dataframes)):

        subregion_allocation_time_begin = time.time()

        print("\n__________________________________________________________")
        print("Performing the allocation for the sub-region number", j,":") 
        
        # Build and optimize the Gurobi model
        self.optimize_milp(dwe_df = list_dwellings_dataframes[j], hhd_df = list_households_dataframes[j]) #gurobi_solve 

        # Check the quality of each assignment made     
        amount_assignments_made_in_round += self.assignments_quality_verification() #verify_quality 

        subregion_allocation_time_end = time.time()
        subregion_allocation_time = subregion_allocation_time_end - subregion_allocation_time_begin
        print("\nThe run time to perform the allocation in sub-region", j, "in the round", n, "is:", round(subregion_allocation_time, 2), "seconds.")
        
      # Update the number of regional decompositions needed          
      n = n + 1 

      # Verify if the amount of assignments made is low. If it is the case, the alpha is reduced if possible
      if amount_assignments_made_in_round <= self.assignments_tol: #assignms_amount
        print("\nThe amount of assignments made was not satisfactory.")

        # If alpha is already zero, then we just pass all the households of this round to the final solution as non-assigned
        if self.alpha == 0 and self.assignments_tol == 0: #quality_zero?
          print("The households in this decomposition round are passed as non-assigned to the final solution.")

          # Update final solution
          for h in self.round_household_df.index.tolist():
            self.final_solution[h] = ["non-assigned", 0.0]
          
          # In this case, set that we dont need to solve remaining problem because there are no more households that
          # can be assigned to some dwelling
          self.need_remaining_problem = False 

          reg_decomp_round_time_end = time.time()
          reg_decomp_round_time = reg_decomp_round_time_end - reg_decomp_round_time_begin
          print("\nThe run time of this round of the regional decomposition is", round(reg_decomp_round_time, 2), "seconds.") 

          break 
        
        print("The alpha parameter is reduced.")
        self.alpha += - self.beta #reduce_quality

        # If the alpha is low, then set as 0
        if self.alpha <= self.gamma: #quality_gamma?
          print("The alpha parameter is set to 0.")
          self.alpha = 0 #quality_and_tol_0

          # In this case, assignments_tol is set to zero in order to guarantee maximality of the final matching
          self.assignments_tol = 0

      reg_decomp_round_time_end = time.time()
      reg_decomp_round_time = reg_decomp_round_time_end - reg_decomp_round_time_begin
      print("\nThe run time of this round of the regional decomposition is", round(reg_decomp_round_time, 2), "seconds.")

    # Check if still existing non-assigned households to define if the solution of remaining problem is needed
    if len(self.round_household_df) == 0:
      self.need_remaining_problem = False

    # Solve the remaining MILP problem
    if self.need_remaining_problem == True: 

      rem_problem_time_begin = time.time()

      print("\n__________________________________________________________")
      print("After performing the regional decomposition and the allocation in each sub-region, it is attempted to make") 
      print("assignments between remaining households and dwellings:")
    
      self.optimize_milp(dwe_df = self.round_dwelling_df, hhd_df = self.round_household_df) #solve_remaining_problem   
                                                                                          
      # Update final solution and the B_k bounds
      self.update_sol_and_Bk(household_df = self.round_household_df, dwelling_df = self.round_dwelling_df)

      rem_problem_time_end = time.time()
      rem_problem_time = rem_problem_time_end - rem_problem_time_begin
      print("\nThe run time for the solution of the remaining problem is:", round(rem_problem_time, 2), "seconds.")

  def run(self):

    """This function runs the heuristic."""

    begin = time.time()
    
    self.hhd_size = max(self.household_df["size"].tolist()) #first_hhd_size

    # Verify if it is not possible to solve the full problem directly
    if len(self.dwelling_df) <= self.amount_to_decompose: #solve_directly?
      
      print("Considering the size of the dataset and the amount_to_decompose value, it is possible to solve the full problem without decomposition procedure:")
      self.optimize_milp(dwe_df = self.dwelling_df,
                        hhd_df = self.household_df) #gurobi_directly

      # Update the final solution, dataframes and the B_k upper bounds
      self.update_sol_and_Bk(household_df = self.household_df, dwelling_df = self.dwelling_df)  
    elif self.hhd_size_decomposition: 
      print("\nThe decomposition by household size is initialized.") #hhd_size_decomposition
      while self.hhd_size > 0: #hhd_size_zero

        time_hhd_size_round_begin = time.time() 

        print("\n______________________________________________________________________________________________________________________")
        print("Household size value considered:", self.hhd_size)
        
        # Take from the main input dataframes the input that will be used in this decomposition round
        self.define_input_to_be_used()  #def_hhd_subproblem

        # Run the regional decomposition if needed
        if len(self.round_dwelling_df) > self.amount_to_decompose and len(self.round_household_df) > 0: #reg_decomp?

          # Update Amin proportionally
          self.assignments_tol = int(round((100/9700)*len(self.round_household_df)))

          self.regional_decomposition()         

          # Set assignments_tol and alpha to the initial value again
          self.assignments_tol = self.assignments_tol_init
          self.alpha = self.alpha_init
        else: # Solve directly the MILP corresponding to the hhd size decomposition round
          
          # Solve the MILP problem using Gurobi
          self.optimize_milp(dwe_df = self.round_dwelling_df, hhd_df = self.round_household_df) #solve_hhd_subproblem  
                                                                                                
          # Update final solution and the B_k bounds
          self.update_sol_and_Bk(household_df = self.round_household_df, dwelling_df = self.round_dwelling_df)   
            
        # Insert back to the main dataframes the information about not assigned households and about dwellings not used yet 
        self.update_main_dwelling_dataframe() 

        # Update hhd size
        if self.hhd_size > 1:
          self.hhd_size = max(self.household_df["size"].tolist())
        else:
          self.hhd_size = 0 # If self.hhd_size == 1 then it is directly set to zero to avoid max([])

        time_hhd_size_round_end = time.time()
        time_hhd_size_round = time_hhd_size_round_end - time_hhd_size_round_begin

        print("\nThe run time of this round of the decomposition by household size is", round(time_hhd_size_round, 2), "seconds.")
    else: # Regional decomposition directly

      # Get dataframes of the round as copies of the initial ones
      self.round_dwelling_df = copy.deepcopy(self.dwelling_df)
      self.round_household_df = copy.deepcopy(self.household_df)

      self.regional_decomposition()

      # Update initial dataframes
      self.dwelling_df = copy.deepcopy(self.round_dwelling_df)
      self.household_df = copy.deepcopy(self.round_household_df)
    
    # Build the final objective value
    self.objective_value()      

    # Print information about solution quality      
    self.quality_of_the_solution() 

    # Transform final solution from index-based into ID-based
    self.label_final_solution()

    # Save the two final solutions in JSON files
    self.save()
          
    end = time.time()

    print("\nThe total run time is", round((end - begin) + self.extra_time, 2), "seconds.")


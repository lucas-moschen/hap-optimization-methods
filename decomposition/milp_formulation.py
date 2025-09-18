# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script builds the MILP Gurobi model and optimizes it. 

    Parameters
    ----------
    w : matrix
        The matrix of assignment weights.
    dwe_df : DataFrame
        Dwelling dataframe.
    hhd_df : DataFrame
        Household dataframe.
    B_k : dict
        Dictionary containing two keys: "hhd" and "per". The value associated to "hhd" is a dictionary where the keys are 
        indices of grid cells associated to the Constraints (5.3d) and the values are their respective B_{k}^{hhd} values.
        Analogously, the value associated to "per" is a dictionary where the keys are indices of grid cells associated to 
        the Constraints (5.3d) and the values are their espective B_{k}^{per} values. 
    gurobi_parameters : dict, optional
        Parameters to be passed to the Gurobi model as {name : value}. 
        The default is None.
    d_restri_cell : list, optional
        List of dwelling indices for which the corresponding Gurobi variables must be integer.
        The default is None.

    Returns
    ----------
    self.sol : dict
        Dictionary which represents the optimal solution in the form 
        {index of household h: [index of dwelling d, weight of the assignment {h,d}]}.
    build_time : float
        Time needed for Gurobi build step.
    optimizing_time : float 
        Time needed for Gurobi optimizing step.
    self.non_int_assignments : dict
        Dictionary which registers all (if any) non-integer variables at the optimal solution found by Gurobi 
        in the form {index of household h: [list of dwelling indices d for which x_{h,d} is non-integer]}.
    self.non_int_dwes : list
        List of all dwelling indices (if any) related to non-integer variable values from Gurobi optimal solution.
    self.non_int_obj_val : float
        Objective value associated to the non-integer part of the optimal solution.
"""

import gurobipy as gp
import numpy as np
import time 
import pandas as pd 
import copy 
import psutil 

class MILPFormulation:

    def __init__(self,
                 dwe_df: pd.DataFrame,
                 hhd_df: pd.DataFrame,
                 B_k: dict, 
                 gurobi_parameters: dict = None,
                 d_restri_cell: list = None):
        
        init_time_1 = time.time() 

        # Initialize the classes
        print("\nStep 1: Load inputs")
        self.gurobi_parameters = gurobi_parameters 
        self.dwe_df = dwe_df
        self.hhd_df = hhd_df
        self.B_k = B_k 
        self.H = self.hhd_df.index.tolist() 
        self.D = self.dwe_df.index.values
        self.K = list(self.dwe_df["grid_cell"].unique())
        self.len_H = len(self.H)
        self.len_D = len(self.D)
        self.d_restri_cell = d_restri_cell 

        # Get inverse index book of dwellings
        self.reverse_index_book_dwe = {d:j for j, d in enumerate(self.D)}

        print("Number of Dwellings:", self.len_D)
        print("Number of Households:", self.len_H)
        print("Number of Grid Cells", len(self.K))

        print("\nStep 2: Create the Gurobi model")
        
        # Initialize Gurobi model
        self.model = gp.Model("MILP Model")

        init_time_2 = time.time()

        self.init_time = init_time_2 - init_time_1

    def create_assignment_problem(self, w):

        print("\n> Creating variables and objective function...")
        # Get dwellings of the round that are located in constrained grid cells
        dwes_round_cell = list(set(self.D) & set(self.d_restri_cell))

        # Define types of variables
        vtypes = np.empty((self.len_H, self.len_D), dtype=object)
        for j in range(self.len_D):
            if self.D[j] in dwes_round_cell:
                vtypes[:,j] = [gp.GRB.BINARY]*self.len_H
            else:
                vtypes[:,j] = [gp.GRB.CONTINUOUS]*self.len_H

        # Get W sub-matrix
        sub_w = w[np.ix_(self.H, self.D)] 

        # Create variables and objective function
        self.x = self.model.addMVar(shape=(self.len_H, self.len_D), vtype=vtypes, obj=sub_w) 

        del vtypes 

        # Set to zero variables corresponding to infeasible assignments
        for i in range(self.len_H):
            row_i_of_sub_w = sub_w[i,:]
            null_element_indices = np.argwhere(row_i_of_sub_w == 0)[:,0]
            self.x[i,null_element_indices].setAttr("ub", 0)

        del sub_w

        # Add constraints which guarantee each household in at most one dwelling
        print("\n> Adding household constraints...")
        self.model.addConstrs((self.x[i, :].sum() <= 1 for i in range(self.len_H)))

        # Add constraints which guarantee each dwelling receiving at most one household
        print("\n> Adding dwelling constraints...")
        self.model.addConstrs((self.x[:, j].sum() <= 1 for j in range(self.len_D)))

    def add_Bk_constraints(self):

        # Group dwellings by grid cell indices
        dwes_groupedby_cell = self.dwe_df.groupby('grid_cell').apply(lambda x: x.index.tolist()).to_dict()

        # Get grid cells of this decomposition round with constraint on number of households
        round_cell_constr_hhd = list(set(self.K) & set(self.B_k["hhd"].keys()))

        # Get grid cells of this decomposition round with constraint on number of people
        round_cell_constr_per = list(set(self.K) & set(self.B_k["per"].keys()))

        # Filter dwes_groupedby_cell to constrained grid cells and transform dwelling indices to column indices
        constrained_cells = list(set(round_cell_constr_hhd) | set(round_cell_constr_per))
        for k in self.K: # equal to keys of dwes_groupedby_cell
            if k in constrained_cells:
                columns_of_dwes_in_k = [self.reverse_index_book_dwe[d] for d in dwes_groupedby_cell[k]]
                dwes_groupedby_cell[k] = copy.deepcopy(columns_of_dwes_in_k)
            else:
                del dwes_groupedby_cell[k]

        # Add grid cell constraints on number of households
        for k in round_cell_constr_hhd:
            self.model.addConstr(self.x[:, dwes_groupedby_cell[k]].sum() <= self.B_k["hhd"][k])

        # Define vector of p_h
        p_h = self.hhd_df["size"].values

        # Add grid cell constraints on number of households
        for k in round_cell_constr_per:
            expr_B_per = 0
            for j in dwes_groupedby_cell[k]:
                expr_B_per += p_h @ self.x[:,j]
            self.model.addConstr(expr_B_per <= self.B_k["per"][k])

    def get_and_check_solution(self, w):

        """
        This function checks if Gurobi solution is integer and generates the solution as a dictionary with elements 
        in the form {h: [d, weight of assignment {h,d}]}.
        """ 

        # Get the new solution dictionary
        self.sol = {} 
        self.non_int_assignments = {}
        self.non_int_dwes = set()
        self.non_int_obj_val = 0

        # Create matrix with values from self.x
        variable_values = self.x.X 
        # Delete self.x
        del self.x 

        for i, h in enumerate(self.H):
            save_d = "non-assigned"

            # Get index of value one in the variable row (if exists) and put into solution
            is_one = np.abs(variable_values[i,:] - 1) < 1e-05
            indices_one = np.where(is_one)[0]
            if len(indices_one) == 1:
                save_d = str(self.D[indices_one[0]])

            # Get indices of non-integer values in the variable row (if exists) and put into the corresponding objects
            is_non_integer = (variable_values[i,:] >= 1e-05) & (variable_values[i,:] <= 1 - 1e-05)
            indices_of_non_integer = np.where(is_non_integer)[0]
            if len(indices_of_non_integer) > 0:
                self.non_int_assignments[h] = list(self.D[indices_of_non_integer])
                self.non_int_dwes.update(self.D[indices_of_non_integer])
                self.non_int_obj_val += w[h, self.D[indices_of_non_integer]] @ variable_values[i, indices_of_non_integer]

            if save_d == "non-assigned":
                self.sol[h] = ["non-assigned", 0.0] 
            else: 
                self.sol[h] = [save_d, w[h, int(save_d)]]

        del variable_values
        
        self.non_int_dwes = list(self.non_int_dwes)

    def solution_quality(self):

        """This function prints some information about the solution quality."""

        print('\nTotal Weight from Gurobi:', round(self.model.objVal, 4))
        
        # Count how many households were assigned
        amount_of_matched_households = 0
        for h in self.H:
            if self.sol[h][0] != "non-assigned":
                amount_of_matched_households += 1

        print("\nGeneral results:")
        print("Amount of assigned households:", amount_of_matched_households, "/", self.len_H)  
        print("Amount of non-assigned dwellings:", self.len_D - amount_of_matched_households) 


    def optimize(self, w):

        begin_1 = time.time() 

        print("\n> Building assignment problem...")
        self.create_assignment_problem(w = w)
            
        print("\n> Adding grid cell constraints...")
        self.add_Bk_constraints()

        print("\n> Set optional Gurobi model parameters:")
        if self.gurobi_parameters is not None:
            for key, value in self.gurobi_parameters.items():
                self.model.setParam(key, value)
        
        self.model.modelSense = gp.GRB.MAXIMIZE
        self.model.update()
        end_1 = time.time() 

        # Compute memory used by Gurobi model + W
        mem = psutil.virtual_memory()
        print(f"\nRAM usage before optimize MILP formulation: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")

        print('\nStep 3: Gurobi optimizes the model.')
        begin_2 = time.time()
        self.model.optimize()
        end_2 = time.time()

        print('\nStep 4: Computing solution...')

        # Generate the solution
        self.get_and_check_solution(w)

        # Print some information about the solution quality
        self.solution_quality()

        build_time = (end_1 - begin_1) + self.init_time
        optimizing_time = end_2 - begin_2

        print("\nBuild time:", round(build_time, 2), "seconds.")
        print("Optimizing time:", round(optimizing_time, 2), "seconds.")

        self.model.dispose()
        del self.model
            
        return self.sol, build_time, optimizing_time, self.non_int_assignments, self.non_int_dwes, self.non_int_obj_val
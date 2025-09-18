# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

If an integer optimal solution was not obtained for the MILP problem through milp_formulation.py neither by milp_postprocessing.py, 
then this script obtains it through the creation and solution of an integer Gurobi model corresponding to the part of the MILP 
problem related to the non-integer values.   

    Parameters
    ----------
    gurobi_parameters : dict
        Parameters to be passed to the Gurobi model as {name : value}.
    build_time : float
        Time spent in the building of Gurobi models for this MILP problem so far.
    optimizing_time : float 
        Time spent in the optimization of this MILP problem so far.
    w : matrix
        The matrix of assignment weights.
    current_x : dict
        Dictionary which represents the current optimal solution in the form 
        {index of household h: [index of dwelling d, weight of the assignment {h,d}]}.
    non_int_assignments : dict
        Dictionary containing the non-integer values of the variables with the form 
        {index of household h: {index of dwelling d: value of the variable x_{h,d}}}.
    total_amount_dwe : int
        Total amount of dwellings related to this MILP problem.

    Returns
    ----------
    current_x : dict
        The parameter current_x updated with the new assignments obtained through the full integer solution.
    self.build_time : float
        The parameter build_time updated with the build time of this integer Gurobi model.
    self.optimizing_time : float 
        The parameter optimizing_time updated with the optimizing time of this integer Gurobi model.
        
"""

import gurobipy as gp 
import time 
import numpy as np
import psutil 

class PostprocessingNonInteger:

    def __init__(self,
                 gurobi_parameters: dict,
                 build_time, 
                 optimizing_time):

        init_time_begin = time.time() 

        # Initialize classes
        print("\nStep 1: Load inputs")
        self.gurobi_parameters = gurobi_parameters
        self.build_time = build_time 
        self.optimizing_time = optimizing_time

        init_time_end = time.time() 

        # Store time of __init__
        self.init_time = init_time_end - init_time_begin

    def create_problem(self, w, non_int_assignments):

        """
        This function generates the Gurobi model.
        """ 

        # Get list of households and dwellings and their lengths
        self.H = list(non_int_assignments.keys())
        self.D = sorted(set(d for dwes in non_int_assignments.values() for d in dwes))
        self.len_H = len(self.H)
        self.len_D = len(self.D)

        # Get inverse index book of dwellings
        self.reverse_index_book_dwe = {d:j for j, d in enumerate(self.D)}

        print("Number of Dwellings:", self.len_D)
        print("Number of Households:", self.len_H)

        print("\nStep 2: Create the Gurobi model")

        # Initializes Gurobi model
        self.model = gp.Model("ILP Model")

        # Get W sub-matrix
        sub_w = w[np.ix_(self.H, self.D)]

        # Create variables and objective function
        self.x = self.model.addMVar(shape=(self.len_H, self.len_D), vtype=gp.GRB.BINARY, obj=sub_w)

        # Delete sub_w
        del sub_w 

        # Set to 0 variables corresponding to assignments that are not contained in non_int_assignments
        for i, h in enumerate(self.H):
            int_assignments_for_h = list(set(self.D) - set(non_int_assignments[h]))
            cols_for_int_assignments_for_h = [self.reverse_index_book_dwe[d] for d in int_assignments_for_h]
            self.x[i, cols_for_int_assignments_for_h].setAttr("ub",0)

        # Add constraints which guarantee each household in at most one dwelling
        print("\n> Adding household constraints...")
        self.model.addConstrs((self.x[i, :].sum() <= 1 for i in range(self.len_H)))

        # Add constraints which guarantee each dwelling receiving at most one household
        print("\n> Adding dwelling constraints...")
        self.model.addConstrs((self.x[:, j].sum() <= 1 for j in range(self.len_D)))

        del non_int_assignments

    def get_and_check_solution(self, w):

        """
        This function generates the solution as a dictionary with elements in the form {h : [d, w[h,d]]}.
        """ 

        # Initialize solution dict
        self.sol = {}

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

            if save_d == "non-assigned":
                self.sol[h] = ["non-assigned", 0.0] 
            else: 
                self.sol[h] = [save_d, w[h, int(save_d)]]

        del variable_values

    def postprocessing_non_int(self,
                               w,
                                current_x: dict, 
                                non_int_assignments,
                                total_amount_dwe):
        
        """
        This function gets the integer optimal solution through the solution of the ILP model.
        """ 

        begin_1 = time.time() 

        self.create_problem(w, non_int_assignments)

        print("\n> Set optional Gurobi model parameters:")
        for key, value in self.gurobi_parameters.items():
            self.model.setParam(key, value)
        self.model.setParam("OutputFlag", 0)
        
        self.model.modelSense = gp.GRB.MAXIMIZE
        self.model.update()
        end_1 = time.time() 

        # Compute memory used by Gurobi model + W
        mem = psutil.virtual_memory()
        print(f"\nRAM usage before optimize ILP formulation: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")

        print('\nStep 3: Gurobi optimizes the model.')
        begin_2 = time.time()
        self.model.optimize()
        end_2 = time.time()

        print('\nStep 4: Computing solution...')

        # Generates the solution
        self.get_and_check_solution(w)

        # Update times
        b_time = (end_1 - begin_1) + self.init_time
        o_time = end_2 - begin_2

        print("\nBuild time:", round(b_time, 2), "seconds.")
        print("Optimizing time:", round(o_time, 2), "seconds.")

        self.build_time += b_time
        self.optimizing_time += o_time

        # Update current solution
        for h in self.sol.keys():
            current_x[h] = self.sol[h].copy()

        print('\nTotal Weight from Gurobi:', round(self.model.objVal, 4)) 

        # Count how many households were assigned
        amount_of_matched_households = 0
        for h in current_x.keys():
            if current_x[h][0] != "non-assigned":
                amount_of_matched_households += 1

        print("\nGeneral results:")
        print("Amount of assigned households:", amount_of_matched_households, "/", len(list(current_x.keys())))  
        print("Amount of non-assigned dwellings:", total_amount_dwe - amount_of_matched_households)

        self.model.dispose()
        del self.model
            
        return current_x, self.build_time, self.optimizing_time
# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

If Gurobi found a non-integer solution in the optimal facet for the allocation problem in milp_formulation.py then this script 
makes a post-processing step in order to obtain a vertex of this optimal facet, thus an integer optimal solution. 
The main function of this script is milp_postprocessing. Its parameters and returns are written below.

    Parameters
    ----------
    w : matrix
        The matrix of assignment weights.
    current_x : dict
        Dictionary in the form {index of household h: [index of dwelling d, w_{h,d}]} containing the allocation made by 
        milp_formulation.py.
    non_int_assignments : dict
        Dictionary which registers all the non-integer variables at the optimal solution found by Gurobi 
        in the form {index of household h: [list of dwelling indices d for which x_{h,d} is non-integer]}.
    opt_time : float
        Optimizing time from milp_formulation.py
    non_int_dwes : list
        List of all dwelling indices related to non-integer variable values from Gurobi optimal solution found in 
        milp_formulation.py.
    non_int_obj_val : float
        The objective value corresponding to the non-integer part of the optimal solution.
    total_amount_hhd : int
        Amount of households related to this MILP problem.
    total_amount_dwe : int
        Amount of dwellings related to this MILP problem.
    limit_it : int
        Upper bound on the amount of iterations that can be done by this algorithm.

    Returns
    ----------
    current_x : dict
        Dictionary in the form {index of household h: [index of dwelling d, w_{h,d}]} containing the allocation made 
        after MILP post-processing.
    optimizing_time : float
        Sum between optimizing time from milp_formulation.py and time used for this MILP post-processing.
    opt_not_found : bool
        True if the algorithm did not find the optimal solution either due to the upper bound on the amount of iterations 
        (limit_it) or due to rounding problems from Gurobi. False otherwise.
"""

import time  
import sys 

def recursive_for(w,
                  non_int_assignments,
                  non_int_dwes,
                  non_int_hhds,
                  i,
                  assignments,
                  non_int_obj_val,
                  sol_found,
                  its,
                  limit_it):
    
    """
    This is a recursive function used by milp_postprocessing function which tests matching possibilities between households and 
    dwellings corresponding to non-integer variable values of the Gurobi model from milp_formulation.py. It runs either until an 
    objective value equal to the optimal one found is obtained or until the amount of iterations reaches the upper bound limit_it.  
    """
    
    # Get the corresponding household
    h = non_int_hhds[i]

    # Update position i
    i += 1

    fixed_non_int_dwes = non_int_dwes.copy()
    dwes_available_feasible_for_h = list(set(non_int_assignments[h]) & set(fixed_non_int_dwes))

    # Add the possibility for this hhd to be non-assigned
    dwes_available_feasible_for_h.append(-1)

    # Loop over all dwellings d which Gurobi variable x_{h,d} was non-integer
    for d in dwes_available_feasible_for_h:

        # Update lists considering the use of current d
        assignments.append(d)
        if d != -1:
            non_int_dwes.remove(d)

        if i == len(non_int_assignments):

            # This is the last function call in the recursion loop, checking the objective value
            obj_val = 0
            for j in range(len(non_int_assignments)):
                dwe = assignments[j]
                if dwe != -1:
                    obj_val += w[non_int_hhds[j], dwe]
            
            print("The integer objective value found is", round(obj_val, 4))
            if abs(obj_val - non_int_obj_val) < 1e-04:
                
                sol_found = True 
                solution = assignments.copy()
                break 

            # Update amount of iterations
            its += 1

            # If the limit of iterations was exceeded then stop
            if its >= limit_it:
                break 
        else:
            # This is not the last function call in the recursion loop, it goes to next level
            solution, its = recursive_for(w, non_int_assignments, non_int_dwes, non_int_hhds, i, assignments, non_int_obj_val, sol_found, its, limit_it)  

            if solution is not None:
                sol_found = True 
                break

            if its >= limit_it:
                break 

        # The arriving here means that matching chosen was not satisfactory so we insert again the current dwelling in lists below
        del assignments[-1]
        if d != -1:
            non_int_dwes.append(d) 
    
    if sol_found:
        return solution, its 
    else:
        return None, its

def milp_postprocessing(w,
                        current_x,
                        non_int_assignments,
                        opt_time,
                        non_int_dwes,
                        non_int_obj_val,
                        total_amount_hhd,
                        total_amount_dwe,
                        limit_it):
    
    """This function calls the recursive function recursive_for to move the solution found by Gurobi in the optimal facet to a 
    vertex (and consequently an integer optimal solution)."""
    
    begin = time.time()

    print("\nThe objective value corresponding to the non-integer part of the solution is", round(non_int_obj_val, 4))

    non_int_hhds = list(non_int_assignments.keys())

    # Initialize position i and list of assignments of post-processing
    i = 0
    assignments = []
    sol_found = False 
    print("Running post-processing...")

    # Set recursion limit to avoid stack overflow
    value_stack = len(non_int_assignments) + 100
    sys.setrecursionlimit(value_stack)

    # Initialize amount of iterations of milp_postprocessing
    its = 0

    # Get assignments made by recursive function
    list_assignments, its = recursive_for(w, non_int_assignments, non_int_dwes, non_int_hhds, i, assignments, non_int_obj_val, sol_found, its, limit_it)

    # Check if the process could find an optimal solution
    if list_assignments is not None:

        # Set flag
        opt_not_found = False

        print("Combination found. Computing final solution...")
        
        # Update final solution
        for j in range(len(non_int_assignments)):
            hhd = non_int_hhds[j]
            dwe = list_assignments[j]
            if dwe != -1:
                current_x[hhd] = [str(dwe), w[hhd, dwe]]
            else:
                current_x[hhd] = ["non-assigned", 0.0]
        print("Final solution obtained.")

        del non_int_assignments
        
        # Count how many households were assigned
        amount_of_matched_households = 0
        for h in current_x.keys():
            if current_x[h][0] != "non-assigned":
                amount_of_matched_households += 1

        # Print general results after post-processing 
        print("\nGeneral results:")
        print("Amount of assigned households:", amount_of_matched_households, "/", total_amount_hhd)  
        print("Amount of non-assigned dwellings:", total_amount_dwe - amount_of_matched_households)
    else:

        print("\nAn integer solution was not found within the number of iterations allowed.")

        # Set flag
        opt_not_found = True 
        
    end = time.time()

    # Update optimizing time
    optimizing_time = opt_time + (end-begin)

    return current_x, optimizing_time, opt_not_found
# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script runs the greedy heuristic.

    Parameters
    ----------
    matrix : matrix
        The matrix of assignment weights.
    B_k : dict
        Dictionary containing two keys: "hhd" and "per". The value associated to "hhd" is a dictionary where the keys are 
        indices of grid cells associated to the Constraints (5.2d) and the values are their respective B_{k}^{hhd} values.
        Analogously, the value associated to "per" is a dictionary where the keys are indices of grid cells associated to 
        the Constraints (5.2e) and the values are their espective B_{k}^{per} values.
    H : list
        List of indices for the households.
    D : list
        List of indices for the dwellings.
    p_h : array
        Array where each element is the number of persons in the household corresponding to its index.
    dwelling_df : DataFrame
        Dwelling dataframe.
    verbose : bool
        If True, details about the operation of the algorithm are printed on the screen.

    Returns
    -------
    final_solution : dict
        Dictionary in the form {index of household h: [index of dwelling d, w_{h,d}]} containing the allocation made by 
        the algorithm. 
    final_weight : float
        Objecive value of the allocation.  
    hhd_assigned_count : int
        Number of households assigned in the allocation. 
"""

import numpy as np 
import time 

def verboseprint(*args, verbose=False, **kwargs):

    """
    This function makes the verbose argument control the messages printed on the screen.  
    """

    if verbose:
        print(*args, **kwargs)

def greedy(matrix,
           B_k,
           H,
           D,
           p_h,
           dwelling_df, 
            verbose: bool):
    
    """
    This function runs the greedy heuristic.  
    """

    # Print initial information
    print("num rows:", matrix.shape[0])
    print("num cols:", matrix.shape[1])

    final_solution = {}
    final_weight = 0

    available_hhds = set(H)
    available_dwes = set(D)

    begin = time.time() 

    print("\nThe matrix is being vectorized and sorted to run the greedy heuristic...")

    # Creates index vector of non-null elements
    idx_linear = np.flatnonzero(matrix)

    # Get corresponding values
    values = matrix.flat[idx_linear]

    # Sorts the indexes by descending value (without copying the entire matrix)
    order = np.argsort(-values)

    # Delete values
    del values 

    # Converts ordered linear indices to coordinates (row, column)
    sorted_indices = np.column_stack(np.unravel_index(idx_linear[order], matrix.shape))
    
    verboseprint(sorted_indices, verbose=verbose)

    # Get amount of available households.
    amount_available_hhds = len(available_hhds)

    end = time.time() 

    print("\nThe time used to vectorize and sort the matrix is: ", round(end-begin, 2))

    print("\nThe greedy heuristic is initialized.")
    print("\n")

    hhd_constrained_cells = list(B_k["hhd"].keys())
    per_constrained_cells = list(B_k["per"].keys())

    it = 1
    # Greedy operation.
    for vec in sorted_indices:

        # Get hhd and dwe indices
        h = vec[0]
        d = vec[1]

        # Set grid cell constraints flag       
        hhd_constr = False
        per_constr = False
        

        verboseprint("\nIteration number:", it, verbose = verbose) 

        verboseprint("The selected weight is matrix[", h, ",", d, "]", verbose = verbose)

        if h in available_hhds and d in available_dwes:
            
            k = dwelling_df.at[int(d), "grid_cell"]

            # Check grid cell constraints
            if k in hhd_constrained_cells:

                hhd_constr = True

                if B_k["hhd"][k] < 1:
                    continue 

            if k in per_constrained_cells:

                per_constr = True

                if B_k["per"][k] < p_h[int(h)]:

                    continue
            
            verboseprint("The assignment is accepted.", verbose = verbose)

            # Update final matching and its weight
            w_val = matrix[h, d]
            final_solution[h] = [str(d), w_val]
            final_weight += w_val

            # Update available hhds and dwes
            available_hhds.remove(h)
            available_dwes.remove(d)
            verboseprint("The new list of available hhds is:", available_hhds, verbose = verbose)
            verboseprint("The new list of available dwes is:", available_dwes, verbose = verbose)

            # Update amount of available households
            amount_available_hhds += -1
            
            # Update grid cell constraints
            if hhd_constr is True:
                B_k["hhd"][k] -= 1

            if per_constr is True:
                B_k["per"][k] -= p_h[int(h)]
            
            if amount_available_hhds == 0:
                verboseprint("\nThere is no more available households!", verbose = verbose)
                break 

        it += 1

        if it % 1000000 == 0:
            print(f"{it} iterations done")

    print("\nThe number of iterations was", it)

    # Put into solution information about non-assigned households
    available_hhds = list(available_hhds)
    for h in available_hhds:
        final_solution[h] = ["non-assigned", 0.0]

    # Get number of hhds assigned
    hhd_assigned_count = len(H) - len(available_hhds)

    verboseprint("The final matching is:", verbose = verbose)
    verboseprint(final_solution, verbose = verbose)

    print("\nAllocation weight:")
    print(round(final_weight, 2))

    end2 = time.time()

    greedy_time = round(end2-begin, 2)

    print("\nThe time to create the vector, sort it, and execute greedy is:")
    print(greedy_time, "seconds.")

    return final_solution, final_weight, hhd_assigned_count

# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script runs the path-growing heuristic.

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

from scipy.sparse import lil_matrix
import numpy as np
import random 
import time
import copy 

def verboseprint(*args, verbose=False, **kwargs):

    """
    This function makes the verbose argument control the messages printed on the screen.  
    """

    if verbose:
        print(*args, **kwargs)

def path_growing_algorithm(matrix,
                            B_k,
                            H,
                            D,
                            p_h,
                            dwelling_df, 
                            verbose: bool):
    
    """
    This function runs the path-growing heuristic.  
    """

    # Set seed
    # We need to randomize list of dwelling indices to avoid trend of let last dwellings of the data set as non-assigned
    random.seed(42)

    num_rows, num_cols = matrix.shape

    # Create two B_k dicts: one for each matching constructed throughout the path-growing heuristic
    B_k_1 = copy.deepcopy(B_k)
    B_k_2 = copy.deepcopy(B_k)

    # Initialize the matchings and their respective weights
    M1 = lil_matrix((num_rows, num_cols), dtype=int)
    M2 = lil_matrix((num_rows, num_cols), dtype=int)
    M1_weight = 0
    M2_weight = 0

    # Initialize the list of available households, the list of available dwellings, and shuffle each one
    available_hhds = np.array(H)
    random.shuffle(available_hhds)
    available_dwes = np.array(D)
    random.shuffle(available_dwes)

    # Initialize counting of number of hhds assigned in M1 and M2
    M1_hhds_assigned_count = 0
    M2_hhds_assigned_count = 0

    # Get constrained grid cells
    hhd_constrained_cells = list(B_k["hhd"].keys())
    per_constrained_cells = list(B_k["per"].keys())

    # Set initial i
    i = 1

    # Start iteration counting
    it = 1

    # Start path-growing heuristic run time 
    path_gr_begin = time.time()

    # Loop to run the algorithm
    while len(available_hhds) > 0 and len(available_dwes) > 0:

        verboseprint("Number of available hhds remaining:", len(available_hhds), verbose = verbose)
        verboseprint("Number of available dwes remaining:", len(available_dwes), verbose = verbose)

        # Defines starting node for a new path
        if i == 1:
            x = available_hhds[0]
            verboseprint("The node", x, "is selected from the available hhds", verbose = verbose)
        else:
            x = available_dwes[0]
            verboseprint("The node", x, "is selected from the available dwes", verbose = verbose)

        # Builds the path
        while True: 

            verboseprint("Iteration number:", it, verbose = verbose) 
            verboseprint("Current evaluated node:", x, verbose = verbose) 
            
            # Current node is a household, we try to match with a dwelling in M1
            if i == 1:
                verboseprint("i == 1", verbose = verbose)

                # Get row of W with weights from available dwellings
                available_dwe_weights_in_row_x = matrix[x, available_dwes]

                # Filter positive values
                positive_bool_vector = available_dwe_weights_in_row_x > 0

                # Verify if degree is not zero
                if np.any(positive_bool_vector):

                    # Flag for side constraints
                    hhd_constr = False
                    per_constr = False

                    # Flag for heaviest node being feasible in relation to side constraints
                    feasible_node_not_found = False

                    # Finding the biggest weight and the corresponding node
                    idx_local = np.argmax(available_dwe_weights_in_row_x[positive_bool_vector])
                    heaviest_node_edge = available_dwes[positive_bool_vector][idx_local]
                    biggest_weight = available_dwe_weights_in_row_x[positive_bool_vector][idx_local]

                    # Checking side constraints
                    k = dwelling_df.at[int(heaviest_node_edge), "grid_cell"]

                    if k in hhd_constrained_cells:

                        hhd_constr = True

                        if B_k_1["hhd"][k] < 1:
                            feasible_node_not_found = True  

                    if k in per_constrained_cells:

                        per_constr = True

                        if B_k_1["per"][k] < p_h[int(x)]:

                            feasible_node_not_found = True 

                    # If heaviest node is infeasible in relation to the side constraints (which for most iterations is not 
                    # the case) then sort all the remaining nodes by their weight and get the heaviest feasible one
                    if feasible_node_not_found == True:

                        sorted_indices = np.argsort(-available_dwe_weights_in_row_x[positive_bool_vector])
                        for idx in sorted_indices:
                            if idx == idx_local:
                                continue  # already tested

                            # Flag for side constraints
                            hhd_constr = False
                            per_constr = False

                            heaviest_node_edge = available_dwes[positive_bool_vector][idx]
                            biggest_weight = available_dwe_weights_in_row_x[positive_bool_vector][idx]

                            # Checking side constraints
                            k = dwelling_df.at[int(heaviest_node_edge), "grid_cell"]

                            if k in hhd_constrained_cells:

                                hhd_constr = True

                                if B_k_1["hhd"][k] < 1:

                                    continue  

                            if k in per_constrained_cells:

                                per_constr = True

                                if B_k_1["per"][k] < p_h[int(x)]:

                                    continue 

                            # Stop when the heaviest feasible node is found
                            feasible_node_not_found = False
                            break 

                    # Check if feasible node was found
                    if feasible_node_not_found == False:

                        # Update side constraints
                        if hhd_constr is True:
                            B_k_1["hhd"][k] -= 1

                        if per_constr is True:
                            B_k_1["per"][k] -= p_h[int(x)]

                        # Update
                        M1[x,heaviest_node_edge] = 1
                        M1_weight += biggest_weight
                        available_hhds = available_hhds[available_hhds != x]
                        x = heaviest_node_edge
                        M1_hhds_assigned_count += 1
                        verboseprint("Heaviest node found:", heaviest_node_edge, verbose = verbose)
                    else:

                        verboseprint("There is no feasible node in relation to the side constraints", verbose = verbose)
                        available_hhds = available_hhds[available_hhds != x] 
                        
                        break
                else:
                    # The node has no neighborhood 
                    verboseprint("We found a null-neighborhood node", verbose = verbose)
                    available_hhds = available_hhds[available_hhds != x] 
                    
                    break

            # Current node is a dwelling, we try to match with a hhd in M2
            if i == 2:
                verboseprint("i==2", verbose = verbose)

                # Flag for side constraints
                hhd_constr = False
                per_constr = False

                # Checking hhd side constraint
                k = dwelling_df.at[int(x), "grid_cell"]

                if k in hhd_constrained_cells:

                    hhd_constr = True

                    if B_k_2["hhd"][k] < 1:

                        verboseprint("There is no feasible node in relation to the side constraints", verbose = verbose)
                        available_dwes = available_dwes[available_dwes != x]
                        break 

                # Get column of W with weights from available households 
                available_hhd_weights_in_column_x = matrix[available_hhds, x]

                # Filter positive values
                positive_bool_vector = available_hhd_weights_in_column_x > 0

                # Verify if degree is not zero
                if np.any(positive_bool_vector):

                    # Flag for heaviest node being feasible in relation to hhd side constraint
                    feasible_node_not_found = False

                    # Finding the biggest weight and the corresponding node
                    idx_local = np.argmax(available_hhd_weights_in_column_x[positive_bool_vector])
                    heaviest_node_edge = available_hhds[positive_bool_vector][idx_local]
                    biggest_weight = available_hhd_weights_in_column_x[positive_bool_vector][idx_local]

                    # Checking persons side constraint
                    if k in per_constrained_cells:

                        per_constr = True

                        if B_k_2["per"][k] < p_h[int(heaviest_node_edge)]:

                            feasible_node_not_found = True

                    # If heaviest node is infeasible in relation to the side constraints (which for most iterations is not 
                    # the case) then sort all the remaining nodes by their weight and get the heaviest feasible one
                    if feasible_node_not_found == True:

                        sorted_indices = np.argsort(-available_hhd_weights_in_column_x[positive_bool_vector])
                        for idx in sorted_indices:
                            if idx == idx_local:
                                continue  # already tested

                            heaviest_node_edge = available_hhds[positive_bool_vector][idx]
                            biggest_weight = available_hhd_weights_in_column_x[positive_bool_vector][idx]

                            # Checking persons side constraint
                            if k in per_constrained_cells:

                                per_constr = True

                                if B_k_2["per"][k] < p_h[int(heaviest_node_edge)]:

                                    continue 

                            feasible_node_not_found = False
                            break
 
                    # Check if feasible node was found
                    if feasible_node_not_found == False:

                        # Update side constraints
                        if hhd_constr is True:
                            B_k_2["hhd"][k] -= 1

                        if per_constr is True:
                            B_k_2["per"][k] -= p_h[int(heaviest_node_edge)]

                        # Update
                        M2[heaviest_node_edge, x] = 1 
                        M2_weight += biggest_weight
                        available_dwes = available_dwes[available_dwes != x]
                        x = heaviest_node_edge
                        M2_hhds_assigned_count += 1
                        verboseprint("Heaviest node found:", heaviest_node_edge, verbose = verbose)
                    else:
 
                        verboseprint("There is no feasible node in relation to the side constraints", verbose = verbose)
                        available_dwes = available_dwes[available_dwes != x] 
                        
                        break
                else:
                    # The node has no neighborhood
                    verboseprint("We found a null-neighborhood node", verbose = verbose)
                    available_dwes = available_dwes[available_dwes != x]

                    break 
                    
            # Update i 
            i = 3-i

            # Update iteration counting
            it += 1 

    # Get algorithm run time
    path_gr_end = time.time()
    path_gr_time = path_gr_end - path_gr_begin
    print("\nPath-growing heuristic executed for this model.")
    print("Run time of the path-growing heuristic:", round(path_gr_time, 2), "seconds.")

    # Print information abour the matchings
    print("Number of households assigned in M1: " + str(M1_hhds_assigned_count) + "/" + str(len(H)))
    print("Objective value of M1:", round(M1_weight, 4))
    print("Number of households assigned in M2:" + str(M2_hhds_assigned_count) + "/" + str(len(H)))
    print("Objective value of M2:", round(M2_weight, 4))

    # Select heaviest matching
    if M1_weight >= M2_weight:
        print("\nThe selected matching is M1")
        matching = M1
        final_weight = M1_weight
        hhds_assigned_count = M1_hhds_assigned_count
        B_k = B_k_1
        del M2
    else:
        print("\nThe selected matching is M2")
        matching = M2
        final_weight = M2_weight
        hhds_assigned_count = M2_hhds_assigned_count
        B_k = B_k_2
        del M1

    # Run post-processing to guarantee SCM matching
    postprocessing_begin_time = time.time()

    print("\nRunning post-processing...")

    # Get list of non-assigned households and dwellings
    free_hhd = sorted(set(H) - set(matching.nonzero()[0]))
    free_dwe = sorted(set(D) - set(matching.nonzero()[1]))
    random.shuffle(free_hhd)
    random.shuffle(free_dwe)

    # For each household, get heaviest dwelling so that the assignment is allowed by the side constraints
    for h in free_hhd:

        best_dwe = None 
        best_weight = 0

        for d in free_dwe:

            # Gets grid cell index of this dwelling
            k = dwelling_df.at[d, "grid_cell"]

            # Check if this grid cell has constraints. If so, verify if they hold
            if k in hhd_constrained_cells:
                if B_k["hhd"][k] < 1:
                    continue 
            if k in per_constrained_cells:
                if B_k["per"][k] < p_h[h]:
                    continue

            # Update heaviest node if necessary
            if matrix[h, d] > best_weight:
                best_dwe = d 
                best_weight = matrix[h, d]
        
        # Update
        if best_dwe is not None:
            
            # Update results
            assignmt_weight = round(matrix[h, best_dwe], 4)
            matching[h,best_dwe] = 1
            final_weight += assignmt_weight
            hhds_assigned_count += 1
            free_dwe.remove(best_dwe)
            
            # Update side constraints if necessary
            k = dwelling_df.at[best_dwe, "grid_cell"]
            if k in hhd_constrained_cells:
                B_k["hhd"][k] -= 1 
            if k in per_constrained_cells:
                B_k["per"][k] -= p_h[h]

    # Get post-processing time
    postprocessing_end_time = time.time()
    print("\nTime used by post-processing:", round(postprocessing_end_time - postprocessing_begin_time, 2), "seconds.")

    # Convert matching matrix to COO form
    matching_coo = matching.tocoo()

    # Initialize final solution
    final_solution  = {} 

    # Get list of assigned households and dwellings and loop over it
    hhd_assigned, dwe_assigned = matching_coo.row, matching_coo.col
    for h, d in zip(hhd_assigned, dwe_assigned):

        # Get weight value
        w_val = matrix[h,d]

        # Update final solution and weight
        w_val = round(w_val, 4) 
        final_solution[h] = [str(d), w_val]

    # Put into solution information about non-assigned households
    for h in list(set(H) - set(hhd_assigned)):
        final_solution[h] = ["non-assigned", 0.0]

    # Print results
    print("\nMatching obtained.")
    print("Objective value:", round(final_weight, 4))
    print("Number of households assigned in the matching: " + str(hhds_assigned_count) + "/" + str(len(H)))

    return final_solution, final_weight, hhds_assigned_count
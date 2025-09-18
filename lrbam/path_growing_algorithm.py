# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script runs the path-growing algorithm described in Section 7.4.1 of the thesis.
It also runs the post-processing step for the LRBAM-IWPP according to Section 7.4.3 of the thesis.

    Parameters
    ----------
    postprocessing : str
        If is "iwpp", the post-processing step of the LRBAM-IWPP is made.
    H : list
        List of household indices.
    D : list
        List of residential dwelling indices.
    kappa : matrix
        Matrix containing the kappa_{h,d} values for each possible assignment of household h to dwelling d. 
    verbose : bool
        If True, detailed information about the operation of the path-growing algorithm is provided.
    warm_start_hhd : list
        List of nodes corresponding to households contained in the list of nodes for warm-starting the next 
        iteration of the LRBAM.
    warm_start_dwe : list
        List of nodes corresponding to dwellings contained in the list of nodes for warm-starting the next 
        iteration of the LRBAM.

    Returns
    ----------
    matching_csc : csc_matrix
        Sparse matrix in CSC format that corresponds to the obtained matching.
    weight_penal : float
        Objective value of the obtained matching. 
    algo_history_hhd : list
        List of the sequence of nodes corresponding to households treated by the path-growing algorithm. 
    algo_history_dwe : list
        List of the sequence of nodes corresponding to dwellings treated by the path-growing algorithm.
    path_gr_time : float
        Run time of the path-growing algorithm in this iteration of the LRBAM
    hhds_assigned_count : int
        Number of assigned households in the matching obtained by the path-growing algorithm.

"""

from scipy.sparse import lil_matrix
import numpy as np
import random 
import time

def verboseprint(*args, verbose=False, **kwargs):

    """
    This function makes the verbose argument control the messages printed on the screen.  
    """

    if verbose:
        print(*args, **kwargs)

def path_growing_algorithm(postprocessing,
                            H,
                            D,
                            kappa, 
                            verbose: bool,
                            warm_start_hhd: list,
                            warm_start_dwe: list):
    
    """
    This function runs the path-growing algorithm in the LRBAM framework.  
    """

    # Set seed
    # We need to randomize list of dwelling indices to avoid trend of let last dwellings of the data set as non-assigned
    random.seed(42)

    num_rows, num_cols = kappa.shape

    # Initialize the matchings built by the algorithm and their respective weights
    M1 = lil_matrix((num_rows, num_cols), dtype=int)
    M2 = lil_matrix((num_rows, num_cols), dtype=int)
    M1_weight = 0
    M2_weight = 0

    # Initialize the list of available households, the list of available dwellings, and shuffle them
    available_hhds = np.array(H)
    random.shuffle(available_hhds)
    available_dwes = np.array(D)
    random.shuffle(available_dwes)

    # Initialize the lists that represent the sequence of nodes trated by the algorithm, which are used for
    # the warm start procedure
    algo_history_hhd = []
    algo_history_dwe = []

    # Initialize the flag that informs if warm start exists
    warm_start_exists = False 

    # Initialize counting of number of hhds assigned in M1 and M2 by warm start
    M1_hhds_assigned_count = 0
    M2_hhds_assigned_count = 0

    # Insert warm start if exists
    if len(warm_start_dwe) > 0:

        warm_start_exists = True 

        # Update algorithm history
        algo_history_hhd = warm_start_hhd.copy()
        algo_history_dwe = warm_start_dwe.copy()

        # Update M1 and its weight
        for j in range(len(warm_start_dwe)):
            h = warm_start_hhd[j] 
            d = warm_start_dwe[j]
            if h != -1:
                if d != -1:
                    M1[h, d] = 1
                    M1_weight += kappa[h, d]
                    M1_hhds_assigned_count += 1

        # Update M2 and its weight
        for j in range(len(warm_start_dwe)):
            h = warm_start_hhd[j+1]
            d = warm_start_dwe[j]
            if d != -1:
                if h != -1:
                    M2[h, d] = 1
                    M2_weight += kappa[h, d]
                    M2_hhds_assigned_count += 1

        print("\nResults after warm start:")
        print("Penalized objective value of M1:", round(M1_weight, 4))
        print("Number of households assigned in M1: " + str(M1_hhds_assigned_count) + "/" + str(len(H)))
        print("Penalized objective value of M2", round(M2_weight, 4))
        print("Number of households assigned in M2: " + str(M2_hhds_assigned_count) + "/" + str(len(H)))

        # Update available households and dwellings after warm start
        available_hhds = available_hhds[~np.isin(available_hhds, warm_start_hhd[:len(warm_start_hhd)-1])]
        available_dwes = available_dwes[~np.isin(available_dwes, warm_start_dwe)]

    # If warm start do not exists then this will be the initial i
    i = 1

    # Initialize iteration counting
    it = 1

    # Initialize time counting
    path_gr_begin = time.time()

    # Initialize loop that runs the path-growing algorithm after the warm-start
    while len(available_hhds) > 0 and len(available_dwes) > 0:

        # Defines the starting node for a new path
        if warm_start_exists:

            if warm_start_hhd[-1] != -1:
                x = warm_start_hhd[-1]
                verboseprint("The node", x, "is selected from the available hhds due to the warm start.", verbose = verbose)
                warm_start_exists = False 
            else:
                x = available_dwes[0]
                algo_history_dwe.append(x)
                verboseprint("The node", x, "is selected to continue the available dwes from warm start.", verbose = verbose)
                i = 2
                warm_start_exists = False 
        else:
            if i == 1:
                x = available_hhds[0]
                algo_history_hhd.append(x)
                verboseprint("The node", x, "is selected from the available hhds", verbose = verbose)
            else:
                x = available_dwes[0]
                algo_history_dwe.append(x)
                verboseprint("The node", x, "is selected from the available dwes", verbose = verbose)

        # Builds the path
        while True: 

            verboseprint("Iteration number:", it, verbose = verbose) 
            verboseprint("Current evaluated node:", x, verbose = verbose) 
            
            # Current node is a household, we try to match with a dwelling in M1
            if i == 1:
                verboseprint("i == 1", verbose = verbose)

                # Get row of kappa with weights from available dwellings 
                available_dwe_weights_in_row_x = kappa[x, available_dwes]

                # Filter positive values
                positive_bool_vector = available_dwe_weights_in_row_x > 0

                # Verify if degree is not zero
                if np.any(positive_bool_vector):
                    # Finding the biggest weight and the corresponding node
                    idx_local = np.argmax(available_dwe_weights_in_row_x[positive_bool_vector])
                    heaviest_node_edge = available_dwes[positive_bool_vector][idx_local]
                    biggest_weight = available_dwe_weights_in_row_x[positive_bool_vector][idx_local]

                    # Update
                    M1[x,heaviest_node_edge] = 1
                    M1_weight += biggest_weight
                    available_hhds = available_hhds[available_hhds != x]
                    x = heaviest_node_edge
                    algo_history_dwe.append(x)
                    M1_hhds_assigned_count += 1
                    verboseprint("Heaviest node found:", heaviest_node_edge, verbose = verbose)
                else:
                    # The node has no neighborhood 
                    verboseprint("We found a null-neighborhood node", verbose = verbose)
                    available_hhds = available_hhds[available_hhds != x] 
                    algo_history_dwe.append(-1)
                    
                    break

            # Current node is a dwelling, we try to match with a hhd in M2
            if i == 2:
                verboseprint("i==2", verbose = verbose)

                # Get column of kappa with weights from available households 
                available_hhd_weights_in_column_x = kappa[available_hhds, x]

                # Filter positive values
                positive_bool_vector = available_hhd_weights_in_column_x > 0

                # Verify if degree is not zero
                if np.any(positive_bool_vector):
                    # Finding the biggest weight and the corresponding node
                    idx_local = np.argmax(available_hhd_weights_in_column_x[positive_bool_vector])
                    heaviest_node_edge = available_hhds[positive_bool_vector][idx_local]
                    biggest_weight = available_hhd_weights_in_column_x[positive_bool_vector][idx_local]
 
                    # Update
                    M2[heaviest_node_edge, x] = 1 
                    M2_weight += biggest_weight
                    available_dwes = available_dwes[available_dwes != x]
                    x = heaviest_node_edge
                    algo_history_hhd.append(x)
                    M2_hhds_assigned_count += 1
                    verboseprint("Heaviest node found:", heaviest_node_edge, verbose = verbose)
                else:
                    # The node has no neighborhood
                    verboseprint("We found a null-neighborhood node", verbose = verbose)
                    available_dwes = available_dwes[available_dwes != x]
                    algo_history_hhd.append(-1)

                    break 
                    
            # Update i 
            i = 3-i

            it += 1 

    # Compute path-growing algorithm run time
    path_gr_end = time.time()
    path_gr_time = path_gr_end - path_gr_begin
    print("\nPath-growing algorithm executed for this model.")
    print("Run time of the path-growing algorithm:", round(path_gr_time, 2), "seconds.")

    # Print results
    print("Number of households assigned in M1: " + str(M1_hhds_assigned_count) + "/" + str(len(H)))
    print("Penalized objective value of M1:", round(M1_weight, 4))
    print("Number of households assigned in M2:" + str(M2_hhds_assigned_count) + "/" + str(len(H)))
    print("Penalized objective value of M2:", round(M2_weight, 4))

    # Select heavier matching
    if M1_weight >= M2_weight:
        print("\nThe selected matching is M1")
        matching = M1
        weight_penal = M1_weight
        hhds_assigned_count = M1_hhds_assigned_count
        del M2
    else:
        print("\nThe selected matching is M2")
        matching = M2
        weight_penal = M2_weight
        hhds_assigned_count = M2_hhds_assigned_count
        del M1

    verboseprint("\nThe history of households is:", verbose = verbose)
    verboseprint(algo_history_hhd, verbose = verbose)
    verboseprint("The history of dwellings is:", verbose = verbose)
    verboseprint(algo_history_dwe, verbose = verbose)

    # If LRBAM-IWPP is running then execute the iteration-wise post-processing to generate maximal matching
    if postprocessing == "iwpp":

        iwpp_begin_time = time.time()

        print("\nRunning iteration post-processing...")

        # Get list of non-assigned households and dwellings and shuffle them
        free_hhd = sorted(set(H) - set(matching.nonzero()[0]))
        free_dwe = sorted(set(D) - set(matching.nonzero()[1]))
        free_dwe = np.array(free_dwe)
        random.shuffle(free_hhd)
        random.shuffle(free_dwe)

        for h in free_hhd:

            # Get row of kappa with weights from free dwellings 
            free_dwe_weights_in_row_h = kappa[h, free_dwe]

            # Filter positive values
            positive_bool_vector = free_dwe_weights_in_row_h > 0

            # Verify if degree is not zero
            if np.any(positive_bool_vector):
                # Finding the biggest weight and the corresponding node
                idx_local = np.argmax(free_dwe_weights_in_row_h[positive_bool_vector])
                heaviest_d = free_dwe[positive_bool_vector][idx_local]
                biggest_weight = free_dwe_weights_in_row_h[positive_bool_vector][idx_local]

                # Update matching, overall weight, and free_dwe
                matching[h, heaviest_d] = 1
                weight_penal += biggest_weight
                free_dwe = free_dwe[free_dwe != heaviest_d]
                hhds_assigned_count += 1

        iwpp_end_time = time.time()
        print("\nThe time used by this round of IWPP is", round(iwpp_end_time - iwpp_begin_time, 2), "seconds.")

    # Convert type of matching matrix to use in lrbam.py
    matching_csc = matching.tocsc()

    print("\nMatching obtained.")
    print("Penalized objective value:", round(weight_penal, 4))
    print("Number of households assigned in the matching obtained in this iteration: " + str(hhds_assigned_count) + "/" + str(len(H)))

    return matching_csc, weight_penal, algo_history_hhd, algo_history_dwe, path_gr_time, hhds_assigned_count
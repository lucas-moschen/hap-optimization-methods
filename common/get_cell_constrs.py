# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

Take dwelling dataframe and create dicts specific_coefficient_for_B_k_hhd and specific_coefficient_for_B_k_per. 
The first dict has as keys the IDs of the n grid cells with largest number of residential 
dwellings and the value prop_hhd as values, while the second has as keys the IDs of the n grid cells with largest sum of 
capacities of dwellings and the value prop_per as values. In the thesis, prop_hhd and prop_per are equal to 
\vartheta (see Section 8.2 of the thesis). 

    Parameters
    ----------
    dwe_df_path : str 
        Path to dwelling dataframe.
    n : int 
        Number of constrained grid cells for each type of side constraints.
    prop_hhd : float
        Value inserted in the values of specific_coefficient_for_B_k_hhd (equal to \vartheta of Section 8.2 of the thesis). 
    prop_per : float
        Value inserted in the values of specific_coefficient_for_B_k_per (equal to \vartheta of Section 8.2 of the thesis).
    verbose : bool
        If True, information about the execution of this code is printed. 

    Returns
    ----------
    specific_coefficient_for_B_k_hhd : dict 
        Dictionary of the form {ID of grid cell: prop_hhd}
    specific_coefficient_for_B_k_per : dict 
        Dictionary of the form {ID of grid cell: prop_per}
"""

import pandas as pd

def get_cell_constrs(dwe_df_path: str, n: int, prop_hhd: float, prop_per: float, verbose: bool):

    if n > 0:

        # Load dwelling dataframe
        dwelling_df = pd.read_csv(dwe_df_path)

        # Compute sum of capacities per grid cell
        capacity_sums = dwelling_df.groupby('Gitter_ID_100m')['capacity'].sum()

        # Select the n grid cells with biggest sum of capacities
        top_cap = capacity_sums.nlargest(n)
        top_cap_ids = capacity_sums.nlargest(n).index

        if verbose == True:
            print("\nThe grid cells with biggest sum of capacities of dwellings:")
            print(top_cap)

        # Create dict for people constraints
        specific_coefficient_for_B_k_per = {grid_id: prop_per for grid_id in top_cap_ids}
            
        # Delete non-residential dwellings
        residential_dwelling_dataframe = dwelling_df[dwelling_df['capacity'] != 0]
        
        # Count dwellings per grid cell
        res_dwe_counts = residential_dwelling_dataframe.groupby('Gitter_ID_100m').size()

        # Select the n grid cells with most residential dwellings
        top_res_dwe = res_dwe_counts.nlargest(n)
        top_res_dwe_ids = res_dwe_counts.nlargest(n).index

        if verbose == True:
            print("\nThe grid cells with biggest number of residential dwellings:")
            print(top_res_dwe)

        # Create dict of hhd constraints
        specific_coefficient_for_B_k_hhd = {grid_id: prop_hhd for grid_id in top_res_dwe_ids}

        if verbose == True:
            print("\nThe dictionaries returned are:")
            print("For hhd grid cell constraints:")
            print(specific_coefficient_for_B_k_hhd)
            print("For per grid cell constraints:")
            print(specific_coefficient_for_B_k_per)
    else:

        specific_coefficient_for_B_k_hhd = None
        specific_coefficient_for_B_k_per = None
    
    return specific_coefficient_for_B_k_hhd, specific_coefficient_for_B_k_per
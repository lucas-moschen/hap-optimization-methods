# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script contains the function that creates the W matrices. 
This function takes the data set files and creates a CSV file with the selected W matrix in dense form.
This script is ready to create 3 different types of W matrix:
* w_0: when the household data sets only have information about their size and the dwelling data sets only have information about 
    their capacity, but the feasibility condition for the assignments is p_{h} <= c_{d} + 1 
* w_1: when the household data sets only have information about their size and the dwelling data sets only have information about 
    their capacity, but the feasibility condition for the assignments is p_{h} <= c_{d}
* w_2: when the household data sets have information about their size, income, and coordinates of head of household's workplace, 
    while the dwelling data sets have information about their capacity, monthly cost, and coordinates.

    Parameters
    ----------
    w_type : string
        String corresponding to the type of W matrix selected to be created from the datasets. 
        It must be w_0, w_1, or w_2. 
    sub_dir : string, optional
        Sub-directory that contains the data sets. It must start from the directory level of this file.
        The default is "data/datasets".       
    data_words : list, optional
        List of sub-strings used to find the data set files.
        The default is None.
    ignore_data_words : list, optional
        List of sub-strings used to ignore data set files.
        The default is None.       
    save_path : string, optional
        Directory where the CSV file of the W matrix created is saved.
        The default is "".
        If the default is selected, the directory chosen will be the sub-directory "data/w_matrices".
    p: list, optional
        The expotents of the formula to compute the weights for the w_3 matrix.
        The default is [2,1,1].
    tau: list, optional
        The "importance coefficients" for the weight formula of the w_3 matrix.
        The default is [0.4, 0.4, 0.2]. 

    Returns 
    ----------
    None.
"""

from common.get_files import get_file_path, get_path_to_folder
from scipy.spatial import distance
import os
import pandas as pd
import csv 
import time
import numpy as np

def w(w_type: str,
        sub_dir: str = "data/datasets",
        data_words: list = None,
        ignore_data_words: list = None,
        save_path: str = "",
        p: list = [2,1,1],
        tau: list = [0.4,0.4,0.2]):
    
    begin = time.time() 
    
    # Get the paths to the data set files
    if data_words is not None:
        path_to_files = get_file_path(sub_dir,
                                      file_words=data_words,
                                      ignore_file_words=ignore_data_words,
                                      num_files=2)
    else:
        path_to_files = get_file_path(sub_dir,
                                      file_words=[".csv"],
                                      ignore_file_words=ignore_data_words,
                                      num_files=2)

    # Initialize the dictionary that will store the data set file paths   
    data_paths_dict = {}

    for path in path_to_files:
        print("\nthe path of the dataset is:", path)
        name_of_file = path.split("/")[-1].split(".")[0].lower()
        print("the correspondent name of file is:", name_of_file)
        type_of_file = ["h" if "hold" in name_of_file
                        else "d" if "house" in name_of_file else None][0]
        if type_of_file:
            data_paths_dict[type_of_file] = path

    # Create w_0 matrix
    if w_type == "w_0":

        # Get dwelling data set
        dwelling_df = pd.read_csv(data_paths_dict["d"])
        dwelling_df = dwelling_df.reset_index(drop=True)

        # Get the capacity of dwellings
        c_d = dwelling_df["capacity"]
            
        del dwelling_df

        # Get household data set
        household_df = pd.read_csv(data_paths_dict["h"])
        household_df = household_df.astype({"size": int})

        # Get the size of the households
        p_h = household_df["size"]
            
        del household_df
        
        len_p_h = len(p_h)
        len_c_d = len(c_d)

        # Take the name of the district or municipality       
        save_str = name_of_file.split("_")[-1].split(".")[0]  
        save_str = "w_0_" + save_str 
        
        # Define the path to save the matrix          
        if save_path == "":
            save_path = get_path_to_folder("data/w_matrices")
            save_path = os.path.join(save_path, save_str + ".csv")
        else:
            save_path = os.path.join(save_path, save_str + ".csv")

        # Open file and write the W matrix
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Create the W matrix elements row per row (i.e., household per household)
            for h in range(len_p_h):
                
                # Initialize row of household h
                row_h = np.zeros(len_c_d)

                # Loop to update the non-null elements
                for d in range(len_c_d):

                    # Verify feasibility of possible assignment
                    if (c_d[d] + 1) >= p_h[h]:
                        if c_d[d] > 0:

                            # Compute value
                            if (c_d[d] + 1) == p_h[h]:
                                val = 0.3333
                            else:
                                val = 1.0/(1 + c_d[d] - p_h[h])
                                val = round(val,4)

                            row_h[d] = val

                # Write row in file
                writer.writerow(row_h)

                # Show progress at each 1000 households
                if h % 1000 == 0:
                    print(f'Line of household {h} saved...')
        
        print("\nSaved the w matrix at:", save_path)
                    
        end = time.time() 
        
        print("The build time of the w matrix is", end-begin) 
                    
        return None

    # Create w_1 matrix
    if w_type == "w_1":

        # Get dwelling data set
        dwelling_df = pd.read_csv(data_paths_dict["d"])
        dwelling_df = dwelling_df.reset_index(drop=True)

        # Get the capacity of dwellings
        c_d = dwelling_df["capacity"]
            
        del dwelling_df

        # Get household data set
        household_df = pd.read_csv(data_paths_dict["h"])
        household_df = household_df.astype({"size": int})

        # Get the size of the households
        p_h = household_df["size"]
            
        del household_df
        
        len_p_h = len(p_h)
        len_c_d = len(c_d)

        # Take the name of the district or municipality       
        save_str = name_of_file.split("_")[-1].split(".")[0]  
        save_str = "w_1_" + save_str 
        
        # Define the path to save the matrix          
        if save_path == "":
            save_path = get_path_to_folder("data/w_matrices")
            save_path = os.path.join(save_path, save_str + ".csv")
        else:
            save_path = os.path.join(save_path, save_str + ".csv")

        # Open file and write the W matrix
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Create the W matrix elements row per row
            for h in range(len_p_h):
                
                # Initialize row of household h
                row_h = np.zeros(len_c_d)

                # Loop to update the non-null elements
                for d in range(len_c_d):

                    # Verify feasibility of possible assignment
                    if c_d[d] >= p_h[h]:

                        # Compute value
                        val = 1.0/(1 + c_d[d] - p_h[h])
                        val = round(val,4)
                        row_h[d] = val

                # Write row in file
                writer.writerow(row_h)

                # Show progress at each 1000 households
                if h % 1000 == 0:
                    print(f'Line of household {h} saved...')
        
        print("\nSaved the w matrix at:", save_path)
                    
        end = time.time() 
        
        print("The build time of the w matrix is", end-begin) 
                    
        return None
    
    # Create w_2 matrix
    if w_type == "w_2":

        # Get dwelling data set
        dwelling_df = pd.read_csv(data_paths_dict["d"])
        dwelling_df = dwelling_df.reset_index(drop=True)

        # Get the X coordinate of dwellings
        x_coord_d = dwelling_df["X"]
        
        # Get the Y coordinate of dwellings
        y_coord_d = dwelling_df["Y"]
        
        # Get the cost of dwellings
        cost = dwelling_df["cost"]
        
        # Get the capacity of dwellings
        c_d = dwelling_df["capacity"]
            
        del dwelling_df

        # Get household data set
        household_df = pd.read_csv(data_paths_dict["h"])
        household_df = household_df.astype({"size": int})
            
        # Create the variable income
        income = household_df["income"]
        
        # Create the variable x coordinate
        x_coord_h = household_df["X"]
        
        # Create the variable y coordinate
        y_coord_h = household_df["Y"]

        # Create the variable p_h (number of people in the household h)
        p_h = household_df["size"]
            
        del household_df
        
        len_p_h = len(p_h)
        len_c_d = len(c_d)

        # Initialize the max and min values of |0.3*income-cost|, capacity-size, and distance between coordinates of 
        # dwelling and coordinates of head of household's workplace
        c_p_max = float("-inf")
        inc_cost_max = float("-inf")
        dist_max = float("-inf")
        c_p_min = float("inf")
        inc_cost_min = float("inf")
        dist_min = float("inf") 

        # Loop to identify maximum and minimum values
        for h in range(len_p_h):
            
            for d in range(len_c_d):

                # Check feasibility of possible assignment
                if c_d[d] >= p_h[h]:
                    if income[h] >= cost[d]:

                        # Update max and min for capacity-size if necessary
                        if c_d[d] - p_h[h] > c_p_max:
                            c_p_max = c_d[d] - p_h[h]
                        if c_d[d] - p_h[h] < c_p_min:
                            c_p_min = c_d[d] - p_h[h]

                        # Update max and min for |0.3*income-cost| if necessary
                        if abs((0.3 * income[h]) - cost[d]) > inc_cost_max:
                            inc_cost_max = abs((0.3 * income[h]) - cost[d]) 
                        if abs((0.3 * income[h]) - cost[d]) < inc_cost_min:
                            inc_cost_min = abs((0.3 * income[h]) - cost[d]) 

                        # Update max and min for distance between coordinates of dwelling and coordinates of head of 
                        # household's workplace if necessary
                        coord_d = (x_coord_d[d], y_coord_d[d])
                        coord_h = (x_coord_h[h], y_coord_h[h])
                        val = distance.euclidean(coord_d, coord_h)
                        if val > dist_max:
                            dist_max = val
                        if val < dist_min:
                            dist_min = val 

            # Show progress at each 1000 households
            if h % 1000 == 0:
                print(f'Household {h} passed by investigation of min and max for the weights...')

        # Take the name of the district or municipality       
        save_str = name_of_file.split("_")[-1].split(".")[0]  
        save_str = "w_2_" + save_str + str(tau) 
        
        # Define the directory where W will be saved         
        if save_path == "":
            save_path = get_path_to_folder("data/w_matrices")
            save_path = os.path.join(save_path, save_str + ".csv")
        else:
            save_path = os.path.join(save_path, save_str + ".csv")

        # Open file and write the W matrix
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Create the W matrix elements row per row (i.e., household per household)
            for h in range(len_p_h):
                
                # Initialize row of household h
                row_h = np.zeros(len_c_d)

                # Loop to update the non-null elements
                for d in range(len_c_d):

                    # Check feasibility of possible assignment
                    if c_d[d] >= p_h[h]:
                        if income[h] >= cost[d]:

                            # Compute part of the weight corresponding to difference between capacity of dwelling and 
                            # size of household
                            w_line = ((c_p_max - (c_d[d] - p_h[h]))/(c_p_max - c_p_min))**p[0]

                            # Compute part of the weight corresponding to difference between cost of dwelling and 
                            # 0.3*(income of household)
                            w_2line = ((inc_cost_max - abs((0.3 * income[h]) - cost[d]))/(inc_cost_max - inc_cost_min))**p[1]

                            # Compute part of the weight corresponding to distance between coordinates of dwelling and 
                            # coordinates of head of household's workplace
                            coord_d = (x_coord_d[d], y_coord_d[d])
                            coord_h = (x_coord_h[h], y_coord_h[h])
                            dist = distance.euclidean(coord_d, coord_h)
                            w_3line = ((dist_max - dist)/(dist_max - dist_min))**p[2]

                            val = (tau[0] * w_line) + (tau[1] * w_2line) + (tau[2] * w_3line)
                            val = round(val, 4)

                            row_h[d] = val

                # Write row in file
                writer.writerow(row_h)

                # Show progress at each 1000 households
                if h % 1000 == 0:
                    print(f'Line of household {h} saved...')   
        
        print("\nSaved the w matrix at:", save_path)
                    
        end = time.time() 
        
        print("The build time of the w matrix is", end-begin)
                    
        return None

if __name__ == "__main__":

    import sys 
    import ast 

    print(sys.argv)
    # Transform lists in string form into lists
    data_words = ast.literal_eval(sys.argv[2])
    tau = ast.literal_eval(sys.argv[3])

    w(w_type = sys.argv[1],
      data_words = data_words,
      tau = tau)
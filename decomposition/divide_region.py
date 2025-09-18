# -*- coding: utf-8 -*-
"""
@author: Lucas Moschen

This script contains the class that uses the grid cell structure to divide the region defined by its input dwelling dataframe.

Parameters
    ----------
    dataframe_to_be_divided : DataFrame
        The input dwelling dataframe that will be divided.
    dict_indices_K : dict
        It is a dictionary where the keys are the grid cell indices and the values are the grid cell IDs.

    Returns
    -------
    self.dataframe_to_be_divided : DataFrame
        A dataframe corresponding to the left half of the area containing the dwellings of dataframe_to_be_divided.
    self.new_dataframe : DataFrame
        A dataframe corresponding to the right half of the area containing the dwellings of dataframe_to_be_divided.

"""

import copy 
import pandas as pd 

class DivideRegion:

    def __init__(self,
                    dataframe_to_be_divided: pd.DataFrame,  
                    dict_indices_K: dict):
        
        # Initialize the parameters
        self.dataframe_to_be_divided = copy.deepcopy(dataframe_to_be_divided)
        self.dict_indices_K = dict_indices_K 

    def create_list_of_vertical_coordinates_and_list_of_horizontal_coordinates(self):

        """
        These coordinates refer to the lower left corner of the grid cells.
        These lists are important to decide if we shall divide the region horizontally or vertically.
        """

        self.list_vertical_coords = []
        self.list_horizontal_coords = []
        self.dict_of_coords = {}

        # Get the list of all grid cell indices
        list_K = list(self.dataframe_to_be_divided["grid_cell"].unique())

        for k in list_K:
            # Get ID of grid cell k
            id = self.dict_indices_K[k]

            # Get north coordinate and east coordinate of the grid cell
            N_coord = copy.deepcopy(id[(id.find("N")+1):id.find("E")])
            E_coord = copy.deepcopy(id[(id.find("E")+1):(len(id)+1)])

            # Add the coordinates to the respective list if they are not there yet
            if N_coord not in self.list_vertical_coords:
                self.list_vertical_coords.append(N_coord)
            if E_coord not in self.list_horizontal_coords:
                self.list_horizontal_coords.append(E_coord)

            self.dict_of_coords[k] = [N_coord, E_coord]

    def decision_if_divide_region_horizontally_or_vertically(self):

        """
        This function decides if we shall divide the region horizontally or vertically.
        This is decided based in which coordinates list is bigger.
        """

        if len(list(set(self.list_vertical_coords))) >= len(list(set(self.list_horizontal_coords))):
            self.list_vertical_coords = sorted(self.list_vertical_coords)
            self.cut_coordinate = self.list_vertical_coords[len(self.list_vertical_coords) // 2]
            self.direction_of_cut = "vertical" 
        else:
            self.list_horizontal_coords = sorted(self.list_horizontal_coords)
            self.cut_coordinate = self.list_horizontal_coords[len(self.list_horizontal_coords) // 2]
            self.direction_of_cut = "horizontal"

    def divide(self):

        """This function makes the division of the region."""

        if self.direction_of_cut == "vertical":
            number = 0
        else:
            number = 1

        # Add the information on the important grid cell coordinates to the dataframe
        imp_coord = [0]*len(self.dataframe_to_be_divided["grid_cell"].tolist())
        position = 0
        for k in self.dataframe_to_be_divided["grid_cell"].tolist():
            imp_coord[position] = self.dict_of_coords[k][number]
            position += 1
        self.dataframe_to_be_divided["imp_coord"] = imp_coord

        # Create the dataframe corresponding to one of the halves
        self.new_dataframe = self.dataframe_to_be_divided.loc[self.dataframe_to_be_divided["imp_coord"] > self.cut_coordinate]
        # Creates the dataframe corresponding to the other half
        self.dataframe_to_be_divided.drop(self.new_dataframe.index.tolist(), inplace = True) 

        # Remove imp_coord column from both dataframes to keep standard dataframe structure
        del self.new_dataframe["imp_coord"]
        del self.dataframe_to_be_divided["imp_coord"] 

    def run(self):
        
        self.create_list_of_vertical_coordinates_and_list_of_horizontal_coordinates()
            
        self.decision_if_divide_region_horizontally_or_vertically()

        self.divide() 

        return self.dataframe_to_be_divided, self.new_dataframe
# Main file

from decomposition.decomposition import Test 
from lrbam.lrbam import LRBAM 
from greedy_heuristic.greedy_heuristic import GREEDY_HEURISTIC
from path_growing_heuristic.path_growing_heuristic import PATH_GROWING_HEURISTIC
from common.get_cell_constrs import get_cell_constrs
from common.get_files import get_file_path, get_path_to_folder
import sys 
import ast 
import os 

print(sys.argv)
# Transform lists in string form into lists
data_words = ast.literal_eval(sys.argv[1])

# Get file_words
file_words_dwe = ["Houses"]
file_words_hhd = ["Households"]
for elem in data_words:
    file_words_dwe.append(elem)
    file_words_hhd.append(elem)

# Get dwelling dataframe path
dwe_df_path_list = get_file_path(sub_dir = "data/datasets", file_words = file_words_dwe)
dwe_df_path = dwe_df_path_list[0]

# Get full W path
w_directory = get_path_to_folder("data/w_matrices")  # The user can change this variable if this sub-directory is not being used 
w_name = sys.argv[2]
w_path = os.path.join(w_directory, w_name)


# Get grid cell dict constraints
specific_coefficient_for_B_k_hhd, specific_coefficient_for_B_k_per = get_cell_constrs(dwe_df_path, 
                                                                                    int(sys.argv[3]), 
                                                                                    float(sys.argv[4]), 
                                                                                    float(sys.argv[5]),
                                                                                    False)

# Run the selected solution process
if sys.argv[6] == "exact":

    method = Test(data_words=data_words,
                    path_to_w=w_path,
                    specific_coefficient_for_B_k_hhd = specific_coefficient_for_B_k_hhd,
                    specific_coefficient_for_B_k_per = specific_coefficient_for_B_k_per,
                    gurobi_verbose = True,
                    hhd_size_decomposition=False,
                    amount_to_decompose = 4000000000000000000000000000)
    
    method.run() 
elif sys.argv[6] == "reg_decomp": 

    method = Test(data_words=data_words,
                    path_to_w=w_path,
                    specific_coefficient_for_B_k_hhd = specific_coefficient_for_B_k_hhd,
                    specific_coefficient_for_B_k_per = specific_coefficient_for_B_k_per,
                    gurobi_verbose = True,
                    hhd_size_decomposition=False,
                    amount_to_decompose = int(sys.argv[7]),
                    alpha = float(sys.argv[8]),
                    beta = float(sys.argv[9]),
                    gamma = float(sys.argv[10]),
                    assignments_tol = int(sys.argv[11]))
    
    method.run()
elif sys.argv[6] == "hhd_decomp": 

    method = Test(data_words=data_words,
                    path_to_w=w_path,
                    specific_coefficient_for_B_k_hhd = specific_coefficient_for_B_k_hhd,
                    specific_coefficient_for_B_k_per = specific_coefficient_for_B_k_per,
                    gurobi_verbose = True,
                    hhd_size_decomposition = True,
                    amount_to_decompose = int(sys.argv[7]),
                    alpha = float(sys.argv[8]),
                    beta = float(sys.argv[9]),
                    gamma = float(sys.argv[10]))
    
    method.run()
elif sys.argv[6] in ["lrbam_spp", "lrbam_iwpp"]:

    # Get post-processing type
    postpr = sys.argv[6].split("_")[-1]

    method = LRBAM(data_words=data_words,
                    path_to_w=w_path,
                    initial_lambdas_hhd=float(sys.argv[7]),
                    initial_lambdas_per=float(sys.argv[8]),
                    increase=float(sys.argv[9]),
                    postprocessing = postpr,
                    specific_coefficient_for_B_k_hhd = specific_coefficient_for_B_k_hhd,
                    specific_coefficient_for_B_k_per = specific_coefficient_for_B_k_per)
    
    method.optimize() 
elif sys.argv[6] == "greedy":

    method = GREEDY_HEURISTIC(data_words=data_words,
                    path_to_w=w_path,
                    specific_coefficient_for_B_k_hhd = specific_coefficient_for_B_k_hhd,
                    specific_coefficient_for_B_k_per = specific_coefficient_for_B_k_per)
    
    method.optimize() 
elif sys.argv[6] == "path_gr":

    method = PATH_GROWING_HEURISTIC(data_words=data_words,
                                    path_to_w=w_path,
                                    specific_coefficient_for_B_k_hhd = specific_coefficient_for_B_k_hhd,
                                    specific_coefficient_for_B_k_per = specific_coefficient_for_B_k_per)
    
    method.optimize()
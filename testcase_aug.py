import numpy as np
from smote import Smote
def change_to_vector(string):
    """
    transform string to vector
    :param string: string 
    """
    string_split = string.split()
    vector_list = [float(x) for x in string_split]
    return vector_list

def cos_sim(vector_a, vector_b):
    """
    compute the similarity of vector_a and vector_b
    :param vector_a: vector a 
    :param vector_b: vector b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if not denom == 0:
        cos = num / denom
    else:
        cos = -1
    sim = 0.5 + 0.5 * cos
    return sim

file_path_component_info = "componentinfo.txt"
file_path_slice_result = "sliceResult.txt"
file_path_matrix = "covMatrix.txt"  # file path
file_path_matrix_global = "covMatrix.txt"  # file path
file_path_error = "error.txt"  # file path
file_path_result = "result/result.txt"  # file path
file_path_cosin_result = "result/cos_result.txt"  # file path
file_result = open(file_path_result,'a')
file_cosin_result = open(file_path_cosin_result,'a')

component_info = []
slice_result = []
matrix_list = []     #matrix file
matrix_global_list = []     #matrix file
error_list = []      #error file
error_matrix_list = []     #error matrix index

for line in open(file_path_component_info):
    component_info.append(line.strip())
for line in open(file_path_slice_result):
    slice_result.append(line.strip()) 
for line in open(file_path_matrix):
    matrix_list.append(line.strip())
for line in open(file_path_error):
    error_list.append(line.strip())
for line in open(file_path_matrix_global):
    matrix_global_list.append(line.strip())


component_info_vector = change_to_vector(component_info[1])
slice_result_vector = change_to_vector(slice_result[1])
slice_value_for_compute = []
for item in component_info_vector:
    if item in slice_result_vector:
        slice_value_for_compute.append('1')
    else:
        slice_value_for_compute.append('0')

for error_element_index in range(len(error_list)):
    if error_list[error_element_index] == '1':
        error_matrix_list.append(error_element_index)
file_result.write(str(len(error_matrix_list)))
file_result.write('\n')

file_result.write(str(error_matrix_list))
file_result.write('\n')
"""
print error_element_index and error_element
"""

for error_element_index in error_matrix_list:
    file_result.write(str(error_element_index))
    file_result.write('\n')
    file_result.write(matrix_list[error_element_index])
    file_result.write('\n')

matrix_global_vector = []
#for element in matrix_global_list:
#    matrix_global_vector.append(change_to_vector(element)) 
for item in  error_matrix_list:
    matrix_global_vector.append(change_to_vector(matrix_global_list[item]))
"""
print cos_sim
"""

for error_element in error_matrix_list:
    file_cosin_result.write("error statement:"+str(error_element))
    file_cosin_result.write('\n')
    target = change_to_vector(matrix_list[error_element])
    for i in error_matrix_list:
        erro_statement_vector = change_to_vector(matrix_list[i])
        file_cosin_result.write(str(i)+':'+str(cos_sim(target,erro_statement_vector)))
        file_cosin_result.write('\n')
    for i in range(0,len(matrix_list)):
        if i not in error_matrix_list:
            correct_statement_vector = change_to_vector(matrix_list[i])
            file_cosin_result.write(str(i)+':'+str(cos_sim(target,correct_statement_vector)))
            file_cosin_result.write('\n')
    file_cosin_result.write('\n')

"""
compute the intersection of all the failed test cases
"""
intersection_array = np.array(change_to_vector(matrix_list[error_matrix_list[0]]))
for error_element in error_matrix_list: 
    for i in range(0, len(intersection_array)):
        intersection_array[i] = intersection_array[i] * np.array(change_to_vector(matrix_list[error_element]))[i]
"""
compute smote test cases
"""
erro_statement_array = np.zeros((len(error_matrix_list), len(erro_statement_vector)),int)    
for i in range(0,len(erro_statement_array)):
    erro_statement_array[i] = change_to_vector(matrix_list[error_matrix_list[i]])


erro_statement_array_clean = []
for i in erro_statement_array.tolist():
    if not i in erro_statement_array_clean:
        erro_statement_array_clean.append(i)

N = round((len(matrix_list)-len(error_matrix_list)*2)/5)
s=Smote(np.array(erro_statement_array_clean),N)              #a为少数数据集，N为倍率，即从k-邻居中取出几个样本点
new_test_cases = s.over_sampling()  

file_path_newtestcases = "result/newMatrix.txt"  # 文件路径
file_newtestcases = open(file_path_newtestcases,'a')
file_path_new_error = "result/newError.txt"  # 文件路径
file_new_error = open(file_path_new_error,'a')
for i in range(0, len(erro_statement_array_clean)*N):
    file_new_error.write('1')
    file_new_error.write('\n')


for element in new_test_cases:
    for i in range(len(element)):
        element[i] = element[i]*float(slice_value_for_compute[i])


for i in range(0, len(new_test_cases)):
    for j in range(0, len(new_test_cases[i])):
        file_newtestcases.write(str(new_test_cases[i][j]))
        file_newtestcases.write(' ')
    file_newtestcases.write('\n')

file_result.close()
file_cosin_result.close()
file_newtestcases.close()
file_new_error.close()

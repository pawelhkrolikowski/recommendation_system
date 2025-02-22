# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:40:23 2025

@author: 48661
"""

import numpy as np
import pandas as pd
import os

a = np.array([[1,2,3],[44,55,66]])
#print(a.shape)

a = np.array([[1],[2],[3],[44],[55],[66]])
#print(np.hstack((a,a)))


a = np.array([545,46,32,333])
b = np.array([2,4,7,0])
#print("aaaa ", a)



new_arr = np.vstack((a,b)).T
print(new_arr)

new_arr = new_arr[new_arr[:, 1].argsort()]

print("NEW :" )
print(new_arr )
print("HELLO ", new_arr[-2:,:])

file_name = "data"
raw_data = pd.read_csv(os.getcwd() + '\\input\\' + file_name + ".csv", index_col=0)

if set(['user_id','movie_id','rating']) <= set(raw_data.columns):
    print("YES there are in ")

def create_pivot_numpy(df_table, index = "feature_name_row", column = "feature_name_column", value = "feature_name_value"):
    

    # check if provided column are within provided df
    if set([index,column,value]) <= set(raw_data.columns):
    
    
        # convert Pandas DataFrame into Numpy Array
        data = df_table.values
        
        # expected format of the input : key & values as column name and column id
        column_map = {index : df_table.columns.get_loc(index),
                      column : df_table.columns.get_loc(column),
                      value : df_table.columns.get_loc(value),
                      }
        # Create USER x MOVIE matrix with RATINGS as values
        # USERs as rows,
        # MOVIEs as columns,
        # RATINGs as values,
        
        # return_inverse = True - return indicies from raw numpy array (values from df pandas)
        rows_unique, row_pos = np.unique(data[:, column_map[index]], return_inverse=True)
        cols_unique, col_pos = np.unique(data[:, column_map[column]], return_inverse=True)

        # create an empty array with desired shape
        pivot_table_raw = np.empty((len(rows_unique), len(cols_unique)), dtype=float)
        # insert nans
        pivot_table_raw[:] = np.nan
        # insert values from pandas df into numy array
        pivot_table_raw[row_pos, col_pos] = data[:, column_map[value]]

        # return numpy pivot table with unique rows value and unique column value
        
        return pivot_table_raw, rows_unique, cols_unique
        
    else:
        print("Provide index, column and value features are no in provided Data Frame !")
        
        return np.array([0])
    
    
pivot_test, row_uniq, col_uniq  = create_pivot_numpy(raw_data,'user_id','movie_id','rating')






def normalize_matrix(numpy_matrix, axis = 1, fill_na = 0):
    
        if axis == 1:
            # normalize each row wise
            average_per_row = np.nanmean(average_per_row,axis=1)
            
            # normalize each row
            numpy_matrix_normalized = numpy_matrix - average_per_row.reshape(self.user_item_raw.shape[0],1)
            
            # replace nans by provided value
            numpy_matrix_normalized[np.isnan(numpy_matrix_normalized)] = fill_na
            
            return numpy_matrix_normalized
            
            
        elif axis == 0:
            # normalize each row wise
            average_per_row = np.nanmean(average_per_row,axis = 0)
            
            # normalize each row
            numpy_matrix_normalized = numpy_matrix - average_per_row.reshape(self.user_item_raw.shape[0],1)
            
            # replace nans by provided value
            numpy_matrix_normalized[np.isnan(numpy_matrix_normalized)] = fill_na
            
            return numpy_matrix_normalized
        else:
            print("Provide corrected value for axis : 0 or 1 !")
            
            return numpy_matrix
            

def similarity_matrix(matrix, method = "cosine"):
    
        """
        
        Assumtions:
            
        input: matrix: Object x Values
        
        output: matrix = [similarity between two objects based on values (two rows)] for each pair of two objects
        
        comment: only one method implemented, cosine similarity
        
        """
        
        if method == 'cosine':
            # dot product per USER__i x USER_j
            similarity = np.dot(matrix, matrix.T)
            
            # 2. LOWER part of the cosine sim formula
            
            # squared magnitude of preference vectors
            square_mag = np.diag(similarity)
            
            # inverse squared magnitude
            inv_square_mag = 1 / square_mag
            
            # inverse of the magnitude
            inv_mag = np.sqrt(inv_square_mag)
            
            # 3. Final User x User similarity matrix
            user_cosine_sim = similarity * inv_mag
            user_cosine_sim = user_cosine_sim.T * inv_mag
        
        
            return user_cosine_sim
        
        else:
            print("There is only one method implemented - cosine similarity")
            
            return np.dot(matrix, matrix.T)
        
        


















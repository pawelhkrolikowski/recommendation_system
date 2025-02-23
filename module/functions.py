# -*- coding: utf-8 -*-
"""
Recommendation system

Written by Pawel Krolikowski
"""

import numpy as np
import pandas as pd
import os

def square(m):
    """
    Inputs:
    ------
        m - numpy array
    Outputs:
    ------
        bool - True if matrix is squared matrix
    
    """

    return m.shape[0] == m.shape[1]


def same_size(m,n,axis = 1):
    """
    Inputs:
    ------
        m - numpy array
        n - numpy array
        axis - 0 or 1
    Outputs:
    ------
        bool - True if m and n have the same size along provided axis
    
    """
    
    if axis == 1 or axis == 0:
        return m.shape[axis] == n.shape[axis]
    else:
        "Error same_size(): Provide correct axis value: 0 or 1 !!! "
        return False

    
def predict_ratings_for_movie(ith_movie, user_user_cosine_sim, all_user_avg_ratings):
    """
    Inputs:
    ------
        ith_movie:  
            numpy column - consists all ratings for ith movie - n users size
        user_user_cosine_sim:
            user-user sim matrix, shape n users x n users shape
        all_user_avg_ratings:
            numpy column : average rating for each user - n users size
        
    Outputs:
    ------
        numpy column, n users size. 
        Predicted ratings of ith movie from each user, based on positive similarity with others users.
    
    Comment:
    ------
        it is column-wise vectorization for predicting x-user rating for ith movie
        it is for better performance
    """
    
 
    # check if all shapes are correct
    if same_size(ith_movie,user_user_cosine_sim,axis = 0) and same_size(ith_movie,all_user_avg_ratings,axis = 0) and same_size(user_user_cosine_sim,user_user_cosine_sim.T):
    
        # get user's indexes, who rated ith movie
        user_index = np.argwhere(~np.isnan(ith_movie))
        
        # get corresponging rating
        movie_ratings = ith_movie[user_index.ravel()]
        
        # get users average rating 
        #user_avg_ratings = all_user_avg_ratings[user_index.ravel()]
        
        # get all similatities from selected indexes
        all_user_similarities_raw = user_user_cosine_sim[:, user_index.ravel()]
        
        # if negative cosin then mask it as 0 - no impact on predicted rating
        all_user_similarities_raw = np.where(all_user_similarities_raw < 0, 0, all_user_similarities_raw)
        
        # sum average similarity per user
        average_similarity = np.sum(all_user_similarities_raw, axis=1)
        
        # calculate weighted average rating 
        predicted_rating_raw = np.matmul(all_user_similarities_raw, movie_ratings) / (average_similarity - 0.0000000001)
        
        return predicted_rating_raw
        
    else:
        print("Error predict_ratings_for_movie(): Please check if provided arrays are with correct shapes:")
        print(ith_movie.shape)
        print(user_user_cosine_sim.shape)
        print(all_user_avg_ratings.shape)
        
        return ith_movie
    
      
def create_pivot_numpy(df_table, index = "feature_name_row", column = "feature_name_column", value = "feature_name_value"):
    """
    
    Inputs:
    ------
        df_table:  
            pandas dataframe with columns : [feature_name_row,feature_name_column,feature_name_value, [other_columns]. 
        index:
            string - name of the column, which unique values will be corresponds to row index in pivot numpy array
        column:
            string - name of the column, which unique values will be corresponds to column index in pivot numpy array
        value:
            string - name of the column, which values will values in pivot numpy array
            
        
    Outputs:
    ------
        pivot_table_raw:
            numpy array - pivot array of size: #{unique values from 'feature_name_row'} x #{unique values from 'feature_name_column'
        rows_unique, cols_unique:
            numpy array - two one dime arrays which holds unique values used to pivot creation
        
    Comment:
    ------
        The function assumes there is no duplicates for [row, column] in pandas df_table.
        It means we dont need aggregation function.
        If there are duplicates, maximum value is taken.
 

    """

    # check if provided column are within provided df
    if set([index,column,value]) <= set(df_table.columns):
    
        # check duplicates of index, column cominations
        unique_combination = df_table[[index,column]].drop_duplicates().shape[0]
        all_combination = df_table[[index,column]].shape[0]

        if unique_combination != all_combination:
            print("Warning create_pivot_numpy(): There are duplicates ! If user rateded one film by [r1,..,rn] then final rating is max([r1,..,rn])")
            
            # remove duplicates, select the higest rating
            df_table = df_table[[index,column,value]].groupby(by=[index, column]).max()
            df_table = df_table.reset_index(drop=False)
        


        # convert Pandas DataFrame into Numpy Array
        data = df_table.values
        
        # expected format of the input : key & values as column name and column id
        column_map = {index : df_table.columns.get_loc(index),
                      column : df_table.columns.get_loc(column),
                      value : df_table.columns.get_loc(value),
                      }

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
        
        return np.array([0]),np.array([0]),np.array([0])


def normalize_matrix(numpy_matrix, axis = 1, fill_na = 0):
    """
    
    Inputs:
    ------
        numpy_matrix:  
            numpy array, can consist np.Nans
        axis:
            integer : direction of normalization: axis=1 means along each rows, axis=0 alonf columns 
        fill_na:
            float - number to fill nans 

    Outputs:
    ------
        numpy_matrix_normalized:
            numpy array - pivot array of size: #{unique values from 'feature_name_row'} x #{unique values from 'feature_name_column'
        averages:
            numpy array - with averages calulated per each row/column (axis 1/axis 0)
        
    Comment:
    ------
        The function calculates mean over NON-NaNs so for [1,np.nan,3] -> 2
        Normalization is along rows or columns.


    """

    if axis == 1:
        # get averages each row wise
        averages = np.nanmean(numpy_matrix,axis=1)
        
        # normalize each row
        numpy_matrix_normalized = numpy_matrix - averages.reshape(numpy_matrix.shape[0],1)
        
        # replace nans by provided value
        numpy_matrix_normalized[np.isnan(numpy_matrix_normalized)] = fill_na
        
        return numpy_matrix_normalized, averages
        
        
    elif axis == 0:
        # get averages  each column wise
        averages = np.nanmean(numpy_matrix,axis = 0)
        
        # normalize each col
        numpy_matrix_normalized = numpy_matrix - averages.reshape(1,numpy_matrix.shape[1])
        
        # replace nans by provided value
        numpy_matrix_normalized[np.isnan(numpy_matrix_normalized)] = fill_na
        
        return numpy_matrix_normalized, averages
    else:
        print("Provide corrected value for axis : 0 or 1 !")
        
        return numpy_matrix


def similarity_matrix(numpy_matrix, method = "cosine"):
    
    """
    
    Inputs:
    ------
        numpy_matrix:  
            numpy array, rows index are objects, columns contains feature value for object
        method:
            string - provide method for similarity for j_th and i_th objects, so for numpy_matrix[j_th, :] and numpy_matrix[i_th, :]
            Defaulted method is cosine similarity

    Outputs:
    ------
        object_cosine_sim:
            squared numpy array of size #objects x #objects. 
            object_cosine_sim[i_th,j-th] contains similarity between two objects.
      
        
    Comment:
    ------
        Only cosine similarity is implemented. Others for future development.

    """
    
    
    
    if method == 'cosine':
        
        # 1. UPPER part o the cosine sim formula 
        
        # dot product per USER__i x USER_j
        similarity = np.dot(numpy_matrix, numpy_matrix.T)
        
        # 2. LOWER part of the cosine sim formula
        
        # squared magnitude of preference vectors
        square_mag = np.diag(similarity)
        
        # inverse squared magnitude
        inv_square_mag = 1 / square_mag
        
        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)
        
        # 3. UPPER/LOWER : final User x User similarity matrix
        
        # broadcasting multiplication row by row
        object_cosine_sim = similarity * inv_mag
        
        # broadcasting multiplication row by row (with transposed matrix)
        object_cosine_sim = object_cosine_sim.T * inv_mag
    
    
        return object_cosine_sim
    
    else:
        print("There is only one method implemented - cosine similarity")
        
        return np.dot(numpy_matrix, numpy_matrix.T)
        
        
def masking_nan(matrix_with_nan, matrix_with_prediction):
    
    """
    
    Inputs:
    ------
        matrix_with_nan:  
            numpy array, n x k size. Contains Nans
        matrix_with_prediction:
            numpy array, n x k size. WITOUT Nans
         

    Outputs:
    ------
        matrix numpy:
            numpy array, n x k size. 
            Nans value from matrix_with_nan are replaced by matrix_with_prediction, element-wise. 
            Then matrix_with_nan is returned.
  

    """
    
    if (same_size(matrix_with_nan,matrix_with_prediction,axis = 1) == True) and (same_size(matrix_with_nan,matrix_with_prediction,axis = 0) == True):
        
        # Mask 1: replace exisitg numbers by 0, nans by 1
        mask_nan = matrix_with_nan.copy()
        mask_nan[~np.isnan(mask_nan)] = 0
        mask_nan[np.isnan(mask_nan)] = 1
    
        # Mask 2: replace nans by 0
        mask_old_values = matrix_with_nan.copy()
        mask_old_values[np.isnan(mask_old_values)] = 0
        
        # apply masks correction
        
        # mask_with_nan*matrix_with_prediction -> get only cells [i,j] with predictions, rest 0
        # previous + mask_old_values -> for rest  cells [i,j] as it was before.
        
        return mask_nan*matrix_with_prediction + mask_old_values
    
    else:
        print("Provided matrixes are of differents shapes, no replacement applied")
        
        return matrix_with_nan

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
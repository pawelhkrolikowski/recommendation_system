# -*- coding: utf-8 -*-
"""
Recommendation system

Written by Pawel Krolikowski
"""

import numpy as np
import pandas as pd
import os

def square(m):
    return m.shape[0] == m.shape[1]

def same_size(m,n,axis = 1):
    if axis == 1 or axis == 0:
        return m.shape[axis] == n.shape[axis]
    else:
        "Provide correct axis value: 0 or 1 !!! "
        return False
    
def predict_ratings_for_movie(ith_movie, user_user_cosine_sim, all_user_avg_ratings):
    """
    Inputs:
        ith_movie:  
            numpy column - consists all ratings for ith movie - n users size
        user_user_cosine_sim:
            user-user sim matrix, shape n users x n users shape
        all_user_avg_ratings:
            numpy column : average rating for each user - n users size
        
    Outputs:
        numpy column, n users size. 
        Predicted ratings of ith movie from each user, based on positive similarity with others users.
    
    Comment:
        it is column-wise vectorization for predicting x-user rating for ith movie
        it is for better performance
    """
    
    """
    
    Parameters
    ----------
    df_table : TYPE
        pandas df .
    index : TYPE, optional
        DESCRIPTION. The default is "feature_name_row".
    column : TYPE, optional
        DESCRIPTION. The default is "feature_name_column".
    value : TYPE, optional
        DESCRIPTION. The default is "feature_name_value".

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    

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
        print("Please check if provided arrays are with correct shapes:")
        print(ith_movie.shape)
        print(user_user_cosine_sim.shape)
        print(all_user_avg_ratings.shape)
        
        return ith_movie
    
    
    
def create_pivot_numpy(df_table, index = "feature_name_row", column = "feature_name_column", value = "feature_name_value"):
    """
    
    The function assumes there is no duplicates for [row, column] in pandas df_table.
    It means we dont need aggregation function.
    
    Parameters
    ----------
    df_table : TYPE
        pandas df .
    index : TYPE, optional
        DESCRIPTION. The default is "feature_name_row".
    column : TYPE, optional
        DESCRIPTION. The default is "feature_name_column".
    value : TYPE, optional
        DESCRIPTION. The default is "feature_name_value".

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    # check if provided column are within provided df
    if set([index,column,value]) <= set(df_table.columns):
    
        # check duplicates of index, column cominations
        unique_combination = df_table[[index,column]].drop_duplicates().shape[0]
        all_combination = df_table[[index,column]].shape[0]

        if unique_combination != all_combination:
            print("There are duplicates ! If user rateded one film by [r1,..,rn] then final rating is max([r1,..,rn])")
            
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
    
    Parameters
    ----------
    df_table : TYPE
        pandas df .
    index : TYPE, optional
        DESCRIPTION. The default is "feature_name_row".
    column : TYPE, optional
        DESCRIPTION. The default is "feature_name_column".
    value : TYPE, optional
        DESCRIPTION. The default is "feature_name_value".

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    if axis == 1:
        # normalize each row wise
        average_per_row = np.nanmean(numpy_matrix,axis=1)
        
        # normalize each row
        numpy_matrix_normalized = numpy_matrix - average_per_row.reshape(numpy_matrix.shape[0],1)
        
        # replace nans by provided value
        numpy_matrix_normalized[np.isnan(numpy_matrix_normalized)] = fill_na
        
        return numpy_matrix_normalized, average_per_row
        
        
    elif axis == 0:
        # normalize each row wise
        average_per_row = np.nanmean(numpy_matrix,axis = 0)
        
        print(average_per_row)
        
        # normalize each row
        numpy_matrix_normalized = numpy_matrix - average_per_row.reshape(1,numpy_matrix.shape[1])
        
        # replace nans by provided value
        numpy_matrix_normalized[np.isnan(numpy_matrix_normalized)] = fill_na
        
        return numpy_matrix_normalized, average_per_row,
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
    
    """
    
    Parameters
    ----------
    df_table : TYPE
        pandas df .
    index : TYPE, optional
        DESCRIPTION. The default is "feature_name_row".
    column : TYPE, optional
        DESCRIPTION. The default is "feature_name_column".
    value : TYPE, optional
        DESCRIPTION. The default is "feature_name_value".
    
    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    
    """
    
    
    if method == 'cosine':
        
        # 1. UPPER part o the cosine sim formula 
        
        # dot product per USER__i x USER_j
        similarity = np.dot(matrix, matrix.T)
        
        # 2. LOWER part of the cosine sim formula
        
        # squared magnitude of preference vectors
        square_mag = np.diag(similarity)
        
        # inverse squared magnitude
        inv_square_mag = 1 / square_mag
        
        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)
        
        # 3. UPPER/LOWER : final User x User similarity matrix
        
        # broadcasting multiplication row by row
        user_cosine_sim = similarity * inv_mag
        
        # broadcasting multiplication row by row (with transposed matrix)
        user_cosine_sim = user_cosine_sim.T * inv_mag
    
    
        return user_cosine_sim
    
    else:
        print("There is only one method implemented - cosine similarity")
        
        return np.dot(matrix, matrix.T)
        
        
def masking_nan(matrix_with_nan, matrix_with_prediction):
    
    """
    
    Parameters
    ----------
    df_table : TYPE
        pandas df .
    index : TYPE, optional
        DESCRIPTION. The default is "feature_name_row".
    column : TYPE, optional
        DESCRIPTION. The default is "feature_name_column".
    value : TYPE, optional
        DESCRIPTION. The default is "feature_name_value".

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
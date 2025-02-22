# -*- coding: utf-8 -*-
"""
Recommendation system

Written by Pawel Krolikowski
"""

import numpy as np
import pandas as pd
import os

import unittest

from module.functions import predict_ratings_for_movie
from module.functions import create_pivot_numpy
from module.functions import normalize_matrix
from module.functions import similarity_matrix
from module.functions import masking_nan

from module.functions import square
from module.functions import same_size

"""

Testing:
    same_size from module.functions 

"""

class TesSameSize(unittest.TestCase):
    def test_case1(self):
        """
        Test rows case
        """
        a = np.array([[1, 26, 3],[1, 42, 53]])
        b = np.array([[1, 23, 3],[1, 32, 34]])
        result = same_size(a,b,axis=1)
        self.assertEqual(result, True)
        
    def test_case2(self):  
        a = np.array([[1, 2, 53],[1, 2, 53]])
        b = np.array([[13, 24, 3],[13, 42, 33]])
        result = same_size(a,b,axis=0)
        self.assertEqual(result, True)
        
    def test_case4(self):  
        a = np.array([[1, 2, 53],[1, 2, 53],[1, 2, 53]])
        b = np.array([[13, 24, 3],[13, 42, 33]])
        result = same_size(a,b,axis=0)
        self.assertEqual(result, False)
        
    def test_case5(self):  
        a = np.array([[1, 2],[1, 2]])
        b = np.array([[13, 24, 3],[13, 42, 33]])
        result = same_size(a,b,axis=1)
        self.assertEqual(result, False)


"""

Testing:
    square from module.functions 
    
"""

        
class TestSquare(unittest.TestCase):
    def test_case1(self):

        a = np.array([[1, 26, 3],[1, 42, 53]])
        result = square(a)
        self.assertEqual(result, False)
        
    def test_case12(self):  
        a = np.array([[1, 2, 53],[1, 2, 53],[1, 2, 53]])
        result = square(a)
        self.assertEqual(result, True)



"""

Testing:
    masking_nan from module.functions 

"""

class TestMaskingNan(unittest.TestCase):
    def test_case1(self):

        matrix_with_nan = np.array([[np.nan, np.nan, np.nan],[np.nan, 42, 543],[16, 42, 53],[45, np.nan, 53]])
        matrix_with_prediction = np.array([[99,999,9999],[777777, 4452, 667],[674, 3442, 3253],[454, 8885555, 553]])
        
        # calc the test case
        result = masking_nan(matrix_with_nan, matrix_with_prediction)
        # convert to TRUE if equall
        result_bool = np.alltrue(result == np.array([[99,999,9999],[777777, 42, 543],[16, 42, 53],[45, 8885555, 53]]))
        
        self.assertEqual(result_bool, True)
   

"""

Testing:
    similarity_matrix from module.functions 

"""

class TestsSimilarity(unittest.TestCase):
    def test_case1(self):

        uset_item_matrix = np.array([[1,2],[3,4]])
        # calc the test case
        result = similarity_matrix(uset_item_matrix, method = "cosine")
        # convert to TRUE if equall
        result_bool = np.alltrue(np.round(result,4) == np.round(np.array([[1,0.98386991],[0.98386991,1]]),4))
        
        self.assertEqual(result_bool, True)
    def test_case2(self):

        user_item_matrix = np.array([[1,1,1],[1,1,1],[1,1,1]])
        # calc the test case
        result = similarity_matrix(user_item_matrix, method = "cosine")
        # convert to TRUE if equall
        result_bool = np.alltrue(np.round(result,4) == np.round(np.array(np.array([[1,1,1],[1,1,1],[1,1,1]])),4))
        
        self.assertEqual(result_bool, True)
   

"""

Testing:
    normalize_matrix from module.functions 

"""

class TestsNormalizaMatrix(unittest.TestCase):
    def test_case1(self):
        # row-wise normalization
        user_item_matrix = np.array([[1,np.nan,3],[3,4,6]])
        # calc the test case
        result, average_per_row = normalize_matrix(user_item_matrix, axis = 1,fill_na = 999)
        # convert to TRUE if equall
        result_bool = np.alltrue(np.round(result,4) == np.round(np.array([[-1,999,1],[-1.33333333,-0.33333333,1.66666667]]),4))
        
        self.assertEqual(result_bool, True)
        
    def test_case2(self):

        # column-wise normalization
        user_item_matrix = np.array([[1,np.nan],[3,4],[5,2]])
        # calc the test case
        result, average_per_row = normalize_matrix(user_item_matrix, axis = 0,fill_na = 0)
        # convert to TRUE if equall
        result_bool = np.alltrue(np.round(result,4) == np.round(np.array([[-2,  0],[ 0,  1],[ 2, -1]]),4))
        
        self.assertEqual(result_bool, True)
   
   
"""

Testing:
    create_pivot_numpy from module.functions 

"""
   
class TestsCreatePivot(unittest.TestCase):
    
    def test_case1(self):
        # unique values
        raw_dataframe = pd.DataFrame({"feature_name_row_fake" : [3,2,4,2,2,3],
                                      "feature_name_column_fake" : [7,5,7,6,7,5],
                                      "feature_name_value_fake" : [745,99,73,0.48,0.34,0.3],
                                      })
        
        # create pivot using pandas
        pandas_pivot_table = pd.pivot_table(raw_dataframe, values='feature_name_value_fake', 
                                                            index='feature_name_row_fake', 
                                                            columns='feature_name_column_fake')
        

        # calc the test case
        pivot_table_raw, rows_unique, cols_unique = create_pivot_numpy(raw_dataframe, "feature_name_row_fake",
                                                                                       "feature_name_column_fake",
                                                                                       "feature_name_value_fake")

        # compare unique values for rows, column and "body" of the pivot so values
        values_bool = np.allclose(pandas_pivot_table.values, pivot_table_raw, equal_nan=True)
        rows_bool = np.allclose(pandas_pivot_table.index.values, rows_unique, equal_nan=True)
        columns_bool = np.allclose(pandas_pivot_table.columns.values, cols_unique, equal_nan=True)
         
        self.assertEqual(values_bool and rows_bool and columns_bool, True)
        
    def test_case2(self):
        # unique values
        raw_dataframe = pd.DataFrame({"feature_name_row_fake" : [3,3,4,4],
                                      "feature_name_column_fake" : [1,1,7,7],
                                      "feature_name_value_fake" : [1,2,3,4],
                                      })
        
        # create pivot usinf pandas
        pandas_pivot_table = pd.pivot_table(raw_dataframe, values='feature_name_value_fake', 
                                                            index='feature_name_row_fake', 
                                                            columns='feature_name_column_fake',
                                                            aggfunc = max)
          

        # calc the test case
        pivot_table_raw, rows_unique, cols_unique = create_pivot_numpy(raw_dataframe,
                                                                       "feature_name_row_fake", 
                                                                       "feature_name_column_fake",
                                                                       "feature_name_value_fake")

        # compare unique values for rows, column and "body" of the pivot so values
        values_bool = np.allclose(pandas_pivot_table.values, pivot_table_raw, equal_nan=True)
        rows_bool = np.allclose(pandas_pivot_table.index.values, rows_unique, equal_nan=True)
        columns_bool = np.allclose(pandas_pivot_table.columns.values, cols_unique, equal_nan=True)
         
        self.assertEqual(values_bool and rows_bool and columns_bool, True)
    
"""

Testing:
    predict_ratings_for_movie from module.functions 
def predict_ratings_for_movie(ith_movie, movies_ids, users_ids, user_user_cosine_sim, all_user_avg_ratings):
"""  

class TestsPredictRatings(unittest.TestCase):
    
    def test_case1(self):

        ith_movie = np.array([np.nan,2,4])
        user_user_cosine_sim = np.array([[1,0.1,-0.1],[0.2,1,0.1],[0.2,0.5,1]])
        all_user_avg_ratings = np.array([3,2.5,4])
        
        results = predict_ratings_for_movie(ith_movie,user_user_cosine_sim, all_user_avg_ratings)
        results_bool = np.allclose(results, [2,2.181818,3.333333], equal_nan=True)

        self.assertEqual(results_bool, True)
        

if __name__ == '__main__':
    unittest.main()




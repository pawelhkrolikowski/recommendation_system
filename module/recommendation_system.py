# -*- coding: utf-8 -*-
"""
Recommendation system

Written by Pawel Krolikowski
"""

import numpy as np
import pandas as pd
import os

from module.functions import predict_ratings_for_movie
from module.functions import create_pivot_numpy
from module.functions import normalize_matrix
from module.functions import similarity_matrix
from module.functions import masking_nan


class RecommendationSystem():
    """
    Inputs:
        file_name.csv - CVS file with [user_id, movie_id, rating, timestamp] data, in \\input\\ folder
        
    calulations:
        user-item matrix pivot array 
        user x user similarity matrix
        prediction of all ratings if NAN
        
    Outputs:
        recommendation(x_user, k_films) method for returning k-recommendation for x-user, based on similarities to other users
        
    source: https://www.youtube.com/watch?v=h9gpufJFF-0&t=732s
        
    """
    
    def __init__(self):
        self.raw_data = pd.DataFrame()
        self.user_item_raw = np.empty((1,1)) 
        self.movies_ids = np.empty((1,1)) 
        self.users_ids =  np.empty((1,1)) 
        self.user_item_normalized = np.empty((1,1)) 
        self.user_item_predicted = np.empty((1,1)) 
        self.average_user_rating = np.empty((1,1)) 
        self.predicted_user_rating = None
    
    def load_data(self, file_name = "data"):
        print("Loading input file .... ",file_name )
        self.raw_data = pd.read_csv(os.getcwd() + '\\input\\' + file_name + ".csv", index_col=0)
        
        
    def user_item(self):

        """
        Create User-Item matrix
        
        """
    
        self.user_item_raw, self.movies_ids, self.users_ids  = create_pivot_numpy(self.raw_data,'user_id','movie_id','rating')
        
    def norm_user_item(self):
        """
        Fill Nans
        - normalize each User by subtracting mean (only based on rated movies)
        - fill nans with 0, as a average 0 for each user
        """
        
        self.user_item_normalized, self.average_user_rating = normalize_matrix(self.user_item_raw, axis = 1, fill_na = 0)
        
    def user_user_sim(self):
        """
        Create User-User similarities (cosine) matrix
        """
        self.user_user_similarity = similarity_matrix(self.user_item_normalized, method = "cosine")
        
    def predict_missing_ratings(self):
        
        # apply column-wise the prediction of ith movie rate for all users
        args = {"movies_ids" : self.movies_ids,
                "users_ids" :self.users_ids,
                "similarity" : self.user_user_similarity,
                "avg_user_rating" : self.average_user_rating,
                }
        
        # apply column-wise prediction of rating for each movie, it overrides existing ratings
        self.user_item_predicted  = np.apply_along_axis(predict_ratings_for_movie, 0, self.user_item_raw,args['similarity'],args['avg_user_rating'])
        
        # replace predictions by true ratings
        self.user_item_predicted = masking_nan(self.user_item_raw, self.user_item_predicted)

        
    def recommendation(self, x_user_id, k_films):
        
        # check if provided user id is within our DataBase
        if x_user_id in self.users_ids:
            
            # get index of the user
            x_user_index = np.argwhere(self.users_ids == x_user_id)[0]

            # select all movies 
            rated_movies = self.user_item_raw[x_user_index,:]
            
            # get all indexes of movies NOT rated by user X
            index_not_rated_movies = np.argwhere(np.isnan(rated_movies)).ravel()
    
            # get all predictions
            predicted_ratings = self.user_item_predicted[x_user_index,index_not_rated_movies]   
            
            # concatenate index & rating
            index_rating_predicted = np.vstack((index_not_rated_movies,predicted_ratings)).T
            
            # sort by ratings
            index_rating_predicted = index_rating_predicted[index_rating_predicted[:, 1].argsort()]
            
           
            # select 5 best rated films
            if index_rating_predicted.shape[0] < k_films:
                # take as it is - case when one user spend his whole life to watch & rate almost all movies xD
                Index_rating_predicted = index_rating_predicted
            else:
                index_rating_predicted = index_rating_predicted[-k_films:,:]
            

            recommended_movie = index_rating_predicted[:,0]
            recommended_movie_rating_prediction = index_rating_predicted[:,1]
            
            print("Recommended movies: " , recommended_movie)
            print("Corresponding rating: ", recommended_movie_rating_prediction)
            
            return {"recommended movies" : recommended_movie, "predicted ratings" : recommended_movie_rating_prediction }
        
        else:
            print("there is no user with id = ", x_user_id)
            return {"recommended movies" : np.nan, "predicted ratings" : np.nan }
            


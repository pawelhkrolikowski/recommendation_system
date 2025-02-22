# -*- coding: utf-8 -*-
"""
Recommendation system

Written by Pawel Krolikowski

"""
import numpy as np
import pandas as pd
import os
from module.recommendation_system import RecommendationSystem

"""

Script serves for running the recommendation model, and providing examples of recommendation.

input:
    file_name.csv - CVS file with [user_id, movie_id, rating, timestamp] data, in \\input\\ folder
    
calculations:
    details in class RecommendationSystem() description
    
output
    examples of recommendations of 5 movies, for users with ID 1,30,45,300,770
    
"""

if __name__ == '__main__':

    # Load Input data
    file_name = 'Data'
    
    # run the model
    model = RecommendationSystem()
    model.load_data(file_name)
    model.user_item()
    model.norm_user_item()
    model.user_user_sim()
    model.predict_missing_ratings()
    
    # examples of recommendations for users id. k_films is a number of films for recommendation.
    model.recommendation(1,k_films = 5)
    model.recommendation(30,k_films = 5)
    model.recommendation(45,k_films = 5)
    model.recommendation(300,k_films = 5)
    model.recommendation(770,k_films = 5)


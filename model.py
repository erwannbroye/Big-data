#!/bin/python3
import os
import argparse

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt

data_path = "./ml-latest-small"
movies_filename = 'movies.csv'
ratings_filename = 'ratings.csv'
tags_filename = 'tags.csv'

class KnnRecommender:
    def __init__(self, movie_rating_thres, user_rating_thres):
        self.path_movies =  os.path.join(data_path, movies_filename)
        self.path_ratings =  os.path.join(data_path, ratings_filename)
        self.path_tags = os.path.join(data_path, tags_filename)
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres
        self.model = NearestNeighbors()

    def SetModelParams(self, n_neighbors, algorithm, metric, n_jobs=None):
        self.model.set_params(**{'n_neighbors': n_neighbors, 'algorithm': algorithm, 'metric': metric})

    def GetData(self, movie_tag):
        movies = pd.read_csv(os.path.join(self.path_movies))
        ratings = pd.read_csv(os.path.join(self.path_ratings))
        tags = pd.read_csv(os.path.join(self.path_movies))

        movies_count = pd.DataFrame(
            ratings.groupby('movieId').size(),
            columns=['count'])
        popular_movies = list(set(movies_count.query('count >= @self.movie_rating_thres').index))
        movies_filter = ratings.movieId.isin(popular_movies).values

        tags["genres"]=tags["genres"].str.split("|")
        tags = tags.explode("genres").reset_index(drop=True)
        tag_sorted = tags.loc[tags['genres'] == movie_tag]['movieId'].tolist()
        tag_filter = ratings.movieId.isin(tag_sorted).values

        users_count = pd.DataFrame(
            ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(users_count.query('count >= @self.user_rating_thres').index))
        users_filter = ratings.userId.isin(active_users).values
        print(len(users_filter))
        movie_user_mat = []

        if movie_tag == '':
            movie_user_mat = ratings[movies_filter & users_filter].pivot(index='movieId', columns='userId', values='rating').fillna(0)
        else:
            movie_user_mat = ratings[movies_filter & users_filter & tag_filter].pivot(index='movieId', columns='userId', values='rating').fillna(0)


        mList = list(movies.set_index('movieId').loc[movie_user_mat.index].title)
        hashmap = {
            movie: i for i, movie in enumerate(mList)
        }
        return csr_matrix(movie_user_mat.values), hashmap

    def FuzzyMatching(self, hashmap, fav_movie):
        match_tuple = []
        match_tuple_60 = []

        for title, idx in hashmap.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 80:
                match_tuple.append((title, idx, ratio))
            if ratio >= 60:
                match_tuple_60.append((title, idx, ratio))
        if not match_tuple:
            match_tuple = match_tuple_60

        match_tuple.sort(key=lambda ratio: ratio[2], reverse=True)
        if not match_tuple:
            print('No match is found for : ' + fav_movie)
            exit()
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def Inference(self, model, data, hashmap, fav_movie, n_recommendations):
        model.fit(data)

        idx = self.FuzzyMatching(hashmap, fav_movie)

        print('Making distance ......\n')
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)
        return sorted(list(zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )), key=lambda x: x[1])[:0:-1]

    def MakeRecommendations(self, fav_movie, n_recommendations, movie_tag):
        movie_user, hashmap = self.GetData(movie_tag)
        raw_recommends = self.Inference(self.model, movie_user, hashmap, fav_movie, n_recommendations)
        reverse_hashmap = {v: k for k, v in hashmap.items()}

        print('Recommendations for ' + fav_movie + ':')

        bars = []
        height = []
        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i+1, reverse_hashmap[idx], dist))
            bars.append(reverse_hashmap[idx])
            height.append(dist)
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars, color='black', rotation=80 , fontsize='9', horizontalalignment='right')
        plt.subplots_adjust(bottom=0.4)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--movie_name', nargs='?', default='',
                        help='provide your favoriate movie name')
    parser.add_argument('--movie_tag', nargs='?', default='',
                        help='provide your tag')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()


args = parse_args()
movie_name = args.movie_name
top_n = args.top_n
movie_tag = args.movie_tag

recommender = KnnRecommender(50, 50)

recommender.SetModelParams(20 , 'auto', 'cosine')

recommender.MakeRecommendations(movie_name, top_n, movie_tag)


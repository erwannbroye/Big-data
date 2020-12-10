import os
import time
import argparse

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

class KnnRecommender:
    def __init__(self, path_movies, path_ratings):
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.movie_rating_thres = 0
        self.user_rating_thres = 0
        self.model = NearestNeighbors()

    def set_filter_params(self, movie_rating_thres, user_rating_thres):
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric})

    def _prep_data(self):
        df_movies = pd.read_csv(
            os.path.join(self.path_movies))
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings))

        df_movies_cnt = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @self.movie_rating_thres').index))
        movies_filter = df_ratings.movieId.isin(popular_movies).values

        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))
        users_filter = df_ratings.userId.isin(active_users).values

        df_ratings_filtered = df_ratings[movies_filter & users_filter]

        movie_user_mat = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)

        hashmap = {
            movie: i for i, movie in
            enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
        }
        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
        return movie_user_mat_sparse, hashmap

    def _fuzzy_matching(self, hashmap, fav_movie):
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

        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
            exit()
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def _inference(self, model, data, hashmap, fav_movie, n_recommendations):
        model.fit(data)

        print('You have input movie:', fav_movie)
        idx = self._fuzzy_matching(hashmap, fav_movie)

        print('......\n')
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]
        return raw_recommends

    def make_recommendations(self, fav_movie, n_recommendations):
        movie_user_mat_sparse, hashmap = self._prep_data()
        raw_recommends = self._inference(
            self.model, movie_user_mat_sparse, hashmap,
            fav_movie, n_recommendations)
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        print('Recommendations for ' + fav_movie + ':')

        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i+1, reverse_hashmap[idx], dist))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--movie_name', nargs='?', default='',
                        help='provide your favoriate movie name')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()


args = parse_args()
data_path = "./ml-latest-small"
movies_filename = 'movies.csv'
ratings_filename = 'ratings.csv'
movie_name = args.movie_name
top_n = args.top_n

recommender = KnnRecommender(
    os.path.join(data_path, movies_filename),
    os.path.join(data_path, ratings_filename))

recommender.set_filter_params(50, 50)
recommender.set_model_params(20 , 'auto', 'cosine')

recommender.make_recommendations(movie_name, top_n)


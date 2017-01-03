# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    movies['genres'] = movies['genres'].astype(str)
    movies['tokens'] = list(map(lambda x: tokenize_string(x), movies['genres']))
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """

    def check(token_list, vocab_list):
        check_list = []
        frequency = dict(Counter(token_list))
        a = 0;
        for i in vocab_list:
            if (i in token_list):
                check_list.append(frequency[i])
            else:
                check_list.append(0)
        return check_list

    def tfidf(movies_list, vocab_list, document_frequency):
        data = []
        column = []
        for i in range(0, len(vocab_list)):
            data.append(movies_list[i] / max(movies_list) * math.log(len(movies.index) / document_frequency[i]))
            column.append(i)
            # tfidf[0,i] = movies_list[i] / max(movies_list) * math.log(len(movies.index)/document_frequency[i])
        row = [0] * len(vocab_list)
        tfidf = csr_matrix((data, (row, column)), shape=(1, len(vocab_list)))
        return (tfidf)

    results = set()
    movies['vocab'] = movies['tokens'].apply(results.update)
    vocab = {}
    r_index = 0;
    for i in sorted(results):
        vocab[i] = r_index
        r_index = r_index + 1
    # vocab = dict(Counter(results))
    # vocab = dict(sorted(vocab.items(), key=lambda x: x[0]))
    # print(vocab)
    vocab_list = list(vocab.keys())
    movies['list'] = list(map(lambda x: check(x, vocab_list), movies['tokens']))
    # print(movies)
    df_result = []
    movies['vocab'] = movies['list'].apply(df_result.append)
    # print(df_result)
    document_frequency = np.array([sum(i) for i in zip(*df_result)])
    # print(document_frequency)
    movies['features'] = list(map(lambda x: tfidf(x, vocab_list, document_frequency), movies['list']))
    movies = movies.drop(['vocab', 'list'], axis=1)
    return (movies, vocab)
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    numerator = np.dot(a.toarray(),b.toarray().T)
    denominator1 = np.linalg.norm(a.toarray())
    denominator2 = np.linalg.norm(b.toarray())
    denominator = denominator1 * denominator2
    return numerator[0][0]/denominator
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ratings_train.is_copy = False
    ratings_train['merge'] = ratings_train[['movieId', 'rating']].apply(tuple, axis=1)
    user = {k: list(v) for k, v in ratings_train.groupby("userId")["merge"]}
    user[0] = []
    rating = {k: list(v) for k, v in ratings_train.groupby("userId")["rating"]}
    mean_rating = {}
    for k, v in rating.items():
        mean_rating[k] = sum(v) / float(len(v))

    df_result = ratings_test['userId'].tolist()
    df_movie = ratings_test['movieId'].tolist()
    r_index = 0
    predicted_rating = []  # np.array((len(df_result),1))

    for i in df_result:
        user_rated = user[i]
        movie_to_rate = movies[movies["movieId"] == df_movie[r_index]]["features"]
        r_index += 1
        movie_to_rate = movie_to_rate.tolist()[0]
        numerator = denominator = avg = 0.0
        for u in user_rated:
            user_movie = movies[movies["movieId"] == u[0]]["features"]
            user_movie = user_movie.tolist()[0]
            movie_sim = cosine_sim(movie_to_rate, user_movie)
            if (movie_sim > 0):
                numerator = numerator + movie_sim * u[1]
                denominator = denominator + movie_sim
                avg = numerator / denominator
        if (avg == 0):
            avg = mean_rating[i]
        predicted_rating.append(avg)

    return np.array(predicted_rating)
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()

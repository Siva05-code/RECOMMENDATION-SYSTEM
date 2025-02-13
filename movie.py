#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# In[7]:


#Loading
movies = pd.read_csv("tmdb_5000_movies.csv")

#Preprocessing
movies = movies[['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count']]
movies.dropna(inplace=True)

#Convert genres from JSON format
movies['genres'] = movies['genres'].apply(lambda x: ' '.join([d['name'] for d in ast.literal_eval(x)]))

#Filtering Using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[13]:


movies


# In[8]:


def recommend_movies_content(movie_title, top_n=10):
    idx = movies[movies['title'].str.lower() == movie_title.lower()].index
    if len(idx) == 0:
        return "Movie not found. Please check the title."
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'vote_average']]

# Collaborative Filtering Using SVD
ratings_data = {'userId': np.random.randint(1, 1000, size=len(movies)),
                'movieId': movies['id'],
                'rating': movies['vote_average']}
ratings = pd.DataFrame(ratings_data)
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)


# In[ ]:


def recommend_movies_collaborative(user_id, top_n=10):
    movie_ids = movies['id'].values
    predictions = [model.predict(user_id, movie_id).est for movie_id in movie_ids]
    recommended_indices = np.argsort(predictions)[::-1][:top_n]
    return movies.iloc[recommended_indices][['title', 'vote_average']]

#Example Usage
y=input("Enter the movie Name: ")
print("Content-Based Recommendations for", y, " :")
print(recommend_movies_content(y))
print('\n')
print('\n')
print("Collaborative Filtering Recommendations for User 1:")
x=float(input("Enter the movie rating: "))
print(recommend_movies_collaborative(x))


from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import pickle
import requests
import pandas as pd
import random
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class MovieRecommendFunction:
    def __init__(self): 
        #Contentbase Filtering
        self.movies = pickle.load(open("D:\Code\Python\my_django\MRS\MovieRS\data\movies.pkl", 'rb'))
        self.similarity = pickle.load(open("D:\Code\Python\my_django\MRS\MovieRS\data\similarity.pkl", 'rb'))
        self.movies_list=self.movies['title'].values
        #Collaborative Filtering
        self.ratings = pickle.load(open("D:\Code\Python\my_django\MRS\MovieRS\data\movies_ratings.pkl", 'rb'))
        self.userid = self.ratings["userId"].unique().astype(int)
        self.getRatingsMatrix()
        self.getSimilarityUUCB()
        
    def getRatingsMatrix(self):
        n_users = int(np.max(self.ratings["userId"]))
        n_items = int(np.max(self.ratings["tmdbId"]))
        
        users = self.ratings["userId"]
        ratings_copy = self.ratings.copy()
        self.mu = np.zeros((n_users,))
        for n in range(n_users):
            ids = np.where(users == n+1)[0]
            ratings_for_mean = self.ratings["rating"].iloc[ids]
            m = np.mean(ratings_for_mean)
            if np.isnan(m):
                m = 0  # để tránh mảng trống và NaN value
            self.mu[n] = m
            # chuẩn hóa
            ratings_copy.loc[ids,"rating"] = ratings_for_mean - self.mu[n]
        self.ratings_matrix = sparse.coo_matrix((ratings_copy["rating"],
                                        (ratings_copy["tmdbId"]-1, ratings_copy["userId"]-1)), shape=(n_items, n_users))
        self.ratings_matrix = self.ratings_matrix.tocsr()

    def getSimilarityUUCB(self):
        self.ratings_similar_uuCB = cosine_similarity(self.ratings_matrix.T, self.ratings_matrix.T)

    def __pred(self, u, i, normalized=1):
        ids = np.where(self.ratings["tmdbId"] == i)[0]
        users_rated_i = self.ratings['userId'].iloc[ids].astype(int)
        sim = self.ratings_similar_uuCB[u-1, users_rated_i-1] # Lấy giá trị similarity lần lượt của u so với từng u trong user_rated_i
        a = np.argsort(sim)[-2:] # Lấy ra vị trí của 2 u có similarity cao nhất
        nearest_s = sim[a] # Lấy ra giá trị similarity của 2 u trên đối với item này
        r = self.ratings_matrix[i-1, users_rated_i.iloc[a]-1] # Lấy rating của 2 u trên đối với item này
        # if normalized:
        #     return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) 
        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def collaborative_recommend(self, u):
        u = int(u)
        items = self.ratings["tmdbId"].unique()
        itemsName = self.ratings["title"].unique()
        ids = np.where(self.ratings["userId"] == u)[0]
        items_rated_by_u = self.ratings["tmdbId"].iloc[ids]
        item = {'id': None, 'name': None, 'poster': None, 'similar': None}
        list_items = []

        def take_similar(elem):
            return elem['similar']
        count = 0
        for i, n in zip(items, itemsName):
            if count == 24:
                break
            if i not in items_rated_by_u.values:
                rating = self.__pred(u, i)
                item['name'] = n
                item['id'] = self.movies[self.movies["tmdbId"]==i].index[0]
                item['poster'] = self.fetch_poster(i)
                item['similar'] = rating
                count+=1
                list_items.append(item.copy())
        sorted_items = sorted(list_items, key=take_similar, reverse=True)
        return sorted_items

    def fetch_poster(self, movie_id):
        url = "https://api.themoviedb.org/3/movie/{}?api_key=19b98f78a51a3924b33b555437599b0c&language=en-US".format(movie_id)
        data=requests.get(url)
        data=data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
        return full_path

    def contentbase_recommend(self, movie):
        index=self.movies[self.movies['title']==movie].index[0]
        distance = sorted(list(enumerate(self.similarity[index])), reverse=True, key=lambda vector:vector[1])
        recommend_movie=[]
        recommend_genre=[]
        recommend_overview=[]
        recommend_poster=[]
        recommend_vote=[]
        recommend_idx=[]
        for i in distance[1:6]:
            movies_id=self.movies.iloc[i[0]].tmdbId
            recommend_movie.append(self.movies.iloc[i[0]].title)
            recommend_poster.append(self.fetch_poster(movies_id))
            recommend_overview.append(self.movies.iloc[i[0]].overview)
            recommend_genre.append(self.movies.iloc[i[0]].genre)
            recommend_vote.append(self.movies.iloc[i[0]].vote_average)
            recommend_idx.append(i[0])
        return recommend_movie, recommend_poster, recommend_overview, recommend_genre, recommend_vote, recommend_idx

    def getMoviesRatedByUser(self, u):
        ids = np.where(self.ratings["userId"] == u)[0]
        itemsID_rated = self.ratings["tmdbId"].iloc[ids]
        itemsRate_rated = self.ratings["rating"].iloc[ids]
        itemsName_rated = self.ratings["title"].iloc[ids]
        items_poster = []
        items_name = []
        items_rate = []
        for i,r,n in zip(itemsID_rated, itemsRate_rated, itemsName_rated):
            items_poster.append(self.fetch_poster(i))
            items_name.append(n)
            items_rate.append(r)
        return items_poster, items_name, items_rate
    
    def getIDnNameRatedMovies(self, u):
        ids = np.where(self.ratings["userId"] == u)[0]
        itemsID_rated = self.ratings["tmdbId"].iloc[ids]
        itemsName_rated = self.ratings["title"].iloc[ids]
        items_id = []
        items_name = []
        for i,n in zip(itemsID_rated, itemsName_rated):
            items_id.append(i)
            items_name.append(n)
        return items_id, items_name

mrs = MovieRecommendFunction()
checkLogin = False
uid = 0

def moviegridfw(request):
    if request.method == 'GET' and 'logout' in request.GET:
        global checkLogin
        checkLogin = False        
    moviesInfo = []
    moviesRating = []
    if checkLogin == True:
        rmdmovies = mrs.collaborative_recommend(uid)
        for x in range(24):
            moviesInfo.append((rmdmovies[x]["poster"],rmdmovies[x]["name"],rmdmovies[x]["id"],round(rmdmovies[x]["similar"],1)))
    else:
        for x in range(24): 
            randNum = random.randint(0,7972)
            moviesInfo.append((mrs.fetch_poster(mrs.movies['tmdbId'][randNum]),mrs.movies['title'][randNum],randNum, moviesRating))
    template = loader.get_template('moviegridfw.html')
    context = {
        'checkLogin': checkLogin,
        'moviesInfo': moviesInfo,
        'uid' : uid,
        'userid' : mrs.userid
    }
    return HttpResponse(template.render(context))

def userrate(request):
    global checkLogin
    checkLogin=True
    template = loader.get_template('userrate.html')
    global uid
    uid = int(request.GET["uid"])        
    moviesPoster, moviesName, moviesRate = mrs.getMoviesRatedByUser(uid)
    context = {
        'checkLogin': checkLogin,
        'userid' : mrs.userid,
        'uid' : uid,
        'moviesRated': zip(moviesPoster,moviesName,moviesRate)
    }
    return HttpResponse(template.render(context))    

def moviesingle(request, idx):
    template = loader.get_template('moviesingle.html')
    template = loader.get_template('moviesingle.html')
    movie_name, movie_poster, movie_overview, movie_genre, movie_vote, movie_idx = mrs.contentbase_recommend(mrs.movies['title'][idx])
    context = {
        'uid' : uid,
        'checkLogin': checkLogin,
        'moviesName': mrs.movies['title'][idx],
        'moviesOverview': mrs.movies['overview'][idx],
        'moviesGenre': mrs.movies['genre'][idx],
        'moviesVote': mrs.movies['vote_average'][idx],
        'moviesPoster': mrs.fetch_poster(mrs.movies['tmdbId'][idx]),
        'range': range(math.floor(float(mrs.movies['vote_average'][idx]))),
        'rangeUnCheck': range(10-math.floor(float(mrs.movies['vote_average'][idx]))),
        'moviesRecommend': zip(movie_idx, movie_name, movie_poster, movie_overview, movie_genre, movie_vote),
        'userid' : mrs.userid
    }
    return HttpResponse(template.render(context))

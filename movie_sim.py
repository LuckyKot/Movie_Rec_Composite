import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#a function that returns all the movies the user have seen
def retrieve_movies(data,User_ID):
    mov_list=[]
    #get all the rows of the user
    user_entries = data.loc[data['User_ID'] == User_ID]
    #get all the movie IDs
    mov_list=user_entries['Movie_ID'].tolist()
    return mov_list

#a function that calculates the cosine similarities for the list of movies
def sim_movies(data, movies, movie_IDs, User_ID):
    #initialize 2 arrays, 1 for similar movie IDs and 1 for their similarities
    sim=np.empty(0)
    sim_scores=np.empty(0)
    #cycle through movie IDs
    for mov in movie_IDs:
        #get the movie and its features
        movie=movies.loc[movies['Movie_ID'] == mov]
        #compute cosine similarity with all the movies
        sim_matrix = cosine_similarity(movies[['Year', 'isAdult', 'runtimeMinutes', 'movie', 'short',
       'tvEpisode', 'tvMiniSeries', 'tvMovie', 'tvSeries', 'tvShort',
       'tvSpecial', 'video', 'Action', 'Adult', 'Adventure', 'Animation',
       'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
       'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music',
       'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi',
       'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']],movie[['Year', 'isAdult', 'runtimeMinutes', 'movie', 'short',
       'tvEpisode', 'tvMiniSeries', 'tvMovie', 'tvSeries', 'tvShort',
       'tvSpecial', 'video', 'Action', 'Adult', 'Adventure', 'Animation',
       'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
       'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music',
       'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi',
       'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']])
        #gather top 50 similar movies
        sim=np.append(sim,sim_matrix.argsort(axis=0)[-50:-1].flatten())
        #gather top 50 scores for the respective movies
        sim_scores=np.append(sim_scores,np.sort(sim_matrix, axis=0)[-50:-1].flatten())
        #repeat, gather 50 similar movies for each movie a user has seen

    #the list gets large fast, so we drop the least similar ones to keep the number of movies computationally feasible
    #between 10 and 500
    if len(sim)>=500:
        rec_num=int(len(sim)/50)
    #sort the scores to keep only the best ones
    #argsort to keep only indexes of most similar movies as we don't really care about the score
    sim_scores =sim_scores.argsort()[-rec_num:]
    out = []
    #retrieve only movie IDs from indexes we determined to contain most similar movies
    for i in sim_scores:
            out.append(sim[i])
    return out

#a function that gathers a list of similar movie IDs and retrieves all the rows that contain them
#note that the movies a target user has seen were filtered out previously so we only gather data on unseen before movies
def retrieve_sim_users(data,sim_ids):
    #initialize the output dataframe
    new_data=pd.DataFrame()
    #cycle through similar movies list
    for i in range(len(sim_ids)):
        #locate a subset of rows for each movie ID
        a=data.loc[data['Movie_ID'] == sim_ids[i]]
        #extract only their user/item/rating triplet
        a=a[['User_ID','Movie_ID','Rating']]
        #add them to the output dataframe
        new_data = pd.concat([new_data,a])
        #in case we somehow got duplicates - drop them
        new_data.drop_duplicates(inplace=True)
        #a progress bar of sort, this method used to take a while until I limited the number of movies
        if i%100==0:
            print(i," movies out of",len(sim_ids),"done")
            print("current new data entries:",len(new_data))
    return new_data

#the final function that utilizes all the previous helper functions and outputs 2 dataframes
#a train set of all the reviews for all similar movies to the ones user watched
#a test set of movies similar to the ones user watched but has not seen before (to predict their scores and return the top n predictions)
def sample_new_data(data,movies,User_ID):
    #start calling helper functions, get the list of movies user has seen
    mov_list=retrieve_movies(data,User_ID)
    print(len(mov_list)," retrieved movies for the user")
    print("finding similar movies...")
    #gather similar movies, remove duplicates
    sim_mov_list=sim_movies(data,movies,mov_list,User_ID)
    sim_mov_list = np.asarray(sim_mov_list,dtype='int')
    sim_mov_list = np.unique(sim_mov_list)
    #also remove those the target user has seen before
    #start cycling through similar movies
    for i in range(len(sim_mov_list)):
        cur=sim_mov_list[i]
        #check if there is such a row that contains target user and the movie
        if not data[(data['User_ID']==User_ID)&(data['Movie_ID']==cur)].empty:
            #if there is - the user has seen the movie and we mark it for deletion
            sim_mov_list[i]=0
    #delete all the seen movies
    sim_mov_list=sim_mov_list[sim_mov_list!=0]
    
    print(len(sim_mov_list)," similar unwatched movies found")
    print("retrieving new data")
    #gather all the reviews to unseen movies
    new_train_data=retrieve_sim_users(data,sim_mov_list)
    #initialize test set that we will predict scores for (all unseen before movies that are similar to the ones watched)
    new_test_data=pd.DataFrame(columns=['User_ID','Movie_ID','Rating'])
    #assemble the test set: target user in every row, put each movie ID in movie column and NA in rating column
    for i in range(0,len(sim_mov_list)):
        new_test_data.loc[i,"User_ID"]=User_ID
        new_test_data.loc[i,"Movie_ID"]=sim_mov_list[i]
        new_test_data.loc[i,"Rating"]=np.nan
    
    return new_train_data, new_test_data
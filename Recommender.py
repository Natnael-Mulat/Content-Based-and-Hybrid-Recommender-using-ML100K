'''
Create content-based recommenders: Feature Encoding, TF-IDF/CosineSim
       using item/genre feature data.
       

Programmer name: << Basel Elzatahry, Rida Shahid, Paul Choi, Natnael Mulat>>

Collaborator/Author: Carlos Seminario

sources: 
https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.XoT9p257k1L

reference:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html


'''

import numpy as np
import pandas as pd
from math import sqrt 
import math
import os
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import pickle
import copy

#SIG_THRESHOLD = 0 # accept all positive similarities > 0 for TF-IDF/ConsineSim Recommender
                  # others: TBD ...
    
def from_file_to_2D(path, genrefile, itemfile):
    ''' Load feature matrix from specified file 
        Parameters:
        -- path: directory path to datafile and itemfile
        -- genrefile: delimited file that maps genre to genre index
        -- itemfile: delimited file that maps itemid to item name and genre
        
        Returns:
        -- movies: a dictionary containing movie titles (value) for a given movieID (key)
        -- genres: dictionary, key is genre, value is index into row of features array
        -- features: a 2D list of features by item, values are 1 and 0;
                     rows map to items and columns map to genre
                     returns as np.array()
    
    '''
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(movies), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    

    # Get movie genre from the genre file, place into genre dictionary indexed by genre index
    genres={} # key is genre index, value is the genre string
   
    try: 
        for line in open(path+'/'+ genrefile):
            #print(line, line.split('|')) #debug
     
            (string,index)=line.split('|')[0:2]
            
            genres[int(index[0:])] = string.strip()
            
    except Exception as ex:
        print (ex)
        print ('Proceeding with len(features)', len(genres))
    ##
    
    # Load data into a nested 2D list
    features = []
    start_feature_index = 5
    try: 
        for line in open(path+'/'+ itemfile, encoding='iso8859'):
            #print(line, line.split('|')) #debug
            fields = line.split('|')[start_feature_index:]
            row = []
            for feature in fields:
                row.append(int(feature))
            features.append(row)
        features = np.array(features)
    except Exception as ex:
        print (ex)
        print ('Proceeding with len(features)', len(features))
        #return {}
    
    #return features matrix
    return movies, genres, features 
   

def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings (value) for each user (key)
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(movies), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}

    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def sim_distance(prefs,person1,person2,n=50): #default weight 
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        -- n: significance weighting
        
        Returns:
        -- Euclidean distance similarity as a float
        
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    factor = 1
    if len(si)==0: 
        return 0
    if len(si) < n:
        factor = len(si)/n
    # Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) 
                        for item in prefs[person1] if item in prefs[person2]])


        
    return factor *(1/(1+sqrt(sum_of_squares)))

def sim_pearson(prefs,p1,p2,n=50):
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        -- n: significance weighting
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    '''
    
    # Get the list of shared_items
    si={p1:[], p2:[]}
    num = 0
    den1 = 0
    den2 = 0
    
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[p1].append(prefs[p1][item])
            si[p2].append(prefs[p2][item])
    factor =1
    
    # if they have no ratings in common, return 0
    if (si[p1])==[]: 
        return 0
    if len(si[p1]) < n:
        factor = len(si[p1])/n

    p1_avg = np.mean(si[p1])
    p2_avg = np.mean(si[p2])
    
    for item in prefs[p1]:
        if item in prefs[p2]:
            num += (prefs[p1][item] - p1_avg)*(prefs[p2][item] - p2_avg)
            den1 += (prefs[p1][item] - p1_avg)**2
            den2 += (prefs[p2][item]- p2_avg)**2
            
    if (sqrt(den1*den2)==0):
        return 0
    else:
        return factor * (num/(sqrt(den1*den2)))


def topMatches(prefs,person,similarity=sim_pearson, n=100):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        
        Returns:
        -- A list of similar matches with 0 or more tuples, 
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
        
    '''     
    scores=[(similarity(prefs,person,other),other) 
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary) 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix, 
           this function returns an I-U matrix
        
    '''     
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d" % (c,len(itemPrefs)))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n)
        result[item]=scores
    return result

def prefs_to_2D_list(prefs):
    '''
    Convert prefs dictionary into 2D list used as input for the MF class
    
    Parameters: 
        prefs: user-item matrix as a dicitonary (dictionary)
        
    Returns: 
        ui_matrix: (list) contains user-item matrix as a 2D list
        
    '''
    ui_matrix = []
    
    user_keys_list = list(prefs.keys())
    num_users = len(user_keys_list)
    #print (len(user_keys_list), user_keys_list[:10]) # debug
    
    itemPrefs = transformPrefs(prefs) # traspose the prefs u-i matrix
    item_keys_list = list(itemPrefs.keys())
    num_items = len(item_keys_list)
    #print (len(item_keys_list), item_keys_list[:10]) # debug
    
    sorted_list = False # <== set manually to test how this affects results
    
    if sorted_list == True:
        user_keys_list.sort()
        item_keys_list.sort()
        print ('\nsorted_list =', sorted_list)
        
    # initialize a 2D matrix as a list of zeroes with 
    #     num users (height) and num items (width)
    
    for i in range(num_users):
        row = []
        for j in range(num_items):
            row.append(0.0)
        ui_matrix.append(row)
          
    # populate 2D list from prefs
    # Load data into a nested list

    for user in prefs:
        for item in prefs[user]:
            user_idx = user_keys_list.index(user)
            movieid_idx = item_keys_list.index(item) 
            
            try: 
                # make it a nested list
                ui_matrix[user_idx][movieid_idx] = prefs [user][item] 
            except Exception as ex:
                print (ex)
                print (user_idx, movieid_idx)   
                
    # return 2D user-item matrix
    return ui_matrix

def to_array(prefs):
    ''' convert prefs dictionary into 2D list '''
    R = prefs_to_2D_list(prefs)
    R = np.array(R)
    print ('to_array -- height: %d, width: %d' % (len(R), len(R[0]) ) )
    return R

def to_string(features):
    ''' convert features np.array into list of feature strings '''
    
    feature_str = []
    for i in range(len(features)):
        row = ''
        for j in range(len (features[0])):
            row += (str(features[i][j]))
        feature_str.append(row)
    print ('to_string -- height: %d, width: %d' % (len(feature_str), len(feature_str[0]) ) )
    return feature_str

def to_docs(features_str, genres):
    ''' convert feature strings to a list of doc strings for TFIDF '''
    print(genres)
    feature_docs = []
    for doc_str in features_str:
        row = ''
        for i in range(len(doc_str)):
            
            if doc_str[i] == '1':
                #print(i)
                row += (genres[i] + ' ') # map the indices to the actual genre string
        feature_docs.append(row.strip()) # and remove that pesky space at the end
        
    print ('to_docs -- height: %d, width: varies' % (len(feature_docs) ) )
    return feature_docs

def cosine_sim(docs):
    ''' Performs cosine sim calcs on features list, aka docs in TF-IDF world.
    
        Parameters:
        -- docs: list of item features
     
        Returns:   
        -- 2-D list containing cosim_matrix: item_feature-item_feature cosine similarity matrix 
    
    
    '''
    
    print()
    print('## Cosine Similarity calc ##')
    print()
    print('Documents:', docs[:10])
    
    print()
    print ('## Count and Transform ##')
    print()
    
    # choose one of these invocations
    tfidf_vectorizer = TfidfVectorizer() # orig
  
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    #print (tfidf_matrix.shape, type(tfidf_matrix)) # debug

    
    print()
    print('Document similarity matrix:')
    cosim_matrix = cosine_similarity(tfidf_matrix[0:], tfidf_matrix)
    print (type(cosim_matrix), len(cosim_matrix))
    print()
    print(cosim_matrix[0:6])
    print()
    
    
    #Uncomment to print examples of similarity angles
    #print('Examples of similarity angles')
    #if tfidf_matrix.shape[0] > 2:
    #    for i in range(6):
    #        cos_sim = cosim_matrix[1][i] #(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))[0][i] 
    #        if cos_sim > 1: cos_sim = 1 # math precision creating problems!
    #        angle_in_radians = math.acos(cos_sim)
    #        print('Cosine sim: %.3f and angle between documents 2 and %d: ' 
    #              % (cos_sim, i+1), end=' ')
    #        print ('%.3f degrees, %.3f radians' 
    #               % (math.degrees(angle_in_radians), angle_in_radians))
    
    #Calculates the cosine similarities and creates matrix
    cosim_List =[]
    count_i = 0
    count_j = 0
    for i in cosim_matrix:
        count_i += 1
        count_j = 0
        for j in i:
            count_j += 1
            if (j>0 and j<=1 and count_i != count_j):
                cosim_List.append(j)
    
    cosim_List.sort()
    
    #Creates a histogram of cosine similarities
    hist,bins = np.histogram(cosim_List,bins =[0.0,0.2,0.4,0.6,0.8,1])
    axes = plt.gca()
    axes.grid(True)
    axes.set_xlim(0,1)
    axes.set_ylim(0,max(hist))
    plt.hist(cosim_List,bins=[0.0,0.2,0.4,0.6,0.8,1],facecolor='g')
    plt.title("Sim Histogram")
    plt.xlabel("Sims")
    plt.ylabel("Occurences")
    plt.show()
    
    #Prints stats of the similarities
    print("Mean: ")
    print(np.mean(cosim_List))
    print("Standard Deviation: ")
    print(np.std(cosim_List))
    
    return cosim_matrix

def movie_to_ID(movies):
    ''' converts movies mapping from "id to title" to "title to id" '''
    movies2 = {}
    for i in movies.keys():
        movies2[movies[i]]=i
    return movies2


def get_TFIDF_recommendations(prefs,cosim_matrix,user,movie_title_to_id,threshold):
    '''
    Calculates TFIDF recommendations for a given user.

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
    -- user: string containing name of user requesting recommendation
    -- movie_title_to_id: dictionary that maps movie title to movieid
    -- threshold: the similarity threshold that determines the neighborhood size of similarities
    Returns:
    -- rankings: A list of recommended items with 0 or more tuples,
        each tuple contains (predicted rating, item name).
        List is sorted, high to low, by predicted rating.
        An empty list is returned when no recommendations have been calc'd.

    '''
    toBeRated = []   # array of tuples of not rated movies, array of recommendations
    preds = []
    #making a list of the unrated movies
    for item, itemID in movie_title_to_id.items():
        if item not in prefs[user].keys():
            toBeRated.append((item, int(itemID)-1))

    for item,itemID in toBeRated:
        den = 0
        num = 0
        for ratedMovie,rating in prefs[user].items():
            cosine_value = cosim_matrix[int(movie_title_to_id[ratedMovie])-1][itemID]
            if cosine_value>float(threshold):
                num+= (cosine_value*rating)
                den+=cosine_value
        if den!=0:
            preds.append((num/den,item))
    preds.sort(reverse=True)
    return preds

def get_TFIDF_rec_single(prefs,cosim_matrix,user,movie_title_to_id,threshold,item):
    '''
    Calculates TFIDF recommendations for a given user and item.

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
    -- user: string containing name of user requesting recommendation
    -- movie_title_to_id: dictionary that maps movie title to movieid
    -- threshold: the similarity threshold that determines the neighborhood size of similarities
    -- item: string containing name of item requesting recommendation
    
    Returns:
    -- rankings: A list of recommended items with 0 or more tuples,
        each tuple contains (predicted rating, item name).
        List is sorted, high to low, by predicted rating.
        An empty list is returned when no recommendations have been calc'd.

    '''
    toBeRated = []   # array of tuples of not rated movies, array of recommendations
    preds = []
    #making a list of the unrated movies
    for i, itemID in movie_title_to_id.items():
        if item == i:
            toBeRated.append((item, int(itemID)-1))

    for item,itemID in toBeRated:
        den = 0
        num = 0
        for ratedMovie,rating in prefs[user].items():
            cosine_value = cosim_matrix[int(movie_title_to_id[ratedMovie])-1][itemID]
            if cosine_value>float(threshold):
                num+= (cosine_value*rating)
                den+=cosine_value
        if den!=0:
            preds.append((num/den,item))
    preds.sort(reverse=True)
    return preds

def get_FE_recommendations(prefs, features, movie_title_to_id, user):
    '''
        Calculates feature encoding recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- features: an np.array whose height is based on number of items
                     and width equals the number of unique features (e.g., genre)
        -- movie_title_to_id: dictionary that maps movie title to movieid
        -- user: string containing name of user requesting recommendation        
        
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    toBeRated = []
    predictions = []
    sumOfRatings = 0
    
    # making a list of the unrated movies
    for item, itemID in movie_title_to_id.items():
        if item not in prefs[user].keys():
            toBeRated.append((item, int(itemID)))
    # calculating the sum of the ratings
    for item, rating in prefs[user].items():
        itemID = int(movie_title_to_id[item])-1
        sumOfRatings += rating * np.count_nonzero(features[itemID] == 1)

    # prediction process
    for item in toBeRated:
        # build feature_preference
        movieFeatures = np.nonzero(features[item[1]-1])[0]
        feature_preference = [[] for i in range(len(movieFeatures))]
        for ratedMovie in prefs[user]:
            for i in range(len(movieFeatures)):
                movie_id = int(movie_title_to_id[ratedMovie])-1
                if features[movie_id][movieFeatures[i]] == 1:
                    feature_preference[i].append(prefs[user][ratedMovie])

        # if there were no genre-sharing rated movies, no prediction can be made
        if feature_preference != [[] for i in range(len(movieFeatures))]:
            # sum the feature_preference matrix columnvalues for all feature attributes
            # calc a normalized_vector of weights for all feature attributes associated 
            # with this user by dividing the column sums by the overall sumof values in the feature_preference matrix
            normalized_weights = [np.sum(featureList)/sumOfRatings for featureList in feature_preference]
            totalWeight = np.sum(normalized_weights)
            # calc normalized_weight by dividingeach of the non-zero feature attribute
            # weights by the above sum
            totalWeightContribs = [genreWeight/totalWeight for genreWeight in normalized_weights]

            # multiply the normalized_weight by the average rating for this user’s item ratings
            genrePred = [np.average(feature_preference[i]) * totalWeightContribs[i] 
                         for i in range(0, len(feature_preference)) if totalWeightContribs[i] != 0]
            prediction = np.sum(genrePred)
            predictions.append((prediction, item[0]))
    
    predictions.sort(reverse=True)
    return predictions  

def get_FE_rec_single(prefs, features, movie_title_to_id, user, item):
    '''
        Calculates feature encoding recommendations for a given user and item.

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- features: an np.array whose height is based on number of items
                     and width equals the number of unique features (e.g., genre)
        -- movie_title_to_id: dictionary that maps movie title to movieid
        -- user: string containing name of user requesting recommendation        
        -- item: string containing name of item requesting recommendation
        
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    toBeRated = []
    predictions = []
    sumOfRatings = 0
    
    # making a list of the unrated movies
    for i, itemID in movie_title_to_id.items():
        if i == item:
            toBeRated.append((item, int(itemID)))
    # calculating the sum of the ratings
    for item, rating in prefs[user].items():
        itemID = int(movie_title_to_id[item])-1
        sumOfRatings += rating * np.count_nonzero(features[itemID] == 1)

    # prediction process
    for item in toBeRated:
        # build feature_preference
        movieFeatures = np.nonzero(features[item[1]-1])[0]
        feature_preference = [[] for i in range(len(movieFeatures))]
        for ratedMovie in prefs[user]:
            for i in range(len(movieFeatures)):
                movie_id = int(movie_title_to_id[ratedMovie])-1
                if features[movie_id][movieFeatures[i]] == 1:
                    feature_preference[i].append(prefs[user][ratedMovie])

        # if there were no genre-sharing rated movies, no prediction can be made
        if feature_preference != [[] for i in range(len(movieFeatures))]:
            # sum the feature_preference matrix columnvalues for all feature attributes
            # calc a normalized_vector of weights for all feature attributes associated 
            # with this user by dividing the column sums by the overall sumof values in the feature_preference matrix
            normalized_weights = [np.sum(featureList)/sumOfRatings for featureList in feature_preference]
            totalWeight = np.sum(normalized_weights)
            # calc normalized_weight by dividingeach of the non-zero feature attribute
            # weights by the above sum
            totalWeightContribs = [genreWeight/totalWeight for genreWeight in normalized_weights]

            # multiply the normalized_weight by the average rating for this user’s item ratings
            genrePred = [np.average(feature_preference[i]) * totalWeightContribs[i] 
                         for i in range(0, len(feature_preference)) if totalWeightContribs[i] != 0]
            prediction = np.sum(genrePred)
            predictions.append((prediction, item[0]))
    
    predictions.sort(reverse=True)
    return predictions  

def get_Hybrid_Recommendations(prefs, cosim_matrix, matrix, user, movie, movie_title_to_id, hybrid_weight,threshold):
    '''
    Calculates hybrid recommendations for a given user.

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
    -- matrix: pre-computed item-item similarity dictionary with tuples
    -- user: string containing name of user requesting recommendation
    -- movie_title_to_id: dictionary that maps movie title to movieid
    -- hybrid_weight: weight to apply to the item-item similarity
    -- threshold: the similarity threshold that determines the neighborhood size of similarities
    
    Returns:
    -- rankings: A list of recommended items with 0 or more tuples,
        each tuple contains (predicted rating, item name).
        List is sorted, high to low, by predicted rating.
        An empty list is returned when no recommendations have been calc'd.

    '''
    toBeRated = []   # array of tuples of not rated movies, array of recommendations
    preds = []
    
    matrix2 = copy.copy(cosim_matrix)
    movie2 = movie_to_ID(movie)

    #converting similarity dictionary to matrix
    for j in matrix:
        key_location = int(movie_title_to_id[j]) -1
        for i in range(len(matrix[j])):
            location = int(movie2[matrix[j][i][1]]) -1
            matrix2[key_location][location]=matrix[j][i][0]
            if j==movie2[matrix[j][i][1]]:
                matrix2[key_location][location] = 1
        
    #making a list of the unrated movies
    for item, itemID in movie_title_to_id.items():
        if item not in prefs[user].keys():
            toBeRated.append((item, int(itemID)-1))

    #prediction process
    for item,itemID in toBeRated:
        den = 0
        num = 0
        for ratedMovie,rating in prefs[user].items():
            i = cosim_matrix[int(movie_title_to_id[ratedMovie])-1][itemID]
            j = matrix2[int(movie_title_to_id[ratedMovie])-1][itemID]
            j = j*float(hybrid_weight)
            #if cosim is 0, use item-item sim multiplied by hybrid weight
            if i == 0 and j>float(threshold):
                num+= (j*rating)
                den+=j
            elif i>float(threshold):
                num+= (i*rating)
                den+=i
        if den!=0:
            preds.append((num/den,item))
    preds.sort(reverse=True)
    return preds

def tuple_handle(matrix, movie, cosim_matrix, movie_title_to_id):
    '''
    Converts given tuple dictionary of similarities into a 2-D matrix.
    
    Parameters:
    -- matrix: pre-computed item-item similarity dictionary with tuples
    -- movie: list of movies
    -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
    -- movie_title_to_id: dictionary that maps movie title to movieid
    
    Returns:
    -- matrix2: matrix of the similarities
    '''
    #copies format of cosim matrix
    matrix2 = copy.copy(cosim_matrix)
    movie2 = movie_to_ID(movie)
    #places each similarity to its position matching cosim
    for j in matrix:
        key_location = int(movie_title_to_id[j]) -1
        for i in range(len(matrix[j])):
            location = int(movie2[matrix[j][i][1]]) -1
            matrix2[key_location][location]=matrix[j][i][0]
            if j==movie2[matrix[j][i][1]]:
                matrix2[key_location][location] = 1
    return matrix2

def get_Hybrid_Rec_single(prefs, cosim_matrix, matrix2, user, movie, movie_title_to_id, hybrid_weight,threshold,item):
    '''
    Calculates hybrid recommendations for a given user and item.

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- cosim_matrix: : pre-computed cosine similarity matrix from TF-IDF command
    -- matrix2: pre-computed item-item similarity matrix
    -- user: string containing name of user requesting recommendation
    -- movie: list of movies
    -- movie_title_to_id: dictionary that maps movie title to movieid
    -- hybrid_weight: weight to apply to the item-item similarity
    -- threshold: the similarity threshold that determines the neighborhood size of similarities
    -- item: string containing name of item requesting recommendation
    
    Returns:
    -- rankings: A list of recommended items with 0 or more tuples,
        each tuple contains (predicted rating, item name).
        List is sorted, high to low, by predicted rating.
        An empty list is returned when no recommendations have been calc'd.

    '''
    toBeRated = []   # array of tuples of not rated movies, array of recommendations
    preds = []
    toBeRated.append((item, (int(movie_title_to_id[item])-1)))

    #prediction process
    for item,itemID in toBeRated:
        den = 0
        num = 0
        for ratedMovie,rating in prefs[user].items():
            i = cosim_matrix[int(movie_title_to_id[ratedMovie])-1][itemID]
            j = matrix2[int(movie_title_to_id[ratedMovie])-1][itemID]
            j = j*float(hybrid_weight)
            #if cosim is 0, use item-item sim multiplied by hybrid weight
            if i == 0 and j>float(threshold):
                num+= (j*rating)
                den+=j
            elif i>float(threshold):
                num+= (i*rating)
                den+=i
        if den!=0:
            preds.append((num/den,item))
    preds.sort(reverse=True)
    return preds

def loo_cv_sim(prefs, algo, sim_matrix, cosim_matrix, movie_title_to_id,features, hybrid_weight, threshold,movies):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, ML-100K, etc.
     
     sim: distance, pearson, etc.
     algo: user-based recommender, item-based recommender, etc.
     sim_matrix: pre-computed item-item similarity matrix as dictionary of tuples
     cosim_matrix: pre-computed cosine similarity matrix
     movie_title_to_id: dictionary that maps movie title to movieid
     hybrid_weight: weight to apply to the item-item similarity
     threshold: the similarity threshold that determines the neighborhood size of similarities
     movies: list of movies
    
    Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    """
   
    error_list = []
    error_listabs = []
    temp_prefs = prefs.copy()
    
    #converts tuple dictionary to matrix for hybrid
    if algo == get_Hybrid_Recommendations:
        sim_matrix = tuple_handle(sim_matrix, movies, cosim_matrix,movie_title_to_id)
    
    #predicts each rating for each user and item
    for user in temp_prefs:
        #leaves one rating out and predicts its ratings
        for item in list(temp_prefs[user].keys()):
            item_del = prefs[user][item]
            del temp_prefs[user][item]
            #chooses algo
            if algo == get_TFIDF_recommendations:
                recc_list = get_TFIDF_rec_single(prefs,cosim_matrix,user,movie_title_to_id,threshold,item)
            if algo == get_FE_recommendations:
                recc_list = get_FE_rec_single(prefs, features, movie_title_to_id, user, item)
            if algo == get_Hybrid_Recommendations:
                
                recc_list = get_Hybrid_Rec_single(prefs, cosim_matrix, sim_matrix, user, movies, movie_title_to_id, hybrid_weight,threshold,item)
            
            #caclulates mse, mae and rmse for each prediction
            for recc in recc_list:
                if item in recc:
                    if item == recc[1]:
                        err = (recc[0]-item_del)**2
                        err_1 = abs(recc[0]-item_del)
                        print('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f'% (user, item, \
                                    recc[0], item_del, err))
                        error_listabs.append(err_1)
                        error_list.append(err)

            temp_prefs[user][item] = item_del
           
    #calculates total errrors 
    error_total = np.mean(error_list)
    error_totalabs = np.mean(error_listabs)
    MSE = error_total
    MSE_list = error_list
    MAE = error_totalabs
    MAE_list = error_listabs
    RMSE = sqrt(error_total)
    RMSE_list = error_list
    return MSE, MSE_list, MAE, MAE_list, RMSE, RMSE_list

'''
scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
 Calculate the T-test for the means of two independent samples of scores.
 This is a two-sided test for the null hypothesis that 2 independent samples 
 have identical average (expected) values. This test assumes that the 
 populations have identical variances by default.
'''

def print_loocv_results(sq_diffs_info):
    ''' Print LOOCV SIM results '''

    error_list = []
    print(sq_diffs_info)
    for user in sq_diffs_info:
        for item in sq_diffs_info[user]:
            for data_point in sq_diffs_info[user][item]:
                #print ('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f' %\
            #      (user, item, data_point[0], data_point[1], data_point[2]))                
                error_list.append(data_point[2]) # save for MSE calc
                
    #print()
    error = sum(error_list)/len(error_list)          
    #print ('MSE =', error)
    
    return(error, error_list)

def main():
    
    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    print()
    prefs = {}
    done = False
    
    while not done:
        print()
        file_io = input('R(ead) critics data from file?, \n'
                        'RML(ead) ml100K data from file?, \n'
                        'FE(ature Encoding) Setup?, \n'
                        'TFIDF(and cosine sim Setup)?, \n'
                        'CBR-FE(content-based recommendation Feature Encoding)?, \n'
                        'CBR-TF(content-based recommendation TF-IDF/CosineSim)? \n'
                        'Sim(ilarity matrix) calc for Item-based recommender? \n'
                        'H(ybrid recommendation TF-IDF/CosineSim)? \n'
                        'LCVSIM(eave one out cross-validation)? \n'
                        'Test (of hypothesis)'
                        '==>> '
                        )
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data' # for userids use 'critics_ratings_userIDs.data'
            itemfile = 'critics_movies.item'
            genrefile = 'critics_movies.genre' # movie genre file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys())) 
            ##
            ## delete this next line when you get genres (above) working
            # genres = {0: 'A', 1: 'Co', 2: 'H', 3: 'T', 4: 'F', 5: 'R', 6: 'A', 7: 'S', 8: 'I', 9: 'C', 10: 'D', 11: 'M'}
            ##
            print ('Number of distinct genres: %d, number of feature profiles: %d' % (len(genres), len(features)))
            print('genres')
            print(genres)
            print('features')
            print(features)
            print('prefs')
            print(prefs)
        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratngs file
            itemfile = 'u.item'  # movie titles file
            genrefile = 'u.genre' # movie genre file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            ##
            ## delete this next line when you get genres (above) working
            # genres = {0: 'unknown', 1: 'A', 2: 'A', 3: 'A', 4: 'C', 5: 'C', 6: 'C', 7: 'D', 8: 'D', 9: 'F', 10: 'F', 11: 'H', 12: 'M', 13: 'M', 14: 'R', 15: 'S', 16: 'Th', 17: 'W', 18: 'W'}
            ##
            print('Number of users: %d\nList of users [0:10]:' 
                  % len(prefs), list(prefs.keys())[0:10] ) 
            print ('Number of distinct genres: %d, number of feature profiles: %d' 
                   % (len(genres), len(features)))
            print('genres')
            print(genres)
            print('features')
            print(features)
        elif file_io == 'FE' or file_io == 'fe':
            
            print()
            
            movie_title_to_id = movie_to_ID(movies)
            #print(movie_title_to_id)
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                
                '''
                # e.g., critics data (CES)
                R = np.array([
                [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
                [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
                [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
                [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
                [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
                [0.0, 4.5, 0.0, 4.0, 1.0, 0.0],
                ])            
                '''      
                print('critics')
                print(R)
                print()
                print('features')
                print(features)

            elif len(prefs) > 10:
                print('ml-100k')   
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                
            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == 'TFIDF' or file_io == 'tfidf':
            print()
            movie_title_to_id = movie_to_ID(movies)
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)                 
                feature_docs = to_docs(feature_str, genres)
                
                '''
                # e.g., critics data (CES)
                R = np.array([
                [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
                [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
                [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
                [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
                [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
                [0.0, 4.5, 0.0, 4.0, 1.0, 0.0],
                ])            
                '''      
                print('critics')
                print(R)
                print()
                print('features')
                print(features)
                print()
                print('feature docs')
                print(feature_docs) 
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print('cosine sim matrix')
                print(cosim_matrix)
                 
                '''
                <class 'numpy.ndarray'> 
                
                [[1.         0.         0.35053494 0.         0.         0.61834884]
                [0.         1.         0.19989455 0.17522576 0.25156892 0.        ]
                [0.35053494 0.19989455 1.         0.         0.79459157 0.        ]
                [0.         0.17522576 0.         1.         0.         0.        ]
                [0.         0.25156892 0.79459157 0.         1.         0.        ]
                [0.61834884 0.         0.         0.         0.         1.        ]]
                '''
                
                #print and plot histogram of similarites


            elif len(prefs) > 10:
                print('ml-100k')   
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)                 
                feature_docs = to_docs(feature_str, genres)
                
                print(R[:3][:5])
                print()
                print('features')
                print(features[0:5])
                print()
                print('feature docs')
                print(feature_docs[0:5]) 
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print('cosine sim matrix')
                print (type(cosim_matrix), len(cosim_matrix))
                print()
                
                 
                '''
                <class 'numpy.ndarray'> 1682
                
                [[1.         0.         0.         ... 0.         0.34941857 0.        ]
                 [0.         1.         0.53676706 ... 0.         0.         0.        ]
                 [0.         0.53676706 1.         ... 0.         0.         0.        ]
                 [0.18860189 0.38145435 0.         ... 0.24094937 0.5397592  0.45125862]
                 [0.         0.30700538 0.57195272 ... 0.19392295 0.         0.36318585]
                 [0.         0.         0.         ... 0.53394963 0.         1.        ]]
                '''
                
                #print and plot histogram of similarites)
                
            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == 'CBR-FE' or file_io == 'cbr-fe':
            n=10
            print()
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                print('critics')
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                preds = get_FE_recommendations(prefs, features, movie_title_to_id, userID)
                print()
                print("Predictions: %s:" % userID)
                print(preds[:n])
                item = input('Enter item: ')
                print("Predictions: %s:" % item)
                print(get_FE_rec_single(prefs, features, movie_title_to_id, userID, item))

            elif len(prefs) > 10:
                print('ml-100k')
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                preds = get_FE_recommendations(prefs, features, movie_title_to_id, userID)
                print()
                print("Predictions: %s:" % userID)
                print(preds[:n])

            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == 'CBR-TF' or file_io == 'cbr-tf':
            n=10
            print()
            # determine the U-I matrix to use ..
            threshold = input('Enter neighbor threshold: ')
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                print('critics')

                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                preds = get_TFIDF_recommendations(prefs,cosim_matrix,userID,movie_title_to_id,threshold)
                
                print("Prediction: %s" %userID,preds[:n])
                item = input('Enter item: ')
                print("Predictions: %s:" % item)
                print(get_TFIDF_rec_single(prefs,cosim_matrix,userID,movie_title_to_id,threshold,item))


            elif len(prefs) > 10:
                print('ml-100k')
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                preds= get_TFIDF_recommendations(prefs,cosim_matrix,userID,movie_title_to_id,threshold)
                print("Prediction: %s" %userID,preds[:n])
            else:
                print ('Empty dictionary, read in some data!')
                print()
                
        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson?')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                    
                 

                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
               
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(itemsim)))
                    print()
                    ##
                    ## enter new code here, or call a new function, 
                    if (sim_method=='sim_distance'):
                        result = calculateSimilarItems(prefs, n=100, similarity=sim_distance)
                  
                    else:
                        result = calculateSimilarItems(prefs)
                    #for item in result:
                        #print(item, result[item])
                    ##    to print the sim matrix
                    ##
                print()
            else:
                print ('Empty dictionary, RML(ead ml100K) in some data!')  
                
        elif file_io == 'h' or file_io == 'H':
            n=input("Enter n, the ranking:")
            print()
            threshold = input('Enter threshold: ')
            hybrid_weight = input('Enter hybrid weight: ')
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                print('critics') 
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                

                preds = get_Hybrid_Recommendations(prefs, cosim_matrix, result, userID, movies, movie_title_to_id, hybrid_weight,threshold)
                
                print("Prediction: %s" %userID,preds[:int(n)])
                
            elif len(prefs) > 10:
                print('ml-100k')   
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                preds = get_Hybrid_Recommendations(prefs, cosim_matrix, result, str(userID), movies, movie_title_to_id, hybrid_weight,threshold)
                
                print("Prediction: %s" %userID,preds[:int(n)])
            else:
                print ('Empty dictionary, read in some data!')
                print()
        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            file_io = input('Enter FE or TFIDF or H(ybrid) algo:')
            n=input("Enter n, the ranking:")
            threshold = input('Enter threshold: ')
            if file_io == 'H' or file_io == 'h':
                hybrid_weight = input('Enter hybrid weight: ')
            else:
                hybrid_weight = 0
            if file_io == 'H' or file_io == 'h':
                if len(prefs) > 0:             
                    print('LOO_CV_SIM Evaluation')
                   
                    prefs_name = 'ML-100K'
                   
                    algo = get_Hybrid_Recommendations #hybrid-based

                    if sim_method == 'sim_pearson': 
                        sim = sim_pearson
                        MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, algo, result, cosim_matrix, movie_title_to_id,features,hybrid_weight,threshold,movies)
                        print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                              RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name,MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                        print()
                        print(MSE_list)
                        pickle.dump(MSE_list, open( "hybrid_pearson%s %s.p"%(hybrid_weight,n), "wb" ))
                    
                    if sim_method == 'sim_distance':
                        sim = sim_distance
                        MSE,MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, algo, result, cosim_matrix, movie_title_to_id,features,hybrid_weight,threshold,movies)
                        print('MSE for %s:%.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name,MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                        pickle.dump(MSE_list, open( "hybrid_distance%s %s.p"%(hybrid_weight,n), "wb" ))
                    else:
                        print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
            if file_io == 'tfidf' or file_io == 'TFIDF':
                if len(prefs) > 0:             
                    print('LOO_CV_SIM Evaluation')
                   
                    prefs_name = 'ML-100K'
                   
                    algo = get_TFIDF_recommendations #hybrid-based

                    result = cosim_matrix
                    MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, algo, result, cosim_matrix, movie_title_to_id,features,hybrid_weight,threshold,movies)
                    print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                          RMSE: %.5f, len(RMSE list): %d,' %(prefs_name, MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list))) 
                    print()
                    pickle.dump(MSE_list, open( "tfidf%s.p"%(threshold), "wb" ))
                    
            if file_io == 'fe' or file_io == 'FE':
                if len(prefs) > 0:             
                    print('LOO_CV_SIM Evaluation')
                   
                    prefs_name = 'ML-100K'
                   
                    algo = get_FE_recommendations

                    result = []
                    cosim_matrix = []
                    MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, algo, result, cosim_matrix, movie_title_to_id,features,hybrid_weight,threshold,movies)
                    print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                          RMSE: %.5f, len(RMSE list): %d,' %(prefs_name, MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list)))
                    print()
                    pickle.dump(MSE_list, open( "fe.p", "wb" ))
                    
                else:
                    print ('Empty dictionary, run RML(ead ml100K) OR Empty Sim Matrix, run Sim!')
        elif file_io == 'TEST' or file_io == 'test':
            print()
            file_io = input('Enter H(ybrid test), TF(tfidf & fe test), HT(hybrid & tfidf), HF(hybrid and fe) algo:, TOP(compare top results)')
            
            if file_io == 'H' or file_io == 'h':
                hybrid_weight = input('Enter hybrid weight: ')
                n=input("Enter n, the ranking:")
                # Load LOOCV results from file, print HYBRID
                try:
                    sq_diffs_info = pickle.load(open( "hybrid_pearson%s %s.p"%(hybrid_weight,n), "rb" ))
                    pearson_errors_u_lcv = sq_diffs_info
                    pearson_errors_u_lcv_MSE = np.mean(pearson_errors_u_lcv)
                    print("Results for sim_person, h, lcv: ")
                    print("MSE = ",pearson_errors_u_lcv_MSE )
                    print()
                    # Load LOOCV results from file, print
                    
                    sq_diffs_info = pickle.load(open( "hybrid_distance%s %s.p"%(hybrid_weight,n), "rb" ))
                    distance_errors_u_lcv = sq_diffs_info
                    distance_errors_u_lcv_MSE = np.mean(distance_errors_u_lcv)
                    print("Results for sim_distance, h, lcv: ")
                    print("MSE = ",distance_errors_u_lcv_MSE )        
                    print()
                    
                    print ('t-test for Hybrid-LCV distance vs pearson',len(distance_errors_u_lcv), len(pearson_errors_u_lcv))
                    print ('Null Hypothesis is that the means (MSE values for User-LCV distance and pearson) are equal')
                    
                    ## Calc with the scipy function
                    t_u_lcv, p_u_lcv = stats.ttest_ind(distance_errors_u_lcv,pearson_errors_u_lcv)
                    print("t = " + str(t_u_lcv))
                    print("p = " + str(p_u_lcv))
                    print()
                    print('==>> Unable to reject null hypothesis that the means are equal') # The two-tailed p-value    
                    print('==>> The means may or may not be equal')
                    check = input('\nContinue to tfidf & fe test Y(es) or N(o)? ', )
                    
                    if check ==  'Y' or check == 'y':
                        file_io = 'TF'
                    else:
                        print("Goodbye")
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!')
                    print()
           
            print()
            
            # Load LOOCV SIM results from file, print TFIDF AND FE
            if file_io == 'TF' or file_io == 'tf':
                threshold = input('Please enter the threshold for tfidf: ', )
                try:
                    print('Results for TFIDF, lcvsim:')
                    sq_diffs_info = pickle.load(open( "tfidf%s.p"%threshold, "rb" ))
                    tfidf_errors_lcvsim = sq_diffs_info
                    tfidf_errors_lcvsim_MSE = np.mean(tfidf_errors_lcvsim)  
                    print("MSE = ",tfidf_errors_lcvsim_MSE )
                    print()
                    # Load LOOCV SIM results from file, print
                    print('Results for FE, lcvsim:')
                    sq_diffs_info = pickle.load(open( "fe.p", "rb" ))
                    fe_errors_lcvsim = sq_diffs_info
                    fe_errors_lcvsim_MSE = np.mean(fe_errors_lcvsim)
                    print("MSE = ",fe_errors_lcvsim_MSE )
                    
                    print()
                    print ('t-test for TFIDF vs FE', len(tfidf_errors_lcvsim), len(fe_errors_lcvsim))
                    print ('Null Hypothesis is that the means (MSE values for TFIDF and FE) are equal')
                    
                    ## Calc with the scipy function
                    t_i_lcvsim, p_i_lcvsim = stats.ttest_ind(tfidf_errors_lcvsim, fe_errors_lcvsim)
                    print("t = " + str(t_i_lcvsim))
                    print("p = " + str(p_i_lcvsim))
                    print('==>> Unable to reject null hypothesis that the means are equal') 
                    print('==>> The means may or may not be equal')
                    
                    check = input('\nContinue to cross test b/n hybrid vs tfidf Y(es) or N(o):', )
                    if check ==  'Y' or check == 'y':
                        file_io = 'HT'
                    else:
                        print("Goodbye")
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!')
                    print()
            
            print()
            if file_io == 'Top' or file_io == 'top':
                print("top resutls are: mfALS, hybrid_pearson1 50, item-item, user-user")
                first = input('Please enter the first top result name: ', )
                second = input('Please enter the second top result name: ', )
                try:
                    print('Results for the first, lcvsim:')
                    sq_diffs_info = pickle.load(open( "%s.p"%(first), "rb" ))
                    first_errors_lcvsim = sq_diffs_info
                    first_errors_lcvsim_MSE = np.mean(first_errors_lcvsim)  
                    print("MSE = ",first_errors_lcvsim_MSE )
                    print()
                    # Load LOOCV SIM results from file, print
                    print('Results for the second, lcvsim:')
                    sq_diffs_info = pickle.load(open( "%s.p"%(second), "rb" ))
                    second_errors_lcvsim = sq_diffs_info
                    second_errors_lcvsim_MSE = np.mean(second_errors_lcvsim)
                    print("MSE = ",second_errors_lcvsim_MSE )
                    
                    print()
                    print ('t-test for first vs second', len(first_errors_lcvsim), len(second_errors_lcvsim))
                    print ('Null Hypothesis is that the means (MSE values for first and second) are equal')
                    
                    ## Calc with the scipy function
                    t_i_lcvsim, p_i_lcvsim = stats.ttest_ind(first_errors_lcvsim, second_errors_lcvsim)
                    print("t = " + str(t_i_lcvsim))
                    print("p = " + str(p_i_lcvsim))
                    print('==>> Unable to reject null hypothesis that the means are equal') 
                    print('==>> The means may or may not be equal')
                    
                    check = input('\nContinue to cross test b/n hybrid vs tfidf Y(es) or N(o):', )
                    if check ==  'Y' or check == 'y':
                        file_io = 'HT'
                    else:
                        print("Goodbye")
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!')
                    print()
            
            print()
            # Load LOOCV SIM results from file, print TFIDF AND TFIDF
            if file_io == 'TFTF' or file_io == 'tftf':
                threshold = input('Please enter the threshold for the first tfidf: ', )
                threshold2 = input('Please enter the threshold for the second tfidf: ', )
                try:
                    print('Results for first TFIDF, lcvsim:')
                    sq_diffs_info = pickle.load(open( "tfidf%s.p"%threshold, "rb" ))
                    tfidf_errors_lcvsim = sq_diffs_info
                    tfidf_errors_lcvsim_MSE = np.mean(tfidf_errors_lcvsim)  
                    print("MSE = ",tfidf_errors_lcvsim_MSE )
                    print()
                    # Load LOOCV SIM results from file, print
                    print('Results for second TFIDF, lcvsim:')
                    sq_diffs_info2 = pickle.load(open( "tfidf%s.p"%threshold2, "rb" ))
                    tfidf_errors_lcvsim2 = sq_diffs_info2
                    tfidf_errors_lcvsim_MSE2 = np.mean(tfidf_errors_lcvsim2)  
                    print("MSE = ",tfidf_errors_lcvsim_MSE2 )
                    print()
                    
                    print()
                    print ('t-test for TFIDF vs TFIDF2', len(tfidf_errors_lcvsim), len(tfidf_errors_lcvsim2))
                    print ('Null Hypothesis is that the means (MSE values for first TFIDF and second TFIDF) are equal')
                    
                    ## Calc with the scipy function
                    t_i_lcvsim, p_i_lcvsim = stats.ttest_ind(tfidf_errors_lcvsim, tfidf_errors_lcvsim2)
                    print("t = " + str(t_i_lcvsim))
                    print("p = " + str(p_i_lcvsim))
                    print('==>> Unable to reject null hypothesis that the means are equal') 
                    print('==>> The means may or may not be equal')
                    
                    check = input('\nContinue to cross test b/n hybrid vs tfidf Y(es) or N(o):', )
                    if check ==  'Y' or check == 'y':
                        file_io = 'HT'
                    else:
                        print("Goodbye")
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!')
                    print()
            
            print()
            if file_io == 'HT' or file_io == 'ht':
                threshold = input('Please enter the threshold for tfidf: ', )
                hybrid_weight = input('Enter hybrid weight: ')
                n=input("Enter n, the ranking:")                
                try:
                    print ('Cross t-tests between hybrid and tfidf')
                    print()
                    print ('Cross t-tests between hybrid and fe')
                    print()
                    sq_diffs_info = pickle.load(open( "hybrid_pearson%s %s.p"%(hybrid_weight,n), "rb" ))
                    pearson_errors_u_lcv = sq_diffs_info
                    pearson_errors_u_lcv_MSE = np.mean(pearson_errors_u_lcv)
                    print("Results for sim_person, h, lcv: ")
                    print("MSE = ",pearson_errors_u_lcv_MSE )
                    print()
                    # Load LOOCV results from file, print
                    
                    sq_diffs_info = pickle.load(open( "hybrid_distance%s %s.p"%(hybrid_weight,n), "rb" ))
                    distance_errors_u_lcv = sq_diffs_info
                    distance_errors_u_lcv_MSE = np.mean(distance_errors_u_lcv)
                    print("Results for sim_distance, h, lcv: ")
                    print("MSE = ",distance_errors_u_lcv_MSE )        
                    print()
                    
                    print('Results for TFIDF, lcvsim:')
                    sq_diffs_info = pickle.load(open( "tfidf%s.p"%threshold, "rb" ))
                    tfidf_errors_lcvsim = sq_diffs_info
                    tfidf_errors_lcvsim_MSE = np.mean(tfidf_errors_lcvsim)  
                    print("MSE = ",tfidf_errors_lcvsim_MSE )
                    print()
                    
                    print ('t-test for hybrid distance vs tfidf',len(distance_errors_u_lcv), len(tfidf_errors_lcvsim))
                    print ('Null Hypothesis is that the means (MSE values for hybrid distance and tfidf) are equal')
                    
                    ## Calc with the scipy function
                    t_u_lcv_i_lcvsim_distance, p_u_lcv_i_lcvsim_distance = stats.ttest_ind(tfidf_errors_lcvsim, distance_errors_u_lcv)
                    
                    print()
                    print('tfidf_lcv_MSE, distance_errors_u_lcv_MSE:', tfidf_errors_lcvsim_MSE, distance_errors_u_lcv_MSE)
                    print("t = " + str(t_u_lcv_i_lcvsim_distance))
                    print("p = " + str(p_u_lcv_i_lcvsim_distance), '==>> Unable to reject null hypothesis that the means are equal')
                    print('==>> The means may or may not be equal')
        
                    print()
                    print ('t-test for hybrid pearson vs tfidf',len(tfidf_errors_lcvsim), len(pearson_errors_u_lcv))
                    print ('Null Hypothesis is that the means (MSE values for hybrid pearson and tfidf) are equal')
                    
                    ## Cross Checking with the scipy function
                    t_u_lcv_i_lcvsim_pearson, p_u_lcv_i_lcvsim_pearson = stats.ttest_ind(tfidf_errors_lcvsim, pearson_errors_u_lcv)
                    print()
                    print('tfidf_MSE, pearson_errors_u_lcv_MSE:', tfidf_errors_lcvsim_MSE, pearson_errors_u_lcv_MSE)   
                    print("t = " + str(t_u_lcv_i_lcvsim_pearson))
                    print("p = " + str(p_u_lcv_i_lcvsim_pearson), '==>> Reject null hypothesis that the means are equal')
                    print('==>> The means are not equal')
                
                    check = input('\nContinue to cross test b/n hybrid & fe Y(es) or N(o) ',)
                    if check ==  'Y' or check == 'y':
                        file_io = 'HF'
                    else:
                        print("Goodbye")
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!')
                    print()
            
            print()
            if file_io == 'HF' or file_io == 'hf':
                hybrid_weight = input('Enter hybrid weight: ')
                n=input("Enter n, the ranking:")                
                try:
                    print ('Cross t-tests between hybrid and fe')
                    print()
                    sq_diffs_info = pickle.load(open( "hybrid_pearson%s %s.p"%(hybrid_weight,n), "rb" ))
                    pearson_errors_u_lcv = sq_diffs_info
                    pearson_errors_u_lcv_MSE = np.mean(pearson_errors_u_lcv)
                    print("Results for sim_person, h, lcv: ")
                    print("MSE = ",pearson_errors_u_lcv_MSE )
                    print()
                    # Load LOOCV results from file, print
                    
                    sq_diffs_info = pickle.load(open( "hybrid_distance%s %s.p"%(hybrid_weight,n), "rb" ))
                    distance_errors_u_lcv = sq_diffs_info
                    distance_errors_u_lcv_MSE = np.mean(distance_errors_u_lcv)
                    print("Results for sim_distance, h, lcv: ")
                    print("MSE = ",distance_errors_u_lcv_MSE )        
                    print()
                    
                    print('Results for FE, lcvsim:')
                    sq_diffs_info = pickle.load(open( "fe.p", "rb" ))
                    fe_errors_lcvsim = sq_diffs_info
                    fe_errors_lcvsim_MSE = np.mean(fe_errors_lcvsim)
                    print("MSE = ",fe_errors_lcvsim_MSE )
                    
                    print()
                    
                    print ('t-test for hybrid distance vs fe',len(distance_errors_u_lcv), len(fe_errors_lcvsim))
                    print ('Null Hypothesis is that the means (MSE values for hybrid distance and tfidf) are equal')
                    
                    ## Calc with the scipy function
                    t_u_lcv_i_lcvsim_distance, p_u_lcv_i_lcvsim_distance = stats.ttest_ind(fe_errors_lcvsim, distance_errors_u_lcv)
                    
                    print()
                    print('fe_lcv_MSE, distance_errors_u_lcv_MSE:', fe_errors_lcvsim_MSE, distance_errors_u_lcv_MSE)
                    print("t = " + str(t_u_lcv_i_lcvsim_distance))
                    print("p = " + str(p_u_lcv_i_lcvsim_distance), '==>> Unable to reject null hypothesis that the means are equal')
                    print('==>> The means may or may not be equal')
        
                    print()
                    print ('t-test for hybrid pearson vs fe',len(fe_errors_lcvsim), len(pearson_errors_u_lcv))
                    print ('Null Hypothesis is that the means (MSE values for hybrid pearson and fe) are equal')
                    
                    ## Cross Checking with the scipy function
                    t_u_lcv_i_lcvsim_pearson, p_u_lcv_i_lcvsim_pearson = stats.ttest_ind(fe_errors_lcvsim, pearson_errors_u_lcv)
                    print()
                    print('fe_MSE, pearson_errors_u_lcv_MSE:', fe_errors_lcvsim_MSE, pearson_errors_u_lcv_MSE)   
                    print("t = " + str(t_u_lcv_i_lcvsim_pearson))
                    print("p = " + str(p_u_lcv_i_lcvsim_pearson), '==>> Reject null hypothesis that the means are equal')
                    print('==>> The means are not equal')
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!')
                    print()
        else:
            done = True
        
    print('Goodbye!')  
    
if __name__ == "__main__":
    main()    
    
    
'''
Sample output ..


==>> cbr-fe
ml-100k

Enter username (for critics) or userid (for ml-100k) or return to quit: 340
rec for 340 = [(5.0, 'Woman in Question, The (1950)'), 
(5.0, 'Wallace & Gromit: The Best of Aardman Animation (1996)'), 
(5.0, 'Thin Man, The (1934)'), 
(5.0, 'Maltese Falcon, The (1941)'), 
(5.0, 'Lost Highway (1997)'), 
(5.0, 'Faust (1994)'), 
(5.0, 'Daytrippers, The (1996)'), 
(5.0, 'Big Sleep, The (1946)'), 
(4.836990595611285, 'Sword in the Stone, The (1963)'), 
(4.836990595611285, 'Swan Princess, The (1994)')]

==>> cbr-tf
ml-100k

Enter username (for critics) or userid (for ml-100k) or return to quit: 340
rec for 340 =  [
(5.000000000000001, 'Wallace & Gromit: The Best of Aardman Animation (1996)'), 
(5.000000000000001, 'Faust (1994)'), 
(5.0, 'Woman in Question, The (1950)'), 
(5.0, 'Thin Man, The (1934)'), 
(5.0, 'Maltese Falcon, The (1941)'), 
(5.0, 'Lost Highway (1997)'), 
(5.0, 'Daytrippers, The (1996)'), 
(5.0, 'Big Sleep, The (1946)'), 
(4.823001861184155, 'Sword in the Stone, The (1963)'), 
(4.823001861184155, 'Swan Princess, The (1994)')]

'''





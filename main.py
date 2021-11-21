from typing import Type
from numpy.lib.function_base import append
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# Retive the data from the files and removes duplicates
df = pd.read_csv("./data/The_Movies_Dataset/movies_metadata.csv")
df = df[["imdb_id","original_title", "original_language", "overview", "genres"]]
df = df[df['overview'].notna()]
df = df.drop_duplicates(subset=['imdb_id'])

# Creates the corpus
corpus = df["overview"].to_list()


# PS: "Uncomment" the plt.show if you want to see the graphs

# Show the length on the plots in the dataset
plt.hist([len(text) for text in corpus], bin=100)
# plt.show()
print(len(df.index))

#Removo the movies that are too short or too long
df = df[df['overview'].map(len) > 140 ]
df = df[df['overview'].map(len) < 800 ]

# Re-index the dataset
df.reindex()
print(len(df.index))

#Shows changes
corpus = df["overview"].to_list()
plt.hist([len(text) for text in corpus], bins=100)
# plt.show()



vect = TfidfVectorizer(stop_words="english",ngram_range=(1,1))
vect.fit_transform(corpus)

# Returns the movie if in the dataset
def findMovie(title):
    row = df.loc[df['original_title'] == title]
    return row

# Find similar movies to a movie
def findSimilarMovie(movie):
    movie =  vect.transform(findMovie(movie)["overview"])
    sim = []
    n =9200  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
    for current_df in list_df:
        title = current_df["original_title"].to_list()
        plots = current_df["overview"].to_list()
        cos = cosine_similarity(movie, vect.transform(plots))
        cos = cos.tolist()[0]
        cos = [(title[i],plots[i],simi) for i,simi in enumerate(cos)]
        sim.extend(cos)
    sim = sorted(sim, key=lambda el:el[2], reverse=True)
    res = pd.DataFrame(sim)
    res.columns =['title', 'plot', 'similarity']
    return res

# Find the movies that are similar to the movie. Compares to all others
def findAllSimilar():
    f = open("./res_Nomarlize.txt","w")
    size = len(df["original_title"].to_list())
    for i,title in enumerate(df["original_title"].to_list()):
        print(i*100/size)
        res = findSimilarMovie(title)
        f.write(f"{str(i)}\n")
        writeSection(f, res)  
        f.write(f"{str(i)}\n")  

# Wiring fuction to write the result to file
def writeSection(f, res):
   for row in (res.iloc[:11].iterrows()):
       row = row[1].to_list()
       f.write(f"{row[0]}\n{row[1]}\n{row[2]}\n")

# findAllSimilar()

# Controll the Twin movies that contains both movies in the dataset
def get_twin_films():
    twins = pd.read_csv("./data/twin_movies.csv")
    first_row  = twins["movie_1"].to_list()
    second_row = twins["movie_2"].to_list()
    movies = df["original_title"].to_list()
    twins_list = []
    for mov1, mov2 in zip(first_row, second_row):
        if mov1 in movies and mov2 in movies:
            twins_list.append((mov1,mov2))
    return twins_list

if __name__ == "__main__":
    #Test Twin Movies
    print(f"Length of Twin Movies: {len(get_twin_films())}")
    for movies in get_twin_films():
        print("----------------------------------------------------------------------------------|")
        similarities = findSimilarMovie(movies[0])
        r = similarities.loc[similarities['title'] == movies[1]]
        print(f"[{movies[0]} -> {movies[1]}]")
        print(r, "\n")
        
        similarities = findSimilarMovie(movies[1])
        r = similarities.loc[similarities['title'] == movies[0]]
        print(f"[{movies[1]} -> {movies[0]}]")
        print(r, "\n")
        print("----------------------------------------------------------------------------------|")
    
    
    # Test what is the most similar
    movies = [item for t in get_twin_films() for item in t]
    f = open("./similar.csv","w")
    f.write("movie,plot,similarity\n")
    for movie in movies:
        print("-----------------------------------------------------------------------------------------")
        m = findMovie(movie).iloc[0].to_numpy().tolist()
        print(f"MOVIE:\t{movie}\nPLOT:\t{m[3]}\n")
        similarities = findSimilarMovie(movie)
        # print(movie)
        r = similarities.iloc[1].to_numpy().tolist()
        print(f"MOVIE:\t{r[0]}\nPLOT:\t{r[1]}\nSIM:\t{r[2]}\n")
        print("-----------------------------------------------------------------------------------------")
    
            
    # White house Fallen plot
    m1 = findMovie("White House Down").iloc[0].to_numpy().tolist()
    m2 = findMovie("Olympus Has Fallen").iloc[0].to_numpy().tolist()
    print(m1)
    
    # q = "No Strings Attached"
    # print(f"Searching for {q}")
    # res = findSimilarMovie(q)
    # print(res)
    # get_twin_films()
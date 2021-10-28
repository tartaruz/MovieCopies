from typing import Type
from numpy.lib.function_base import append
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("./data/The_Movies_Dataset/movies_metadata.csv")
df = df[["imdb_id","original_title", "original_language", "overview", "genres"]]
df = df[df['overview'].notna()]
df = df.drop_duplicates(subset=['imdb_id'])

corpus = df["overview"].to_list()
vect = TfidfVectorizer(stop_words="english",ngram_range=(1,1))
vect.fit_transform(corpus)

def findMovie(title):
    row = df.loc[df['original_title'] == title]
    return row

def findSimilarMovie(movie):
    sim = []
    n = 4000  #chunk row size
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
    print(res[:10])


memento = vect.transform(findMovie("Child's Play")["overview"])

findSimilarMovie(memento)
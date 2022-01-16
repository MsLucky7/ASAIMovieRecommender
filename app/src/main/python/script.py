from math import sqrt

def main(number1, number2):
    num1 = int(number1)
    num2 = int(number2)

    sum = num1 + num2

    srt = sqrt(sum)

    return "sum is " +str(srt)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movie_dataset.csv")  #TODO: Dataset richtig verknüpfen

features = ['keywords','cast','genres','director']

# Kombiniere gewählte Columns in einen String
def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]


# Säubere NaNs aus String
for feature in features:
    df[feature] = df[feature].fillna('')
df["combined_features"] = df.apply(combine_features,axis=1)

# Erstelle Count Vectorizer Matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
# Normalisiere mit Cosinus um Werte zwischen 0 und 1 zu bekommen
cosine_sim = cosine_similarity(count_matrix)

# Helferfunktionen um aus Dataset Filminfos zu ziehen
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

movie_user_likes = "Avatar" #TODO: Hier muss die eingegebene Info vom User integriert werden
movie_index = get_index_from_title(movie_user_likes)

# Gehe durch alle Ähnlichkeitswerte für genannten Film und gleiche ab
#ACHTUNG, Film der verglichen wird, ist hier mit enthalten
similar_movies =  list(enumerate(cosine_sim[movie_index]))

#Sortiere Ähnlichekitswerte von 1 abwärts, schneide erstes Element ab weil das der zu vergleichende Film ist
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

#TODO: Gebe ähnlichste Filme an Androidscreen weiter und zeige sie dort an
i=0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>=5:
        break
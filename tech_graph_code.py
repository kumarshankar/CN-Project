# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:56:36 2018

@author: SHANKAR
"""

import networkx as nx
import string
import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import itertools

#nltk.download()

data_tech = pd.read_csv("G:\\CN Project\\Labelled_Tech - Labelled_Tech.csv")
data_music = pd.read_csv("G:\\CN Project\\Labelled_Music.csv")
data_ent = pd.read_csv("G:\\CN Project\\Labelled_Ent.csv")
data_tech['tokenized_sents'] = data_tech.apply(lambda row: nltk.word_tokenize(row['comment_text']), axis=1)
data_music['tokenized_sents'] = data_music.apply(lambda row: nltk.word_tokenize(row['comment_text']), axis=1)
data_ent['tokenized_sents'] = data_ent.apply(lambda row: nltk.word_tokenize(row['comment_text']), axis=1)

#Remove NULL rows    
#for index,row in data_tech.iterrows():
#    if(row['comment_text'] == '<NULL>'):
#        data_tech = data_tech.drop(index,axis=0)
#        
#for index,row in data_music.iterrows():
#    if(row['comment_text'] == '<NULL>'):
#        data_music = data_music.drop(index,axis=0)
        
        
stop_words = set(stopwords.words('english'))
      
extra_stop_words = ['.',',','!','...','?',':',"'",'-','I','i']
for x in extra_stop_words:
    stop_words.add(x)


for index,row in data_tech.iterrows():
    for token in row['tokenized_sents']:
        if token in stop_words:
            row['tokenized_sents'].remove(token)
            
for index,row in data_music.iterrows():
    for token in row['tokenized_sents']:
        if token in stop_words:
            row['tokenized_sents'].remove(token)
        
for index,row in data_ent.iterrows():
    for token in row['tokenized_sents']:
        if token in stop_words:
            row['tokenized_sents'].remove(token)
        
g_tech = nx.Graph()
g_music = nx.Graph()
g_ent = nx.Graph()

#########################################################################    
for index,row in data_tech.iterrows():
    for token in row['tokenized_sents']:
        if(token not in g_tech):
            g_tech.add_node(token)
        
for index,row in data_tech.iterrows():
    for pair in itertools.combinations(row['tokenized_sents'], r=2):
        g_tech.add_edge(*pair)
        
##########################################################################        
for index,row in data_music.iterrows():
    for token in row['tokenized_sents']:
        if(token not in g_music):
            g_music.add_node(token)
        
for index,row in data_music.iterrows():
    for pair in itertools.combinations(row['tokenized_sents'], r=2):
        g_music.add_edge(*pair)
        
##########################################################################        
for index,row in data_ent.iterrows():
    for token in row['tokenized_sents']:
        if(token not in g_ent):
            g_ent.add_node(token)
        
for index,row in data_ent.iterrows():
    for pair in itertools.combinations(row['tokenized_sents'], r=2):
        g_ent.add_edge(*pair)
###########################################################################
#below 3 lines are only for drawing the graph comment them while running on the server
nx.draw_networkx(g_tech,font_size =5)
nx.draw_networkx(g_music,font_size =5)
nx.draw_networkx(g_ent,font_size =5)

nx.draw(g_tech,node_size=13,node_color = 'blue')
nx.draw(g_music,node_size=13,node_color= 'blue')
nx.draw(g_ent,node_size=13,node_color = 'blue')

###########################################################################
#computing the degree centrality only once ,store it in a variable and use it 
tech_deg = nx.degree_centrality(g_tech)
music_deg = nx.degree_centrality(g_music)
ent_deg = nx.degree_centrality(g_ent)


def check_degree_centrality(word,cat_index):
    if cat_index == 0:
        if word in (tech_deg).keys():
            return (tech_deg)[word]
        return 0
    if cat_index == 1:
        if word in (music_deg).keys():
            return (music_deg)[word]
        return 0
    if cat_index == 3:
        if word in (ent_deg).keys():
            return (ent_deg)[word]
        return 0
    return 0

tech_deg_cent = []
for index,row in data_tech.iterrows():
    sum_tech = 0
    count_tech = 0
    for token in row['tokenized_sents']:
        sum_tech = sum_tech + (check_degree_centrality(token,0))
        count_tech = count_tech + 1
    tech_deg_cent.append(sum_tech/count_tech if count_tech != 0  else 0)  
        
data_tech['degree_centrality']   =  pd.Series(np.array(tech_deg_cent)) 

music_deg_cent = []
for index,row in data_music.iterrows():
    sum_music = 0
    count_music = 0
    for token in row['tokenized_sents']:
        sum_music = sum_music + (check_degree_centrality(token,1))
        count_music = count_music + 1
    #row['degree_centrality'] = sum_music/count_music
    music_deg_cent.append(np.round((sum_music/count_music),decimals=3) if count_music != 0  else 0) 

data_music['degree_centrality']   =  pd.Series(np.array(music_deg_cent))


ent_deg_cent = []
for index,row in data_ent.iterrows():
    sum_ent = 0
    count_ent = 0
    for token in row['tokenized_sents']:
        sum_ent = sum_ent + (check_degree_centrality(token,3))
        count_ent = count_ent + 1
    #row['degree_centrality'] = sum_music/count_music
    ent_deg_cent.append(np.round((sum_ent/count_ent),decimals=3) if count_ent != 0  else 0) 

data_ent['degree_centrality']   =  pd.Series(np.array(ent_deg_cent))

###############################################################################
#computing the betweenness centrality only once ,store it in a variable and use it 
tech_bet = nx.betweenness_centrality(g_tech)
music_bet = nx.betweenness_centrality(g_music)
ent_bet = nx.betweenness_centrality(g_ent)

def check_betweenness_centrality(word,cat_index):
    if cat_index == 0:
        if word in (tech_bet).keys():
            return (tech_bet)[word]
        return 0
    if cat_index == 1:
        if word in (music_bet).keys():
            return (music_bet)[word]
        return 0
    if cat_index == 2:
        if word in (ent_bet).keys():
            return (ent_bet)[word]
        return 0
    return 0

tech_bet_cent= []
music_bet_cent = []
ent_bet_cent = []

for index,row in data_tech.iterrows():
    sum_tech = 0
    count_tech = 0
    for token in row['tokenized_sents']:
        sum_tech = sum_tech + (check_betweenness_centrality(token,0))
        count_tech = count_tech + 1
    tech_bet_cent.append(sum_tech/count_tech if count_tech != 0  else 0)  
    
data_tech['betweenness_centrality']   =  pd.Series(np.array(tech_bet_cent)) 


for index,row in data_music.iterrows():
    sum_music = 0
    count_music = 0
    for token in row['tokenized_sents']:
        sum_music = sum_music + (check_betweenness_centrality(token,1))
        count_music = count_music + 1
    music_bet_cent.append(sum_music/count_music if count_music != 0  else 0) 

data_music['betweenness_centrality']   =  pd.Series(np.array(music_bet_cent))

for index,row in data_ent.iterrows():
    sum_ent = 0
    count_ent = 0
    for token in row['tokenized_sents']:
        sum_ent = sum_ent + (check_betweenness_centrality(token,2))
        count_ent = count_ent + 1
    ent_bet_cent.append(sum_ent/count_ent if count_ent != 0  else 0) 

data_ent['betweenness_centrality']   =  pd.Series(np.array(ent_bet_cent))

##################################################################################
#computing the clustering coefficient only once ,store it in a variable and use it 

tech_clus = nx.clustering(g_tech)
music_clus = nx.clustering(g_music)
ent_clus = nx.clustering(g_ent)

def check_clustering_coefficient(word,cat_index):
    if cat_index == 0:
        if word in (tech_clus).keys():
            return (tech_clus)[word]
        return 0
    if cat_index == 1:
        if word in (music_clus).keys():
            return (music_clus)[word]
        return 0
    if cat_index == 2:
        if word in (ent_clus).keys():
            return (ent_clus)[word]
        return 0
    return 0

tech_clus_coeff= []
music_clus_coeff = []
ent_clus_coeff = []

for index,row in data_tech.iterrows():
    sum_tech = 0
    count_tech = 0
    for token in row['tokenized_sents']:
        sum_tech = sum_tech + (check_clustering_coefficient(token,0))
        count_tech = count_tech + 1
    tech_clus_coeff.append(sum_tech/count_tech if count_tech != 0  else 0)  
    
data_tech['clustering_coefficient']   =  pd.Series(np.array(tech_clus_coeff)) 


for index,row in data_music.iterrows():
    sum_music = 0
    count_music = 0
    for token in row['tokenized_sents']:
        sum_music = sum_music + (check_clustering_coefficient(token,1))
        count_music = count_music + 1
    music_clus_coeff.append(sum_music/count_music if count_music != 0  else 0) 

data_music['clustering_coefficient']   =  pd.Series(np.array(music_clus_coeff))

for index,row in data_ent.iterrows():
    sum_ent = 0
    count_ent = 0
    for token in row['tokenized_sents']:
        sum_ent = sum_ent + (check_clustering_coefficient(token,2))
        count_ent = count_ent + 1
    ent_clus_coeff.append(sum_ent/count_ent if count_ent != 0  else 0) 

data_ent['clustering_coefficient']   =  pd.Series(np.array(ent_clus_coeff))
###############################################################################
# store the updated dataframes in the csv format       
data_music.to_csv("G:\\CN Project\\updated_music.csv",index=False)
data_tech.to_csv("G:\\CN Project\\updated_tech.csv",index=False)
data_ent.to_csv("G:\\CN Project\\updated_ent.csv",index=False)


###############################################################################
    
music_deg = data_music.groupby('comment_class',as_index=False)['degree_centrality'].mean()    
music_count = data_music.groupby('comment_class',as_index=False).count()


tech_deg = data_tech.groupby('comment_class',as_index=False)['degree_centrality'].mean()    
tech_count = data_tech.groupby('comment category',as_index=False).count()

ent_deg = data_ent.groupby('comment_class',as_index=False)['degree_centrality'].mean()    
ent_count = data_ent.groupby('category',as_index=False).count()

music_deg.to_csv("G:\\CN Project\\music_deg.csv",index=False)
tech_deg.to_csv("G:\\CN Project\\tech_deg.csv",index=False)
ent_deg.to_csv("G:\\CN Project\\ent_deg.csv",index=False)
ent_count.to_csv("G:\\CN Project\\ent_count.csv",index=False)

###############################################################################

music_bet = data_music.groupby('category',as_index=False)['betweenness_centrality'].mean()    
music_count = data_music.groupby('comment_class',as_index=False).count()


tech_bet = data_tech.groupby('comment category',as_index=False)['betweenness_centrality'].mean()    
tech_count = data_tech.groupby('comment category',as_index=False).count()

ent_bet = data_ent.groupby('category',as_index=False)['betweenness_centrality'].mean()    
ent_count = data_ent.groupby('category',as_index=False).count()

music_deg.to_csv("G:\\CN Project\\music_deg.csv",index=False)
tech_deg.to_csv("G:\\CN Project\\tech_deg.csv",index=False)
ent_deg.to_csv("G:\\CN Project\\ent_deg.csv",index=False)
ent_count.to_csv("G:\\CN Project\\ent_count.csv",index=False)

###############################################################################
music_cls = data_music.groupby('category',as_index=False)['clustering_coefficient'].mean()    
music_count = data_music.groupby('comment_class',as_index=False).count()


tech_cls = data_tech.groupby('comment_class',as_index=False)['clustering_coefficient'].mean()    
tech_count = data_tech.groupby('comment category',as_index=False).count()

ent_cls = data_ent.groupby('category',as_index=False)['clustering_coefficient'].mean()    
ent_count = data_ent.groupby('category',as_index=False).count()

music_deg.to_csv("G:\\CN Project\\music_deg.csv",index=False)
tech_deg.to_csv("G:\\CN Project\\tech_deg.csv",index=False)
ent_deg.to_csv("G:\\CN Project\\ent_deg.csv",index=False)
ent_count.to_csv("G:\\CN Project\\ent_count.csv",index=False)


import operator
sorted_x = sorted((nx.degree_centrality(g)).items(), key=operator.itemgetter(1),reverse=True)

      

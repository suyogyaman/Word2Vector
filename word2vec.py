# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:30:02 2020

@author: suyog
"""

#Example of Word2Vector

#Importing Libraries
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
#nltk.download('punkt')

paragraph = """Word2vec is a group of related models that are used to produce word embeddings. 
            These models are shallow, two-layer neural networks that are trained to reconstruct linguistic 
            contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, 
            typically of several hundred dimensions, with each unique word in the corpus being assigned 
            a corresponding vector in the space. Word vectors are positioned in the vector space such that 
            words that share common contexts in the corpus are located close to one another in the space.
            Word2vec was created and published in 2013 by a team of researchers led by Tomas Mikolov at Google 
            and patented. The algorithm has been subsequently analysed and explained by other 
            researchers. Embedding vectors created using the Word2vec algorithm have many advantages 
            compared to earlier algorithms such as latent semantic analysis.Word2vec can utilize either 
            of two model architectures to produce a distributed representation of 
            words: continuous bag-of-words (CBOW) or continuous skip-gram. In the continuous bag-of-words 
            architecture, the model predicts the current word from a window of surrounding context words. 
            The order of context words does not influence prediction (bag-of-words assumption). In the 
            continuous skip-gram architecture, the model uses the current word to predict the surrounding 
            window of context words."""

#Preprocessing the data
text = re.sub(r'\[0-9]*\]',' ',paragraph) #remove any numbers
text = re.sub(r'\s+',' ',text) #remove the space
text = text.lower() #convert all strings in text to lower case
text = re.sub(r'\d',' ',text) #remove digits
text = re.sub(r'\s',' ',text) #remove spaces

#Preparing the dataset
sentences = nltk.sent_tokenize(text) #convert paragraph to sentences

sentences = [nltk.word_tokenize(sentence) for sentence in sentences] #convert sentences to word

#remove the stopwords from our sentences
for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

#training our word2vec model
model = Word2Vec(sentences,min_count=1)

#See the word vectors
words = model.wv.vocab

#Finding the word vectors
vector = model.wv['continuous']    

#Find most similar word
similar = model.wv.most_similar('continuous')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








 
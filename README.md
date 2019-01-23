# Introduction
This project deals with SMS texts and uses Natural Language Procesing and Deep Learning to clasify the SMS as Spam or NonSpam. 
Here Preprocessing is done via Python, Pandas, Regex and Numpy and the main neural network code uses Keras library.

# Dataset
The Dataset consists of 11134 instances of text and given type. It can be downloaded from: https://drive.google.com/file/d/1OaFDBhd0zIxbf0o_xbOgBEdUz_MiVNrl/view?usp=sharing

# Word Vectors
The word vectors that have been used here are of Gensim Word2Vec Model. The Preprocess code is dynamic and we can also invoke the 
Google Word2Vec Model also from Gensim. But to invoke the Google Word2Vec model, one must have the gogle word2vec bin file 
which can be dowmloaded here: https://code.google.com/archive/p/word2vec
The difference between the gensim's own vectors and google word vectors is that Google word vectors are more expressive as they 
have been trained on many corpuses where as gensim's vectors are trained on the existing data that has been passed to it.

# Summary

Train Accuracy: 98.12% and 
Test Accuracy : 97% 

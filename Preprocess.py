
def load(file_name):
    import pandas as pd
    import os
    #Preparing dataset as type and text sms
    name_list = ['Type','Text']
    path = os.getcwd()+'/'+file_name
    data_frame = pd.read_csv(path, sep = ',', header = None, names = name_list)
    data_frame = data_frame.drop([0], axis = 0)
    return data_frame

def create_labels(data_frame): 
    '''Enamurating the gold labels and storing in a seperated dataframe'''
    import numpy as np
    data_frame["Type"] = data_frame["Type"].astype('category')
    data_frame["Type"] = data_frame["Type"].cat.codes
    data_frame["Type"] = data_frame["Type"].astype('int64')
    labels = np.reshape(np.array(data_frame["Type"]),(len(data_frame["Type"]),1))
    return labels

def make_word_container(data_frame):
    '''function arguments are data_frame and a string which is column of data_frame fow which word_container
       is to be constructed'''
    import re
    max_len_sentence = 0
    word_container = []
    sentence_list = list(data_frame['Text'])
    for i in range(len(sentence_list)):
        w = sentence_list[i]
        word_container.append(re.split('\W',str(w)))
    for sentence in range(len(word_container)):
        for word in word_container[sentence]:
            if word == '':
                word_container[sentence].remove('')
    for i in range(len(word_container)):
        length = len(word_container[i])
        if length >= max_len_sentence:
            max_len_sentence = length
    return word_container,max_len_sentence

def No_Word_padding(word_container, max_len_sentence):
    '''function for padding no_word so that shape remains equal for all word_vec 
       that will be obtained after Word2Vec embedding. The argument max_len_sentence is 
       global max length obtained from 'max_dataset_sentence_length' function. Use after concatenate'''
    for i in range(len(word_container)):
        while len(word_container[i]) < max_len_sentence:
            word_container[i].append('nan')
    return word_container

def Word2Vec(word_container, word_vector_size, min_count):
    '''this function takes arguments word_container, min count it returns the word2vec model, 
       vocabulary.'''
    import numpy as np
    import gensim
    from gensim.models import Word2Vec
    #model training
    model = Word2Vec(word_container, size = word_vector_size, min_count = min_count)
    print('Word2Vec Model details = {}'.format(model))
    vocabs = list(model.wv.vocab)
    return model, vocabs

def Google_Word2Vec():
    '''this function  returns the Google Word2Vec model'''
    import os
    import gensim
    from gensim.models import Word2Vec
    path = os.getcwd()
    google_model = gensim.models.KeyedVectors.load_word2vec_format(path + '/GoogleNews-vectors-negative300.bin', binary=True)  
    google_vocabs = list(google_model.wv.vocab)
    return google_model, google_vocabs

def Word_Embedding(model, word_vector_size, word_container):
    '''function to embedd words to vectors according to the input word2vec model (model can be local or Google Word2Vec)
       model. It returns word_vector_matrix (zero padded) that can be directly fed to LSTM, GRU etc.
       use after concatenate, No_word padding and then after Word2Vec
       Note ::  While using Google_Word2Vec, make vector size = 300'''
    import numpy as np
    sentence_number = len(word_container)
    vector_size = word_vector_size
    max_len_sentence = len(word_container[1])
    for i in range(len(word_container)):
        for j in range(len(word_container[i])):
            try:
                a = model[word_container[i][j]]
            except KeyError:
                word_container[i][j] = 'nan'
    word_vector_matrix = np.zeros(shape = (sentence_number,max_len_sentence,vector_size), dtype = np.float32)
    for i in range(sentence_number):
        for j in range(max_len_sentence):
            word_vector_matrix[i][j] = model[word_container[i][j]]
    return word_vector_matrix


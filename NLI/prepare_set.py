#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:20:13 2019

@author: loretta

generate train, dev, test sets
"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import os
import gc

def get_vector(documents,mode):
    if mode == "Word2Vec":
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        new_str = []
        for i in documents:
            new_str.append(tokenizer.tokenize(i.lower()))
        model = Word2Vec(new_str, min_count=1, size=300)
        print("word2vec model is ready...")

    if mode == "Glove":
        print("loading Glove...")
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format('w2v/gensim_glove_vectors.txt',binary=False)
    #    embedding_dim=500
    #    word_index = tokenizer.word_index
        print("Glove model is ready...")        
        
    word_vector = model.wv
    return word_vector


def create_embedding_matrix(tokenizer, word_vectors,mode):
    if mode == "Word2Vec" or mode=="Glove":
        embedding_dim = 300
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors[word]
    #        print(word+' is in')
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except:
#            if embedding_vector is not None:
            embedding_matrix[i] = np.zeros(embedding_dim)
#            print(word+' is not in vocabulary')
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
#    gc.collect()
    return embedding_matrix
    
    
    
def word_embed_meta_data(documents,mode):
    tokenizer = Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'',
                                   lower=True,
                                   split=" ",
                                   char_level=False)    
    tokenizer.fit_on_texts(documents)
    word_vector = get_vector(documents,mode)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector,mode)
    del word_vector
#    gc.collect()
    return tokenizer, embedding_matrix


def create_train_dev_set(tokenizer, sentences_pair, sim, left_sequence_length, \
                         right_sequence_length, validation_split_ratio):
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    
    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=left_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=right_sequence_length)

    train_labels = np.array(sim)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))


    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, val_data_1, val_data_2, labels_val


def create_train_dev_from_files(tokenizer, sentences_pair_train, sim_train, \
                                sentences_pair_dev,sim_dev,left_sequence_length,\
                                right_sequence_length):
    """
    train_set & dev_set from 2 different files
    """
    sentences1 = [x[0] for x in sentences_pair_train]
    sentences2 = [x[1] for x in sentences_pair_train]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    
    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=left_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=right_sequence_length)

    train_labels = np.array(sim_train)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]

    sentences1_dev = [x[0] for x in sentences_pair_dev]
    sentences2_dev = [x[1] for x in sentences_pair_dev]
    dev_sequences_1 = tokenizer.texts_to_sequences(sentences1_dev)
    dev_sequences_2 = tokenizer.texts_to_sequences(sentences2_dev)
    
    dev_padded_data_1 = pad_sequences(dev_sequences_1, maxlen=left_sequence_length)
    dev_padded_data_2 = pad_sequences(dev_sequences_2, maxlen=right_sequence_length)

    dev_labels = np.array(sim_dev)

    shuffle_indices = np.random.permutation(np.arange(len(dev_labels)))
    dev_data_1_shuffled = dev_padded_data_1[shuffle_indices]
    dev_data_2_shuffled = dev_padded_data_2[shuffle_indices]
    dev_labels_shuffled = dev_labels[shuffle_indices]
    
#    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))
#
#
#    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
#    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
#    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]

    return train_data_1_shuffled, train_data_2_shuffled, train_labels_shuffled, \
            dev_data_1_shuffled, dev_data_2_shuffled, dev_labels_shuffled


def create_test_data(tokenizer, test_sentences_pair, left_sequence_length, right_sequence_length):
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    
    test_data_1 = pad_sequences(test_sequences_1, maxlen=left_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=right_sequence_length)

    return test_data_1, test_data_2


#=======================================
#============FOR LSTM ONLY==============
#=======================================
def create_train_dev_set_lstm(tokenizer, sentences_pair, sim, left_sequence_length,right_sequence_length, validation_split_ratio=0.1):
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=left_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=right_sequence_length)
    train_labels = np.array(sim)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val

def create_train_dev_from_files_lstm(tokenizer, sentences_pair_train, sim_train, \
                                sentences_pair_dev,sim_dev,left_sequence_length,\
                                right_sequence_length):
    
    sentences1 = [x[0] for x in sentences_pair_train]
    sentences2 = [x[1] for x in sentences_pair_train]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=left_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=right_sequence_length)
    train_labels = np.array(sim_train)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]
    
    sentences1_dev = [x[0] for x in sentences_pair_dev]
    sentences2_dev = [x[1] for x in sentences_pair_dev]
    dev_sequences_1 = tokenizer.texts_to_sequences(sentences1_dev)
    dev_sequences_2 = tokenizer.texts_to_sequences(sentences2_dev)
    leaks_dev = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(dev_sequences_1, dev_sequences_2)]

    dev_padded_data_1 = pad_sequences(dev_sequences_1, maxlen=left_sequence_length)
    dev_padded_data_2 = pad_sequences(dev_sequences_2, maxlen=right_sequence_length)
    dev_labels = np.array(sim_dev)
    leaks_dev = np.array(leaks_dev)

    shuffle_indices = np.random.permutation(np.arange(len(dev_labels)))
    dev_data_1_shuffled = dev_padded_data_1[shuffle_indices]
    dev_data_2_shuffled = dev_padded_data_2[shuffle_indices]
    dev_labels_shuffled = dev_labels[shuffle_indices]
    leaks_dev_shuffled = leaks_dev[shuffle_indices]


    return train_data_1_shuffled, train_data_2_shuffled, train_labels_shuffled, \
            leaks_shuffled, dev_data_1_shuffled, dev_data_2_shuffled, dev_labels_shuffled, \
            leaks_dev_shuffled
            
def create_original_train_dev_from_files_lstm(sentences_pair_train, sim_train, \
                                sentences_pair_dev,sim_dev):
    
    sentences1 = [x[0] for x in sentences_pair_train]
    sentences2 = [x[1] for x in sentences_pair_train]
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(sentences1, sentences2)]
    
    sentences1 = np.array(sentences1, dtype=object)[:, np.newaxis]
    sentences2 = np.array(sentences2, dtype=object)[:, np.newaxis]
    
    train_labels = np.array(sim_train)
#    train_labels = train_labels.reshape((train_labels.shape[0], train_labels.shape[1], 1))
    leaks = np.array(leaks)

    sentences1_dev = [x[0] for x in sentences_pair_dev]
    sentences2_dev = [x[1] for x in sentences_pair_dev]
    leaks_dev = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(sentences1_dev, sentences2_dev)]

    sentences1_dev = np.array(sentences1_dev, dtype=object)[:, np.newaxis]
    sentences2_dev = np.array(sentences2_dev, dtype=object)[:, np.newaxis]

    dev_labels = np.array(sim_dev)
    leaks_dev = np.array(leaks_dev)
    
#    dev_labels = dev_labels.reshape((dev_labels.shape[0], dev_labels.shape[1], 1))



    return sentences1, sentences2, train_labels, leaks, sentences1_dev, sentences2_dev,\
            dev_labels, leaks_dev


def create_test_data_lstm(tokenizer, test_sentences_pair, left_sequence_length,right_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=left_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=right_sequence_length)

    return test_data_1, test_data_2, leaks_test
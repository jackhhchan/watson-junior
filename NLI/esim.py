#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:50:19 2019

@author: loretta
"""

from __future__ import print_function
from keras import backend as K
from keras.models import Model
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping,ModelCheckpoint
import time
import os
from keras.models import load_model
import tqdm
import json
from keras.layers import LSTM, Bidirectional, GlobalAveragePooling1D, Input,\
      concatenate, Lambda, subtract, multiply, Dense, TimeDistributed, Embedding
from NLI.attention import DotProductAttention
from NLI.abcnn import word_embed_meta_data, create_train_dev_set,create_test_data,\
    create_train_dev_from_files
from sentence_selection.generateTrainingFile import getPage_index,readOneFile      


def plot(*args, **kwargs):

    from keras.utils import plot_model as plt
    plt(*args, **kwargs)



def buildESIM(tokenizer,sentences_pair_train,sim_train,sentences_pair_dev=None,sim_dev=None,\
              embed_dimensions,embedding_matrix,left_sequence_length,right_sequence_length,\
              num_classes,epoch,batch_size):
    
    input_premise = Input(shape=(left_sequence_length,))
    input_hypothesis = Input(shape=(right_sequence_length,))
    
#    embedding = ELMoEmbedding(output_mode=elmo_output_mode, idx2word=self.config.idx2token, mask_zero=mask_zero,
#                              hub_url=elmo_model_url, elmo_trainable=elmo_trainable)
#    premise_embed = embedding(input_premise)
#    hypothesis_embed = embedding(input_hypothesis)
    
    if sentences_pair == None:
        train_data_x1, train_data_x2, train_labels, \
        val_data_x1, val_data_x2, val_labels  = create_train_dev_set(
                tokenizer, sentences_pair_train, sim_train, left_sequence_length, right_sequence_length, \
                validation_split_ratio=0)
    else:
        train_data_x1, train_data_x2, train_labels, \
        val_data_x1, val_data_x2, val_labels  = create_train_dev_from_files(
                tokenizer, sentences_pair_train, sim_train, sentences_pair_dev, \
                sim_dev,left_sequence_length, right_sequence_length)
    
    nb_words = len(tokenizer.word_index) + 1
    emb_layer = Embedding(nb_words, embed_dimensions,
#                          input_length=max_len,
                          weights=[embedding_matrix],
                          trainable=False)
#                          mask_zero=True)
    
    premise_embed = emb_layer(input_premise)
    hypothesis_embed = emb_layer(input_hypothesis)

    bilstm_1 = Bidirectional(LSTM(units=300, return_sequences=True))
    premise_hidden = bilstm_1(premise_embed)
    hypothesis_hidden = bilstm_1(hypothesis_embed)

    # local inference collected over sequences
    premise_attend, hypothesis_attend = DotProductAttention()([premise_hidden, hypothesis_hidden])

    # enhancement of local inference information
    premise_enhance = concatenate([premise_hidden, premise_attend, 
                                   subtract([premise_hidden, premise_attend]),
                                   multiply([premise_hidden, premise_attend])])
    hypothesis_enhance = concatenate([hypothesis_hidden, hypothesis_attend,
                                      subtract([hypothesis_hidden, hypothesis_attend]),
                                      multiply([hypothesis_hidden, hypothesis_attend])])

    # inference composition
    feed_forward1 = TimeDistributed(Dense(units=300, activation='relu'))
    feed_forward2 = TimeDistributed(Dense(units=300, activation='relu'))

    bilstm_2 = Bidirectional(LSTM(units=300, return_sequences=True))
    premise_compose = bilstm_2(feed_forward1(premise_enhance))
    hypothesis_compose = bilstm_2(feed_forward2(hypothesis_enhance))
#    premise_compose = bilstm_2(premise_enhance)
#    hypothesis_compose = bilstm_2(hypothesis_enhance)
    

    global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
    premise_avg = GlobalAveragePooling1D()(premise_compose)
    premise_max = global_max_pooling(premise_compose)
    hypothesis_avg = GlobalAveragePooling1D()(hypothesis_compose)
    hypothesis_max = global_max_pooling(hypothesis_compose)

    
    inference_compose = concatenate([premise_avg, premise_max, hypothesis_avg, hypothesis_max])

    dense = Dense(units=300, activation='tanh')(inference_compose)
    output = Dense(num_classes, activation='softmax')(dense)

    model = Model([input_premise, input_hypothesis], output)
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=8)

    checkpoint_dir = './trained_model/ESIM/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + str(int(time.time())) + '.h5'

    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_acc', mode='auto',\
                                       save_best_only=True, save_weights_only=False)

    his = model.fit([train_data_x1, train_data_x2], train_labels,
              validation_data=([val_data_x1, val_data_x2], val_labels),
              epochs=epoch, batch_size=batch_size, shuffle=False,verbose=1,
              callbacks=[early_stopping,model_checkpoint]
                )
    return model,his

if __name__ == '__main__':
    """
    loretta's playgournd
    """
    
    
    with open('resource/train.json') as data:
        json_data = json.load(data)
        
#    =========================
#    prepare training file
#    =========================

    page_index, page_size = getPage_index()
    
    claims = []
    evidences = []
    labels = []
    for j,item in enumerate(json_data):
        if j < 100:
        # only get first 100, [FOR NOW]
            evds = []
            if json_data[item]['evidence'] == []:
                continue
            for k,evd in enumerate(json_data[item]['evidence']):
                if evd[0] in page_index:
                    evds.append(readOneFile(page_index[evd[0]],evd[0],evd[1]))
            claims.append(json_data[item]['claim'])
            evidences.append(' '.join(evds))
            labels.append(json_data[item]['label'])
    
    label_transfer = {'SUPPORTS':0,'REFUTES':1}
    label_v2 = []
    for l in labels:
        if l in label_transfer:
            label_v2.append(label_transfer[l])
                
    
#    sampled_file.info()
            
#    =========================
#    prepare inputs
#    =========================

    tokenizer, embedding_matrix = word_embed_meta_data(sampled_file.claim+sampled_file.evidences,mode = "Glove")
    # return the embedding_matrix for sentences.
    sentences_pair = [(x1, x2) for x1, x2 in zip(sampled_file.claim, sampled_file.evidences)]
            
    left_sequence_length = 32
    right_sequence_length = 64     # max length of one sentence
    num_samples = len(claims)
    num_classes = 2 # supports, refutes
    embed_dimensions = 300
    epoch = 50
    # batch_size = 1024
    batch_size = 256
    
    sim = keras.utils.to_categorical(sampled_file.labels, num_classes)
                 
    model,his = buildESIM(tokenizer,sentences_pair,sim,embed_dimensions,embedding_matrix,\
                          left_sequence_length,right_sequence_length,num_classes,epoch,batch_size)
    
    his = his.history
    
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    plt.plot(his['acc'], 'r:')
#    plt.plot(his['val_acc'], 'g-')
#    plot(model, to_file='esim1.png')
#    model = load_model('/Users/loretta/watson-junior/trained_model/ESIM/1558325833.h5',\
#                       custom_objects={'DotProductAttention':DotProductAttention})
#    
#    """
#    testing
#    """
    with open('resource/random-devset.json') as data:
            json_data_dev = json.load(data)
    
    claims_dev = []
    evidences_dev = []
    labels_dev = []
    for j,item in enumerate(json_data_dev):
        if j < 100:
            evds = []
            if json_data_dev[item]['evidence'] == []:
                continue
            for k,evd in enumerate(json_data_dev[item]['evidence']):
                if evd[0] in page_index:
                    evds.append(readOneFile(page_index[evd[0]],evd[0],evd[1]))
            claims_dev.append(json_data_dev[item]['claim'])
            evidences_dev.append(' '.join(evds))
            labels_dev.append(json_data_dev[item]['label'])
    
    label_v2_dev = []
    for l in labels_dev:
        if l in label_transfer:
            label_v2_dev.append(label_transfer[l])
    
    
    test_file = pd.DataFrame({'claim':claims_dev,'evidences':evidences_dev,'labels':label_v2_dev})
    test_file.labels.value_counts()
    
    tokenizer.word_index
    test_sentences_pair = [(x1, x2) for x1, x2 in zip(claims[:5], evidences[:5])]
    test_claim,test_evidence = create_test_data(tokenizer, test_sentences_pair, \
                                                left_sequence_length, right_sequence_length)
    pred = model.predict([test_claim,test_evidence])
    from sklearn.metrics import mean_squared_error,confusion_matrix,f1_score
    pred = np.argmax(pred,axis=1)
    f1_score(pred,test_file.labels)
#    plot(model, to_file='esim.png')

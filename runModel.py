#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:58:14 2019

@author: loretta


run model
"""
from NLI.esim import buildESIM, plot
from NLI.normalLSTM import buildLSTM
from NLI.abcnn import word_embed_meta_data, create_test_data
from sentence_selection.generateTrainingFile import getPage_index,readOneFile   
import json
import pandas as pd
import keras

if __name__ == '__main__':
    with open('resource/train.json') as data:
            json_data = json.load(data)
        
    # read the json_file
        
    page_index, page_size = getPage_index()
    
    claims = []
    evidences = []
    labels = []
    for j,item in enumerate(json_data):
#        if j < 100:
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
                
#    training_file = pd.DataFrame({'claim':claims,'evidences':evidences,'labels':label_v2})  
#    training_file.labels.value_counts()  
#    training_file.to_csv('resource/training_file_v3.csv')    
            
    tokenizer, embedding_matrix = word_embed_meta_data(claims+evidences,mode = "Glove")
    # return the embedding_matrix for sentences.
    sentences_pair = [(x1, x2) for x1, x2 in zip(claims, evidences)]
            
    max_len = 64     # max length of one sentence
    num_samples = len(claims)
    num_classes = 2 # supports, refutes
    embed_dimensions = 300
    epoch = 50
    batch_size = 1024
    
    sim = keras.utils.to_categorical(label_v2, num_classes)
            
#"""
#ESIM 
#"""        
    model,his = buildESIM(tokenizer,sentences_pair,sim,embed_dimensions,embedding_matrix,\
                          max_len,num_classes,epoch,batch_size)
    
#    plot(model, to_file='esim.png')

#"""
#siamese LSTM
#"""
    number_lstm_units = 50
    rate_drop_lstm = 0.17
    rate_drop_dense = 0.15
    number_dense_units = 100
    model,his = buildLSTM(tokenizer,sentences_pair,sim,embed_dimensions,embedding_matrix,\
                number_lstm_units,rate_drop_lstm, rate_drop_dense, number_dense_units,\
                max_len,num_classes,epoch,batch_size)
#    plot(model,'normalLSTM.png')
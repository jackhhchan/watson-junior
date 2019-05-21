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

def plot_acc(his):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(his.history['acc'], 'r:')
    plt.plot(his.history['val_acc'], 'g-')
    return fig

if __name__ == '__main__':
    with open('resource/train.json') as data:
        json_data = json.load(data)
        
    # read the json_file
        
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
                
    
    training_file = pd.DataFrame({'claim':claims,'evidences':evidences,'labels':label_v2})  
#    training_file.labels.value_counts()  
    
#    """
#    down sampling
#    """
    negative = training_file[training_file['labels'] == 1]
    positive = training_file[training_file['labels'] == 0]
    positive_sampled = positive.sample(n=negative.shape[0])
    sampled_file = pd.concat([positive_sampled,negative])
    sampled_file.claim
    sampled_file.to_csv('resource/sampled_file.csv')
#    training_file.to_csv('resource/training_file_v3.csv')  
    training_file = pd.read_csv('resource/training_file_v3.csv')
    training_file.labels.value_counts()
      

    #"""
    #prepare inputs, word_embedding
    #"""      
    tokenizer, embedding_matrix = word_embed_meta_data(sampled_file.claim+sampled_file.evidences,mode = "Glove")
    # return the embedding_matrix for sentences.
    sentences_pair = [(x1, x2) for x1, x2 in zip(sampled_file.claim, sampled_file.evidences)]
            
    left_sequence_length = 32   # max length of claims
    right_sequence_length = 64     # max length of evidences
    num_samples = len(claims)
    num_classes = 2 # supports, refutes
    embed_dimensions = 300
    epoch = 50
    # batch_size = 1024
    batch_size = 256
    
    sim = keras.utils.to_categorical(sampled_file.labels, num_classes)
            
    #"""
    #ESIM 
    #"""        
    model,his = buildESIM(tokenizer,sentences_pair,sim,embed_dimensions,embedding_matrix,\
                          left_sequence_length,right_sequence_length,num_classes,epoch,batch_size)
    
#    plot(model, to_file='esim1.png')
#    test_sentences_pair = [(x1, x2) for x1, x2 in zip(sampled_file.claim[:10], sampled_file.evidences[:10])]
#    test_claim,test_evidence = create_test_data(tokenizer, test_sentences_pair, \
#                                                left_sequence_length, right_sequence_length)
#    pred = model.predict([test_claim,test_evidence])
    
    #plot the damn stuff
    fig = plot_acc(his)
    fig.show()
    
    #"""
    #siamese LSTM
    #"""
    number_lstm_units = 50
    rate_drop_lstm = 0.17
    rate_drop_dense = 0.15
    number_dense_units = 100
    model,his = buildLSTM(tokenizer,sentences_pair,sim,embed_dimensions,embedding_matrix,\
                number_lstm_units,rate_drop_lstm, rate_drop_dense, number_dense_units,\
                left_sequence_length,right_sequence_length,num_classes,epoch,batch_size)
#    plot(model,'normalLSTM.png')
    
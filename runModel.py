#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:58:14 2019

@author: loretta


run model
"""
from NLI.esim import buildESIM
from NLI.normalLSTM import buildLSTM
from NLI.abcnn import word_embed_meta_data, create_test_data
from utils import load_pickle, save_pickle
from NLI.attention import DotProductAttention
#from sentence_selection.generateTrainingFile import getPage_index,readOneFile   
import json
import pandas as pd
import keras
from keras.models import load_model
import pickle
from tqdm import tqdm
import time
from sklearn.metrics import mean_squared_error,confusion_matrix,f1_score
import matplotlib.pyplot as plt
import numpy as np

#define folder directory
mask_dir = 'resource/training_data/'
claims_supports = mask_dir + 'train_claims_supports_downsampled.pkl'
claims_refutes = mask_dir + 'train_claims_refutes.pkl'
evidences_supports = mask_dir + 'train_evidences_supports_downsampled.pkl'
evidences_refutes = mask_dir + 'train_evidences_refutes.pkl'
labels_supports = mask_dir + 'train_labels_supports_downsampled.pkl'
labels_refutes = mask_dir + 'train_labels_refutes.pkl'
dev_claims = mask_dir + 'dev_claims.pkl'
dev_evidences = mask_dir + 'dev_evidences.pkl'
dev_labels = mask_dir + 'dev_labels.pkl'

def plot_acc(his,name,index):
    fig = plt.figure(index)
    plt.plot(his.history['acc'], 'r:')
    plt.plot(his.history['val_acc'], 'g-')
    plt.plot(his.history['loss'], 'r:')
    plt.plot(his.history['val_loss'],'g-')
    plt.savefig(name+".png")

if __name__ == '__main__':
#    with open('resource/train.json') as data:
#        json_data = json.load(data)
        
    # read the json_file
        
#    page_index, page_size = getPage_index()
#    

#    for j,item in enumerate(json_data):
#        if j < 100:
#        # only get first 100, [FOR NOW]
#            evds = []
#            if json_data[item]['evidence'] == []:
#                continue
#            for k,evd in enumerate(json_data[item]['evidence']):
#                if evd[0] in page_index:
#                    evds.append(readOneFile(page_index[evd[0]],evd[0],evd[1]))
#            claims.append(json_data[item]['claim'])
#            evidences.append(' '.join(evds))
#            labels.append(json_data[item]['label'])

    claims = []
    evidences = []
    labels = []

    for item in [claims_supports,claims_refutes]:
        claims += load_pickle(item) 
    for item in [evidences_supports,evidences_refutes]:
        evidences += load_pickle(item)
    for item in [labels_supports,labels_refutes]:
        labels += load_pickle(item)
    
    claims_dev = load_pickle(dev_claims)
    evidences_dev = load_pickle(dev_evidences)
    labels_dev = load_pickle(dev_labels)
    
#        label_transfer = {'SUPPORTS':0,'REFUTES':1}
#        label_v2 = []
#        for l in labels:
#            if l in label_transfer:
#                label_v2.append(label_transfer[l])
#                
    
#    training_file = pd.DataFrame({'claim':claims,'evidences':evidences,'labels':labels})  
#    training_file.labels.value_counts()  
    
#    """
#    down sampling
#    """
#    negative = training_file[training_file['labels'] == 1]
#    positive = training_file[training_file['labels'] == 0]
#    positive_sampled = positive.sample(n=negative.shape[0])
#    sampled_file = pd.concat([positive_sampled,negative])
#    sampled_file.claim
#    sampled_file.to_csv('resource/sampled_file.csv'
                        
#    training_file.to_csv('resource/training_file_v3.csv')  
#    training_file = pd.read_csv('resource/training_file_v3.csv')
#    training_file.labels.value_counts()
#      
#    new_claims = []
#    new_evidences = []
#    new_labels = []
#    evd = []
#    for c,e,l in zip(claims,evidences,labels):
#        if not c in new_claims and not evd == []:
#            new_claims.append(c)
#            new_labels.append(l)
#            new_evidences.append(' '.join(evd))
#            evd = []
#            evd.append(e)
#        else:
#            evd.append(e)
        
    
        
    

    #"""
    #prepare inputs, word_embedding
    #"""      
    tokenizer, embedding_matrix = word_embed_meta_data(claims+evidences,mode = "Glove")
    # return the embedding_matrix for sentences.
    sentences_pair_train = [(x1, x2) for x1, x2 in zip(claims, evidences)]
    
    left_sequence_length = 32   # max length of claims
    right_sequence_length = 64     # max length of evidences
    num_samples = len(claims)
    num_classes = 2 # supports, refutes
    embed_dimensions = 300
    epoch = 50
    # batch_size = 1024
    batch_size = 512
    
    sim_train = keras.utils.to_categorical(labels, num_classes)
            
    sentences_pair_dev = [(x1, x2) for x1, x2 in zip(claims_dev, evidences_dev)]
    sim_dev = keras.utils.to_categorical(labels_dev, num_classes)
#    test_claim,test_evidence = create_test_data(tokenizer, test_sentences_pair, \
#                                                left_sequence_length, right_sequence_length)
#    
    #"""
    #ESIM 
    #"""        
    model,his = buildESIM(tokenizer,sentences_pair_train,sim_train,sentences_pair_dev,\
                          sim_dev,embed_dimensions,embedding_matrix,left_sequence_length,\
                          right_sequence_length,num_classes,epoch,batch_size)
    
#    plot(model, to_file='esim1.png')
#    test_sentences_pair = [(x1, x2) for x1, x2 in zip(sampled_file.claim[:10], sampled_file.evidences[:10])]
#    test_claim,test_evidence = create_test_data(tokenizer, test_sentences_pair, \
#                                                left_sequence_length, right_sequence_length)
#    pred = model.predict([test_claim,test_evidence])
    
    name = 'ESIM_'+str(int(time.time())) + '.png'
    plot_acc(his,name,0)

    
#    model = load_model('/Users/loretta/watson-junior/trained_model/ESIM/1558325833.h5',\
#                       custom_objects={'DotProductAttention':DotProductAttention})

#    pred = model.predict([test_claim,test_evidence])
#    pred = np.argmax(pred,axis=1)
#    print('mean_squared_error: '+mean_squared_error(pred,labels_dev))
#    print('f1_score: '+f1_score(pred,labels_dev))
#    print("confusion_matrix: ")
#    print(confusion_matrix(pred,labels_dev))
#    print("=============================")

    
    #"""
    #siamese LSTM
    #"""
    number_lstm_units = 50
    rate_drop_lstm = 0.17
    rate_drop_dense = 0.15
    number_dense_units = 100
    model,his = buildLSTM(tokenizer,sentences_pair_train,sim_train,sentences_pair_dev,\
                          sim_dev,embed_dimensions,embedding_matrix,number_lstm_units,\
                          rate_drop_lstm, rate_drop_dense, number_dense_units,\
                          left_sequence_length,right_sequence_length,num_classes,epoch,batch_size)
#    plot(model,'normalLSTM.png')
    name = 'LSTM_'+str(int(time.time())) + '.png'
    plot_acc(his,name,1)
    
#    pred = model.predict([test_claim,test_evidence])
#    pred = np.argmax(pred,axis=1)
#    print('mean_squared_error: '+ str(mean_squared_error(pred,labels_dev)))
#    print('f1_score: '+str(f1_score(pred,labels_dev)))
#    print("confusion_matrix: ")
#    print(confusion_matrix(pred,labels_dev))

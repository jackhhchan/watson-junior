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
import os
from NLI.train import get_training_data

#define folder directory
# mask_dir = 'resource/training_data/'
MASK_DIR = 'training_data/'

#============STAGE 2================
#=====SENTENCE SELECTION============

SS_CLAIMS = MASK_DIR + 'sentence_selection_train_claims.pkl'
SS_EVIDENCES = MASK_DIR + 'sentence_selection_train_evidences.pkl'
SS_LABELS = MASK_DIR + 'sentence_selection_train_labels.pkl'

#============STAGE 3================
claims_supports = MASK_DIR + 'train_claims_supports_downsampled.pkl'
claims_refutes = MASK_DIR + 'train_claims_refutes.pkl'
evidences_supports = MASK_DIR + 'train_evidences_supports_downsampled.pkl'
evidences_refutes = MASK_DIR + 'train_evidences_refutes.pkl'
labels_supports = MASK_DIR + 'train_labels_supports_downsampled.pkl'
labels_refutes = MASK_DIR + 'train_labels_refutes.pkl'
dev_claims = MASK_DIR + 'dev_claims.pkl'
dev_evidences = MASK_DIR + 'dev_evidences.pkl'
dev_labels = MASK_DIR + 'dev_labels.pkl'



def plot_acc(his,fig_dir,index):
    """
    :param his: the output of a training model
    :param fig_dir: the directory to store the image
    :param index: current figure. Ascending. 
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig = plt.figure(index)
    name = fig_dir.split('/')[-1] + '_' + str(int(time.time()))
    #either be 'ESIM_{timestamp}' or 'LSTM_{timestamp}'
    fig.suptitle(name)
    plt.subplot(211)
    plt.plot(his.history['acc'], 'r:',label='acc')
    plt.plot(his.history['val_acc'], 'g-',label='val_acc')
    legend = plt.legend(loc='lower right')
    plt.subplot(212)
    plt.plot(his.history['loss'], 'r:',label = 'loss')
    plt.plot(his.history['val_loss'],'g-',label = 'val_loss')
    legend = plt.legend(loc='upper right')
    plt.savefig(fig_dir+'/'+name+".png")



if __name__ == '__main__':
#    """
#    prepare dataset
#    """
    claims, evidences, labels = get_training_data(claims_path=SS_CLAIMS,
                                                  evidences_path=SS_EVIDENCES,
                                                  labels_path=SS_LABELS)
        
    

    #"""
    #prepare inputs, word_embedding
    #"""      
    tokenizer, embedding_matrix = word_embed_meta_data(claims+evidences,mode = "Glove")
    # return the embedding_matrix for sentences.
    # only for the training set
    
    
#    """
#    model hyperparameters
#    """
    left_sequence_length = 32   # max length of claims
    right_sequence_length = 32     # max length of evidences
                                # if evidence is concatenate, the length will be 64
    num_samples = len(claims)
    num_classes = 2 # supports, refutes
    embed_dimensions = 300
    epoch = 50
    # batch_size = 1024
    batch_size = 512
    
#    ==================
    mode = 'regression' 
#    mode = 'classification'
#    regression: sentence selection
#    classification: Text Entailment Recognition 
    isDev = False
#    ==================
#    
#    """
#    prepare dataset
#    """
    sentences_pair_train = [(x1, x2) for x1, x2 in zip(claims, evidences)]
    sim_train = keras.utils.to_categorical(labels, num_classes) if not mode == 'regression' else labels

    if isDev:
        claims_dev, evidences_dev, labels_dev = get_training_data(claims_path=dev_claims,
                                                                  evidences_path=dev_evidences,
                                                                  labels_path=dev_labels)
        sentences_pair_dev = [(x1, x2) for x1, x2 in zip(claims_dev, evidences_dev)]
        sim_dev = keras.utils.to_categorical(labels_dev, num_classes) if not mode == 'regression' else labels_dev
    
    
#    test_claim,test_evidence = create_test_data(tokenizer, test_sentences_pair, \
#                                                left_sequence_length, right_sequence_length)
#    
    #"""
    #ESIM 
    #"""        
    model,his = buildESIM(tokenizer,sentences_pair_train,sim_train,sentences_pair_dev,\
                          sim_dev,embed_dimensions,embedding_matrix,left_sequence_length,\
                          right_sequence_length,num_classes,epoch,batch_size,mode)
    
#    test_sentences_pair = [(x1, x2) for x1, x2 in zip(sampled_file.claim[:10], sampled_file.evidences[:10])]
#    test_claim,test_evidence = create_test_data(tokenizer, test_sentences_pair, \
#                                                left_sequence_length, right_sequence_length)
#    pred = model.predict([test_claim,test_evidence])
    
    plot_acc(his,'figure/ESIM',0)

    
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
                          left_sequence_length,right_sequence_length,num_classes,\
                          epoch,batch_size,mode)
#    plot(model,'normalLSTM.png')
    plot_acc(his,'figure/LSTM',1)
    
#    pred = model.predict([test_claim,test_evidence])
#    pred = np.argmax(pred,axis=1)
#    print('mean_squared_error: '+ str(mean_squared_error(pred,labels_dev)))
#    print('f1_score: '+str(f1_score(pred,labels_dev)))
#    print("confusion_matrix: ")
#    print(confusion_matrix(pred,labels_dev))

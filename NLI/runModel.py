#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:58:14 2019

@author: loretta


run model
"""
import sys
sys.path.append(sys.path[0] + '/..')
from NLI.esim import buildESIM
from NLI.normalLSTM import buildLSTM
from NLI.abcnn import buildABCNN
from utils import load_pickle, save_pickle
from NLI.attention import DotProductAttention
from NLI.prepare_set import word_embed_meta_data, create_train_dev_set,create_test_data,\
    create_train_dev_from_files
from NLI.train import get_training_data
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
from utils import get_timestamp

#define folder directory
# mask_dir = 'resource/training_data/'
# MASK_DIR = 'resource/training_data/combined_fixed/'
MASK_DIR = 'resource/training_data/sentence_selection/'

#============STAGE 2================
#=====SENTENCE SELECTION============

SS_CLAIMS = MASK_DIR + 'sentence_selection_train_claims.pkl'
SS_EVIDENCES = MASK_DIR + 'sentence_selection_train_evidences.pkl'
SS_LABELS = MASK_DIR + 'sentence_selection_train_labels.pkl'

SS_CLAIMS_DEV = MASK_DIR + 'sentence_selection_devset_claims.pkl'
SS_EVIDENCES_DEV = MASK_DIR + 'sentence_selection_devset_evidences.pkl'
SS_LABELS_DEV = MASK_DIR + 'sentence_selection_devset_labels.pkl'

#============STAGE 3================
# claims_supports = MASK_DIR + 'separated/train_claims_supports_downsampled_concatenate_True.pkl'
# claims_refutes = MASK_DIR + 'separated/train_claims_refutes_concatenate_True.pkl'
# evidences_supports = MASK_DIR + 'separated/train_evidences_supports_downsampled_concatenate_True.pkl'
# evidences_refutes = MASK_DIR + 'separated/train_evidences_refutes_concatenate_True.pkl'
# labels_supports = MASK_DIR + 'separated/train_labels_supports_downsampled_concatenate_True.pkl'
# labels_refutes = MASK_DIR + 'separated/train_labels_refutes_concatenate_True.pkl'
dev_claims = MASK_DIR + 'dev_claims_concatenate_True.pkl'
dev_evidences = MASK_DIR + 'dev_evidences_concatenate_True.pkl'
dev_labels = MASK_DIR + 'dev_labels_concatenate_True.pkl'

train_claims = MASK_DIR + 'train_claims_all_concatenate_True.pkl'
train_evidences = MASK_DIR + 'train_evidences_all_concatenate_True.pkl'
train_labels = MASK_DIR + 'train_labels_all__concatenate_True.pkl'

def plot_acc(his,fig_dir,index,items):
    """
    :param his: the output of a training model
    :param fig_dir: the directory to store the image
    :param index: current figure. Ascending. 
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig = plt.figure(index)
    name = fig_dir.split('/')[-1] + '_' + get_timestamp()
    #either be 'ESIM_{timestamp}' or 'LSTM_{timestamp}'
    fig.suptitle(name)
    plt.subplot(211)
    plt.plot(his.history[items[0]], 'r:',label=items[0])
    plt.plot(his.history[items[1]], 'g-',label=items[1])
    legend = plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(his.history[items[2]], 'r:',label =items[2])
    plt.plot(his.history[items[3]],'g-',label =items[3])
    legend = plt.legend(loc='best')
    plt.savefig(fig_dir+'/'+name+".png")



if __name__ == '__main__':
#    """
#    prepare dataset
#    """
    claims, evidences, labels = get_training_data(claims_path=SS_CLAIMS,
                                                  evidences_path=SS_EVIDENCES,
                                                  labels_path=SS_LABELS)

    # claims = []
    # evidences = []
    # labels = []   
    # for item in [claims_supports,claims_refutes]:
    #     claims += load_pickle(item)
    # for item in [evidences_supports,evidences_refutes]:
    #     evidences += load_pickle(item)
    # for item in [labels_supports,labels_refutes]:
    #     labels += load_pickle(item)

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
    # mode = 'regression' 
    mode = 'classification'
#    regression: sentence selection
#    classification: Text Entailment Recognition 
    isDev = True
#    ==================
#    
#    """
#    prepare dataset
#    """
    sentences_pair_train = [(x1, x2) for x1, x2 in zip(claims, evidences)]
    sim_train = keras.utils.to_categorical(labels, num_classes) if mode == 'classification' else labels
    sentences_pair_dev = None
    sim_dev = None

    if isDev:
        print("[INFO]prepare dev_set...")
        claims_dev, evidences_dev, labels_dev = get_training_data(claims_path=SS_CLAIMS_DEV,
                                                                  evidences_path=SS_EVIDENCES_DEV,
                                                                  labels_path=SS_LABELS_DEV)
        sentences_pair_dev = [(x1, x2) for x1, x2 in zip(claims_dev, evidences_dev)]
        sim_dev = keras.utils.to_categorical(labels_dev, num_classes) if mode == 'classification' else labels_dev
    
    items = ['acc','val_acc','loss','val_loss'] if mode == 'classification' else \
            ['mean_absolute_error','val_mean_absolute_error','loss','val_loss']
    
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
    
    plot_acc(his,'figure/ESIM',0,items)

    
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
    # number_lstm_units = 50
    # rate_drop_lstm = 0.17
    # rate_drop_dense = 0.15
    # number_dense_units = 100
    # model,his = buildLSTM(tokenizer,sentences_pair_train,sim_train,sentences_pair_dev,\
    #                       sim_dev,embed_dimensions,embedding_matrix,number_lstm_units,\
    #                       rate_drop_lstm, rate_drop_dense, number_dense_units,\
    #                       left_sequence_length,right_sequence_length,num_classes,\
    #                       epoch,batch_size,mode)

    # plot_acc(his,'figure/LSTM',1,items)
    
#    pred = model.predict([test_claim,test_evidence])
#    pred = np.argmax(pred,axis=1)
#    print('mean_squared_error: '+ str(mean_squared_error(pred,labels_dev)))
#    print('f1_score: '+str(f1_score(pred,labels_dev)))
#    print("confusion_matrix: ")
#    print(confusion_matrix(pred,labels_dev))

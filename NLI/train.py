#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:58:14 2019

@author: loretta

run model(s)
"""
import sys
sys.path.append(sys.path[0]+"/..")
import os
import json
from enum import Enum

from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from sklearn.metrics import mean_squared_error, confusion_matrix, f1_score

from utils import load_pickle, save_pickle
from NLI.esim import buildESIM
from NLI.normalLSTM import buildLSTM
from NLI.abcnn import word_embed_meta_data, create_test_data
from NLI.attention import DotProductAttention

#### ! MANUAL CHANGES ! ####
concatenate = False                                         # CHANGE THIS TO CHANGE DATASETS

#### PATHs ####
MASK_DIR = 'resource/training_data/'
TRAIN_CLAIMS_PATH = MASK_DIR + "train_claims_all_concatenate_{}.pkl".format(concatenate)
TRAIN_EVIDENCES_PATH = MASK_DIR + "train_evidences_all_concatenate_{}.pkl".format(concatenate)
TRAIN_LABELS_PATH = MASK_DIR + "train_labels_all_concatenate_{}.pkl".format(concatenate)
DEV_CLAIMS_PATH = MASK_DIR + "dev_claims_concatenate_{}.pkl".format(concatenate)
DEV_EVIDENCES_PATH = MASK_DIR + "dev_evidences_concatenate_{}.pkl".format(concatenate)
DEV_LABELS_PATH = MASK_DIR + "dev_labels_concatenate_{}.pkl".format(concatenate)

class NeuralNetwork(Enum):
    """ Neural Networks used in the Fact Verification System """
    SENTENCE_SELECTION = "sentence"
    ENTAILMENT_RECOGNIZER = "entailment"

    @staticmethod
    def list(): return list(map(lambda case: case.value, NeuralNetwork))



def main():
    """Train models"""

    train_claims, train_evidences, train_labels = get_training_data(claims_path=TRAIN_CLAIMS_PATH,
                                                                    evidences_path=TRAIN_EVIDENCES_PATH,
                                                                    labels_path=TRAIN_LABELS_PATH)

    dev_claims, dev_evidences, dev_labels = get_training_data(claims_path=TRAIN_CLAIMS_PATH,
                                                              evidences_path=TRAIN_EVIDENCES_PATH,
                                                              labels_path=TRAIN_LABELS_PATH)



def train(network, esim=True, lstm=True):
    assert network in NeuralNetwork.list(), "Invalid network selected, choose from {}".format(NeuralNetwork.list())
    

    if esim:
        pass
    
    if lstm:
        pass



def get_training_data(claims_path, evidences_path, labels_path):
    claims = load_pickle(claims_path) 
    evidences = load_pickle(evidences_path)
    labels = load_pickle(labels_path)

    return claims, evidences, labels
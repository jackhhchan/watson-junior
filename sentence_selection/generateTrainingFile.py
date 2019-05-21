#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:55:05 2019

@author: loretta
"""
import os
import sys
import json
from tqdm import tqdm
# import pandas as pd
# import numpy as np

import mongodb_query    
#from watson_junior.utils.utils import load_file
#from watson_junior.utils.wiki_parser import parse_raw_line


#list(json_data.keys())[:5]

#folders_name = 'resource'
#folders = os.listdir(folders_name)
#num_folders = len(folders)
#for idx, wiki_folder in tqdm(enumerate(folders)):
#    folder_path = "{}/{}".format(folders_name, wiki_folder)
#    
#    print("[INFO] Parsing the wiki docs in {}...".format(folder_path))
#    try:
#        pages_collection = list(wiki_parser.parse_wiki_docs(folder_name=folder_path).values())
#    except:
#        pass
#
#pages_collection[0]

def parse_raw_line(raw_line):
    """ Extract page id, passage index and tokens from raw line to build Passage objects 
    
        Helper function
    """
    page_id = raw_line[0]
    passage_idx = raw_line[1]
    tokens = extract_tokens(raw_line[2:])

    return page_id, passage_idx, tokens

def extract_tokens(passage):
    """ Extract lower case tokens from the passage """

    return [token.lower() for token in passage]

from enum import Enum

class encoding(Enum):
    UTF8 = "UTF-8"


def load_file(f_path, encoding=encoding.UTF8.value):
    """ Load text file from path and return raw array of split strings"""

    raw_lines = []
    with open(f_path, 'r', encoding=encoding) as f_handle:
        for raw_line in f_handle:
            # split line into tokens
            raw_lines.append(raw_line.split())

    print("[INFO] Extracted {}, len(raw_lines) = {}".format(f_path, len(raw_lines)))
    return raw_lines

    
def getPages(folder_name = 'resource_model/wiki_files'):
    pages = {}      # {page_id, Page object}
    for file_name in os.listdir(folder_name):
        # load file
        path = "{}/{}".format(folder_name, file_name)
        raw_lines = load_file(path)
        # loop through raw lines in file
        for raw_line in raw_lines:
            page_id, passage_idx, tokens = parse_raw_line(raw_line)
            try:
                pages[(page_id,int(passage_idx))] = ' '.join(tokens)
            except:
                pass
    return pages
        

#list(page_index.keys())[0]
#list(page_index.values())[0]
#page_index['Alexander_McNair']

def getPage_index(folder_name = 'resource_model/wiki_files'):
    page_index = {}      # {page_id, path}
    page_size = {}
    for file_name in os.listdir(folder_name):
        # load file
        path = "{}/{}".format(folder_name, file_name)
        raw_lines = load_file(path)
        # loop through raw lines in file
        page_size[path] = len(raw_lines)
        for raw_line in raw_lines:
            page_id, _, _ = parse_raw_line(raw_line)
            try:
                if not page_id in page_index:
                    page_index[page_id] = path
            except:
                pass
        # break       ## add for debugging runModel.py script (runs entire script without loading all txt files)
    return page_index, page_size
 
def readOneFile(path,pg_id,index):
    with open(path, 'r', encoding=encoding.UTF8.value) as data:
        for line in data:
            content = line[:-1].split(' ')
            if content[0] == pg_id and content[1] == str(index):
                return ' '.join(content[2:])

#readOneFile(page_index['Alexander_McNair'],'Alexander_McNair',1)

def generateRandom(fileList,page_size):
    file = str(np.random.choice(fileList))
    randomK = int(np.random.choice(page_size[file]))
    with open(file) as data:
        for k,line in enumerate(data):
            if k == randomK:
                content = line[:-1].split(' ')
                return ' '.join(content[2:])

#generateRandom(fileList)


# if __name__ == '__main__':

#     with open('resource/train.json') as data:
#         json_data = json.load(data)
        
#     pages = getPages()
#     page_index, page_size = getPage_index()
#     fileList = list(set(list(page_index.values())))
    
    
#     training_corpus = pd.DataFrame()
#     claims = []
#     evidences = []
#     labels = []
#     for evd in json_data['75397']['evidence']:
#         if evd[0] in page_index:
#             claims.append(json_data['75397']['claim'])
#             evidences.append(readOneFile(page_index[evd[0]],evd[0],evd[1]))
#             labels.append(1)
#     while len(evidences) < 5:
#         claims.append(json_data['75397']['claim'])
#         evidences.append(generateRandom(fileList))
#         labels.append(0)
            
#     training_corpus = pd.DataFrame({'claims':claims,'evidences':evidences,'label':labels})
#     training_corpus.info()
#     training_corpus.to_csv('training_random.csv')
#     claims.pop(-1)
    
#     ('Fox_Broadcasting_Company', 0) in pages
    
#     claims = []
#     evidences = []
#     labels = []
#     for item in tqdm(json_data):
#         for k,evd in enumerate(json_data[item]['evidence']):
#             if evd[0] in page_index:
#                 claims.append(json_data[item]['claim'])
#                 evidences.append(readOneFile(page_index[evd[0]],evd[0],evd[1]))
#                 labels.append(1)
#         while k < 4:
#             claims.append(json_data[item]['claim'])
#             evidences.append(generateRandom(fileList))
#             labels.append(0)    
#             k += 1



#####################################################
######## Generate Training Data with MongoDB ########
#####################################################

class JSONField(Enum):
    # nested field identifiers
    claim = 'claim'
    evidence = 'evidence'
    label = 'label'

def generate_training_data_from_db(train_json, concatenate, threshold):
    """ Pull tokens from DB based on train.json script
    Arguments:
    ----------
    cocatenate (Bool)
        Takes a boolean, if true then evidences are cocatenated else other wise.
    Returns:
    ----------
    train_claims
        A list of claims
    """
    
    # connect to db -- host=192.168.1.10 for ubuntu, port=27017 is default
    mydb, mycol = mongodb_query._connected_db(host="localhost",port="27017")       # throws exception
    print("[INFO] collections in db: {}".format(mydb.list_collection_names()))

    train_claims = []
    train_evidences = []
    train_labels = []
    
    file_idx = 0
    print("[INFO] Extracting training data from db...")
    for idx, data in tqdm(enumerate(train_json)):
        label = data[JSONField.label.value]
        claim = data[JSONField.claim.value]
        evidences = data[JSONField.evidence.value]

        if label == 'NOT ENOUGH INFO': continue     # skip data points with no evidences

        label = encoded_label(label)                # encode label, SUPPORTS=0, REFUTES=1
        if concatenate:
            # evidences tokens are concatenated
            train_claims.append(claim)
            train_labels.append(label)
            concatenated_tokens = []
            [concatenated_tokens.extend(get_tokens(evidence, mycol)) for evidence in evidences]
            train_evidences.append(concatenated_tokens)        
        else:
            # evidences tokens are not concatenated
            for evidence in evidences:
                tokens = get_tokens(evidence, mycol)
                if tokens == None:
                    continue
                train_evidences.append(tokens)

                train_claims.append(claim)          # these two appends must be after token, for db check
                train_labels.append(label)
        
        if idx >= threshold:            # threshold for downsampling
            break    

    print("[INFO] complete.")
    return train_claims, train_evidences, train_labels

                    
        

def get_tokens(evidence, collection):
    page_id = evidence[0]
    passage_idx = evidence[1]
    doc = mongodb_query.query(collection=collection, page_id=page_id, passage_idx=passage_idx)
    
    if doc is None: 
        return None

    tokens = doc.get('tokens')
    assert tokens is not None

    tokens_string = ''
    for token in tokens:
        tokens_string = tokens_string + token + ' '
        
    return tokens_string

class Label(Enum):
    SUPPORTS = 'SUPPORTS'
    REFUTES = 'REFUTES'

labels = {'SUPPORTS':0, 'REFUTES':1}    
def encoded_label(label):
    return labels.get(label)

def load_json(json_path):
    # parse train.json file
    assert json_fname.endswith('.json')
    try:
        with open(json_path, 'r') as handle:
            json_data = json.load(handle)
    except:
        print("Unable to load train.json")

    return json_data

def parse_json(json_file, separate):

    if not separate:
        data_array = []
        for key in tqdm(json_file.keys()):
            label = json_file.get(key.get('label'))
            if label == Label.SUPPORTS.value or label == Label.REFUTES.value:
                data_array.append(json_file.get(key))
            else:
                continue
    else:
        supports_train_json = []
        refutes_train_json = []
        for key in tqdm(json_file.keys()):
            label = json_file.get(key).get('label')
            if label == Label.SUPPORTS.value:
                supports_train_json.append(json_file.get(key))
            elif label == Label.REFUTES.value:
                refutes_train_json.append(json_file.get(key))
            else:
                continue        # skip NOT ENOUGH INFO

    return supports_train_json, refutes_train_json


def save_pickle(obj, name):
    assert name.endswith('.pkl')
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle)


import pickle
if __name__ == '__main__' :
    # train_json = load_train_json()
    # supports_train_json, refutes_train_json = parse_train_json(train_json)

    # print("Number of SUPPORTS: {}".format(len(supports_train_json)))
    # print("Number of REFUTES: {}".format(len(refutes_train_json)))

    # train_claims, train_evidences, train_labels = generate_training_data_from_db(refutes_train_json, False)
    # save_pickle(train_claims, 'train_claims_refutes.pkl')
    # save_pickle(train_evidences, 'train_evidences_refutes.pkl')
    # save_pickle(train_labels, 'train_labels_refutes.pkl')

    # train_claims, train_evidences, train_labels = generate_training_data_from_db(supports_train_json, False, len(refutes_train_json))
    # save_pickle(train_claims, 'train_claims_supports_downsampled.pkl')
    # save_pickle(train_evidences, 'train_evidences_supports_downsampled.pkl')
    # save_pickle(train_labels, 'train_labels_supports_downsampled.pkl')


    ##### DEVELOPMENT SET #####
    dev_json = load_json('resource_train/devset.json')
    dev_array = parse_json(dev_json, False)
    dev_claims, dev_evidences, dev_labels = generate_training_data_from_db(dev_json, False, len(dev_json))
    save_pickle(dev_claims, 'dev_claims.pkl')
    save_pickle(dev_evidences, 'dev_evidences.pkl')
    save_pickle(dev_labels, 'dev_labels.pkl')






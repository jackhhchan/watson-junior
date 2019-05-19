#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:55:05 2019

@author: loretta
"""

import json
from tqdm import tqdm
#from watson_junior.utils.wiki_parser import parse_raw_line
import os
import pandas as pd
import numpy as np
#from watson_junior.utils.utils import load_file




    
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

if __name__ == '__main__':

    with open('resource/train.json') as data:
        json_data = json.load(data)
        
    pages = getPages()
    page_index, page_size = getPage_index()
    fileList = list(set(list(page_index.values())))
    
    
    training_corpus = pd.DataFrame()
    claims = []
    evidences = []
    labels = []
    for evd in json_data['75397']['evidence']:
        if evd[0] in page_index:
            claims.append(json_data['75397']['claim'])
            evidences.append(readOneFile(page_index[evd[0]],evd[0],evd[1]))
            labels.append(1)
    while len(evidences) < 5:
        claims.append(json_data['75397']['claim'])
        evidences.append(generateRandom(fileList))
        labels.append(0)
            
    training_corpus = pd.DataFrame({'claims':claims,'evidences':evidences,'label':labels})
    training_corpus.info()
    training_corpus.to_csv('training_random.csv')
    claims.pop(-1)
    
    ('Fox_Broadcasting_Company', 0) in pages
    
    claims = []
    evidences = []
    labels = []
    for item in tqdm(json_data):
        for k,evd in enumerate(json_data[item]['evidence']):
            if evd[0] in page_index:
                claims.append(json_data[item]['claim'])
                evidences.append(readOneFile(page_index[evd[0]],evd[0],evd[1]))
                labels.append(1)
        while k < 4:
            claims.append(json_data[item]['claim'])
            evidences.append(generateRandom(fileList))
            labels.append(0)    
            k += 1
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:19:18 2019

@author: loretta
"""
from enum import Enum

# from allennlp.predictors import Predictor

import utils



predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
#https://allennlp.org/models
#constituency parse

class Keys(Enum):
    KOI = 'hierplane_tree'      # key of interest
    nested_KOI = 'root'


def get_NER_tokens(string, stacked=False):
    """ Returns a list of Noun Phrase using AllenNLP Named Entity Recognizer"""

    NER_tokens = []
    predicted_dict = predictor.predict(string)
    NER_tokens.extend(get_NP(predicted_dict[Keys.KOI.value][Keys.nested_KOI.value], []))

    if stacked:
        NER_list = [NER_token for NER_token in NER_tokens] 
        return NER_list

    return NER_tokens




##### HELPER FUNCTIONS ######

def get_NP(tree, nps):
    """
    recursively read the tree
    """
    
    if isinstance(tree, dict):
        if "children" not in tree:
            if tree['nodeType'] == "NP":
                nps.append(tree['word'])
        elif "children" in tree:
            if tree['nodeType'] == "NP":
                nps.append(tree['word'])
                get_NP(tree['children'], nps)
            else:
                get_NP(tree['children'], nps)
    elif isinstance(tree, list):
        for sub_tree in tree:
            get_NP(sub_tree, nps)

    return nps

def get_subjects(tree):
    subject_words = []
    subjects = []
    for subtree in tree['children']:
        if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
            subjects.append(' '.join(subject_words))
            subject_words.append(subtree['word'])
        else:
            subject_words.append(subtree['word'])
    return subjects


def json_transfer(claim=None,evidences=None,label=None):
    return {
        'claim':claim,
        'evidences':evidences,
        'label':label
    }

# if __name__ == '__main__':  
#     claim = 'Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.'
    
    
#     with open('resource/test-unlabelled.json') as data:
#             json_data = json.load(data)
            
#     result_json = {}
    
    
#     for k,item in tqdm(enumerate(json_data)): 
# #        if k < 2:
#         if not item in result_json:
#             start_time = time.time()
#             tokens = predictor.predict(json_data[item]['claim'])
#             nps = []
#             noun_phrases = []
#             tree = tokens['hierplane_tree']['root']
#             noun_phrases = get_NP(tree, nps)
#             subjects = get_subjects(tree)
#             for subject in subjects:
#                 if len(subject) > 0:
#                     noun_phrases.append(subject)
#             NER_time = time.time()
#             print("NER part "+str(NER_time - start_time))
#             output = {}
#             for claim in noun_phrases:
#                 final_output = {}
#     #                print("claim: {}".format(claim))
#                 output[claim] = query(inverted_index, page_ids_idx_dict,claim, final_output)
#                 output[claim] = sorted_dict(output[claim])[:5]
            
#             evidences = []
#             evidences = list(set( g for cc in list(output.values()) for g in cc ))
#             evidences = [[evd[0],0] for evd in evidences]
#             print("inverted index part "+str(time.time() - NER_time))

#             result_json[item] = json_transfer(json_data[item]['claim'],evidences,'SUPPORTS')
        
#     json_data_dev['91198']    
#     result_json['110000']      

#     query(inverted_index, page_ids_idx_dict,'good', final_output)

#     #"""
#     #wikipedia API 
#     #"""
#     predicted_pages = []
#     for np in noun_phrases:
#         doc = wikipedia.search(np)
#         predicted_pages.extend(doc)
#     predicted_pages = list(set(predicted_pages))
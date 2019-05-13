#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:19:18 2019

@author: loretta
"""

from allennlp.predictors import Predictor
predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
#https://allennlp.org/models
#constituency parse

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

if __name__ == '__main__':  
    claim = 'Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.'
    tokens = predictor.predict(claim)
    nps = []
    noun_phrases = []
    tree = tokens['hierplane_tree']['root']
    noun_phrases = get_NP(tree, nps)
    subjects = get_subjects(tree)
    for subject in subjects:
        if len(subject) > 0:
            noun_phrases.append(subject)
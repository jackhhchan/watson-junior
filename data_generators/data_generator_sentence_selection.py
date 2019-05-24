import sys
sys.path.append(sys.path[0] + '/..')            # allow parent dir imports
import os
import random
from enum import Enum

import pickle
from tqdm import tqdm

import utils
from mongodb.mongodb_query import WikiQuery



class Label(Enum):
    RELEVANT = 'RELEVANT'
    IRRELEVANT = 'IRRELEVANT'

    @staticmethod
    def list(): return list(map(lambda case: case.value, Label))

    @staticmethod
    def encode(label):
        assert label in Label.list(), "Label must be {}".format([label for label in Label.list()])
        encoder_dict = {
            Label.RELEVANT.value : 0,
            Label.IRRELEVANT.value : 1
            }
        return encoder_dict.get(label)


class JSONField(Enum):
    # nested field identifiers
    claim = 'claim'
    evidence = 'evidence'
    label = 'label'


#############################################################
######## Generate Training Data pulled from MongoDB #########
#############################################################

def main():
    """" main method"""
    # load and parse json file
    train_json = utils.load_json('resource/train/train.json')
    train_array = parse_json(json_file=train_json)

    # connect to db and create query object
    wiki_query = WikiQuery()
    
    train_claims, train_evidences, train_labels = generate_data(train_array, wiki_query)

    utils.save_pickle(train_claims, 'sentence_selection_train_claims.pkl')
    utils.save_pickle(train_evidences, 'sentence_selection_train_evidences.pkl')
    utils.save_pickle(train_labels, 'sentence_selection_train_labels.pkl')


def generate_data(json_array, query_object):
    train_claims = []
    train_evidences = []
    train_labels = []

    page_ids_idx_dict = utils.load_pickle('page_ids_idx_dict_normalized_proper_fixed.pkl')
    for data in tqdm(json_array):
        claim = data.get(JSONField.claim.value)

        relevant_page_ids = []
        relevant_evidences = data.get(JSONField.evidence.value)
        ## append data for label: 'RELEVANT'
        for relevant_evidence in relevant_evidences:
            token_string = get_tokens_from_db(relevant_evidence, query_object)
            if token_string is None:                                # IMPORTANT: this handles query returning None
                continue
            train_evidences.append(token_string)
            
            train_claims.append(claim)
            train_labels.append(Label.encode(Label.RELEVANT.value))
            
            relevant_page_ids.append(relevant_evidence[0])       # keep page_ids in the evidences
    
        # get as many irrelevant evidences as there are relevant evidences for each claim
        irrelevant_page_ids = get_irrelevant_page_ids(relevant_page_ids, page_ids_idx_dict, len(relevant_page_ids))
        assert len(irrelevant_page_ids) == len(relevant_page_ids)
        for irrelevant_page_id in irrelevant_page_ids:
            irrelevant_passage_string = get_irrelevant_passage(irrelevant_page_id, query_object)
            if irrelevant_passage_string is None:                   # IMPORTANT: this handles query returning None
                continue
            train_evidences.append(irrelevant_passage_string)

            train_claims.append(claim)
            train_labels.append(Label.encode(Label.IRRELEVANT.value))

    return train_claims, train_evidences, train_labels
        


def get_tokens_from_db(evidence, query_object):
    page_id = evidence[0]
    passage_idx = evidence[1]
    doc = query_object.query(page_id=page_id, passage_idx=passage_idx)
    # returned doc logger
    if doc is None:
        message = "[DB] database returned None for page_id: {}, passage_idx: {}".format(page_id, passage_idx)
        utils.log(message)
        return None

    # returned tokens logger
    tokens = doc.get('tokens')
    if tokens is None:
        message = "[DB] tokens returned None for page_id: {}, passage_idx: {}".format(page_id, passage_idx)
        utils.log(message)

    # cocatenate the split tokens to a single string
    tokens_string = ''
    for token in tokens:
        tokens_string = tokens_string + token + ' '
        
    return tokens_string

def parse_json(json_file):
    """ Returns an array of train identifier dictionaries"""
    json_array = []
    for key in tqdm(json_file.keys()):
        label = json_file.get(key).get('label')
        if label == "SUPPORTS" or label == "REFUTES":
            json_array.append(json_file.get(key))
        else:
            continue
    return json_array

    

####################################################################
###################### GETTING IRRELEVANT DATA #####################
####################################################################

def get_irrelevant_page_ids(relevant_page_ids, page_ids_idx_dict, number_of_irrelevant):
    """ Get irrelevant evidences for training data """

    irrelevant_page_ids = []
    if number_of_irrelevant == 0:
        return irrelevant_page_ids
    # generate random number
    max_page_ids_idx = len(page_ids_idx_dict) - 1
    idx = 0
    while idx < max_page_ids_idx:
        random_num = random.randint(0, max_page_ids_idx)
        page_id = page_ids_idx_dict.get(random_num)

        # check if random page id is one of the relevant evidences' page ids
        if page_id not in relevant_page_ids:
            irrelevant_page_ids.append(page_id)
            if len(irrelevant_page_ids) >= number_of_irrelevant: 
                return irrelevant_page_ids              # return irrelevant page ids
        idx += 1
    
    return None

def get_irrelevant_passage(relevant_page_id, query_object):
    """ Pulls a passage from the database for a single page id"""
    # pull from database a passage from each relevant page id
    doc = query_object.query_page_id_only(page_id=relevant_page_id)

    # returned doc logger
    if doc is None:
        message = "[DB] database returned None for page_id: {}".format(relevant_page_id)
        utils.log(message)
        return None

    # returned tokens logger
    tokens = doc.get('tokens')
    if tokens is None:
        message = "[DB] tokens returned None for page_id: {}".format(relevant_page_id)
        utils.log(message)

    # cocatenate the split tokens to a single string
    tokens_string = ''
    for token in tokens:
        tokens_string = tokens_string + token + ' '
        
    return tokens_string



if __name__ == '__main__' :
    main()
import sys
sys.path.append(sys.path[0] + '/..')            # allow parent dir imports
import os
import random
from enum import Enum
from collections import defaultdict
import math

import pickle
from tqdm import tqdm

import utils
from mongodb.mongodb_query import WikiQuery, WikiIdxQuery

# PATHS #
data_name = 'devset'
data_json_path = 'resource/train/{}.json'.format(data_name)           # NOTE: THIS IS THE ONLY THING THAT NEEDS TO CHANGE
page_ids_idx_dict_path = 'page_ids_idx_dict_normalized_proper_fixed.pkl'        # This is REQUIRED to convert page idx to page id

class Label(Enum):
    RELEVANT = 'RELEVANT'
    IRRELEVANT = 'IRRELEVANT'

    @staticmethod
    def list(): return list(map(lambda case: case.value, Label))

    @staticmethod
    def encode(label):
        assert label in Label.list(), "Label must be {}".format([label for label in Label.list()])
        encoder_dict = {
            Label.RELEVANT.value : 1,
            Label.IRRELEVANT.value : 0
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
    train_json = utils.load_json(data_json_path)
    train_array = parse_json(json_file=train_json)

    # connect to db and create query object
    wiki_query = WikiIdxQuery()
    
    # train_claims, train_evidences, train_labels = generate_data(train_array, wiki_query)
    train_claims, train_evidences, train_labels = generate_data_from_same_page_ids(train_array, wiki_query)

    utils.save_pickle(train_claims, 'sentence_selection_{}_claims.pkl'.format(data_name))
    utils.save_pickle(train_evidences, 'sentence_selection_{}_evidences.pkl'.format(data_name))
    utils.save_pickle(train_labels, 'sentence_selection_{}_labels.pkl'.format(data_name))


def generate_data(json_array, query_object):
    train_claims = []
    train_evidences = []
    train_labels = []

    page_ids_idx_dict = utils.load_pickle(page_ids_idx_dict_path)
    for data in tqdm(json_array):
        claim = data.get(JSONField.claim.value)

        relevant_page_ids = []
        relevant_evidences = data.get(JSONField.evidence.value)
        ## append data for label: 'RELEVANT'
        for relevant_evidence in relevant_evidences:
            token_string = get_tokens_string_from_db(relevant_evidence, query_object)
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
        

def generate_data_from_same_page_ids(json_array, query_object):
    train_claims = []
    train_evidences = []
    train_labels = []

    num_relevant = 0
    num_irrelevant = 0

    for data in tqdm(json_array):
        claim = data.get(JSONField.claim.value)

        relevant_dict = defaultdict(list)
        relevant_evidences = data.get(JSONField.evidence.value)
        ## append data for label: 'RELEVANT'
        for relevant_evidence in relevant_evidences:
            token_string = get_tokens_string_from_db(relevant_evidence, query_object)
            if token_string is None:                                # IMPORTANT: this handles query returning None
                continue
            train_evidences.append(token_string)
            
            train_claims.append(claim)
            train_labels.append(Label.encode(Label.RELEVANT.value))
            num_relevant += 1

            relevant_dict[relevant_evidence[0]].append(str(relevant_evidence[1]))
        
        for page_id, passage_indices in relevant_dict.items():
            appended = passage_indices.copy()
            for _ in range(math.ceil(len(appended)*5)):
                irrelevant_passage_string, appended = get_irrelevant_passage_same_page_id(page_id, 
                                                                                          appended, 
                                                                                          query_object)
                if irrelevant_passage_string is None:
                    continue
                train_evidences.append(irrelevant_passage_string)
                
                train_claims.append(claim)
                train_labels.append(Label.encode(Label.IRRELEVANT.value))
                num_irrelevant += 1


    print("Number of relevant: {}".format(num_relevant))
    print("Number of irrelevant: {}".format(num_irrelevant))
    return train_claims, train_evidences, train_labels



def get_tokens_string_from_db(evidence, query_object):
    page_id = evidence[0]
    passage_idx = evidence[1]
    doc = query_object.query(page_id=page_id, passage_idx=passage_idx)
    # returned doc logger
    if doc is None:
        message = "[DB] database returned None for page_id: {}, passage_idx: {}".format(page_id, passage_idx)
        utils.log(message)
        return None

    tokens = get_tokens(doc)
    # cocatenate the split tokens to a single string
    tokens_string = cocatenate(tokens)

    return tokens_string

def get_tokens(doc):
    # returned tokens logger
    tokens = doc.get('tokens')
    return tokens

def cocatenate(tokens):
    tokens_string = ''
    for token in tokens:
        tokens_string = tokens_string + token + ' '
        
    return tokens_string



##### JSON #####
def parse_json(json_file):
    """ Returns an array of train identifier dictionaries"""
    print("[INFO - DataGen] Parsing json file...")
    json_array = []
    for key in tqdm(json_file.keys()):
        label = json_file.get(key).get('label')
        if label == "SUPPORTS" or label == "REFUTES":
            json_array.append(json_file.get(key))
        else:
            continue
    print("[INFO] Length of json array: {}".format(len(json_array)))
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

def get_irrelevant_passage_same_page_id(page_id, appended, query_object):
    # return all passage indices
    # return a random passage index not in appended.
    cursor = query_object.query_page_id_only(page_id, single=False)
    if cursor is None:
        return None, appended
    for c in cursor:
        passage_idx = c.get(query_object.WikiField.passage_idx.value)
        if passage_idx not in appended:
            irrelevant_passage_tokens = c.get(query_object.WikiField.tokens.value)
            if irrelevant_passage_tokens is None:
                return None
            irrelevant_passage_string = cocatenate(irrelevant_passage_tokens)
            appended.append(passage_idx)
            return irrelevant_passage_string, appended
    return None, appended



def get_irrelevant_passage(relevant_page_id, query_object):
    """ Pulls a passage from the database for a single page id"""
    # pull from database a passage from each relevant page id
    doc = query_object.query_page_id_only(page_id=relevant_page_id, single=True)

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


########################################
########### Watson-Junior ##############
############## main.py #################
########################################

def get_passages_from_db(page_id, query_object, output_string=True):
    """ Returns passage_indices, tokens_string_list from db"""
    passage_indices = []
    tokens_string_list = []

    passages_dict = query_object.query_page_id_only(page_id=page_id, single=False)
    if passages_dict is None:
        message = "Page id: {} returned None".format(page_id)
        utils.log(message)
        return None, None
    if passages_dict.count() > 10:
        print("page id with > passages: {}, number of passages: {}".format(page_id, passages_dict.count()))
    for passage in passages_dict:
        tokens = passage.get(WikiQuery.WikiField.tokens.value)
        passage_idx = passage.get(WikiQuery.WikiField.passage_idx.value)

        try:
            if int(passage_idx) > 50:
                continue
        except:
            continue

        if output_string:
            tokens = cocatenate(tokens)     # technically tokens_string
        
        passage_indices.append(passage_idx)
        tokens_string_list.append(tokens)

    return passage_indices, tokens_string_list
from enum import Enum

import os
import pymongo
from tqdm import tqdm
import pickle
import json

import utils
import wiki_parser

# run server --
# mongod --dbpath /Users/jackchan/downloads/mongodb/data/db

class WikiField(Enum):
    """ Field types used in the wiki collection """
    page_id = "page_id"
    passage_idx = "passage_idx"
    tokens = "tokens"

class InvertedIndexField(Enum):
    """ Field types used in the InvertedIndex collection """
    term = "term"
    page_id = "page_id"
    tfidf = "tfidf"

class Host(Enum):
    ubuntu = "192.168.1.10"
    localhost = "127.0.0.1"

def get_ip_address(host):
    return {
        Host.ubuntu.name : Host.ubuntu.value,
        Host.localhost.name : Host.localhost.value
    }.get(host)


###################################
######## DATABASE SET UP ##########
###################################

def _connected_db(host, port, col_name):
    """
    Arguments:
    ---------
    host
        host name
    port
        port number
    col_name
        collection name in the database
    """
    port = str(port)
    DATABASE_NAME = "wikiDatabase"
    database_connect_format = "mongodb://{}:{}/".format(host, port)
    try:
        print("[INFO] Connecting to {} at port {}...".format(host, port))
        myclient = pymongo.MongoClient(database_connect_format)
    except:
        print("Unable to connect to host: {} at port: {}".format(host, port))
        print("Make sure bind_ip option includes the ip address of this machine.")
    
    print("[INFO] Connected.")
    mydb = myclient[DATABASE_NAME]            # create database
    mycol = mydb[col_name]                    # create collection

    return mydb, mycol


def populate_wiki_db(collection, folder_name='resource'):
    """ Populates the database with passages loaded from wiki txt files """
    # loop through files insert each passage
    # with
    wiki_files = os.listdir(folder_name)
    num_files = len(wiki_files)
    for idx, wiki_file in enumerate(wiki_files):
        path = "{}/{}".format(folder_name, wiki_file)

        raw_lines = utils.load_file(path)
        for raw_line in tqdm(raw_lines):
            page_id, passage_idx, tokens = wiki_parser.parse_raw_line(raw_line)

            doc_json = wiki_doc_formatted(page_id, passage_idx, tokens)
            mycol.insert_one(doc_json)
        
        print("[INFO] {}/{} complete".format(idx+1, num_files))


def populate_inverted_index_db(collection, inverted_index, page_ids_idx_dict):
    assert collection.name == "InvertedIndex"

    length = len(inverted_index)
    for idx, (term, postings) in tqdm(enumerate(inverted_index.items())):
        if not term.isalpha(): continue

        for (page_idx, tfidf)  in postings.items():
            page_id = page_ids_idx_dict.get(page_idx)
            doc_json = inverted_index_doc_formatted(term, page_id, tfidf)
            mycol.insert_one(doc_json)

        if idx%100 == 0:
            print("{}/{} complete transfer to db".format(idx, length))



def wiki_doc_formatted(page_id, passage_idx, tokens):
    """ Returns the formatted dictionary to store each passage """
    return {
        WikiField.page_id.value : page_id, 
        WikiField.passage_idx.value : passage_idx, 
        WikiField.tokens.value : tokens
    }


def inverted_index_doc_formatted(term, page_id, tfidf):
    """ Returns the formatted dictionary to store each term in the inverted index"""
    return {
        InvertedIndexField.term.value : term,
        InvertedIndexField.page_id.value : page_id,
        InvertedIndexField.tfidf.value: tfidf
    }

###################################
############ QUERY ################
###################################

def query(collection, page_id, passage_idx):
    """ Returns the query cursor for the query matching the page_id and passage_idx """
    return collection.find_one({WikiField.page_id.value : str(page_id), 
                                WikiField.passage_idx.value : str(passage_idx)})

####################################

def save_pickle(obj, name):
    assert name.endswith('.pkl')
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(name):
    assert name.endswith('.pkl')
    with open(name, 'rb') as handle:
        data = pickle.load(handle)

    return data

def save_json(obj, name):
    assert name.endswith('.json')
    with open(name, 'w') as handle:
        json.dump(obj, name)


###########################################
def json_dump_inverted_index(inverted_index):
    # split the inverted index and save it into 100 parts
    file_idx = 0
    json_file = {}
    f_name = "inverted_index_json_{}.json".format(file_idx)

    for idx, (term, postings) in tqdm(enumerate(inverted_index.items())):
        if not term.isalpha(): continue     # skip if not alpha

        for (page_idx, tfidf)  in postings.items():
            page_id = page_ids_idx_dict.get(page_idx)
            doc_json = inverted_index_doc_formatted(term, page_id, tfidf)
            json_file.update(doc_json)


        if idx+1%1000:
            save_json(json_file, f_name)
            file_idx += 1
            json_file = {}
    
    save_json(json_file, f_name)



if __name__ == "__main__":

    # SPLIT DUMP INVERTED INDEX TO MULTIPLE JSON FILES TO BE IMPORTED TO MONGODB
    INVERTED_INDEX_FNAME = "inverted_index_tf_normalized_proper_fixed.pkl"
    inverted_index = load_pickle(INVERTED_INDEX_FNAME)

    json_dump_inverted_index(inverted_index)
    return

    #### IGNORE THE REST OF THE SCRIPT ####
    #### this is for populating the db one by one. (slow) ####

    collection_names ["wiki", "InvertedIndex"]
    collection_name = None                                   #!!!!!! CHANGE THIS TO CONNECT TO DIFF COLLECTION
    
    # connect to db, return db and the collection
    hosts = ['ubuntu', 'localhost']                                     # available hosts
    host = get_ip_address(host=hosts[1])                                #!!!!!! CHANGE THIS FOR DIFF HOST
    mydb, mycol = _connected_db(host, 27017, collection_name)
    print("Current collections in the database: {}".format(mydb.list_collection_names()))

    if collection_name == "wiki":
        # populate the database with wiki txt file passages
        # populate_db(collection=mycol)                                 # COMMENT THIS TO NOT POPULATE DATABASE AGAIN.

        # test query
        query_cursor = query(collection=mycol, page_id="Alexander_McNair", passage_idx="0")
        
        for data in query_cursor:
            print(data.get(WikiField.tokens.value))

    elif collection_name == "InvertedIndex":

        print("[INFO] Loading inverted index from pickle...")
        INVERTED_INDEX_FNAME = "inverted_index_tf_normalized_proper_fixed.pkl"
        PAGE_IDS_IDX_DICT_FNAME = "page_ids_idx_dict_normalized_proper_fixed.pkl"
        inverted_index = load_pickle(INVERTED_INDEX_FNAME)
        page_ids_idx_dict = load_pickle(PAGE_IDS_IDX_DICT_FNAME)

        
        print("[INFO] Populating the {} collection in the wikiDatabase...".format(collection_name))
        populate_inverted_index_db(mycol, inverted_index, page_ids_idx_dict)

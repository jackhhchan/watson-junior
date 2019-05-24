""" IGNORE THIS MODULE """
import sys
sys.path.append(sys.path[0] + '/..')
from enum import Enum
import os

import pymongo
import pickle
import json
from tqdm import tqdm


import utils
from IR import wiki_parser

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


def populate_wiki_db(collection, folder_name='resource/wiki-txt-files'):
    """ Populates the database with passages loaded from wiki txt files """
    # loop through files insert each passage
    # with
    wiki_files = os.listdir(folder_name)
    num_files = len(wiki_files)
    db_page_ids_idx_dict = {}
    page_idx = 0
    for idx, wiki_file in enumerate(wiki_files):
        path = "{}/{}".format(folder_name, wiki_file)

        raw_lines = utils.load_file(path)
        for raw_line in tqdm(raw_lines):

            page_id, passage_idx, tokens = wiki_parser.parse_raw_line(raw_line)

            if db_page_ids_idx_dict.get(page_id) is None:
                db_page_ids_idx_dict[page_id] = page_idx
                page_idx += 1
            
            db_page_idx = db_page_ids_idx_dict.get(page_id)

            doc_json = wiki_doc_formatted(db_page_idx, passage_idx, tokens)
            mycol.insert_one(doc_json)
        
        print("[INFO] {}/{} complete".format(idx+1, num_files))
    
    db_page_ids_idx_dict_reversed = dict((v, k) for k, v in db_page_ids_idx_dict.items())
    utils.save_pickle(db_page_ids_idx_dict_reversed, 'db_page_ids_idx_dict.pkl')



def wiki_doc_formatted(page_id, passage_idx, tokens):
    """ Returns the formatted dictionary to store each passage """
    return {
        WikiField.page_id.value : page_id, 
        WikiField.passage_idx.value : passage_idx, 
        WikiField.tokens.value : tokens
    }

def save_pickle(obj, name):
    assert name.endswith('.pkl')
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(name):
    assert name.endswith('.pkl')
    with open(name, 'rb') as handle:
        data = pickle.load(handle)

    return data

def save_json(obj, name, mode='w'):
    assert name.endswith('.json')
    with open(name, mode) as handle:
        json.dump(obj, handle)



###########################################



if __name__ == "__main__":
    #### IGNORE THE REST OF THE SCRIPT ####
    #### this is for populating the db one by one. (slow) ####

    collection_names = ["wiki", "wiki_idx"]
    collection_name = "wiki_idx"                                   #!!!!!! CHANGE THIS TO CONNECT TO DIFF COLLECTION
    
    # connect to db, return db and the collection
    hosts = ['ubuntu', 'localhost']                                     # available hosts
    host = get_ip_address(host=hosts[1])                                #!!!!!! CHANGE THIS FOR DIFF HOST
    mydb, mycol = _connected_db(host, 27017, collection_name)
    print("Current collections in the database: {}".format(mydb.list_collection_names()))

    if collection_name == "wiki_idx":
        # populate the database with wiki txt file passages
        populate_wiki_db(collection=mycol)                                 # COMMENT THIS TO NOT POPULATE DATABASE AGAIN.

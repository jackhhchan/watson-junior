from enum import Enum

import os
import pymongo
from tqdm import tqdm
import pickle

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
    postings = "postings"


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
        myclient = pymongo.MongoClient(database_connect_format)
    except:
        print("Unable to connect to host: {} at port: {}".format(host, port))
        print("Make sure bind_ip option includes the ip address of this machine.")
    
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

def populate_inverted_index_db(index_pkl, page_id_dict_pkl, collection):
    assert collection.name == "InvertedIndex"

    with open(index_pkl, 'rb') as handle:
        inverted_index = pickle.load(handle)
    with open(page_id_dict_pkl, 'rb') as handle:
        page_id_idx_dict = pickle.load(handle)

    for term, postings in tqdm(inverted_index.items()):
        # convert back from page_idx to page_id (i.e. strings)
        db_postings = {}
        for page_idx, tfidf in postings.items():
            db_postings.update({page_id_idx_dict.get(page_idx): tfidf})
        
        doc_json = inverted_index_doc_formatted(term, db_postings)
        mycol.insert_one(doc_json)



def wiki_doc_formatted(page_id, passage_idx, tokens):
    """ Returns the formatted dictionary to store each passage """
    return {WikiField.page_id.value : page_id, 
            WikiField.passage_idx.value : passage_idx, 
            WikiField.tokens.value : tokens}


def inverted_index_doc_formatted(term, postings):
    """ Returns the formatted dictionary to store each term in the inverted index"""
    return {InvertedIndexField.term.value : term,
            InvertedIndexField.postings.value : postings}

###################################
############ QUERY ################
###################################

def query(collection, page_id, passage_idx):
    """ Returns the query cursor for the query matching the page_id and passage_idx """
    return collection.find_one({WikiField.page_id.value : str(page_id), 
                                WikiField.passage_idx.value : str(passage_idx)})

####################################

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ho', '--host', help="host name of mongodb server")
    parser.add_argument('-p', '--port', help="port number of mongodb server", default=27017)
    args = parser.parse_args()
    
    collection_name = "InvertedIndex"                                   # CHANGE THIS TO CONNECT TO DIFF COLLECTION
    # connect to db, return db and the 'wiki' collection
    mydb, mycol = _connected_db(args.host, args.port, collection_name)
    print(mydb.list_collection_names())

    if collection_name == "wiki":
        # populate the database with wiki txt file passages
        # populate_db(collection=mycol)                           # COMMENT THIS TO NOT POPULATE DATABASE AGAIN.

        # test query
        query_cursor = query(collection=mycol, page_id="Alexander_McNair", passage_idx="0")
        
        for data in query_cursor:
            print(data.get(WikiField.tokens.value))

    elif collection_name == "InvertedIndex":
        INVERTED_INDEX_PKL = "inverted_index_tf_normalized_proper_fixed.pkl"
        PAGE_IDS_IDX_DICT_PKL = "page_ids_idx_dict_normalized_proper_fixed.pkl"
        populate_inverted_index_db(INVERTED_INDEX_PKL, PAGE_IDS_IDX_DICT_PKL, mycol)


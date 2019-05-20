from enum import Enum

import os
import pymongo
from tqdm import tqdm

import utils
import wiki_parser

# run server --
# mongod --dbpath /Users/jackchan/downloads/mongodb/data/db

class Field(Enum):
    """ Field types used in the collection """
    def __str__(self):
        return str(self.value)      # string conversion

    page_id = "page_id"
    passage_idx = "passage_idx"
    tokens = "tokens"

###################################
######## DATABASE SET UP ##########
###################################

def _connected_db(host, port):
    database_connect_format = "mongodb://{}:{}/".format(host, port)
    try:
        myclient = pymongo.MongoClient(database_connect_format)
    except:
        print("Unable to connect to host: {} at port: {}".format(host, port))
        print("Make sure bind_ip option includes the ip address of this machine.")
    
    mydb = myclient["wikiDatabase"]         # create database
    mycol = mydb["wiki"]                    # create collection

    return mydb, mycol


def populate_db(collection, folder_name='resource'):
    """ Populates the database with passages loaded from wiki txt files """
    # loop through files insert each passage
    # with
    wiki_files = os.listdir(folder_name)
    for wiki_file in wiki_files:
        path = "{}/{}".format(folder_name, wiki_file)

        raw_lines = utils.load_file(path)
        for raw_line in tqdm(raw_lines):
            page_id, passage_idx, tokens = wiki_parser.parse_raw_line(raw_line)

            json = db_formatted(page_id, passage_idx, tokens)
            mycol.insert_one(json)
        break


def db_formatted(page_id, passage_idx, tokens):
    """ Returns the formatted dictionary format to store each passage """
    return {Field.page_id.value : page_id, 
            Field.passage_idx.value : passage_idx, 
            Field.tokens.value : tokens}


###################################
############ QUERY ################
###################################

def query(collection, page_id, passage_idx):
    """ Returns the query cursor for the query matching the page_id and passage_idx """
    return collection.find_one({Field.page_id.value : str(page_id), 
                            Field.passage_idx.value : str(passage_idx)})

####################################

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ho', '--host', help="host name of mongodb server")
    parser.add_argument('-p', '--port', help="port number of mongodb server", default=27017)
    args = parser.parse_args()
    
    # connect to db, return db and the 'wiki' collection
    mydb, mycol = _connected_db(args.host, args.port)
    print(mydb.list_collection_names())

    # populate the database with wiki txt file passages
    # populate_db(collection=mycol)                           # COMMENT THIS TO NOT POPULATE DATABASE AGAIN.

    # test query
    query_cursor = query(collection=mycol, page_id="Alexander_McNair", passage_idx="0")
    
    for data in query_cursor:
        print(data.get(Field.tokens.value))
    
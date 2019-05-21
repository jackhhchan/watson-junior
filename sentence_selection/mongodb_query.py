from enum import Enum

import os
import pymongo
from tqdm import tqdm


# run server --
# mongod --dbpath /Users/jackchan/downloads/mongodb/data/db

class WikiField(Enum):
    """ Field types used in the collection """
    def __str__(self):
        return str(self.value)      # string conversion

    page_id = "page_id"
    passage_idx = "passage_idx"
    tokens = "tokens"

###################################
######## DATABASE SET UP ##########
###################################

class Host(Enum):
    ubuntu = "192.168.1.10"
    localhost = "127.0.0.1"

def get_ip_address(host):
    return {
        Host.ubuntu.name : Host.ubuntu.value,
        Host.localhost.name : Host.localhost.value
    }.get(host)


def _connected_db(host, port=27017):
    host = get_ip_address(host)
    port = str(port)
    connected_data_format = "mongodb://{}:{}/".format(host, port)
    myclient = pymongo.MongoClient(connected_data_format)
    mydb = myclient["wikiDatabase"]         # create database
    mycol = mydb["wiki"]                    # create collection

    return mydb, mycol



###################################
############ QUERY ################
###################################

def query(collection, page_id, passage_idx):
    """ Returns the query cursor for the query matching the page_id and passage_idx """
    return collection.find_one({WikiField.page_id.value : str(page_id), 
                                WikiField.passage_idx.value : str(passage_idx)})

def query_page_id_only(collection, page_id):
    return collection.find_one({WikiField.page_id.value: page_id})

####################################


if __name__ == "__main__":
    # connect to db, return db and the 'wiki' collection
    available_hosts = ['ubuntu', 'localhost']
    mydb, mycol = _connected_db(host=available_hosts[1])
    print(mydb.list_collection_names())

    # populate the database with wiki txt file passages
    # populate_db(collection=mycol)                           # COMMENT THIS TO NOT POPULATE DATABASE AGAIN.

    # test query
    query_cursor = query(collection=mycol, page_id="Alexander_McNair", passage_idx="0")
    
    for data in query_cursor:
        print(data.get(WikiField.tokens.value))
    
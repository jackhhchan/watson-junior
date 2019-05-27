import sys
sys.path.append(sys.path[0] + "/..")
import os
from enum import Enum

import pymongo
from pymongo.errors import ConnectionFailure

import utils


""" 
    DATABASE INFO:
    Database: 'wikiDatabase'
    Collections: 
        'wiki'          -- contains page_id (index), passage_idx, tokens
        'InvertedIndex' -- contains term (index), page_id, tfidf


    USAGE: QUERY
    contain objects for each collection, instantiate to connect to the database and use its methods for queries.

    --------------------------------------
    HELPER FUNCTIONS: CONNECT DATABASE
    contain methods to connect to the database and its containing collections
"""

###################################
############ QUERY ################
###################################

class WikiQuery(object):
    """ Query object for the collection 'wiki' """
    class WikiField(Enum):
        """ Field types used in the collection """
        page_id = "page_id"             # indexed
        passage_idx = "passage_idx"
        tokens = "tokens"


    def __init__(self):
        self.db_name = 'wikiDatabase'
        self.col_name = 'wiki'

        client = connected(host='localhost')
        db = get_database(client, self.db_name)
        self.col = get_collection(db, self.col_name)

    # used for entailment_recognition
    def query(self, page_id, passage_idx):
        """ Returns the query cursor for the query matching the page_id and passage_idx """
        return self.col.find_one({
            self.WikiField.page_id.value : str(page_id), 
            self.WikiField.passage_idx.value : str(passage_idx)
                })

    # used for sentence_selection
    def query_page_id_only(self, page_id, single):
        """ Returns the a json with the matching page_id picked from the first doc
        Args:
        single -- if True, returns only 1 passage.
        """
        if single:
            return self.col.find_one({
                self.WikiField.page_id.value: page_id
                })
        else:
            return self.col.find({
                self.WikiField.page_id.value: page_id
            })

class WikiIdxQuery(object):
    class WikiField(Enum):
        page_id = "page_idx"             # indexed
        passage_idx = "passage_idx"
        tokens = "tokens"


    def __init__(self):
        self.db_name = 'wikiDatabase'
        self.col_name = 'wiki_idx'

        self.idx_id_mapper_path = "resource/wiki_idx_mapper/wiki_idx_page_idx_id_mapper.pkl"
        self.id_idx_mapper_path = "resource/wiki_idx_mapper/wiki_idx_page_id_idx_mapper.pkl"
        assert os.path.exists(self.idx_id_mapper_path) and os.path.exists(self.id_idx_mapper_path), \
        "[WikiIndexQuery] -- paths must exist for mapper required for 'wiki_idx'"
        # self.idx_id_mapper = utils.load_pickle(self.idx_id_mapper_path)           # {page_idx : page_id} mapper
        self.id_idx_mapper = utils.load_pickle(self.id_idx_mapper_path)           # {page_id: page_idx} mapper
        assert type(self.id_idx_mapper) == dict, \
            "[WikiIndexQuery] mapper must be a dictionary"

        # connect to DB
        client = connected(host='localhost')
        db = get_database(client, self.db_name)
        self.col = get_collection(db, self.col_name)

    def query(self, page_id, passage_idx):
        """ Returns the query cursor for the query matching the page_id and passage_idx """
        page_idx = self.id_idx_mapper.get(str(page_id))         # gets the idx required for the db
        
        return self.col.find_one({
            self.WikiField.page_id.value : page_idx, 
            self.WikiField.passage_idx.value : str(passage_idx)
                })
    def query_page_id_only(self, page_id, single):
        """ Returns the a json with the matching page_id picked from the first doc
        Args:
        single -- if True, returns only 1 passage.
        """
        page_idx = self.id_idx_mapper.get(str(page_id))
        if single:
            return self.col.find_one({
                self.WikiField.page_id.value: page_idx
                })
        else:
            return self.col.find({
                self.WikiField.page_id.value: page_idx
            })

class WikiIdxRawQuery(object):
    class WikiField(Enum):
        page_idx = "page_idx"             # indexed
        passage_idx = "passage_idx"
        tokens = "tokens"


    def __init__(self):
        self.db_name = 'wikiDatabase'
        self.col_name = 'wiki_idx_raw'

        # connect to DB
        client = connected(host='localhost')
        db = get_database(client, self.db_name)
        self.col = get_collection(db, self.col_name)

    def query(self, page_idx, passage_idx):
        """ Returns the query cursor for the query matching the page_id and passage_idx """
       
        return self.col.find_one({
            self.WikiField.page_idx.value : page_idx, 
            self.WikiField.passage_idx.value : str(passage_idx)
                })
    def query_page_id_only(self, page_idx, single):
        """ Returns the a json with the matching page_id picked from the first doc
        Args:
        single -- if True, returns only 1 passage.
        """
        if single:
            return self.col.find_one({
                self.WikiField.page_idx.value: page_idx
                })
        else:
            return self.col.find({
                self.WikiField.page_idx.value: page_idx
            })


class InvertedIndexQuery(object):
    """ Query object for the collection 'InvertedIndex' """
    
    class InvertedIndexField(Enum):
        """ Field types used in the collection """
        term = "term"           # indexed
        page_id = "page_id"
        tfidf = "tfidf"

    def __init__(self):
        self.db_name = 'wikiDatabase'
        self.col_name = 'InvertedIndex'

        client = connected(host='localhost')
        db = get_database(client, self.db_name)
        self.col = get_collection(db, self.col_name)
    
    def get_postings(self, term, limit=None, verbose=False):
        """ Return cursor of postings from the term """
        if limit is None:
            postings  = self.col.find(filter={self.InvertedIndexField.term.value: term})
        postings = self.col.find(filter={self.InvertedIndexField.term.value: term})
        postings.sort(InvertedIndexQuery.InvertedIndexField.tfidf.value, -1).limit(limit)     # sort descending
        if verbose:
            print("[DB] Term: {}; Postings returned: {}".format(term, postings.count()))
        return postings

####################################

###################################
######## CONNECT DATABASE #########
###################################

class Host(Enum):
    ubuntu = "192.168.1.10"
    localhost = "127.0.0.1"
    
    @staticmethod
    def list():
        return list(map(lambda case: case.name, Host))


def connected(host, port=27017):
    """ Returns the client from connecting to the database """
    assert host in Host.list(), "Invalid host: {}".format(host)
    host = get_ip_address(host)
    port = str(port)
    connected_data_format = "mongodb://{}:{}/".format(host, port)
    print("[DB] Connecting to {} at port {}...".format(host, port))
    try:
        client = pymongo.MongoClient(connected_data_format)
    except:
        print("[DB] Unable to connect.")
        quit()
    try:
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("[DB] Connected.")
    except ConnectionFailure:
        print("Server not available")
    
    return client


def get_database(client, db_name):
    """ Returns the database of the connected client """
    if db_name in client.database_names():
        print("[DB] {} found".format(db_name))
    else:
        print("[DB] {} not found, it can be instantiated by inserting a doc".format(db_name))

    db_name = str(db_name)
    db = client[db_name]
    return db

def get_collection(database, col_name):
    """ Returns the collection of the database """

    col_name = str(col_name)
    col = database[col_name]
    return col

def get_ip_address(host):
    return {
        Host.ubuntu.name : Host.ubuntu.value,
        Host.localhost.name : Host.localhost.value
    }.get(host)




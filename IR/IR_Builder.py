""" IR_Builder.py


"""
import sys
sys.path.append(sys.path[0] + '/..')
import os
from enum import Enum
from collections import Counter

import pymongo
from nltk.corpus import stopwords
from tqdm import tqdm

from IR.IR_Wiki_Parser import preprocess_tokens_list
from IR.IR_utils import int_encode
from mongodb.mongodb_query import connected, get_database, get_collection
from mongodb.mongodb_query import WikiIdxRawQuery
from utils import load_pickle, save_pickle
import utils

#### TODO --- INDEX THE PAGE IDXS AND PASSAGE IDXS FIRST -- RENAME PAGE_ID TO PAGE_IDX

class InvIdxRawDB(object):
    class InvIdxRawField(Enum):
        term = "term"             # indexed
        page_idx = "page_idx"
        raw_count = "raw_count"
        df_t = "df_t"


    def __init__(self):
        self.db_name = 'wikiDatabase'
        self.col_name = 'InvIdxRaw'

        # connect to DB
        client = connected(host='localhost')
        db = get_database(client, self.db_name)
        self.col = get_collection(db, self.col_name)

    def insert(self, term, df_t):
        return self.col.insert_one({
                                self.InvIdxRawField.term.name: term, 
                                self.InvIdxRawField.df_t.name : df_t
        })

    def query(self, term, page_idx):
        return self.col.find_one({self.InvIdxRawField.term: term})
    
    def addPosting(self, term, posting):
        return self.col.update_one({self.InvIdxRawField.term.name : term},
                                    { "$addToSet": {"postings": posting}})      # add new if posting not exist

wikiIdxRaw = WikiIdxRawQuery()
InvIdxRawDB = InvIdxRawDB()

############### DB VERSION ###########################

def doc_term_freqs_build():
    unique_page_indices_path = 'raw_db_page_ids_idx_dict.pkl'
    unique_page_indices = load_pickle(unique_page_indices_path)                    # get this from passing all docs -- 
    # loop through all the unique page ids
    counter = Counter()
    doc_term_freqs = {}
    stop_words = stopwords.words('english')

    #### BUILD DOC TERM FREQS ####
    print("[INFO] Building doc term freqs...")
    for page_idx in tqdm(unique_page_indices):
        # pull tokens_string from page_idx
        page_idx_docs = list(wikiIdxRaw.query_page_id_only(page_idx, single=False))
        for doc in page_idx_docs:
            tokens_string = doc.get("tokens")
            # preprocess string
            terms = preprocess_tokens_list(tokens_string.split(), stem=True)        # --> stemmed
            terms = [t for t in terms if t not in stop_words]                       # remove stop words

            terms_set = set(terms)
            # c = list(Counter(terms_set))

            for unique_term in terms_set:
            # update doc term freqs
                if doc_term_freqs.get(unique_term) is None:
                    doc_term_freqs[unique_term] = int_encode(0)
                    raw_count = int_encode(doc_term_freqs.get(unique_term) + 1)
                    doc_term_freqs[unique_term] = raw_count
                else:
                    raw_count = int_encode(doc_term_freqs.get(unique_term) + 1)
                    doc_term_freqs[unique_term] = raw_count
            
            counter.update(terms)           # keeps track of all terms

    save_pickle(doc_term_freqs, "doc_term_freqs_stemmed.pkl")

# doc_term_freqs_build()
# doc_term_freqs = load_pickle("doc_term_freqs_stemmed.pkl")

# # insert all unique terms, then index it.
# print("[INFO Inserting all unique terms into the db.")
# for unique_term in tqdm(doc_term_freqs.keys()):
#     df_t = int(doc_term_freqs.get(unique_term))
#     InvIdxRawDB.insert(unique_term, df_t)


# print("PLEASE INDEX THE TERMS IN DB!")

def build_inv_idx_db():
    stop_words = set(stopwords.words('english'))
    unique_page_indices_path = 'raw_db_page_ids_idx_dict.pkl'
    unique_page_indices = load_pickle(unique_page_indices_path)                    # get this from passing all docs -- 
    for page_idx in tqdm(unique_page_indices.keys()):
        # pull tokens_string from page_idx
        page_idx_docs = list(wikiIdxRaw.query_page_id_only(page_idx, single=False)) # Get tokens_string
        for doc in page_idx_docs:
            tokens_string = doc.get("tokens")
            # preprocess string
            terms = preprocess_tokens_list(tokens_string.split(), stem=True)        # --> stemmed
            terms = [t for t in terms if t not in stop_words]                       # remove stop words

            c = Counter(terms)        # unique terms
            for (term, raw_count) in c.items():
                posting = {page_idx: raw_count}
                InvIdxRawDB.addPosting(term, posting)

# build_inv_idx_db()


    # preprocess tokens_string -> [stemmed terms]

    ###### NOTE: the following terms are all stemmed tokens #######

    # {term: ___ , 
    # page_indices: {
    #                 page_idx: {
    #                 raw_count: ____
    #                     }
    #                },
    # dft: ____ }

    # find term in db
        # if term not in db
            # insert term into db as index
    
    # find term with current page_idx in db
        # if term, page_idx not in db
            # insert term, page_idx

    # update term's raw count in term, page_idx by 1



############## RAM VERSION ###################
##### shut down DB first to recover RAM ######

CHECKPOINT_FOLDER = "resource/IR_Inv_Builder_Checkpoints"
COMPLETED_PAGES_NAME = "completed_pages"
log_file = "IR_Idx_Builder.txt"

inv_index = {}
doc_term_freqs = {}
completed_pages = []

stop_words = set(stopwords.words('english'))
unique_page_indices_path = 'raw_db_page_ids_idx_dict.pkl'
unique_page_indices = load_pickle(unique_page_indices_path)

split = 0
idx = 0
for page_idx in tqdm(unique_page_indices):
    # {term: postings {page_idx: raw_count, ...}, df_t: __}
    # pull tokens_string from page_idx
    page_idx_docs = list(wikiIdxRaw.query_page_id_only(page_idx, single=False))

    page_idx = int_encode(page_idx)             # compress page_idx
    for doc in page_idx_docs:
        tokens_string = doc.get("tokens")
        # preprocess string
        terms = preprocess_tokens_list(tokens_string.split(), stem=True)
        terms = [t for t in terms if t not in stop_words]

        counter = Counter(terms)
        for unique_term in counter.keys():
            if unique_term == 'lrb' or unique_term == 'rrb':
                continue
            # update inverted index
            if inv_index.get(unique_term) is None:
                inv_index[unique_term] = {}                                       # instantiate new key
            if inv_index[unique_term].get(page_idx) is None:
                inv_index[unique_term][page_idx] =  int_encode(0)                 # instantiate new dict value in key

            raw_count = inv_index[unique_term].get(page_idx)
            raw_count = int_encode(raw_count + counter[unique_term])
            inv_index[unique_term][page_idx] = raw_count

            # # update doc term freqs
            # if doc_term_freqs.get(unique_term) is None:
            #     doc_term_freqs[unique_term] = int_encode(0)
            # raw_count = int_encode(doc_term_freqs.get(unique_term) + 1)
            # doc_term_freqs[unique_term] = raw_count
    
    if (idx + 1) % 10000 == 0:
        # -- Checkpoint
        print("[INFO] Writing checkpoint to disk...")
        save_array = [inv_index]
        save_path = "{}/{}.pkl".format(CHECKPOINT_FOLDER, split)
        if not os.path.isdir(CHECKPOINT_FOLDER): os.mkdir(CHECKPOINT_FOLDER)
        save_pickle(save_array, save_path)

        split += 1

        message = ("{} successfully parsed, saved to {}\n"
            "Format: [inv_index, doc_term_freqs]").format(page_idx, 
                                                            save_path)
        utils.log(message, log_file)

        completed_pages.append(str(page_idx) + '.pkl')
        save_pickle(completed_pages, "{}/{}.pkl".format(CHECKPOINT_FOLDER, COMPLETED_PAGES_NAME))
    idx += 1
    

        


        


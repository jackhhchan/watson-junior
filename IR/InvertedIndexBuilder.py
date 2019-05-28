""" Inverted Index 

This module builds the inverted index for the Automatic Fact Verification System, watson-junior.

Note:
-- preprocessing only lower-case

RUN THIS SCRIPT
"""
import sys
sys.path.append(sys.path[0] + '/..')
import math
import os
import json
import gc
import time
# gc.set_debug(gc.DEBUG_LEAK)

import pickle
from tqdm import tqdm

from IR import wiki_parser

#### CHANGE THIS FOR A DIFFERENT SAVE FILE NAME
INVERTED_INDEX_FNAME = "inverted_index_stemmed.pkl"
PAGE_IDS_IDX_DICT_FNAME = "page_idx_ids_dict_stemmed.pkl"
DOC_TERM_FREQS_FNAME = 'doc_term_freqs_stemmed.pkl'
# INVERTED_INDEX_FNAME = "mock-inv-index.pkl"


def inverted_index_builder():
    """ Build Inverted Index from the collection of Page objects loaded from the folder
    
    returns
    inverted_index object

    Note:
    inverted_index.inverted_index; {term: {page_id:weight, page_id:weight}}
    """
    folders_name = 'resource'
    folders = os.listdir(folders_name+"/wiki")
    inverted_index = {}
    page_ids_idx_dict = {}
    num_folders = len(folders)


    inverted_index_builder = InvertedIndexBuilder()

    # # construct doc-term-freqs by portions
    # for idx, wiki_folder in enumerate(folders):
    #     folder_path = "{}/wiki/{}/".format(folders_name, wiki_folder)

    #     print("[INFO] Parsing the wiki docs in {}...".format(folder_path))
    #     pages_collection = list(wiki_parser.parse_wiki_docs(folder_name=folder_path).values()) 

    #     # build up doc term freqs and page collection
    #     print("[INFO] Building {}/{} of the doc term freqs...".format(idx + 1, num_folders))
    #     inverted_index_builder.parse_pages(pages_collection)

    # with open(DOC_TERM_FREQS_FNAME, 'wb') as handle:
    #     pickle.dump(inverted_index_builder.doc_term_freqs, handle)

    with open(DOC_TERM_FREQS_FNAME, 'rb') as handle:
        doc_term_freqs = pickle.load(handle)

    inverted_index_builder.doc_term_freqs = doc_term_freqs
    # construct inverted index by portions
    page_idx = 0
    for idx, wiki_folder in enumerate(folders):
        folder_path = "{}/wiki/{}".format(folders_name, wiki_folder)
        
        print("[INFO] Building portion {}/{} of the inverted index...".format(idx + 1, num_folders))
        pages_collection = list(wiki_parser.parse_wiki_docs(folder_name=folder_path).values()) 
        page_idx = inverted_index_builder.build(pages_collection, page_idx)

    inverted_index = inverted_index_builder.inverted_index
    page_ids_idx_dict = inverted_index_builder.page_ids_idx_dict

    return inverted_index, page_ids_idx_dict




class InvertedIndexBuilder(object):

    def __init__(self, pages_collection = [], inverted_index={}, doc_term_freqs={}, page_ids_idx_dict={}):
        self.doc_term_freqs = doc_term_freqs                    #{token: df_t}  i.e. df_t = frequency of term in entire collection
        
        self.inverted_index = inverted_index        #{token: {page_id:weight, page_id:weight, ...}}
        self.page_ids_idx_dict = page_ids_idx_dict  # page-idx : page-id look up dictionary to reduce inverted index size    

        self.pages_collection = pages_collection


    def build(self, pages_collection, page_idx):
        """ Builds the Inverted Index
        
        ONLY CALL THIS METHOD AFTER PARSING ALL FILES ONCE TO BUILD DOC_TERM_FREQS
        """
        N = 5396106           # HARD CODED

        print("[INFO] Building inverted index...")
        inverted_index = self.inverted_index
        for page in tqdm(pages_collection):            # PROBLEM: -- pages_collection here
            self.page_ids_idx_dict[page_idx] = page.page_id

            term_freqs_dict = self.create_term_freqs_dict(page)
            for term, tf in term_freqs_dict.items():
                # compute tfidf
                df_t = self.doc_term_freqs.get(term)
                tfidf = self.compute_tfidf(tf, df_t, N)

                # posting = {page_idx: tfidf}
                posting = {page_idx: tfidf}

                # update inverted_index
                if inverted_index.get(term) is None:
                    inverted_index[term] = {"DocIDs": [posting]}
                else:
                    # inverted_index[term].update(posting)
                    inverted_index[term]["DocIDs"].append(posting)
                    
            page_idx += 1

        return page_idx


    def parse_pages(self, pages_collection):
        """ Parse the collection of pages and construct the term_freqs_dicts and doc_term_freqs """

        # for every page, for each unique term add 1 
        for page in tqdm(pages_collection):
            term_freqs_dict = self.create_term_freqs_dict(page)                    # create & update page.term_freqs_dict
            
            for term in term_freqs_dict.keys():                                     # all terms are unique
                self.doc_term_freqs[term] = self.doc_term_freqs.get(term, 0) + 1    # update doc_term_freqs {term: doc_freqs}


    def create_term_freqs_dict(self, page):
        """ return term frequency dictionary for the page """

        term_freqs_dict = {}                        # {term: freqs}
        passages = page.passages                    # get passages from page
        
        tokens = self.get_BOW(passages)  # flatten tokens in list
        for token in tokens:
            term_freqs_dict[token] = term_freqs_dict.get(token, 0) + 1

        return term_freqs_dict

    def get_BOW(self, passages):
        """ return a bag of preprocessed tokens from the list of passages """

        return [token for passage in passages for token in passage.tokens]


    def compute_tfidf(self, tf, df_t, N, normalize=True):
        if normalize:
            tf = math.log(1+tf, 2)
        idf = self.compute_idf(df_t, N)
        return tf*idf

    def compute_idf(self, df_t, N):
        idf = math.log(N/df_t, 2)          # log base 2
        return idf


class Page(object):
    __slots__ = ('page_id', 'passages', 'term_freqs_dict')   # prevents other properties from being added to objects of this class
    def __init__(self, page_id):
        self.page_id = page_id
        self.passages = []
        self.term_freqs_dict = {}


class Passage(object):
    __slots__ = ('page_id', 'passage_idx', 'tokens')       
    def __init__(self, passage_idx, tokens):
        # self.page_id = page_id
        self.passage_idx = passage_idx
        self.tokens = tokens






if __name__ == "__main__":
    inverted_index, page_ids_idx_dict = inverted_index_builder()
    # # term = inverted_index.test_term
    # # print("Term: {}, postings list: {}".format(term, inv_idx[term]))
    with open(INVERTED_INDEX_FNAME, 'wb') as inv_handle:
        pickle.dump(inverted_index, inv_handle)                     # dump inverted index
    with open(PAGE_IDS_IDX_DICT_FNAME, 'wb') as page_ids_idx_dict_handle:
        pickle.dump(page_ids_idx_dict, page_ids_idx_dict_handle)    # dump page id idx dict

    with open(INVERTED_INDEX_FNAME, 'rb') as inv_handle:
        inverted_index = pickle.load(inv_handle)                     # dump inverted index
    with open(PAGE_IDS_IDX_DICT_FNAME, 'rb') as page_ids_idx_dict_handle:
        page_ids_idx_dict = pickle.load(page_ids_idx_dict_handle)    # dump page id idx dict


    json_handle = open("inverted_index_stemmed.json", "a")
    for (term, DocIDs) in tqdm(inverted_index.items()):
        # print("Number of postings: {}".format(len(postings.items())))

        # for (page_idx, tfidf) in tqdm(postings.items()):
        #     doc = {"term": term, 'page_idx': page_idx, 'tfidf': tfidf}
        #     json.dump(doc, json_handle)
        doc = {"term": term, "DocIDs": DocIDs}
        json.dump(doc, json_handle)
        
        # json_handle.flush()

    json_handle.close()

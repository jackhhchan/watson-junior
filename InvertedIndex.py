""" Inverted Index 

This module builds the inverted index for the Automatic Fact Verification System, watson-junior.

Note:
-- preprocessing only lower-case

RUN THIS SCRIPT
"""

import math
import os

import pickle
from tqdm import tqdm

import utils
import wiki_parser

#### CHANGE THIS FOR A DIFFERENT SAVE FILE NAME
INVERTED_INDEX_FNAME = "inverted_index.pkl"
# INVERTED_INDEX_FNAME = "mock-inv-index.pkl"



def inverted_index_builder():
    """ Build Inverted Index from the collection of Page objects loaded from the folder
    
    returns
    inverted_index object

    Note:
    inverted_index.inverted_index; {term: {page_id:weight, page_id:weight}}
    """
    folders_name = 'resource'
    folders = os.listdir(folders_name)
    inverted_index = {}
    for idx, wiki_folder in enumerate(folders):
        folder_path = "{}/{}".format(folders_name, wiki_folder)
        
        print("[INFO] Parsing the wiki docs in {}...".format(folder_path))
        pages_dict = wiki_parser.parse_wiki_docs(folder_name=folder_path)
        pages_collection = list(pages_dict.values())
        print("[INFO] Building portion {} of the inverted index...".format(idx))
        inverted_index_object = InvertedIndex(pages_collection=pages_collection, inverted_index=inverted_index)
        inverted_index = inverted_index_object.inverted_index
        pages_collection = None

    return inverted_index




class InvertedIndex():

    def __init__(self, pages_collection, inverted_index = {}):
        self.N = len(pages_collection)
        self.doc_term_freqs = {}                    #{token: df_t}  i.e. df_t = frequency of term in entire collection
        
        self.inverted_index = inverted_index        #{token: {page_id:weight, page_id:weight, ...}}
        
        self.test_term = ""                         # test term
        self.build(pages_collection)
        
        pages_collection = None                     # clear memory



    def build(self, pages_collection):
        """ Builds the Inverted Index"""
        print("[INFO] Creating term_freqs and doc_freqs dictionaries from pages collection to build the inverted index...")
        self.parse_pages(pages_collection)

        print("[INFO] Building inverted index...")
        inverted_index = self.inverted_index
        for page in tqdm(pages_collection):
            for term, tf in page.term_freqs_dict.items():
                # compute tfidf
                df_t = self.doc_term_freqs.get(term)
                tfidf = self.compute_tfidf(tf, df_t)
                posting = {page.page_id: tfidf}
                # update inverted_index
                if inverted_index.get(term) is None:
                    inverted_index[term] = posting
                else:
                    inverted_index[term].update(posting)

                self.test_term = term

    def parse_pages(self, pages_collection):
        """ Parse the collection of pages and construct the term_freqs_dicts and doc_term_freqs """
        for page in tqdm(pages_collection):
            page.term_freqs_dict = self.create_term_freqs_dict(page)                    # create & update page.term_freqs_dict
            
            seen_terms = []
            for term in page.term_freqs_dict.keys():
                if term not in seen_terms:
                    seen_terms.append(term)
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


    def compute_tfidf(self, tf, df_t):
        idf = self.compute_idf(df_t)
        return tf*idf

    def compute_idf(self, df_t):
        idf = math.log(self.N/df_t, 2)          # log base 2
        return idf


class Page:
    def __init__(self, page_id):
        self.page_id = page_id
        self.passages = []
        self.term_freqs_dict = {}


class Passage:
    def __init__(self, page_id, passage_idx, tokens):
        self.page_id = page_id
        self.passage_idx = passage_idx
        self.tokens = tokens






if __name__ == "__main__":
    inverted_index = inverted_index_builder()
    inv_idx = inverted_index.inverted_index
    term = inverted_index.test_term
    print("Term: {}, postings list: {}".format(term, inv_idx[term]))
    pickle.dump(inv_idx, open(INVERTED_INDEX_FNAME, 'wb'))     # dump inverted index

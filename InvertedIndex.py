""" Inverted Index 

This module builds the inverted index for the Automatic Fact Verification System, watson-junior.

Note:
-- preprocessing only lower-case

RUN THIS SCRIPT
"""

import math
import os
import sys
import gc
# gc.set_debug(gc.DEBUG_LEAK)

import pickle
from tqdm import tqdm

import wiki_parser

#### CHANGE THIS FOR A DIFFERENT SAVE FILE NAME
INVERTED_INDEX_FNAME = "inverted_index_tf_normalized_proper_fixed.pkl"
PAGE_IDS_IDX_DICT_FNAME = "page_ids_idx_dict_normalized_proper_fixed.pkl"
# INVERTED_INDEX_FNAME = "mock-inv-index.pkl"


def inverted_index_builder():
    """ Build Inverted Index from the collection of Page objects loaded from the folder
    
    returns
    inverted_index object

    Note:
    inverted_index.inverted_index; {term: {page_id:weight, page_id:weight}}
    """
    # folders_name = 'resource'
    # folders = os.listdir(folders_name)
    # inverted_index = {}
    # page_ids_idx_dict = {}
    # doc_term_freqs = {}
    # num_folders = len(folders)
    # for idx, wiki_folder in enumerate(folders):
    #     folder_path = "{}/{}".format(folders_name, wiki_folder)
        
    #     print("[INFO] Parsing the wiki docs in {}...".format(folder_path))
    #     pages_collection = list(wiki_parser.parse_wiki_docs(folder_name=folder_path).values())               # change this to pages_collection = wiki_parser.parse_wiki_docs(folder_name=folder_path).values()
    #     print("[INFO] Building portion {}/{} of the inverted index...".format(idx + 1, num_folders))
    #     inverted_index_object = InvertedIndex(pages_collection=pages_collection, 
    #                                           inverted_index=inverted_index, 
    #                                           doc_term_freqs=doc_term_freqs, 
    #                                           page_ids_idx_dict=page_ids_idx_dict)
    #     inverted_index = inverted_index_object.inverted_index
    #     page_ids_idx_dict = inverted_index_object.page_ids_idx_dict
    #     doc_term_freqs = inverted_index_object.doc_term_freqs

    #     # force garbage collection
    #     for page in pages_collection: 
    #         del page
    #     del pages_collection
    #     del inverted_index_object
    #     gc.collect()

    # folder_name = 'resource_doc_term_matrix'
    # print("[INFO] Parsing the wiki docs in {}...".format(folder_name))
    # pages_collection = list(wiki_parser.parse_wiki_docs(folder_name=folder_name).values())
    # inverted_index_object = InvertedIndex(pages_collection=pages_collection)
    # inverted_index = inverted_index_object.inverted_index
    # page_ids_idx_dict = inverted_index_object.page_ids_idx_dict


    folders_name = 'resource'
    folders = os.listdir(folders_name)
    inverted_index = {}
    page_ids_idx_dict = {}
    num_folders = len(folders)


    # inverted_index_object = InvertedIndex()
    with open('3.pkl', 'rb') as handle:
        inverted_index_object = pickle.load(handle)

    for idx, wiki_folder in enumerate(folders):
        if wiki_folder in ['0', '1', '2', '3', '10']: 
            continue
        folder_path = "{}/{}".format(folders_name, wiki_folder)

        print("[INFO] Parsing the wiki docs in {}...".format(folder_path))
        pages_collection = list(wiki_parser.parse_wiki_docs(folder_name=folder_path).values()) 
        print("[INFO] Building portion {}/{} of the inverted index...".format(idx + 1, num_folders))

        # build up term freqs and doc term freqs and page collection
        inverted_index_object.parse_pages(pages_collection)
        with open(wiki_folder+'.pkl', 'wb') as handle:
            pickle.dump(inverted_index_object, handle)


    # with open('parsed_inv_object.pkl', 'wb') as handle:
    #     pickle.dump(inverted_index_object, handle)

    inverted_index_object.build()
    inverted_index = inverted_index_object.inverted_index
    page_ids_idx_dict = inverted_index_object.page_ids_idx_dict

    return inverted_index, page_ids_idx_dict




class InvertedIndex(object):

    def __init__(self, pages_collection = [], inverted_index={}, doc_term_freqs={}, page_ids_idx_dict={}):
        self.doc_term_freqs = doc_term_freqs                    #{token: df_t}  i.e. df_t = frequency of term in entire collection
        
        self.inverted_index = inverted_index        #{token: {page_id:weight, page_id:weight, ...}}
        self.page_ids_idx_dict = page_ids_idx_dict  # page-idx : page-id look up dictionary to reduce inverted index size    

        self.pages_collection = pages_collection


    def build(self):
        """ Builds the Inverted Index"""
        pages_collection = self.pages_collection
        N = len(pages_collection)

        print("[INFO] Building inverted index...")
        inverted_index = self.inverted_index
        for page_idx, page in tqdm(enumerate(pages_collection)):
            self.page_ids_idx_dict[page_idx] = page.page_id

            for term, tf in page.term_freqs_dict.items():
                # compute tfidf
                df_t = self.doc_term_freqs.get(term)
                tfidf = self.compute_tfidf(tf, df_t, N)

                posting = {page_idx: tfidf}

                # update inverted_index
                if inverted_index.get(term) is None:
                    inverted_index[term] = posting
                else:
                    inverted_index[term].update(posting)



    def parse_pages(self, pages_collection):
        """ Parse the collection of pages and construct the term_freqs_dicts and doc_term_freqs """
        self.pages_collection.extend(pages_collection)
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
    def __init__(self, page_id, passage_idx, tokens):
        self.page_id = page_id
        self.passage_idx = passage_idx
        self.tokens = tokens






if __name__ == "__main__":
    inverted_index, page_ids_idx_dict = inverted_index_builder()
    # term = inverted_index.test_term
    # print("Term: {}, postings list: {}".format(term, inv_idx[term]))
    with open(INVERTED_INDEX_FNAME, 'wb') as inv_handle:
        pickle.dump(inverted_index, inv_handle)                     # dump inverted index
    with open(PAGE_IDS_IDX_DICT_FNAME, 'wb') as page_ids_idx_dict_handle:
        pickle.dump(page_ids_idx_dict, page_ids_idx_dict_handle)    # dump page id idx dict

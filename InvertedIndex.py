""" Inverted Index Construction

This module builds the inverted index for the Automatic Fact Verification System, watson-junior.

"""
import math
import os

import utils
import wiki_parser

@staticmethod 
def inverted_index_builder():

    # get array of tokens with each row being an individual page_id
    pages_dict = wiki_parser.parse_wiki_docs()
    pages_collection = list(pages_dict.values())

    inverted_index = InvertedIndex(pages_collection=pages_collection)     # placeholder

    return inverted_index




class InvertedIndex():

    def __init__(self, pages_collection):
        self.pages_collection = []
        self.N = len(pages_collection)
        self.doc_term_freqs = []
        self.inverted_index = {}        #{token: {page_id:weight, page_id:weight, ...}}
        self.doc_term_freqs = {}        #{token: df_t}  i.e. df_t = frequency of term in entire collection


    def build(self):
        """ Builds the Inverted Index"""
        inverted_index = self.inverted_index
        for page in self.pages_collection:

            page.term_freqs_dict = self.create_term_freqs_dict(page)    # create & update page.term_freqs_dict

            for term, tf in page.term_freqs_dict.items():
                # compute tfidf
                df_t = self.doc_term_freqs.get(term)
                tfidf = self.compute_tfidf(tf, df_t)
                posting = {page.page_id: tfidf}
                # update inverted_index
                inverted_index[term] = inverted_index.get(term, {}).update(posting)



    def create_term_freqs_dict(self, page):
        """ return term frequency dictionary for the page """

        term_freqs_dict = {}                        # {term: freqs}
        passages = page.passages                    # get passages from page
        
        tokens = self.get_BOW(passages)  # flatten tokens in list
        for token in tokens:
            term_freqs_dict[token] = term_freqs_dict.get(token, 0) + 1

        return term_freqs_dict

    def get_BOW(self, passages):
        """ return a bag of preprocessed tokens from the list of passages
            
            Preprocess include (lower case & non-numeric filtering)
        """

        return [token.lower() for passage in passages for token in passage if token.isalpha()]


    def compute_tfidf(self, tf, df_t):
        idf = self.compute_idf(df_t)
        return tf*idf

    def compute_idf(self, df_t):
        idf = math.log(self.N/df_t)
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
    inverted_index_builder()
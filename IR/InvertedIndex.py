import sys
sys.path.append(sys.path[0] + '/..')
from collections import OrderedDict

import pickle
import json
from tqdm import tqdm
from nltk.corpus import stopwords

import nltk
nltk.download("stopwords")

import utils
from NER import get_NER_tokens
from mongodb.mongodb_query import InvertedIndexQuery

class InvertedIndex(object):
    def __init__(self):
        self.db_query = InvertedIndexQuery()
    
    #########################
    #### RANKED PAGE IDS ####
    #########################

    def get_ranked_page_ids(self, raw_claim, tfidf=False):
        """ 
        Returns a list of ranked page ids given claim
        Args:
        raw_claim   -- (Str) claim in a single string format
        tfidf       -- (Bool) Set True to return the tfidf as well. (default: False)
        """
        assert type(tfidf) == bool, "tfidf argument if True also returns the tfidf"
        page_ids_tfidf = self.get_page_ids_tfidf(raw_claim)
        ranked_page_ids_tfidf = self.ranked_page_ids_tfidf(page_ids_tfidf)      # array of dicts in order

        if tfidf:
            return ranked_page_ids_tfidf
        else:
            ranked_page_ids = [page_id for page_id_tfidf in ranked_page_ids_tfidf for page_id in page_ids_tfidf.keys()]
            return ranked_page_ids


    def get_page_ids_tfidf(self, raw_claim):
        output ={}

        claim = raw_claim.lower().split()
        print("[INFO] Number of tokens in claim: {}".format(len(claim)))

        for term in claim:
            postings = self.db_query.get_postings(term=term, verbose=True)
            for posting in postings:
                page_id = posting.get(self.db_query.InvertedIndexField.page_id.value)
                tfidf = posting.get(self.db_query.InvertedIndexField.tfidf.value)
                output[page_id] = output.get(page_id, 0) + tfidf
        
        return output

    def ranked_page_ids_tfidf(self, page_ids_tfidf):
        return utils.sorted_dict(page_ids_tfidf)

    ##############################
    ##### QUERY REFORMULATION ####
    ##############################

    def query_reformulation(self, raw_claim):
        """ Reformulates the query, NER & (Query Expansion //TODO) """
        raw_claim = raw_claim.lower()
        
        # named entities linking
        NER_tokens = self.get_named_entities(raw_claim)

        # handle remove stop words
        tokens_without_stopwords = self.removed_stop_words(NER_tokens)

        ## TODO -- QUERY EXPANSION
        
        processed_claim_tokens = tokens_without_stopwords
        return processed_claim_tokens

    def get_named_entities(self, raw_claim):
        """ Returns the named entities outputted by the AllenNLP NER"""
        NER_tokens = get_NER_tokens(raw_claim)
        if len(NER_tokens) <= 0:
            message = "Nothing returned from NER for claim: {}".format(raw_claim)
            utils.log(message)

        return NER_tokens
    
    def removed_stop_words(self, tokens):
        stop_words = stopwords.words('english')
        removed = [token for token in tokens if token not in stop_words]
        return removed







################ DATABASE TEST ####################
def test():
    """ Test inverted index with a pre-defined raw claim """

    claims = [
        "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company",
        "Roman Atwood is a content creator",
        "Nikolaj Coster Waldau Fox Broadcasting Company",
        "Nikolaj Coster-Waldau",
        "Fox Broadcasting Company"
    ]

    # Databatase query object
    db_query = InvertedIndexQuery()

    
    raw_claim = claims[2]               # CHANGE THIS
    claim = raw_claim.lower().split()
    print("[INFO] Number of tokens in claim: {}".format(len(claim)))

    # return all relevant page ids and their tfidf scores
    output ={}
    for term in claim:
        postings = db_query.get_postings(term=term, verbose=True)
        for posting in postings:
            page_id = posting.get(db_query.InvertedIndexField.page_id.value)
            tfidf = posting.get(db_query.InvertedIndexField.tfidf.value)
            output[page_id] = output.get(page_id, 0) + tfidf
    
    print(list(output)[:19])            # NOTE: This is not ranked!

if __name__ == '__main__':
    print(utils.get_elapsed_time(test))
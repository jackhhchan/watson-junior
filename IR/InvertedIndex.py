import sys
sys.path.append(sys.path[0] + '/..')
import re

import pickle
import json
from tqdm import tqdm
from nltk.corpus import stopwords

# import nltk
# nltk.download("stopwords")

import utils
from IR.NER import get_NER_tokens
from mongodb.mongodb_query import InvertedIndexQuery

class InvertedIndex(object):
    def __init__(self, verbose=False):
        self.db_query = InvertedIndexQuery()
        self.verbose = verbose
    
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

        claim_terms = self.query_reformulated(raw_claim)
        page_ids_tfidf = self.get_page_ids_tfidf(claim_terms)
        ranked_page_ids_tfidf = self.ranked_page_ids_tfidf(page_ids_tfidf)      # array of dicts in order

        if tfidf:
            return ranked_page_ids_tfidf
        else:
            ranked_page_ids = [page_id for (page_id, tfidf) in ranked_page_ids_tfidf]
            return ranked_page_ids


    def get_page_ids_tfidf(self, claim_terms):
        output ={}
        if self.verbose:
            print("[INFO - InvIdx] Query: {}".format(claim_terms))
            print("[INFO - InvIdx] Number of tokens in claim: {}".format(len(claim_terms)))

        for term in claim_terms:
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

    def query_reformulated(self, raw_claim):
        """ Reformulates the query, NER & (Query Expansion //TODO) """
        raw_claim = self.remove_punctuations(raw_claim, string=True)        # return in string format
        
        # named entities linking
        tokens = self.get_named_entities(raw_claim)
        tokens = self.remove_duplicates(tokens)      # NER sometimes return duplicates

        # handle remove stop words
        tokens = self.removed_stop_words(tokens)

        ## TODO -- QUERY EXPANSION

        processed_claim_tokens = tokens
        return processed_claim_tokens

    def remove_punctuations(self, raw_claim, string=True):
        """ Returns raw claim with removed punctuations in a string format (unless specified otherwise)"""
        raw_claim = re.split('-|_|.|,|#|?|*|&|^|%|#|@|!', raw_claim)
        return " ".join(token for token in raw_claim).strip()


    def get_named_entities(self, raw_claim):
        """ Returns the named entities outputted by the AllenNLP NER"""
        raw_NER_tokens = get_NER_tokens(raw_claim)[0]
        if len(raw_NER_tokens) <= 0:
            message = "Nothing returned from NER for claim: {}".format(raw_claim)
            utils.log(message)
        # further split raw token strings returned from NER
        NER_tokens = utils.extract_processed_tokens(raw_NER_tokens)
        return NER_tokens
    
    def removed_stop_words(self, tokens):
        stop_words = stopwords.words('english')
        removed = [token for token in tokens if token not in stop_words]
        return removed

    def remove_duplicates(self, tokens):
        unique = []
        for token in tokens:
            if token not in unique:
                unique.append(token)
        return unique







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
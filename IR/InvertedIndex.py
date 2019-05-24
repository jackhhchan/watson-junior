import sys
sys.path.append(sys.path[0] + '/..')
from collections import OrderedDict

import pickle
import json
from tqdm import tqdm

import utils
from mongodb.mongodb_query import InvertedIndexQuery

class InvertedIndex(object):
    def __init__(self):
        self.db_query = InvertedIndexQuery()


    def get_ranked_page_ids(self, raw_claim, tfidf=False):
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


    def query_reformulation(self, query):
        """ reformulates the query, NER or Query Expansion"""
        # TODO






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
import sys
sys.path.append(sys.path[0] + '/..')

import pickle
import json
from tqdm import tqdm

import utils
from mongodb.mongodb_query import InvertedIndexQuery


claims = [
    "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company",
    "Roman Atwood is a content creator",
    "Nikolaj Coster Waldau Fox Broadcasting Company",
    "Nikolaj Coster-Waldau",
    "Fox Broadcasting Company"
]

def test():
    inv_index_query = InvertedIndexQuery()

    # return all relevant page ids and their tfidf scores
    start = utils.get_time()
    output = query(claims[2], inv_index_query)
    print("[INFO] Elapsed query time: {}".format(utils.get_time() - start))

    # prints ranked array
    sorted_output = sorted_dict(output)
    print(sorted_output[:19])


def query(claim, query_object):
    output ={}

    claim = claim.lower().split()
    print("[INFO] Number of tokens in claim: {}".format(len(claim)))

    for term in claim:
        postings = query_object.get_postings(term=term, verbose=True)
        for posting in postings:
            page_id = posting.get(query_object.InvertedIndexField.page_id.value)
            tfidf = posting.get(query_object.InvertedIndexField.tfidf.value)
            output[page_id] = output.get(page_id, 0) + tfidf
    
    return output


def sorted_dict(dictionary):
    """ Return array of sorted dictionary based on value and return in tuple format (key, value) """
    sorted_dictionary = sorted(dictionary.items(), key=lambda kv:kv[1], reverse=True)
    return sorted_dictionary

def query_reformulation(query):
    """ reformulates the query, NER or Query Expansion"""
    # TODO
    

if __name__ == '__main__':
    test()
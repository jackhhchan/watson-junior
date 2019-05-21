import pickle
from tqdm import tqdm

from utils import load_pickle, save_pickle

INVERTED_INDEX_FNAME = "inverted_index_tf_normalized_proper_fixed.pkl"
PAGE_IDS_IDX_DICT_FNAME = "page_ids_idx_dict_normalized_proper_fixed.pkl"
DOC_TERM_FREQS_FNAME = 'doc_term_freqs.pkl'

claims = [
    "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company",
    "Roman Atwood is a content creator",
    "Nikolaj Coster-Waldau Fox Broadcasting Company",
    "Nikolaj Coster-Waldau",
    "Fox Broadcasting Company"
]

def query(inverted_index, page_ids_idx_dict, claim, output={}):
    claim = claim.lower().split()

    for word in claim:
        print("{}".format(word))
        try:
            page_indices = inverted_index[word]
            for page_idx in page_indices:
                page_id = page_ids_idx_dict.get(page_idx)
                output[page_id] = output.get(page_id, 0) + inverted_index[word][page_idx]
        except:
            print(word+' is not in the inverted index')

    return output


def sorted_dict(dictionary):
    """ Sort dictionary based on value and return in tuple format (key, value) """
    sorted_dictionary = sorted(dictionary.items(), key=lambda kv:kv[1], reverse=True)
    return sorted_dictionary

def query_reformulation(query):
    """ reformulates the query, NER or Query Expansion"""
    # TODO
    pass


if __name__ == '__main__':
    inverted_index = load_pickle(INVERTED_INDEX_FNAME)
    page_ids_idx_dict = load_pickle(PAGE_IDS_IDX_DICT_FNAME)
    doc_term_freqs = load_pickle(DOC_TERM_FREQS_FNAME)

    tuple_inverted_index = {}
    for term in tqdm(inverted_index.keys()):
        postings = [(page_idx, tfidf) for (page_idx, tfidf) in inverted_index[term].items()]
        tuple_inverted_index[term] = postings

    save_pickle(tuple_inverted_index, 'tuple_inverted_index.pkl')



    # claims = [claims[3], claims[4]]
    claims = [claims[0]]

    final_output = {}
    for claim in claims:
        print("claim: {}".format(claim))
        final_output.update(query(inverted_index, page_ids_idx_dict, claim, final_output))

    final_output = sorted_dict(final_output)

    print(final_output[:19])        # prints the first returned documents
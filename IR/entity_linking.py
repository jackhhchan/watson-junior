# Get entity from Allen NER
# Remove stopwords
# See if there are any matches in unigrams
import sys
sys.path.append(sys.path[0] + '/..')
import re
from collections import defaultdict

from tqdm import tqdm

from nltk.corpus import stopwords
from IR.NER import get_NER_tokens
import utils

def parse_test_json(test_json):
    """ Returns a list of the json values """
    test_array = []
    for test_data in test_json.values():
        test_array.append(test_data.get('claim'))

    return test_array

def removed_stop_words(tokens):
    stop_words = stopwords.words('english')
    kept = [token for token in tokens if token not in stop_words]
    return kept

def remove_punctuations(raw_claim, string=True):
    """ Returns raw claim with removed punctuations in a string format (unless specified otherwise)"""
    raw_claim = re.split('-|_|\\s', raw_claim)
    return " ".join(token for token in raw_claim).strip()


claims = [
    "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company",
    "Roman Atwood is a content creator",
    "Nikolaj Coster Waldau Fox Broadcasting Company",
    "Nikolaj Coster-Waldau",
    "Fox Broadcasting Company"
]

json_path = "resource/train/devset.json"

def main():
    # claim = claims[0]
    # print(claim)
    test_json = utils.load_json(json_path)          
    raw_claims = parse_test_json(test_json)
    page_ids_string_dict = utils.load_pickle("page_ids_string_dict.pkl")
    matched_page_ids = defaultdict(set)
    matched_idx_claim = {}

    for idx, raw_claim in tqdm(enumerate(raw_claims)):
        raw_claim = remove_punctuations(raw_claim, string=True)

        NER_list = get_NER_tokens(raw_claim, stacked=True)
        NER_list = list(map(lambda string: string.lower(), NER_list))     # split string for each name entity

        # print("[INFO] Number of name entities: {}".format(len(NER_list)))
        # print(NER_list)
        # NER_tokens_list = list(map(lambda string: string.split(), NER_list))     # split string for each name entity

        # try to find exact match
        for entity in  NER_list:
            # for token in NER_tokens:
            page_id = page_ids_string_dict.get(entity)
            if page_id is not None:          # check if exact match
                matched_page_ids[idx].add(page_id)
        
        matched_idx_claim[idx] = raw_claim


    utils.save_pickle(matched_page_ids, "matched_page_ids.pkl")
    utils.save_pickle(matched_idx_claim, "matched_idx_claim.pkl")
    print(matched_page_ids)


    # remove stop words and try to find exact match

main()



# tokens = utils.extract_processed_tokens(NER_tokens)
# tokens = removed_stop_words(tokens)         # 1. remove stop words in entity, exact match
                                            # 2. remove stop words in 
                                            # search for pageID with > 1 tokens in claim -- takes ages.
                                            # invertedindex tfidf for the pageid -- takes long too.
# print(tokens)

def build_page_ids_string_dict():
    page_ids_idx_dict = utils.load_pickle("page_ids_idx_dict_normalized_proper_fixed.pkl")

    page_ids_string_dict = {}
    for raw_page_id in tqdm(page_ids_idx_dict.values()):
        page_id_tokens = re.split("_|-", raw_page_id)
        page_id_tokens = list(map(lambda token: token.lower(), page_id_tokens))
        page_id_string = ' '.join(token for token in page_id_tokens).strip()
        # print(page_id_string)
        page_ids_string_dict[page_id_string] = raw_page_id

    utils.save_pickle(page_ids_string_dict, "page_ids_string_dict.pkl")






"""
Run this to run watson-junior

"""
from tqdm import tqdm

import utils
from IR.InvertedIndex import InvertedIndex
from mongodb.mongodb_query import WikiQuery
from data_generators.data_generator_sentence_selection import get_passages_from_db

###### PATHS ######
json_path = "resource/test/test-unlabelled.json"            # test set
json_path = "resource/train/devset.json"                    # dev set

###### PARAMS TO CHANGE ######
# Inverted Index
page_ids_threshold = 15             # only return this many page ids from inverted index
verbose = True

# Passage Selection
confidence_threshold = None
passage_ids_threshold = None        # gets thresholded to 6 in score.py anyway

# Entailment Recognizer
confidence_threshold = None         # maybe?


def main():
    # Load test claims
    test_json = utils.load_json(json_path)          
    raw_claims = parse_test_json(test_json)

    ##### DOCUMENT SELECTION #####
    # get relevant page_ids from the inverted index
    print("[INFO - Main] Getting ranked page ids from inverted index...")
    inv_index = InvertedIndex(verbose=verbose)
    wiki_query = WikiQuery()

    total_test_claims = []
    total_test_evidences = []
    total_test_indices = []

    for idx, raw_claim in tqdm(enumerate(raw_claims)):
        print("[INFO] Claim: {}".format(raw_claim))
        start = utils.get_time()
        ranked_page_ids = inv_index.get_ranked_page_ids(raw_claim)
        print(utils.get_elapsed_time(start, utils.get_time()))

        start = utils.get_time()
        ranked_page_ids = process_ranked_page_ids(ranked_page_ids, page_ids_threshold)
        print(utils.get_elapsed_time(start, utils.get_time()))

        print("[INFO] Returned ranked page ids: \n{}".format(ranked_page_ids))

        start = utils.get_time()
        test_claims, test_evidences, test_indices = get_passage_selection_data(raw_claim=raw_claim, 
                                                                               page_ids=ranked_page_ids, 
                                                                               query_object=wiki_query)
        print(utils.get_elapsed_time(start, utils.get_time()))

        total_test_claims.extend(test_claims)
        total_test_evidences.extend(test_evidences)
        total_test_indices.extend(test_indices)

        if idx >= 49:
            break

    utils.save_pickle(total_test_claims, "test_claims.pkl")
    utils.save_pickle(total_test_evidences, "test_evidences.pkl")
    utils.save_pickle(total_test_indices, "test_indices.pkl")
   
    # format into the proper format to be passed into the passage selection NN

     ##### RELEVANT PASSAGE SELECTION #####

    # pass data into the sentence NN object
        # use an NN object
        # output: {page_id, passage_idx, classification, confidence}
    
    
    ##### ENTAILMENT RECOGNIZER #####
    # format into the proper format to be passed into the entailment recognizer NN

    # pass data into the entailment NN object

    
    ##### OUTPUT #####
    # output format: {id: {'label': LABEL, 
    #                       'evidence': [page_id, passage_idx]
    #                      }}     
    #                      #NOTE: Max 6 evidences allowed in score.py




#### DOCUMENT SELECTION ####
def process_ranked_page_ids(ranked_page_ids, threshold):
    length = len(ranked_page_ids)
    if length <= 0:
        print("[INFO - Main] No relevant page id returned.")
        return 
    else:
        if length <= threshold:
            print("[INFO - Main] Returned page_ids: {}".format(length))
            return ranked_page_ids
        else:
            print("[INFO - Main] Returned page_ids: {}, thresholded to {}".format(length, threshold))
            return ranked_page_ids[:threshold-1]

# Formatter to be passed to Passage Selection
def get_passage_selection_data(raw_claim, page_ids, query_object):
    """ Returns passage selection data in the format to be fed to its NN
    
    """
    test_claims = []
    test_evidences = []
    test_indices = []           # to be kept for output later in format [page_id, passage_idx]

    for page_id in page_ids:
        passage_indices, tokens_string_list = get_passages_from_db(page_id, query_object, output_string=True)
        if passage_indices is None or tokens_string_list is None:       # handles None, also logged
            print("[ERROR] {} query to db returned None".format(page_id))
            continue
        for i, token_string in enumerate(tokens_string_list):
            test_evidences.append(token_string)
            test_claims.append(raw_claim)
            test_indices.append([page_id, passage_indices[i]])

    assert len(test_claims) == len(test_evidences) and len(test_claims) == len(test_indices), \
        "Output data arrays to Passage Selection NN are not the same length!"

    return test_claims, test_evidences, test_indices



#### PASSAGE SELECTION ####



#### ENTAILMENT RECOGNIZER ####




#### JSON  ####
def parse_test_json(test_json):
    """ Returns a list of the json values """
    test_array = []
    for test_data in test_json.values():
        test_array.append(test_data.get('claim'))

    return test_array
    


if __name__ == "__main__":
    main()
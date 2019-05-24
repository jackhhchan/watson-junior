"""
Run this to run watson-junior

"""
import utils
from IR.InvertedIndex import InvertedIndex
from data_generators.data_generator_sentence_selection import get_passages_from_db

###### PATHS ######
test_json_path = "resource/test/test-unlabelled.json"

###### PARAMS TO CHANGE ######
# Inverted Index
page_ids_threshold = 20             # only return this many page ids from inverted index
verbose = True

# Passage Selection
confidence_threshold = None
passage_ids_threshold = None        # gets thresholded to 6 in score.py anyway

# Entailment Recognizer
confidence_threshold = None         # maybe?


def main():
    # Load test claims
    test_json = utils.load_json(test_json_path)
    test_claims = parse_test_json(test_json)

    ##### DOCUMENT SELECTION #####
    # get relevant page_ids from the inverted index
    print("[INFO - Main] Getting ranked page ids from inverted index...")
    inv_index = InvertedIndex(verbose=verbose)

    for claim in test_claims:
        ranked_page_ids = (inv_index.get_ranked_page_ids(claim))
        ranked_page_ids = process_ranked_page_ids(ranked_page_ids, page_ids_threshold)
        print(ranked_page_ids)
        break
   
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
def get_passage_selection_data(raw_claim, page_ids):
    """ Returns passage selection data in the format to be fed to its NN"""
    train_claims = []
    train_evidences = []

    # TODO
    print()


    return train_claims, train_evidences


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
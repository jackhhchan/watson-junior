"""
Run this to run watson-junior

"""
import utils
from IR.InvertedIndex import InvertedIndex

###### PATHS ######
test_json_path = "resource/test/test-unlabelled.json"

###### PARAMS TO CHANGE ######
# Inverted Index
page_ids_threshold = 20

# Passage Selection

# 


def main():
    # Load test claims
    test_json = utils.load_json(test_json_path)
    test_claims = parse_test_json(test_json)

    ##### DOCUMENT SELECTION #####
    # get relevant page_ids from the inverted index
    print("[INFO] Getting ranked page ids from inverted index...")
    inv_index = InvertedIndex()

    ranked_page_ids = []
    for claim in test_claims:
        ranked_page_ids.append(inv_index.get_ranked_page_ids(claim))

    ranked_page_ids = process_ranked_page_ids(ranked_page_ids, page_ids_threshold)
    ##### RELEVANT PASSAGE SELECTION #####
    # format into the proper format to be passed into the passage selection NN

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
        print("[INFO] No relevant page id returned.")
        return 
    else:
        if length <= threshold:
            print("[INFO] Returned page_ids: {}".format(length))
            return ranked_page_ids
        else:
            print("[INFO] Returned page_ids: {}, thresholded to {}".format(length, threshold))
            return ranked_page_ids[:threshold-1]

#### PASSAGE SELECTION ####



#### ENTAILMENT RECOGNIZER ####




#### JSON  ####
def parse_test_json(test_json):
    """ Returns a list of the json values """
    test_array = []
    for test_data in test_json.values():
        test_array.append(test_data)

    return test_array
    


if __name__ == "__main__":
    main()
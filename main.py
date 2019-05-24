"""
Run this to run watson-junior

"""
import utils

test_json_path = "resource/test/test-unlabelled.json"

def main():
    # Load test claims
    test_json = utils.load_json(test_json_path)
    test_claims = parse_test_json(test_json)

    ##### DOCUMENT SELECTION #####
    # get relevant page_ids from the inverted index
        # use an inverted index object

    
    ##### RELEVANT PASSAGE SELECTION #####
    # format into the proper format to be passed into the passage selection NN

    # pass data into the sentence NN object
        # use an NN object
        # output: {page_id, passage_idx, classification, confidence}
    
    
    ##### ENTAILMENT #####
    # format into the proper format to be passed into the entailment recognizer NN

    # pass data into the entailment NN object

    
    ##### OUTPUT #####
    # output format: 

    pass

def parse_test_json(test_json):
    """ Returns a list of the json values """
    test_array = []
    for test_data in test_json.values():
        test_array.append(test_data)

    return test_array
    


if __name__ == "__main__":
    main()
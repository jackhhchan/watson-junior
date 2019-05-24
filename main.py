"""
Run this to run watson-junior

"""
import utils
from IR.InvertedIndex import InvertedIndex
from keras import load_model
from NLI.attention import DotProductAttention
from NLI.prepare_set import create_test_data
from utils import load_pickle, save_pickle
import numpy as np

###### PATHS ######
test_json_path = "resource/test/test-unlabelled.json"
SENTENCE_SELECTION_MODEL = ''
SENTENCE_SELECTION_TKN = ''
ENTAILMENT_RECOGNIZER_MODEL = ''
ENTAILMENT_RECOGNIZER_TKN = ''

###### PARAMS TO CHANGE ######
# Inverted Index
page_ids_threshold = 20             # only return this many page ids from inverted index

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

#    can we have a dataframe to store page_id, passage_idx, raw_sentence?
        
#    ==================================
#    =======DEFINE PARAMETERS==========
#    ==================================
    which_model = 'ESIM'
    left_sequence_length = 32
    right_sequence_length = 32
    
    test_set = pd.DataFrame({'claims':claims,'evidences','page_id':PAGE_ID,\
                             'passage_id':PASSAGE_ID})
    
    sentences_pair = [(x1, x2) for x1, x2 in zip(test_set.claims, test_set.evidences)]

    pred = get_model_prediction(model_dir,tkn_dir,which_model,sentences_pair,left_sequence_length,\
                      right_sequence_length)
    pred = np.where(pred>0.5,1,0) # round up :)
    
    test_set['pred'] = pred
    relevant_test_set = test_set[test_set['pred']==0]

    ##### ENTAILMENT RECOGNIZER #####
    # format into the proper format to be passed into the entailment recognizer NN

    # pass data into the entailment NN object
    
#    1. preprocess, concatenate evidence
    relevant_test_set_concatenate
    
    
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
def get_model_prediction(model_dir,tkn_dir,which_model,sentences_pair,left_sequence_length,\
                      right_sequence_length):
    if which_model == 'ESIM':
        model = load_model(model_dir,custom_objects={'DotProductAttention':DotProductAttention})
    elif which_model == 'LSTM':
        model = load_model(model_dir)
    else:
        raise ValueError('Model Type Not Understood:{}'.format(which_model))
        
    tokenizer = load_pickle(tkn_dir)
    test_claim,test_evidence = create_test_data(tokenizer, sentences_pair, \
                                                left_sequence_length, right_sequence_length)
    pred = model.predict([test_claim,test_evidence])
    return pred


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
"""
Run this to run watson-junior

"""
import utils
from IR.InvertedIndex import InvertedIndex
from data_generators.data_generator_sentence_selection import get_passages_from_db
from keras import load_model
from NLI.attention import DotProductAttention
from NLI.prepare_set import create_test_data,create_test_data_lstm
import numpy as np
import pandas as pd
from NLI.train import get_training_data
###### PATHS ######
test_json_path = "resource/test-unlabelled.json"
PASSAGE_SELECTION_MODEL_ESIM = 'trained_model/ESIM/PassageSelection/ESIM_model.h5'
PASSAGE_SELECTION_TKN_ESIM = 'trained_model/ESIM/PassageSelection/ESIM_tokenizer.pkl'
PASSAGE_SELECTION_MODEL_LSTM = 'trained_model/LSTM/PassageSelection/LSTM_model.h5'
PASSAGE_SELECTION_TKN_LSTM = 'trained_model/LSTM/PassageSelection/LSTM_tokenizer.pkl'

ENTAILMENT_RECOGNIZER_MODEL_ESIM = ''
ENTAILMENT_RECOGNIZER_TKN_ESIM = ''
ENTAILMENT_RECOGNIZER_MODEL_LSTM = ''
ENTAILMENT_RECOGNIZER_TKN_LSTM = ''
###### PARAMS TO CHANGE ######
# Inverted Index
page_ids_threshold = 15             # only return this many page ids from inverted index
verbose = True

# Passage Selection
confidence_threshold = None
passage_ids_threshold = None        # gets thresholded to 6 in score.py anyway
which_model = 'ESIM' #{‘ESIM','LSTM}
one_sentence_length = 32
multiple_sentence_length = 64
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
    claims, raw_evidences, page_info = get_training_data(claims_path='resource/test/test_claims.pkl',
                                                         evidences_path='resource/test/test_evidences.pkl',
                                                         labels_path='resource/test/test_indices.pkl')
     ##### RELEVANT PASSAGE SELECTION #####

    # pass data into the sentence NN object
        # use an NN object
        # output: {page_id, passage_idx, classification, confidence}
    
    test_set = pd.DataFrame({'claim':claims,'raw_evidence':raw_evidences,'evidence':page_info})
    
    sentences_pair = [(x1, x2) for x1, x2 in zip(test_set.claim, test_set.raw_evidence)]

    if which_model == 'ESIM':
        pred = get_model_prediction(model_dir=PASSAGE_SELECTION_MODEL_ESIM,tkn_dir=PASSAGE_SELECTION_TKN_ESIM,\
                                    which_model='ESIM',sentences_pair=sentences_pair,\
                                    left_sequence_length=one_sentence_length,\
                                    right_sequence_length=one_sentence_length)
    elif which_model == 'LSTM':
        pred = get_model_prediction(model_dir=PASSAGE_SELECTION_MODEL_LSTM,tkn_dir=PASSAGE_SELECTION_TKN_LSTM,\
                                which_model='LSTM',sentences_pair=sentences_pair,\
                                left_sequence_length=one_sentence_length,\
                                right_sequence_length=one_sentence_length)
        
    else:
        raise ValueError('Model Type Not Understood:{}'.format(which_model))
    
#    pred = np.where(pred>0.5,1,0) # round up :)
        
    test_set['relevance'] = pred
    relevant_test_set = test_set[test_set['relevance']<0.1]
#    tmp = test_set[test_set['claim']=='Andrew Kevin Walker is only Chinese.']
    
#    tmp = test_set.sample(n=100)
#    new_tmp = concatenate_evidence_df(tmp)

    ##### ENTAILMENT RECOGNIZER #####
    # format into the proper format to be passed into the entailment recognizer NN

    # pass data into the entailment NN object
    relevant_test_set_concatenate = concatenate_evidence_df(relevant_test_set)
    
    sentences_pair = [(x1, x2) for x1, x2 in zip(relevant_test_set_concatenate.claim, \
                      relevant_test_set_concatenate.evidence)]

    if which_model == 'ESIM':
        pred = get_model_prediction(model_dir=ENTAILMENT_RECOGNIZER_MODEL_ESIM,tkn_dir=ENTAILMENT_RECOGNIZER_TKN_ESIM,\
                                which_model='ESIM',sentences_pair=sentences_pair,\
                                left_sequence_length=one_sentence_length,\
                                right_sequence_length=multiple_sentence_length)
    elif which_model == 'LSTM':
        pred = get_model_prediction(model_dir=ENTAILMENT_RECOGNIZER_MODEL_LSTM,tkn_dir=ENTAILMENT_RECOGNIZER_MODEL_LSTM,\
                                which_model='LSTM',sentences_pair=sentences_pair,\
                                left_sequence_length=one_sentence_length,\
                                right_sequence_length=multiple_sentence_length)
        
    else:
        raise ValueError('Model Type Not Understood:{}'.format(which_model))
                                
    pred = np.argmax(pred,axis=1)
    
    relevant_test_set_concatenate['label'] = pred
    
    
    
    ##### OUTPUT #####
    # output format: {id: {'label': LABEL, 
    #                       'evidence': [page_id, passage_idx]
    #                      }}     
    #                      #NOTE: Max 6 evidences allowed in score.py

    relevant_test_set_concatenate.to_json('testoutput.json',orient='index')



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
def get_model_prediction(model_dir,tkn_dir,which_model,sentences_pair,left_sequence_length,\
                      right_sequence_length):
    if which_model == 'ESIM':
        model = load_model(model_dir,custom_objects={'DotProductAttention':DotProductAttention})
        print("[INFO] ESIM model is loaded.")
        tokenizer = utils.load_pickle(tkn_dir)
        test_claim,test_evidence = create_test_data(tokenizer, sentences_pair, \
                                                left_sequence_length, right_sequence_length)
        pred = model.predict([test_claim,test_evidence])

    elif which_model == 'LSTM':
        model = load_model(model_dir)
        print("[INFO] LSTM model is loaded.")
        tokenizer = utils.load_pickle(tkn_dir)
        test_claim,test_evidence,test_leaks = create_test_data_lstm(tokenizer, sentences_pair, \
                                                left_sequence_length, right_sequence_length)
        pred = model.predict([test_claim,test_evidence,test_leaks])

    else:
        raise ValueError('Model Type Not Understood:{}'.format(which_model))
    
    return pred


#### ENTAILMENT RECOGNIZER ####




#### JSON  ####
def parse_test_json(test_json):
    """ Returns a list of the json values """
    test_array = []
    for test_data in test_json.values():
        test_array.append(test_data.get('claim'))

    return test_array

### utils ###
def concatenate_evidence_df(df):
    claims = []
    evidences = []
    page_info = []
    for c in list(df.claim.values):
        if not c in claims:
            claims.append(c)
            tmp = df[df['claim'] == c]
            evidences.append(' '.join(list(tmp['raw_evidence'].values)))
            page_info.append(list(tmp['evidence'].values))
    return pd.DataFrame({'claim':claims,'raw_evidence':evidences,'evidence':page_info})


if __name__ == "__main__":
    main()
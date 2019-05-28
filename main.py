"""
Run this to run watson-junior

"""
from tqdm import tqdm

import utils
# from IR.InvertedIndex import InvertedIndex
# from mongodb.mongodb_query import WikiQuery
# from IR.InvertedIndex import InvertedIndex
# from IR.entity_linking import get_title_entity_match
# from mongodb.mongodb_query import WikiQuery, WikiIdxQuery
# from data_generators.data_generator_sentence_selection import get_passages_from_db
from keras.models import load_model
from NLI.attention import DotProductAttention
from NLI.prepare_set import create_test_data,create_test_data_lstm
import numpy as np
import pandas as pd
from NLI.train import get_training_data

###### PATHS ######
json_file = "test-unlabelled"                                        # test-unlabelled, devset
json_path = "resource/test/{}.json".format(json_file)            

######PRE-TRAINED MODEL######
PASSAGE_SELECTION_MODEL_ESIM = 'trained_model/ESIM/CLF_26-05-2019--23-03-58/ESIM_model.h5'
PASSAGE_SELECTION_TKN_ESIM = 'trained_model/ESIM/CLF_26-05-2019--23-03-58/ESIM_tokenizer.pkl'

PASSAGE_SELECTION_MODEL_LSTM = 'trained_model/LSTM/CLF_26-05-2019--20-37-42/LSTM_model.h5'
PASSAGE_SELECTION_TKN_LSTM = 'trained_model/LSTM/CLF_26-05-2019--20-37-42/LSTM_tokenizer.pkl'

ENTAILMENT_RECOGNIZER_MODEL_ESIM = 'trained_model/ESIM/NLI/ESIM_model.h5'
ENTAILMENT_RECOGNIZER_TKN_ESIM = 'trained_model/ESIM/NLI/ESIM_tokenizer.pkl'
ENTAILMENT_RECOGNIZER_MODEL_LSTM = 'trained_model/LSTM/NLI/LSTM_model.h5'
ENTAILMENT_RECOGNIZER_TKN_LSTM = 'trained_model/LSTM/NLI/LSTM_tokenizer.pkl'
###### PARAMS TO CHANGE ######
verbose = True
# Inverted Index
entity = False                       # use entity title matching
page_ids_threshold = 15             # only return this many page ids from inverted index
inv_index_verbose = False
posting_limit = 1000                # limit to postings returned per term -- we could be missing a lot of page ids with the same tfidf scores. (1 term)

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
    # test_json = utils.load_json(json_path)          
    # raw_claims, page_ids = parse_test_json(test_json, output_page_ids=True)
    # print("[INFO] Number of unique claims: {}".format(len(raw_claims)))

    # ##### PAGE ID RETRIEVAL #####
    # # exact match entity linking
    # page_ids_string_dict = utils.load_pickle("page_ids_string_dict.pkl")


    # # get relevant page_ids from the inverted index

    # inv_index = InvertedIndex(verbose=inv_index_verbose)
    # wiki_query = WikiQuery()

    # total_test_claims = []
    # total_test_evidences = []
    # total_test_indices = []

    # total_true_page_ids_length = 0
    # total_true_pos = 0
    # for idx, raw_claim in tqdm(enumerate(raw_claims)):
    #     if entity: 
    #         print("[INFO - MAIN] Getting exact match entity links")
    #         matched = get_title_entity_match(raw_claim, page_ids_string_dict)


    #     print("[INFO - Main] Getting ranked page ids from inverted index...")
    #     if verbose:
    #         print("[INFO - Main] Claim: \n{}".format(raw_claim))
    #     ranked_page_ids = inv_index.get_ranked_page_ids(raw_claim, posting_limit=posting_limit, tfidf=False)        # tfidf nust be false for production
    #     ranked_page_ids = set(process_ranked_page_ids(ranked_page_ids, page_ids_threshold, verbose=verbose))
    #     if entity:
    #         ranked_page_ids = ranked_page_ids.union(matched)
        
    #     true_page_ids_length = len(page_ids[idx])
    #     if not true_page_ids_length <= 0:
    #         true_pos = 0
    #         for page_id in page_ids[idx]:
    #             if page_id in ranked_page_ids:
    #                 true_pos += 1

    #         percentage = float(true_pos)/float(true_page_ids_length)*100.0
    #         print("[DEBUG INFO] {}'%' returned, {}/{}".format(percentage, true_pos, true_page_ids_length))
    #         total_true_page_ids_length += true_page_ids_length
    #         total_true_pos += true_pos
    #         recall = float(total_true_pos)/float(total_true_page_ids_length)*100.0


    #     if verbose:
    #         print("[INFO - Main] Returned ranked page ids: \n{}".format(ranked_page_ids))

    #     test_claims, test_evidences, test_indices = get_passage_selection_data(raw_claim=raw_claim, 
    #                                                                            page_ids=ranked_page_ids, 
    #                                                                            query_object=wiki_query)

    #     total_test_claims.extend(test_claims)
    #     total_test_evidences.extend(test_evidences)
    #     total_test_indices.extend(test_indices)


    # avg_evidence_per_claim = float(len(total_test_evidences))/float(idx+1)
    # message = "Entity Linking: {}\nThreshold: {}, Recall: {}\nAvg evidences per claim: {}\n".format(entity,
    #                                             page_ids_threshold,
    #                                             recall,
    #                                             avg_evidence_per_claim)
    # utils.log(message, "inv_index_log.txt")


    # utils.save_pickle(total_test_claims, "test_{}_entity_{}_claims.pkl".format(json_file, entity))
    # utils.save_pickle(total_test_evidences, "test_{}_entity_{}_evidences.pkl".format(json_file, entity))
    # utils.save_pickle(total_test_indices, "test_{}_entity_{}_indices.pkl".format(json_file, entity))
   
    # # format into the proper format to be passed into the passage selection NN
    # claims, raw_evidences, page_info = get_training_data(claims_path='resource/training_data/test/test_devset_claims.pkl',
    #                                                      evidences_path='resource/training_data/test/test_devset_evidences.pkl',
    #                                                      labels_path='resource/training_data/test/test_devset_indices.pkl')
     
    ##### RELEVANT PASSAGE SELECTION #####

    # pass data into the sentence NN object
        # use an NN object
        # output: {page_id, passage_idx, classification, confidence}
    

    # test_set = pd.DataFrame({'claim':claims,'raw_evidence':raw_evidences,'evidence':page_info})
    print("start loading..")
    test_set = pd.read_csv('resource/unlabelled_test/test_set_filtered_50.csv')
    print("finish loading.")
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
        
    test_set['relevance'] = [p[1] for p in pred]
    relevant_test_set = test_set[test_set['relevance']>=0.8]
    relevant_test_set.to_csv('relevant_test_testset.csv')
    #manully set the threshold here
#    tmp = test_set[test_set['claim']=='Andrew Kevin Walker is only Chinese.']



    ##### ENTAILMENT RECOGNIZER #####
    # format into the proper format to be passed into the entailment recognizer NN

    # pass data into the entailment NN object
#     relevant_test_set_concatenate = concatenate_evidence_df(relevant_test_set)
    
#     sentences_pair = [(x1, x2) for x1, x2 in zip(relevant_test_set_concatenate.claim, \
#                       relevant_test_set_concatenate.evidence)]
    

#     if which_model == 'ESIM':
#         pred = get_model_prediction(model_dir=ENTAILMENT_RECOGNIZER_MODEL_ESIM,\
#                                     tkn_dir=ENTAILMENT_RECOGNIZER_TKN_ESIM,\
#                                     which_model='ESIM',sentences_pair=sentences_pair,\
#                                     left_sequence_length=one_sentence_length,\
#                                     right_sequence_length=multiple_sentence_length)
#     elif which_model == 'LSTM':
#         pred = get_model_prediction(model_dir=ENTAILMENT_RECOGNIZER_MODEL_LSTM,\
#                                     tkn_dir=ENTAILMENT_RECOGNIZER_MODEL_LSTM,\
#                                     which_model='LSTM',sentences_pair=sentences_pair,\
#                                     left_sequence_length=one_sentence_length,\
#                                     right_sequence_length=multiple_sentence_length)
        
#     else:
#         raise ValueError('Model Type Not Understood:{}'.format(which_model))
                                
#     pred = np.argmax(pred,axis=1)
    
#     relevant_test_set_concatenate['label'] = pred
    
    
    
#     ##### OUTPUT #####
#     # output format: {id: {'claim':CLAIMS,
# #                           'label': LABEL, 
#     #                       'evidence': [page_id, passage_idx]
#     #                      }}     
#     #                      #NOTE: Max 6 evidences allowed in score.py

#     relevant_test_set_concatenate[['claim','evidence','label']].to_json('testoutput.json',orient='index')
    



#### DOCUMENT SELECTION ####
def process_ranked_page_ids(ranked_page_ids, threshold, verbose):
    length = len(ranked_page_ids)
    if length <= 0:
        print("[INFO - Main] No relevant page id returned.")
        return 
    else:
        if length <= threshold+1:
            if verbose:
                print("[INFO - Main] Returned page_ids: {}".format(length))
            return ranked_page_ids
        else:
            if verbose:
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
def get_model_prediction(model_dir,tkn_dir,which_model,sentences_pair,left_sequence_length,\
                      right_sequence_length):
    """
    :param model_dir,tkn_dir: the directory to pre-trained model & tokenizer
    :param which_model, choose between 'ESIM' and 'LSTM
    :param sentences_pair: (claim, evidence) pair
    :param left_sequence_length: the max length of claims, default is 32
    :param right_sequence_length: the max lemgth of evidence. If not concatenate, set to default;
                            if evidence is concatenate, the length is 64
    """
    if which_model == 'ESIM':
        model = load_model(model_dir,custom_objects={'DotProductAttention':DotProductAttention})
        print("[INFO] ESIM model is loaded.")
        tokenizer = utils.load_pickle(tkn_dir)
        test_claim,test_evidence = create_test_data(tokenizer, sentences_pair, \
                                                left_sequence_length, right_sequence_length)
        pred = model.predict([test_claim,test_evidence],batch_size=1024,verbose=1 )

    elif which_model == 'LSTM':
        model = load_model(model_dir)
        print("[INFO] LSTM model is loaded.")
        tokenizer = utils.load_pickle(tkn_dir)
        test_claim,test_evidence,test_leaks = create_test_data_lstm(tokenizer, sentences_pair, \
                                                left_sequence_length, right_sequence_length)
        pred = model.predict([test_claim,test_evidence,test_leaks],batch_size=1024,verbose=1)

    else:
        raise ValueError('Model Type Not Understood:{}'.format(which_model))
    
    return pred


#### ENTAILMENT RECOGNIZER ####




#### JSON  ####
def parse_test_json(test_json, output_page_ids=False):
    """ Returns a list of the json values """
    test_array = []
    test_page_ids = []
    for test_data in test_json.values():
        test_array.append(test_data.get('claim'))
        page_ids = set()
        [page_ids.add(ev[0]) for ev in test_data.get('evidence')]
        # page_ids = [evidence[0] for evidence in test_data.get('evidence')]
        test_page_ids.append(page_ids)

    if output_page_ids:
        return test_array, test_page_ids

    return test_array
    
### utils ###
def concatenate_evidence_df(df):
    """
    concatenate the 'evidence' column in a dataframe.
    """
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
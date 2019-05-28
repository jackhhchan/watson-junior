#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:58:05 2019

@author: loretta
"""

from tqdm import tqdm

import utils
from keras.models import load_model
from NLI.attention import DotProductAttention
from NLI.prepare_set import create_test_data,create_test_data_lstm
import numpy as np
import pandas as pd
from NLI.train import get_training_data
import json

###### PATHS ######
json_path = "resource/test/test-unlabelled.json"            # test set
#json_path = "resource/devset.json"                    # dev set

######PRE-TRAINED MODEL######
PASSAGE_SELECTION_MODEL_ESIM = 'trained_model/ESIM/PassageSelection/CLF_26-05-2019--23-03-58/ESIM_model.h5'
PASSAGE_SELECTION_TKN_ESIM = 'trained_model/ESIM/PassageSelection/CLF_26-05-2019--23-03-58/ESIM_tokenizer.pkl'

PASSAGE_SELECTION_MODEL_LSTM = 'trained_model/LSTM/PassageSelection/CLF_26-05-2019--20-37-42/LSTM_model.h5'
PASSAGE_SELECTION_TKN_LSTM = 'trained_model/LSTM/PassageSelection/CLF_26-05-2019--20-37-42/LSTM_tokenizer.pkl'

ENTAILMENT_RECOGNIZER_MODEL_ESIM = 'trained_model/ESIM/NLI/ESIM_model.h5'
ENTAILMENT_RECOGNIZER_TKN_ESIM = 'trained_model/ESIM/NLI/ESIM_tokenizer.pkl'
ENTAILMENT_RECOGNIZER_MODEL_LSTM = 'trained_model/LSTM/NLI/LSTM_model.h5'
ENTAILMENT_RECOGNIZER_TKN_LSTM = 'trained_model/LSTM/NLI/LSTM_tokenizer.pkl'
###### PARAMS TO CHANGE ######
# Inverted Index
page_ids_threshold = 15             # only return this many page ids from inverted index
verbose = False
posting_limit = 1000                # limit to postings returned per term

# Passage Selection
PS_confidence_threshold = 0.75
passage_ids_threshold = None        # gets thresholded to 6 in score.py anyway
PS_which_model = 'LSTM' #{‘ESIM','LSTM}
one_sentence_length = 32
multiple_sentence_length = 64

# Entailment Recognizer
ER_confidence_threshold = 0.35         # maybe?
ER_which_model = 'LSTM'

def main():
#    ======prepare test_unlabelled json======
    test_json = utils.load_json(json_path) 
    test_standard = pd.DataFrame.from_dict(test_json, orient='index')
    test_standard = test_standard.reset_index()
      
#    =========================================
#    ============prepare test set=============
#    =========================================
    
    claims, raw_evidences, page_info = get_training_data(claims_path='resource/test/full test dev data -- suffix + syns/total_test_claims_0.pkl',
                                                         evidences_path='resource/test/full test dev data -- suffix + syns/total_test_evidences_0.pkl',
                                                         labels_path='resource/test/full test dev data -- suffix + syns/total_test_indices_0.pkl')
    
    test_set = pd.DataFrame({'claim':claims,'raw_evidence':raw_evidences,'evidence':page_info})
#    len(list(set(test_set.claim.values)))
#    test_set.any().isnull()
    test_set = test_set[test_set.apply(lambda row: len(row['raw_evidence'])>=4, axis=1)]
#    test_set['evidence'] = test_set.apply(lambda row: [row['evidence'][0],int(row['evidence'][1])],axis=1)
    test_set['pg_id'] = test_set.apply(lambda row: row['evidence'][0],axis=1)
    test_set['indices'] = test_set.apply(lambda row: row['evidence'][1],axis=1)
#    test_set.indices.describe()
#    test_set = test_set[test_set.apply(lambda row: int(row['indices'])<=50,axis=1)]
    
    indices_int = []
    for k,item in tqdm(test_set.iterrows()):
        try:
            if int(item['indices']) <= 20:
                indices_int.append(int(item['indices']))
            else:
                indices_int.append(-1)
        except:
            indices_int.append(-1)
        
    test_set['indices_int'] = indices_int
    test_set_filterd = test_set[test_set['indices_int']>-1]
    
    test_set_filterd['evidence'] = test_set_filterd.apply(lambda row: [row['pg_id'],row['indices_int']],axis=1)
    
    ##### RELEVANT PASSAGE SELECTION #####
    ##### RELEVANT PASSAGE SELECTION #####
    ##### RELEVANT PASSAGE SELECTION #####
    ##### RELEVANT PASSAGE SELECTION #####

    sentences_pair = [(x1, x2) for x1, x2 in zip(test_set_filterd.claim, test_set_filterd.raw_evidence)]

    
    if PS_which_model == 'ESIM':
        pred = get_model_prediction(model_dir=PASSAGE_SELECTION_MODEL_ESIM,\
                                    tkn_dir=PASSAGE_SELECTION_TKN_ESIM,\
                                    which_model='ESIM',sentences_pair=sentences_pair,\
                                    left_sequence_length=one_sentence_length,\
                                    right_sequence_length=one_sentence_length)
    elif PS_which_model == 'LSTM':
        pred = get_model_prediction(model_dir=PASSAGE_SELECTION_MODEL_LSTM,\
                                    tkn_dir=PASSAGE_SELECTION_TKN_LSTM,\
                                    which_model='LSTM',sentences_pair=sentences_pair,\
                                    left_sequence_length=one_sentence_length,\
                                    right_sequence_length=one_sentence_length)
        
    else:
        raise ValueError('Model Type Not Understood:{}'.format(PS_which_model))
    
#    pred = np.where(pred>0.9,1,0) # round up :)
#    pred = np.argmax(pred,axis=1)
#    test_set_filterd['relevance'] = np.argmax(pred,axis=1)
    test_set_filterd['relevance'] = [p[1] for p in pred]
#    len(list(set(test_set_filterd.claim.values)))
#    test_set_filterd.to_csv('test_set_filterd.csv')
    relevant_test_set = test_set_filterd[test_set_filterd['relevance']>=PS_confidence_threshold]
    
#    ============prepare Not enough info set============
    NEI_list = []
    for k,item in test_standard.iterrows():
        if item['claim'] not in list(relevant_test_set.claim.values) and not item['claim'] in NEI_list:
            NEI_list.append(item['claim'])
    NEI_df = pd.DataFrame({'claim':NEI_list})
    NEI_df['evidence'] = [ [] for i in range(len(NEI_list)) ]
    NEI_df['raw_evidence'] = [ [] for i in range(len(NEI_list)) ]
    NEI_df['label'] = 'NOT ENOUGH INFO'
    
    ##### ENTAILMENT RECOGNIZER #####
    ##### ENTAILMENT RECOGNIZER #####
    ##### ENTAILMENT RECOGNIZER #####
    ##### ENTAILMENT RECOGNIZER #####
    ##### ENTAILMENT RECOGNIZER #####
    
    relevant_test_set_concatenate = concatenate_evidence_df(relevant_test_set)
    
    sentences_pair = [(x1, x2) for x1, x2 in zip(relevant_test_set_concatenate.claim, \
                      relevant_test_set_concatenate.raw_evidence)]
    


    if ER_which_model == 'ESIM':
        pred = get_model_prediction(model_dir=ENTAILMENT_RECOGNIZER_MODEL_ESIM,\
                                    tkn_dir=ENTAILMENT_RECOGNIZER_TKN_ESIM,\
                                    which_model='ESIM',sentences_pair=sentences_pair,\
                                    left_sequence_length=one_sentence_length,\
                                    right_sequence_length=multiple_sentence_length)
    elif ER_which_model == 'LSTM':
        pred = get_model_prediction(model_dir=ENTAILMENT_RECOGNIZER_MODEL_LSTM,\
                                    tkn_dir=ENTAILMENT_RECOGNIZER_TKN_LSTM,\
                                    which_model='LSTM',sentences_pair=sentences_pair,\
                                    left_sequence_length=one_sentence_length,\
                                    right_sequence_length=multiple_sentence_length)
        
    else:
        raise ValueError('Model Type Not Understood:{}'.format(ER_which_model))
                                
#    pred = np.argmax(pred,axis=1)
    
    pred_argmax = []
    for l0,l1 in pred:
        if abs(l0-l1) < ER_confidence_threshold:
            pred_argmax.append(2)
        else:
            pred_argmax.append(np.argmax([l0,l1]))
            
    relevant_test_set_concatenate['label'] = pred_argmax
    
    #=========================================================
    #=========================================================
    #=====merge the prediction result back to the test file===
    #=========================================================
    #=========================================================
    relevant_test_set_concatenate['label'] = relevant_test_set_concatenate.\
            label.map({0:'SUPPORTS',1:'REFUTES',2:'NOT ENOUGH INFO'})
    result = pd.concat([relevant_test_set_concatenate,NEI_df])
#    relevant_test_set_concatenate.label.value_counts()
#    result.label.value_counts()
    
    result = pd.merge(test_standard,result,on='claim').set_index('index')
#    result.dropna(subset=['label_y'],inplace=True)    
    print("the distribution of labels:")
    print(result.label.value_counts())
    
    ##### OUTPUT #####
    
    result[['claim','evidence','label']].to_json('testoutput.json',orient='index')
    print("done!")

    actual = json.load(open(json_path))
    predicted = json.load(open('testoutput.json'))
    
    assert set(actual.keys()) == set(predicted.keys())
    
    
    
    
    
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
        pred = model.predict([test_claim,test_evidence],batch_size=512,verbose=1)

    elif which_model == 'LSTM':
        model = load_model(model_dir)
        print("[INFO] LSTM model is loaded.")
        tokenizer = utils.load_pickle(tkn_dir)
        test_claim,test_evidence,test_leaks = create_test_data_lstm(tokenizer, sentences_pair, \
                                                left_sequence_length, right_sequence_length)
        pred = model.predict([test_claim,test_evidence,test_leaks],batch_size=512,verbose=1)

    else:
        raise ValueError('Model Type Not Understood:{}'.format(which_model))
    
    return pred


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
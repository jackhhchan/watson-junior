import sys
sys.path.append(sys.path[0] + '/..')        # allow parent directory imports
from enum import Enum
from tqdm import tqdm

import utils
from mongodb.mongodb_query import WikiQuery

class Label(Enum):
    SUPPORTS = 'SUPPORTS'
    REFUTES = 'REFUTES'

    @staticmethod
    def list(): return list(map(lambda case: case.value, Label))

    @staticmethod
    def encode(label):
        assert label in Label.list(), "Label must be {}".format([label for label in Label.list()])
        encoder_dict = {
            Label.SUPPORTS.value : 0,
            Label.REFUTES.value : 1
            }
        return encoder_dict.get(label)


### Fields supplied in the .json files ####
class JSONField(Enum):
    # nested field identifiers
    claim = 'claim'
    evidence = 'evidence'
    label = 'label'


def main():
    """ From loading json file, connecting to db for evidence tokens, to saving as pkl"""
    wikiQuery = WikiQuery()

    ####### TRAINING SET ########
    # train_json = utils.load_json('resource/train/train.json')
    # supports_train_json, refutes_train_json = parse_train_json(train_json)

    # print("Number of SUPPORTS: {}".format(len(supports_train_json)))
    # print("Number of REFUTES: {}".format(len(refutes_train_json)))

    # train_claims, train_evidences, train_labels = generate_data(json_file=refutes_train_json,
    #                                                             query_object=wikiQuery,
    #                                                             concatenate=False,
    #                                                             threshold=None)    # save_pickle(train_claims, 'train_claims_refutes.pkl')
    # save_pickle(train_evidences, 'train_evidences_refutes.pkl')
    # save_pickle(train_labels, 'train_labels_refutes.pkl')

    # train_claims, train_evidences, train_labels = generate_data(json_file=supports_train_json,
    #                                                             query_object=wikiQuery,
    #                                                             concatenate=False,
    #                                                             threshold=len(refutes_train_json))    # downsample to the same as refutes
    # save_pickle(train_claims, 'train_claims_supports_downsampled.pkl')
    # save_pickle(train_evidences, 'train_evidences_supports_downsampled.pkl')
    # save_pickle(train_labels, 'train_labels_supports_downsampled.pkl')


    ##### DEVELOPMENT SET #####
    # dev_json = load_json('resource_train/devset.json')
    # dev_array = parse_json(dev_json, False)
    # dev_claims, dev_evidences, dev_labels = generate_data(dev_array, False, len(dev_array))
    # save_pickle(dev_claims, 'dev_claims.pkl')
    # save_pickle(dev_evidences, 'dev_evidences.pkl')
    # save_pickle(dev_labels, 'dev_labels.pkl')



def generate_data(data_json, query_object, concatenate, threshold=None):
    """ Compile the training data according to the json dictionary identifier (train.json)
    
    Arguments:
    ----------
    data_json
        dictionary containing the training data identifiers (e.g. loaded from train.json)
    query_object
        the collection query object imported from mongodb_query.py
    cocatenate (Bool)
        Takes a boolean, if true then evidences are cocatenated else other wise.
    threshold (Int)
        The threshold for downsampling.
    Returns:
    ----------
    train_claims
        A list of claims
    """

    if threshold is None:
        threshold = len(data_json)

    train_claims = []
    train_evidences = []
    train_labels = []

    print("[INFO] Extracting training data from db...")
    for idx, data in tqdm(enumerate(data_json)):
        label = data[JSONField.label.value]
        claim = data[JSONField.claim.value]
        evidences = data[JSONField.evidence.value]

        if label == 'NOT ENOUGH INFO': continue     # skip data points with no evidences

        label = Label.encode(label)                # encode label, SUPPORTS=0, REFUTES=1
        if concatenate:
            # evidences tokens are concatenated
            train_claims.append(claim)
            train_labels.append(label)
            concatenated_tokens = []
            [concatenated_tokens.extend([get_tokens_from_db(evidence, query_object) for evidence in evidences])]
            train_evidences.append(concatenated_tokens)        
        else:
            # evidences tokens are not concatenated
            for evidence in evidences:
                tokens = get_tokens_from_db(evidence, query_object)
                train_evidences.append(tokens)
                train_claims.append(claim)          # these two appends must be after token, for db check
                train_labels.append(label)
        
        if idx >= threshold:            # threshold for downsampling
            break    

    print("[INFO] complete.")
    return train_claims, train_evidences, train_labels

                        

def get_tokens_from_db(evidence, query_object):
    page_id = evidence[0]
    passage_idx = evidence[1]
    doc = query_object.query(page_id=page_id, passage_idx=passage_idx)
    # returned doc logger
    if doc is None:
        message = "[DB] database returned None for page_id: {}, passage_idx: {}".format(page_id, passage_idx)
        utils.append_logger(message)
        return None

    # returned tokens logger
    tokens = doc.get('tokens')
    if tokens is None:
        message = "[DB] tokens returned None for page_id: {}, passage_idx: {}".format(page_id, passage_idx)
        utils.append_logger(message)

    # cocatenate the split tokens to a single string
    tokens_string = ''
    for token in tokens:
        tokens_string = tokens_string + token + ' '
        
    return tokens_string


def parse_json(json_file, separate):

    if not separate:
        dev_jsons = []
        for key in tqdm(json_file.keys()):
            label = json_file.get(key).get('label')
            if label == Label.SUPPORTS.value or label == Label.REFUTES.value:
                dev_jsons.append(json_file.get(key))
            else:
                continue
        return dev_jsons
    else:
        supports_train_json = []
        refutes_train_json = []
        for key in tqdm(json_file.keys()):
            label = json_file.get(key).get('label')
            if label == Label.SUPPORTS.value:
                supports_train_json.append(json_file.get(key))
            elif label == Label.REFUTES.value:
                refutes_train_json.append(json_file.get(key))
            else:
                continue        # skip NOT ENOUGH INFO

    return supports_train_json, refutes_train_json


if __name__ == '__main__':
    main()
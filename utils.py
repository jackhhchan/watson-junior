""" Utility functions used for watson-junior

This module contains a list of helper functions to be used by watson-junior

"""
from enum import Enum
import os

import pickle
import re
import json
import time
from datetime import datetime

class encoding(Enum):
    UTF8 = "UTF-8"


def load_file(f_path, encoding=encoding.UTF8.name):
    """ Load text file from path and return raw array of split strings"""

    raw_lines = []
    with open(f_path, 'r', encoding=encoding) as f_handle:
        for raw_line in f_handle:
            # split line into tokens
            raw_lines.append(raw_line.split())

    print("[INFO] Extracted {}, len(raw_lines) = {}".format(f_path, len(raw_lines)))
    return raw_lines


def extract_tokens(passage):
    """ Extract lower case tokens from the passage """
    tokens = []
    for token in passage:
        token_s = re.split("-|\\s", token.lower())              # split up tokens that are hyphenated, may return a list
        token_s = [t for t in token_s if t.isalnum()]           # keep only if token is alphanumerical
        tokens.extend(token_s)

    return tokens

########## PICKLE ########## 
def save_pickle(obj, name):
    assert name.endswith('.pkl')
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle)

def load_pickle(name):
    assert name.endswith('.pkl')
    with open(name, 'rb') as handle:
        data = pickle.load(handle)

    return data


########## JSON ##########
def append_json(name, function):
    with open(name, 'a') as handle:
        function(handle)

def load_json(json_path):
    # parse train.json file
    assert json_path.endswith('.json')
    try:
        with open(json_path, 'r') as handle:
            json_data = json.load(handle)
    except:
        print("Unable to load {}".format(json_path))

    return json_data



########## LOGGING ##########
def log(string):
    """ Append to logger in logs directory"""
    folder_dir = "logs"
    if not os.path.isdir(folder_dir):
        os.makedirs(folder_dir)

    with open("{}/logs.txt".format(folder_dir), 'a') as handle:
        string = "{}  -  {}\n".format(get_timestamp(), string)
        handle.write(string)

def get_timestamp():
    """ Returns timestamp with format e.g.'2019-05-24T00:30:02.438162'"""
    return datetime.now().isoformat()

def get_time():
    return time.time()

def get_elapsed_time(function):
    """ Return elapsed time of completing the function"""
    start = time.time()
    function()
    end = time.time()
    return "{} seconds".format(end-start)

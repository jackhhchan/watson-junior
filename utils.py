""" Utility functions used for watson-junior

This module contains a list of helper functions to be used by watson-junior

"""
from enum import Enum
import os
import sys

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
            raw_lines.append(raw_line)

    print("[INFO] Extracted {}, len(raw_lines) = {}".format(f_path, len(raw_lines)))
    return raw_lines


########## USEFUL FUNCTIONS #########
def sorted_dict(dictionary):
    """ Return array of sorted dictionary based on value and return in tuple format (key, value) """
    sorted_dictionary = sorted(dictionary.items(), key=lambda kv:kv[1], reverse=True)
    return sorted_dictionary


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
def log(string, f_name="logs.txt"):
    """ Append to logger in logs directory"""
    folder_dir = "logs"
    if not os.path.isdir(folder_dir):
        os.makedirs(folder_dir)

    with open("{}/{}".format(folder_dir, f_name), 'a') as handle:
        string = "{}  -  {}\n".format(get_timestamp(), string)
        handle.write(string)


###### TIMER ########
def get_timestamp():
    """ Returns timestamp with format e.g.'25-05-2019--01-02-14'"""
    return datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

def get_time():
    return time.time()

def get_elapsed_time_in(function):
    """ Return elapsed time of completing the function"""
    start = time.time()
    function()
    end = time.time()
    return "{} seconds elapsed".format(end-start)

def get_elapsed_time(start, end):
    return "{} seconds elapsed".format(end-start)


######## Useful Utilities ########
def reverse_key_value_dict(orig_dict):
    """ Return a dictionary with key and value reversed """
    return dict([v,k] for k,v in orig_dict.items())

def get_size(obj, verbose=False, obj_name=""):
    size = sys.getsizeof(obj)
    if verbose:
        print("{} bytes {}".format(size, obj_name))
    return size
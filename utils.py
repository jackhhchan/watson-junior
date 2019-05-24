""" Utility functions used for watson-junior

This module contains a list of helper functions to be used by watson-junior

"""
from enum import Enum

from nltk.tokenize import word_tokenize
import pickle
import re


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


def save_pickle(obj, name):
    assert name.endswith('.pkl')
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(name):
    assert name.endswith('.pkl')
    with open(name, 'rb') as handle:
        data = pickle.load(handle)

    return data
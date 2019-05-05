""" Utility functions used for watson-junior

This module contains a list of helper functions to be used by watson-junior

"""
from enum import Enum

class encoding(Enum):
    UTF8 = "UTF-8"



def load_file(f_path, encoding=encoding.UTF8.name):
    """ Load text file from path and return raw array of split strings"""

    raw_lines = []
    f_handle = open(f_path, 'r', encoding=encoding)
    for raw_line in f_handle:
        # split line into tokens
        raw_lines.append(raw_line.split())

    print("[INFO] Extracted {}, len(raw_lines) = {}".format(f_path, len(raw_lines)))
    return raw_lines


def extract_tokens(passage):
    """ Extract lower case tokens from the passage """

    return [token.lower() for token in passage]

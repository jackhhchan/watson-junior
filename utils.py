""" Utility functions used for watson-junior

This module contains a list of helper functions to be used by watson-junior

"""


def load_file(f_path):
    """ Load text file from path and return raw array of split strings"""

    raw_lines = []
    f_handle = open(f_path, 'r', encoding='UTF-8')
    for raw_line in f_handle:
        # split line into tokens
        raw_lines.append(raw_line.split())

    print("[INFO] Extracted {}, len(raw_lines) = {}".format(f_path, len(raw_lines)))
    return raw_lines


def extract_tokens(passage):
    """ Extract non-numeric tokens from the passage """

    return [token.lower() for token in passage if token.isalpha()]

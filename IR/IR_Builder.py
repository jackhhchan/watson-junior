import sys
sys.path.append(sys.path[0] + "/..")
import os

import utils
from IR.IR_utils import int_encode

FOLDER_NAME = ""

def parse_wiki_docs(folder_name=FOLDER_NAME):
    """ Parse wiki docs and return pages dictionary; {page_id: Page object}
        
        This method loads and parses wiki docs according to their structure. 
        It first extract page_id, passage_idx, tokens to build Passage object,
        then append Passage object (value) to the passages list in its corresponding Page object.
    """
    pages = {}      # {page_id, Page object}
    page_idx_id_mapper = {}     # {page_idx : page_id}
    page_idx = int_encode(0)
    for file_name in os.listdir(folder_name):
        # load file
        path = "{}/{}".format(folder_name, file_name)
        raw_lines = utils.load_file(path)
        # loop through raw lines in file
        for raw_line in raw_lines:
            page_id, passage_idx, tokens = parse_raw_line(raw_line)
            
            



def parse_raw_line(raw_line):
    """ Extract page id, passage index and tokens from raw line to build Passage objects 

        Returns:
        --------
        page_id
            the page id of the line in a string
        passage_idx
            the passage index of the line in a string
        tokens
            a list of tokens associated with the passage (lower-cased)
    """
    page_id = raw_line[0]
    passage_idx = raw_line[1]
    tokens = utils.extract_processed_tokens(raw_line[2:])

    return page_id, passage_idx, tokens
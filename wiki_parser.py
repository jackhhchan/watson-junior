import os

import utils
from InvertedIndex import Page
from InvertedIndex import Passage



def parse_wiki_docs(folder_name=FOLDER_NAME):
    """ Parse wiki docs and return pages dictionary; {page_id: Page object}
        
        This method loads and parses wiki docs according to their structure. 
        It first extract page_id, passage_idx, tokens to build Passage object,
        then append Passage object (value) to the passages list in its corresponding Page object.
    """
    pages = {}      # {page_id, Page object}
    for file_name in os.listdir(folder_name):
        # load file
        path = "{}/{}".format(folder_name, file_name)
        raw_lines = utils.load_file(path)
        # loop through raw lines in file
        for raw_line in raw_lines:
            page_id, passage_idx, tokens = parse_raw_line(raw_line)
            passage = Passage(page_id, passage_idx, tokens)         # instantiate new passage objet

            pages[page_id] = pages.get(page_id, Page(page_id))      # instantiate page if not exist
            pages[page_id].passages.append(passage)                 # append passage to page's passages list

    return pages
            

def parse_raw_line(raw_line):
    """ Extract page id, passage index and tokens from raw line to build Passage objects 
    
        Helper function
    """
    page_id = raw_line[0]
    passage_idx = raw_line[1]
    tokens = utils.extract_tokens(raw_line[2:])

    return page_id, passage_idx, tokens



#### PATHS ####
FOLDER_NAME = "resource"

FILE_NAME = "wiki-001.txt"          # for testing
PATH = "{}/{}".format(FOLDER_NAME, FILE_NAME) 
###################
import sys
sys.path.append(sys.path[0] + '/..')
import os

from tqdm import tqdm
from nltk.stem import PorterStemmer

import utils
from IR.InvertedIndexBuilder import Page
from IR.InvertedIndexBuilder import Passage



def parse_wiki_docs(folder_name):
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
            

def parse_raw_line(raw_line, log_file="wiki-parser.txt"):
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
    raw_line = raw_line.split()
    page_id = raw_line[0]
    passage_idx = raw_line[1]
    tokens = preprocess_tokens_list(raw_line[2:])

    try:
        passage_idx = int(passage_idx)
    except:
        message = "Passage idx {} can't be converted to integer in page id {}".format(passage_idx, page_id)
        utils.log(message, log_file)

    # assert type(passage_idx) == int, "Passage index is not an integer."

    return page_id, passage_idx, tokens

def preprocess_tokens_list(tokens_list, stem=False):
    """ Returns page_id, passage_idx, tokens from raw passage
        
        Preprocessing for each token includes
        - Remove punctuation
        - Lower casing
        - Porter Stemming (optional, args: stem)
    """
    if stem:
        ps = PorterStemmer()
    tokens = []
    for token in tokens_list:
        token_s = substitute_punctuations(token).split()      # remove punctuations, may return list
        if stem:
            token_s = [ps.stem(t.lower()) for t in token_s]   # lower + stem
        else:
            token_s = [t.lower() for t in token_s]            # lower only
        tokens.extend(token_s)

    return tokens

def substitute_punctuations(s, sub=' '):
    """ Return string with all puncuations substituted with (default=' ') """
    return re.sub(r'[^\w\s]',sub,s)




####                 ####
#### Doc-term matrix ####
####                 ####

import re


def get_page_ids_term_freq_dicts(folder_name=FOLDER_NAME_DOC_TERM_MATRIX):
    """ Parses the wiki docs and extract and process the page-ids only.
        
    
        returns:
        page_ids_term_freq_dicts --  list of term frequencies dictionaries of each page-id
    """
    page_ids_term_freq_dicts = []       # to be returned
    page_idx = 0
    page_idx_id_dict = {}               # {page_idx: page_id}

    # seen_page_ids = list
    for file_name in os.listdir(folder_name):
        if not file_name.endswith(".txt"):
            continue
        path = "{}/{}".format(folder_name, file_name)
        raw_lines = utils.load_file(path)
        for raw_line in tqdm(raw_lines):
            page_id, _, _ = parse_raw_line(raw_line)

            if page_idx_id_dict.get(page_id) is None:
                page_idx_id_dict[page_id] = page_idx
                page_idx += 1

                page_id_tokens = re.split('_|-', page_id)
                page_id_tokens = [token.lower() for token in page_id_tokens if token.isalpha() and not token == "LRB" and not token == "RRB"]

                page_ids_term_freq_dicts.append(get_BOW(page_id_tokens))
    
    page_idx_id_dict = {page_idx:page_id for page_id, page_idx in page_idx_id_dict.items()}         # reverse dictionary keys to page_idx : page_id for ease of use.

    return page_ids_term_freq_dicts, page_idx_id_dict


def get_BOW(tokens_list):
    """ return bag of words in tokens list with words lowered."""
    BOW = {}
    for word in tokens_list:
        BOW[word.lower()] = BOW.get(word.lower(), 0) + 1            # should this be binary for doc-term-matrix? i.e. either 0 or 1 not raw count
    return BOW

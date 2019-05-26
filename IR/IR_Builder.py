import sys
sys.path.append(sys.path[0] + "/..")
import os
import re

from nltk.stem import PorterStemmer

import utils
from utils import load_pickle, save_pickle, get_size
from IR.IR_utils import int_encode

FOLDER_NAME = ""
stem = True                 # stem tokens?

def main():
    pages, page_id_idx_mapper, page_idx_id_mapper = parse_wiki_docs(folder_name=FOLDER_NAME) 
    save_pickle(pages, "pages.pkl")
    save_pickle(page_id_idx_mapper, "page_id_idx_mapper.pkl")
    save_pickle(page_idx_id_mapper, "page_idx_id_mapper.pkl")

    get_size(page_id_idx_mapper, verbose=True)
    get_size(page_idx_id_mapper, verbose=True)


def parse_wiki_docs(folder_name):
    """ Parse wiki docs and return pages dictionary; {page_id: Page object}
        
        This method loads and parses wiki docs according to their structure. 
        It first extract page_id, passage_idx, tokens to build Passage object,
        then append Passage object (value) to the passages list in its corresponding Page object.
    """
    pages = {}                  # {page_idx: {passage_idx: [tokens]}, ...}
    page_id_idx_mapper = {}     # {page_id: page_idx, ...}
    page_idx_id_mapper = {}     # {page_idx: page_id, ...}
    page_idx_iter = int_encode(0)                                    # encoded page idx
    for file_name in os.listdir(folder_name):
        # load file
        path = "{}/{}".format(folder_name, file_name)
        raw_lines = utils.load_file(path)
        # loop through raw lines in file
        for raw_line in raw_lines:
            page_id, passage_idx, tokens = parse_raw_line(raw_line, stem=stem)
            passage_idx = int_encode(passage_idx)                    # encoded passage idx
            
            # populate mappers
            if page_id_idx_mapper.get(page_id) is None:
                page_id_idx_mapper[page_id] = page_idx_iter          # map both idx->id and id->idx
                page_idx_id_mapper[page_idx_iter] = page_id          
                page_idx_iter = int_encode(page_idx_iter + 1)        # encoded page idx iterator
            
            # populate pages
            cur_page_idx = page_id_idx_mapper.get(page_id)
            if pages.get(cur_page_idx) is None:
                pages[cur_page_idx] = {passage_idx: tokens}
            else:
                pages[cur_page_idx].update({passage_idx: tokens})

    
    return pages, page_id_idx_mapper, page_idx_id_mapper
    


def parse_raw_line(raw_line, stem=True):
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
    raw_line = raw_line.split()        # substitute punctuations with blank space
    page_id = raw_line[0]
    passage_idx = raw_line[1]
    tokens = preprocess_tokens_list(raw_line[2:], stem=True)

    try:
        passage_idx = int(passage_idx)
    except:
        message = "Passage idx {} can't be converted to integer for page id {}".format(passage_idx, page_id)
        utils.log(message, "invidxbuilder.txt")

    assert type(passage_idx) == int, "Passage index is not an integer."

    return page_id, passage_idx, tokens


def preprocess_tokens_list(tokens_list, stem=True):
    """ Returns page_id, passage_idx, tokens from raw passage
        
        Preprocessing includes 
        - Porter Stemming
        - Lower casing
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


if __name__=="__main__":
    main()
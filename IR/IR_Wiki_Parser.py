import sys
sys.path.append(sys.path[0] + "/..")
import os
import re

from tqdm import tqdm
from nltk.stem import PorterStemmer

# INDEPENDENT MODULE -- only uses utils; but used by InvertedIndex.py
import utils
from utils import load_pickle, save_pickle, get_size
from IR.IR_utils import int_encode

FOLDER_NAME = "resource/wiki-txt-files"
CHECKPOINT_FOLDER = "resource/builder_checkpoint"
COMPLETED_FILES = "completed_files"
stem = True                 # stem tokens?
log_file = "IR_Builder_log.txt"



def main():
    # logging
    starter_message = "IR_Builder.py -- main() initiatied; Stemming = {}".format(stem)
    utils.log(starter_message, log_file)

    # start variables
    pages = {}
    page_id_idx_mapper = {}
    page_idx_id_mapper = {}
    page_idx_iter = 398342
    completed_files = []
    
    completed_path = "{}/{}.pkl".format(CHECKPOINT_FOLDER, COMPLETED_FILES)
    if os.path.exists(completed_path):
        completed_files = load_pickle(completed_path)
        last_file = str(completed_files[-1])
        print("[INFO] Loading checkpoint from {}...".format(last_file))
        pages, page_id_idx_mapper, page_idx_id_mapper, page_idx_iter = load_checkpoint_array("{}/{}".format(
                                                                                            CHECKPOINT_FOLDER,
                                                                                            last_file
                                                                                            ))

     # parse all wiki docs
    pages, page_id_idx_mapper, page_idx_id_mapper = parse_wiki_docs(folder_name=FOLDER_NAME,
                                                                    pages=pages,
                                                                    page_id_idx_mapper=page_id_idx_mapper,
                                                                    page_idx_id_mapper=page_idx_id_mapper,
                                                                    page_idx_iter=page_idx_iter,
                                                                    completed_files=completed_files) 

    # save final pickle files
    save_pickle(pages, "pages.pkl")
    save_pickle(page_id_idx_mapper, "page_id_idx_mapper.pkl")
    save_pickle(page_idx_id_mapper, "page_idx_id_mapper.pkl")

    get_size(page_id_idx_mapper, verbose=True)
    get_size(page_idx_id_mapper, verbose=True)


def parse_wiki_docs(folder_name, pages, page_id_idx_mapper, page_idx_id_mapper, page_idx_iter, completed_files):
    """ Parse wiki txt files within folder and return pages dictionary; {page_id: Page object}
        
        This method loads and parses wiki docs according to their structure. 
        It first extract page_id, passage_idx, tokens to build Passage object,
        then append Passage object (value) to the passages list in its corresponding Page object.
    """
    pages = pages                  # {page_idx: {passage_idx: [tokens]}, ...}
    page_id_idx_mapper = page_id_idx_mapper     # {page_id: page_idx, ...}
    page_idx_id_mapper = page_idx_id_mapper     # {page_idx: page_id, ...}
    page_idx_iter = int_encode(page_idx_iter)                       # encoded page idx iterator
    completed_files = completed_files
    for file_name in os.listdir(folder_name):
        tmp = file_name.rstrip('.txt')
        tmp = tmp + '.pkl'
        if tmp in completed_files: continue           # skip to checkpoint
        # load file
        path = "{}/{}".format(folder_name, file_name)
        raw_lines = utils.load_file(path)
        # loop through raw lines in file
        for raw_line in tqdm(raw_lines):
            # Structure relevant information 
            page_id, passage_idx, tokens = parse_raw_line(raw_line, stem=stem)
            if not type(passage_idx) == int:
                message = "Page_id {}, Passage_idx {} skipped.".format(page_id, passage_idx)
                utils.log(message, log_file)
                continue 
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

        
        # -- Checkpoint
        print("[INFO] Writing checkpoint to disk...")
        save_array = [pages, page_id_idx_mapper, page_idx_id_mapper, page_idx_iter]
        save_path = "{}/{}.pkl".format(CHECKPOINT_FOLDER,file_name.rstrip('.txt'))
        if not os.path.isdir(CHECKPOINT_FOLDER): os.mkdir(CHECKPOINT_FOLDER)
        save_pickle(save_array, save_path)

        message = ("{} successfully parsed, saved to {}\n"
            "Format: [pages, page_id_idx_mapper, page_idx_id_mapper, page_idx_iter]").format(file_name, 
                                                                                             save_path)
        utils.log(message, log_file)
        
        tmp =  file_name.rstrip('.txt')
        tmp = tmp + '.pkl'
        completed_files.append(tmp)
        save_pickle(completed_files, "{}/{}.pkl".format(CHECKPOINT_FOLDER, COMPLETED_FILES))

    
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
        message = "Passage idx {} can't be converted to integer in page id {}".format(passage_idx, page_id)
        utils.log(message, log_file)

    # assert type(passage_idx) == int, "Passage index is not an integer."

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



 


def load_checkpoint_array(path):
    checkpoint_array = load_pickle(path)
    assert len(checkpoint_array) == 4, "{} checkpoint array is not length size 4."
    pages = checkpoint_array[0]
    page_id_idx_mapper = checkpoint_array[1]
    page_idx_id_mapper = checkpoint_array[2]
    page_idx_iter = checkpoint_array[3]
    assert len(page_id_idx_mapper) == page_idx_iter, "Page_idx_iter is invalid, check mapper length" # might need to rebuild

    return pages, page_id_idx_mapper, page_idx_id_mapper, page_idx_iter

if __name__=="__main__":
    main()
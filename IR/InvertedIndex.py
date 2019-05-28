import sys
sys.path.append(sys.path[0] + '/..')
import re
from itertools import chain

import pickle
import json
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
# import nltk
# nltk.download('wordnet')

# import nltk
# nltk.download("stopwords")

import utils
from IR.NER import get_NER_tokens
from IR.IR_Wiki_Parser import preprocess_tokens_list
from mongodb.mongodb_query import InvertedIndexQuery

class InvertedIndex(object):
    def __init__(self, verbose=False):
        self.db_query = InvertedIndexQuery()
        self.stemmer = PorterStemmer()
        self.verbose = verbose
    
    #########################
    #### RANKED PAGE IDS ####
    #########################

    def get_ranked_page_ids(self, raw_claim, posting_limit=None, tfidf=False):
        """ 
        Returns a list of ranked page ids given claim
        Args:
        raw_claim   -- (Str) claim in a single string format
        tfidf       -- (Bool) Set True to return the tfidf as well. (default: False)
        """
        assert type(tfidf) == bool, "tfidf argument if True also returns the tfidf"

        claim_terms, NER_tokens, synonyms = self.query_reformulated(raw_claim)
        page_ids_tfidf = self.get_page_ids_tfidf(claim_terms, posting_limit, NER_tokens, synonyms)
        ranked_page_ids_tfidf = self.ranked_page_ids_tfidf(page_ids_tfidf)      # array of dicts in order

        if tfidf:
            return ranked_page_ids_tfidf
        else:
            ranked_page_ids = [page_id for (page_id, tfidf) in ranked_page_ids_tfidf]
            return ranked_page_ids


    def get_page_ids_tfidf(self, claim_terms, limit, NER_tokens, synonyms):
        output ={}
        if self.verbose:
            print("[INFO - InvIdx] Query: {}".format(claim_terms))
            print("[INFO - InvIdx] Number of tokens in claim: {}".format(len(claim_terms)))

        for term in claim_terms:
            postings = list(self.db_query.get_postings(term=term, limit=limit, verbose=self.verbose))
            if len(postings) <= 0:
                continue

            for i, posting in enumerate(postings):
                page_id = posting.get(self.db_query.InvertedIndexField.page_id.value)
                if term in NER_tokens:
                    tfidf = posting.get(self.db_query.InvertedIndexField.tfidf.value)*1.5
                else:
                    if term in synonyms:
                        tfidf = posting.get(self.db_query.InvertedIndexField.tfidf.value)*0.5
                    else:
                        tfidf = posting.get(self.db_query.InvertedIndexField.tfidf.value)

                output[page_id] = output.get(page_id, 0) + tfidf
                if i >= limit-1:
                    continue
        return output

    def ranked_page_ids_tfidf(self, page_ids_tfidf):
        return utils.sorted_dict(page_ids_tfidf)

    ##############################
    ##### QUERY REFORMULATION ####
    ##############################

    def query_reformulated(self, raw_claim):
        """ Reformulates the query, NER & (Query Expansion //TODO) """
        # raw_claim = self.remove_punctuations(raw_claim, string=True)        # return in string format

        tokens_list = self.IR_Builder_formatter(raw_claim)
        tokens_string = self.concatenate(tokens_list)
        
        # named entities linking
        NER_tokens = self.get_named_entities(tokens_string)
        # NER_tokens = self.remove_duplicates(NER_tokens)         # NER sometimes return duplicates

        # handle stop words
        NER_tokens = self.removed_stop_words(NER_tokens)

        tokens_list.extend(NER_tokens)
        tokens = self.removed_stop_words(tokens_list)
        tokens = self.remove_duplicates(tokens)

        ## TODO -- QUERY EXPANSION
        processed_claim_tokens = []
        synonyms = []
        # for token in tokens:
        #     syns = self.get_synonyms(token)
        #     synonyms.extend(syns)
        #     processed_claim_tokens.extend(syns)
        
        for tokens in processed_claim_tokens:
            processed_claim_tokens.extend(self.add_suffixes(token))
        
        return set(processed_claim_tokens), set(NER_tokens), set(synonyms)

    def IR_Builder_formatter(self, raw_claim):
        """ Return format used to build the TFIDF"""
        raw_claim = raw_claim.split()
        tokens_list = preprocess_tokens_list(raw_claim, stem=False)
        return tokens_list

    def remove_punctuations(self, raw_claim, string=True):
        """ Returns raw claim with removed punctuations in a string format (unless specified otherwise)"""
        raw_claim = re.split('-|_|\\s', raw_claim)
        return " ".join(token for token in raw_claim if token.isalnum()).strip()


    def get_named_entities(self, raw_claim):
        """ Returns the named entities outputted by the AllenNLP NER"""
        raw_NER_tokens = get_NER_tokens(raw_claim)
        if len(raw_NER_tokens) <= 0:
            message = "Nothing returned from NER for claim: {}".format(raw_claim)
            utils.log(message)

        NER_tokens = []
        for raw_NER_token in raw_NER_tokens:
            NER_tokens.extend(re.split('\\s', raw_NER_token))
        
        return NER_tokens
    
    def removed_stop_words(self, tokens):
        stop_words = stopwords.words('english')
        removed = [token for token in tokens if token not in stop_words]
        return removed

    def remove_duplicates(self, tokens):
        unique = []
        for token in tokens:
            if token not in unique:
                unique.append(token)
        return unique
    
    def concatenate(self, tokens_list):
        tokens_string = ''
        for token in tokens_list:
            tokens_string = tokens_string + token + ' '
            
        return tokens_string
    
    def add_suffixes(self, token):
        suffixes = ['s', 'ed', 'acy', 'al', 'ance', 'ence', 'en', 'fy', 'ify', 'able', 'ible', 'al']
        expanded = []
        expanded.append(token)      # keep original token
        token = self.stemmer.stem(token)
        expanded.append(token)
        for suf in suffixes:
            expanded.append(token+suf)
        return expanded

    def get_synonyms(self, word):
        syn_sets = wordnet.synsets(word)
        # syn_sets = set(chain.from_iterable([word.lemma_names() for word in syn_sets]))
        syns = []
        for word in syn_sets:
            syns.extend(word.lemma_names())
        synonyms = set()
        for syn in syns:
            if self.remove_syn_doubles(syn): continue
            # syn = self.substitute_puncs(syn).split()
            # for s in syn:
            #     synonyms.add(s)
            synonyms.add(syn)
        return synonyms
    
    def substitute_puncs(self, word):
        return re.sub(r'[^\w\s|]|_|-',' ', word)
    
    def remove_syn_doubles(self, word):
        if '_' in word:
            return True
        return False







################ DATABASE TEST ####################
def test():
    """ Test inverted index with a pre-defined raw claim """

    claims = [
        "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company",
        "Roman Atwood is a content creator",
        "Nikolaj Coster Waldau Fox Broadcasting Company",
        "Nikolaj Coster-Waldau",
        "Fox Broadcasting Company"
    ]

    # Databatase query object
    db_query = InvertedIndexQuery()

    
    raw_claim = claims[2]               # CHANGE THIS
    claim = raw_claim.lower().split()
    print("[INFO] Number of tokens in claim: {}".format(len(claim)))

    # return all relevant page ids and their tfidf scores
    output ={}
    for term in claim:
        postings = db_query.get_postings(term=term, verbose=True)
        for posting in postings:
            page_id = posting.get(db_query.InvertedIndexField.page_id.value)
            tfidf = posting.get(db_query.InvertedIndexField.tfidf.value)
            output[page_id] = output.get(page_id, 0) + tfidf
    
    print(list(output)[:19])            # NOTE: This is not ranked!

if __name__ == '__main__':
    print(utils.get_elapsed_time(test))
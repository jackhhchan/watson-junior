import pickle

INVERTED_INDEX_FNAME = "inverted_index_tf_normalized_proper_fixed.pkl"
PAGE_IDS_IDX_DICT_FNAME = "page_ids_idx_dict_normalized_proper_fixed.pkl"

def main():

    inverted_index = pickle.load(open(INVERTED_INDEX_FNAME, 'rb'))
    page_ids_idx_dict = pickle.load(open(PAGE_IDS_IDX_DICT_FNAME, 'rb'))

    claim = "Nikolaj Coster Waldau worked with the Fox Broadcasting Company"
    claim = "Roman Atwood is a content creator"
    claim = "Nikolaj Coster Waldau Fox Broadcasting Company"
    claim = claim.lower().split()

    output = {}
    for word in claim:
        print("{}".format(word))
        page_indices = inverted_index[word]
        for page_idx in page_indices:
            page_id = page_ids_idx_dict.get(page_idx)
            output[page_id] = output.get(page_id, 0) + inverted_index[word][page_idx]

    sorted_output = sorted(output.items(), key=lambda kv:kv[1], reverse=True)
 
    print(sorted_output)



main()
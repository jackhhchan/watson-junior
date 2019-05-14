import wiki_parser

import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

def main():
    page_ids_term_freq_dicts, page_idx_id_dict = wiki_parser.get_page_ids_term_freq_dicts()
    pickle.dump(page_ids_term_freq_dicts, open("page_ids_term_freq_dicts.pkl", 'wb'))
    pickle.dump(page_idx_id_dict, open("page_idx_id_dict.pkl", "wb"))

    page_ids_term_freq_dicts = pickle.load(open("page_ids_term_freq_dicts.pkl", 'rb'))
    page_idx_id_dict = pickle.load(open("page_idx_id_dict.pkl", 'rb'))
    # dicts of feature: feature values -> matrix (row: feature vector, columns: features, values: feature values)
    dictVectorizer = DictVectorizer()
    wiki_matrix = dicts_to_matrix(dictVectorizer, page_ids_term_freq_dicts)
    
    # TF-IDF
    transformer = TfidfTransformer(smooth_idf=False, norm=None)
    wiki_matrix_tfidf = tfidf_transform_matrix(transformer, wiki_matrix)

    # Truncate using single value decomposition
    svd = TruncatedSVD(n_components=100)
    wiki_matrix_truncated = truncate_matrix(svd, wiki_matrix_tfidf)
    # pickle.dump(wiki_matrix_truncated, open("wiki_matrix_truncated.pkl", 'wb'))

    query = "Nikoloj is an actor in fox broadcasting company"
    query_vector = transform_query(dictVectorizer, transformer, svd, query)

    doc_query_sims = []
    for page_idx, page_vector in enumerate(wiki_matrix_truncated):
        cos_sim = cosine_sim(query_vector, page_vector)
        doc_query_sims.append((cos_sim, page_idx))

    doc_query_sims.sort(reverse=True)
    for (_, page_idx) in doc_query_sims[:9]:
        page_id = page_idx_id_dict.get(page_idx)
        print("{}:{}".format(page_id, page_ids_term_freq_dicts[page_idx]))






def dicts_to_matrix(dictVectorizer, page_ids_term_freq_dicts):

    wiki_matrix = dictVectorizer.fit_transform(page_ids_term_freq_dicts)

    print("wiki matrix shape: {}".format(wiki_matrix.shape))

    return wiki_matrix


def tfidf_transform_matrix(transformer, matrix):
    matrix_tfidf = transformer.fit_transform(matrix)

    return matrix_tfidf


def truncate_matrix(svd, matrix):

    truncated_matrix = svd.fit_transform(matrix)

    return truncated_matrix



#### QUERY ####
def transform_query(dictVectorizer, transformer, svd, query):

    wiki_parser.get_BOW(query.split())

    return svd.transform(transformer.transform(dictVectorizer.transform([wiki_parser.get_BOW(query.split())])))[0]

# from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
def cosine_sim(X, Y):
    """ 
    Parameters:
        X - feature vector X
        Y - feature vector Y
    """
    return 1 - spatial.distance.cosine(X, Y)

def get_best_doc_num(query, m):
    dists = np.dot(m, query) / np.sqrt(np.einsum('ij,ij->i', m, m))
    # the above finds q . m[i] for all rows, then normalises (element-wise) by each row's 2-norm, m[i].m[i]
    best_doc = np.argmax(dists)
    return best_doc



if __name__ == "__main__":
    main()
import wiki_parser

import pickle

def main():
    query = "Nikoloj is an actor in fox broadcasting company"
    print(transform_query(query))

    wiki_matrix = wiki_parser.get_page_ids_term_freq_dicts()
    pickle.dump(wiki_matrix, open("wiki_matrix.pkl", 'wb'))
    wiki_matrix_tfidf = tfidf_transform_matrix(wiki_matrix)
    pickle.dump(wiki_matrix_tfidf, open("wiki_matrix_tfidf.pkl", 'wb'))
    wiki_matrix_truncated = truncate_matrix(wiki_matrix_tfidf)
    pickle.dump(wiki_matrix_truncated, open("wiki_matrix_truncated.pkl", 'wb'))



from sklearn.feature_extraction import DictVectorizer

def dicts_to_matrix(page_ids_term_freq_dicts):

    dictVectorizer = DictVectorizer()
    wiki_matrix = dictVectorizer.fit_transform(page_ids_term_freq_dicts)

    print("wiki matrix shape: {}".format(wiki_matrix.shape))

    return wiki_matrix

from sklearn.feature_extraction.text import TfidfTransformer

def tfidf_transform_matrix(matrix, smooth_idf=False, norm=None):
    transformer = TfidfTransformer(smooth_idf=smooth_idf, norm=norm)
    matrix_tfidf = transformer.fit_transform(matrix)

    return matrix_tfidf


from sklearn.decomposition import TruncatedSVD
def truncate_matrix(matrix, n_components=100):

    svd = TruncatedSVD(n_components=n_components)
    truncated_matrix = svd.fit_transform(matrix)

    return truncated_matrix

def transform_query(query):
    transformer = TfidfTransformer(smooth_idf=False, norm=None)
    svd = TruncatedSVD(n_components=100)
    vectorizer = DictVectorizer()

    return svd.transform(transformer.transform(vectorizer.transform([wiki_parser.get_BOW(query.split())])))[0]


if __name__ == "__main__":
    main()
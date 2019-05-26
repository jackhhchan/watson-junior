import utils

idx_id_mapper = utils.load_pickle("resource/wiki_idx_mapper/wiki_idx_page_idx_id_mapper.pkl")

id_idx_mapper = dict([v,k] for k,v in idx_id_mapper.items())

utils.save_pickle(id_idx_mapper, "resource/wiki_idx_mapper/wiki_idx_page_id_idx_mapper.pkl")
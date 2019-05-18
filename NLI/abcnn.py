#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:39:06 2019

@author: loretta

This script to build ABCNN model

"""

from __future__ import print_function
from keras import backend as K
from keras.layers import Input,Embedding, Convolution1D, Convolution2D, AveragePooling1D, \
    GlobalAveragePooling1D, Dense, Lambda, TimeDistributed, RepeatVector, Permute, \
    ZeroPadding1D, ZeroPadding2D, Reshape, Dropout, BatchNormalization
from keras.models import Model
import numpy as np
from keras.layers.merge import concatenate,multiply,dot
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping,ModelCheckpoint
import time
import os
import json
from sentence_selection.generateTrainingFile import getPage_index,readOneFile


def get_vector(documents,mode):
    if mode == "Word2Vec":
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        new_str = []
        for i in documents:
            new_str.append(tokenizer.tokenize(i.lower()))
        model = Word2Vec(new_str, min_count=1, size=300)
        print("word2vec model is ready...")

    if mode == "Glove":
        print("loading Glove...")
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format('w2v/gensim_glove_vectors.txt',binary=False)
    #    embedding_dim=500
    #    word_index = tokenizer.word_index
        print("Glove model is ready...")        
        
    word_vector = model.wv
    return word_vector


def create_embedding_matrix(tokenizer, word_vectors,mode):
    if mode == "Word2Vec" or mode=="Glove":
        embedding_dim = 300
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors[word]
    #        print(word+' is in')
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except:
            if embedding_vector is not None:
                embedding_matrix[i] = np.zeros(embedding_dim)
                print(word+' is not in vocabulary')
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
#    gc.collect()
    return embedding_matrix
    
    
    
def word_embed_meta_data(documents,mode):
    tokenizer = Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'',
                                   lower=True,
                                   split=" ",
                                   char_level=False)    
    tokenizer.fit_on_texts(documents)
    word_vector = get_vector(documents,mode)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector,mode)
    del word_vector
#    gc.collect()
    return tokenizer, embedding_matrix


def create_train_dev_set(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    
    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)

    train_labels = np.array(is_similar)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))


    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, val_data_1, val_data_2, labels_val


def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2


def compute_cos_match_score(l_r):
    l, r = l_r
    return K.batch_dot(
        K.l2_normalize(l, axis=-1),
        K.l2_normalize(r, axis=-1),
        axes=[2, 2]
    )


def compute_euclidean_match_score(l_r):
    l, r = l_r
    denominator = 1. + K.sqrt(
        -2 * K.batch_dot(l, r, axes=[2, 2]) +
        K.expand_dims(K.sum(K.square(l), axis=2), 2) +
        K.expand_dims(K.sum(K.square(r), axis=2), 1)
    )
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator


# def compute_cos_match_score(l_r):
#     # K.batch_dot(
#     #     K.l2_normalize(l, axis=-1),
#     #     K.l2_normalize(r, axis=-1),
#     #     axes=[2, 2]
#     # )
#
#     l, r = l_r
#     denominator = K.sqrt(K.batch_dot(l, l, axes=[2, 2]) *
#                          K.batch_dot(r, r, axes=[2, 2]))
#     denominator = K.maximum(denominator, K.epsilon())
#     output = K.batch_dot(l, r, axes=[2, 2]) / denominator
#     # output = K.expand_dims(output, 1)
#     # denominator = K.maximum(denominator, K.epsilon())
#     return output
def out_shape(shapes):
    return (None, shapes[0][1], shapes[1][1])

def MatchScore(l, r, mode="euclidean"):
    if mode == "euclidean":
        return Lambda(compute_euclidean_match_score, output_shape=out_shape)([l, r])

    elif mode == "cos":
        return Lambda(compute_cos_match_score, output_shape=out_shape)([l, r])

    elif mode == "dot":
        return dot([l, r],axes=-1)
    else:
        raise ValueError("Unknown match score mode %s" % mode)
        
def ABCNN_w2v(
        sentences_pair,sim,tokenizer,embedding_matrix,left_seq_len, 
        right_seq_len, embed_dimensions, number_classes,num_samples, nb_filter, filter_widths,
        depth=2, dropout=0.2, abcnn_1=True, abcnn_2=True, collect_sentence_representations=False, 
        mode="euclidean", batch_normalize=True
        ):
    assert depth >= 1, "Need at least one layer to build ABCNN"
    assert not (depth == 1 and abcnn_2), "Cannot build ABCNN-2 with only one layer!"
    if type(filter_widths) == int:
        filter_widths = [filter_widths] * depth
    assert len(filter_widths) == depth

    print("Using %s match score" % mode)
    print("start training...")
    left_sentence_representations = []
    right_sentence_representations = []
    left_input = Input(shape=(left_seq_len,))
    right_input = Input(shape=(right_seq_len,))
    train_data_x1, train_data_x2, train_labels, \
        val_data_x1, val_data_x2, val_labels  = create_train_dev_set(
                tokenizer, sentences_pair, sim, left_seq_len, validation_split_ratio=0.1)
    nb_words = len(tokenizer.word_index) + 1
    emb_layer = Embedding(nb_words, embed_dimensions,input_length=left_seq_len,
                          weights=[embedding_matrix],
                          trainable=False)
    left_embed = emb_layer(left_input)
    right_embed = emb_layer(right_input)
#    =============================

    if batch_normalize:
        left_embed = BatchNormalization()(left_embed)
        right_embed = BatchNormalization()(right_embed)

    filter_width = filter_widths.pop(0)
    if abcnn_1:
        match_score = MatchScore(left_embed, right_embed, mode=mode)

        # compute attention
        attention_left = TimeDistributed(
            Dense(embed_dimensions, activation="relu"), input_shape=(left_seq_len, right_seq_len))(match_score)
        match_score_t = Permute((2, 1))(match_score)
        attention_right = TimeDistributed(
            Dense(embed_dimensions, activation="relu"), input_shape=(right_seq_len, left_seq_len))(match_score_t)

        left_reshape = Reshape((1, attention_left._keras_shape[1], attention_left._keras_shape[2]))
        right_reshape = Reshape((1, attention_right._keras_shape[1], attention_right._keras_shape[2]))

        attention_left = left_reshape(attention_left)
        left_embed = left_reshape(left_embed)

        attention_right = right_reshape(attention_right)
        right_embed = right_reshape(right_embed)

        # concat attention
        # (samples, channels, rows, cols)
#        left_embed = merge([left_embed, attention_left], mode="concat", concat_axis=1)
        left_embed = concatenate([left_embed, attention_left],axis = 1)
#        right_embed = merge([right_embed, attention_right], mode="concat", concat_axis=1)
        right_embed = concatenate([right_embed, attention_right],axis = 1)

        # Padding so we have wide convolution
        left_embed_padded = ZeroPadding2D((filter_width - 1, 0),data_format = 'channels_first')(left_embed)
        right_embed_padded = ZeroPadding2D((filter_width - 1, 0),data_format = 'channels_first')(right_embed)

        # 2D convolutions so we have the ability to treat channels. Effectively, we are still doing 1-D convolutions.
        conv_left = Convolution2D(
            nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dimensions, activation="tanh", border_mode="valid",
            dim_ordering="th"
        )(left_embed_padded)

        # Reshape and Permute to get back to 1-D
        conv_left = (Reshape((conv_left._keras_shape[1], conv_left._keras_shape[2])))(conv_left)
        conv_left = Permute((2, 1))(conv_left)

        conv_right = Convolution2D(
            nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dimensions, activation="tanh",
            border_mode="valid",
            dim_ordering="th"
        )(right_embed_padded)

        # Reshape and Permute to get back to 1-D
        conv_right = (Reshape((conv_right._keras_shape[1], conv_right._keras_shape[2])))(conv_right)
        conv_right = Permute((2, 1))(conv_right)

    else:
        # Padding so we have wide convolution
        left_embed_padded = ZeroPadding1D(filter_width - 1)(left_embed)
        right_embed_padded = ZeroPadding1D(filter_width - 1)(right_embed)
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(left_embed_padded)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(right_embed_padded)

    # if batch_normalize:
    #     conv_left = BatchNormalization()(conv_left)
    #     conv_right = BatchNormalization()(conv_right)

    conv_left = Dropout(dropout)(conv_left)
    conv_right = Dropout(dropout)(conv_right)

    pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
    pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

#    assert pool_left._keras_shape[1] == left_seq_len, "%s != %s" % (pool_left._keras_shape[1], left_seq_len)
#    assert pool_right._keras_shape[1] == right_seq_len, "%s != %s" % (pool_right._keras_shape[1], right_seq_len)

    if collect_sentence_representations or depth == 1:  # always collect last layers global representation
        left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
        right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-1 ### #
    # ###################### #

    for i in range(depth - 1):
        filter_width = filter_widths.pop(0)
        pool_left = ZeroPadding1D(filter_width - 1)(pool_left)
        pool_right = ZeroPadding1D(filter_width - 1)(pool_right)
        # Wide convolution
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_left)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_right)

        if abcnn_2:
            conv_match_score = MatchScore(conv_left, conv_right, mode=mode)

            # compute attention
            conv_attention_left = Lambda(lambda match: K.sum(match, axis=-1), output_shape=(conv_match_score._keras_shape[1],))(conv_match_score)
            conv_attention_right = Lambda(lambda match: K.sum(match, axis=-2), output_shape=(conv_match_score._keras_shape[2],))(conv_match_score)

            conv_attention_left = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_left))
            conv_attention_right = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_right))

            # apply attention  TODO is "multiply each value by the sum of it's respective attention row/column" correct?
            conv_left = multiply([conv_left, conv_attention_left])
            conv_right = multiply([conv_right, conv_attention_right])

        # if batch_normalize:
        #     conv_left = BatchNormalization()(conv_left)
        #     conv_right = BatchNormalization()(conv_right)

        conv_left = Dropout(dropout)(conv_left)
        conv_right = Dropout(dropout)(conv_right)

        pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
        pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

#        assert pool_left._keras_shape[1] == left_seq_len
#        assert pool_right._keras_shape[1] == right_seq_len

        if collect_sentence_representations or (i == (depth - 2)):  # always collect last layers global representation
            left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
            right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-2 ### #
    # ###################### #

    # Merge collected sentence representations if necessary
    left_sentence_rep = left_sentence_representations.pop(-1)
    if left_sentence_representations:
        left_sentence_rep = concatenate([left_sentence_rep] + left_sentence_representations)

    right_sentence_rep = right_sentence_representations.pop(-1)
    if right_sentence_representations:
        right_sentence_rep = concatenate([right_sentence_rep] + right_sentence_representations)

    global_representation = concatenate([left_sentence_rep, right_sentence_rep])
    global_representation = Dropout(dropout)(global_representation)

    # Add logistic regression on top.
    classify = Dense(number_classes, activation="softmax")(global_representation)
    model = Model([left_input, right_input], output=classify)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    checkpoint_dir = './trained_model/ABCNN' 

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + str(int(time.time())) + '.h5'

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=False, save_weights_only=False)

    his = model.fit([train_data_x1, train_data_x2], train_labels,
              validation_data=([val_data_x1, val_data_x2], val_labels),
              epochs=50, batch_size=300, shuffle=False,verbose=1,
              callbacks=[early_stopping,model_checkpoint]
                )
    return model,his


if __name__ == '__main__':
    
    with open('train.json') as data:
        json_data = json.load(data)
    
    
    page_index, page_size = getPage_index()
    
    claims = []
    evidences = []
    labels = []
    for j,item in enumerate(json_data):
        if j < 10000:
            evds = []
            if json_data[item]['evidence'] == []:
                continue
            for k,evd in enumerate(json_data[item]['evidence']):
                if evd[0] in page_index:
                    evds.append(readOneFile(page_index[evd[0]],evd[0],evd[1]))
            claims.append(json_data[item]['claim'])
            evidences.append(' '.join(evds))
            labels.append(json_data[item]['label'])
    
    label_transfer = {'SUPPORTS':0,'REFUTES':1}
    label_v2 = []
    for l in labels:
        if l in label_transfer:
            label_v2.append(label_transfer[l])
            
    # model hyperparameters        
    num_samples = len(claims)
    num_classes = 2 # supports, refutes
    left_seq_len = 64 #maxium size
    right_seq_len = 64
    
    embed_dimensions = 300
    
    nb_filter = 300
    filter_width = [4, 3] 
    
    
    tokenizer, embedding_matrix = word_embed_meta_data(claims+evidences,mode = "Glove")
    sentences_pair = [(x1, x2) for x1, x2 in zip(claims, evidences)]
    label = keras.utils.to_categorical(label_v2, num_classes)
    
    
    model,his = ABCNN_w2v(
            sentences_pair,label,tokenizer,embedding_matrix,left_seq_len, 
            right_seq_len, embed_dimensions, num_classes,num_samples, nb_filter, filter_widths=[4,3],
            depth=2, dropout=0.2, abcnn_1=False, abcnn_2=True, collect_sentence_representations=True, 
            mode="euclidean", batch_normalize=True
            )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:33:13 2019

@author: loretta
"""
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from gensim.models.keyedvectors import KeyedVectors
from keras import regularizers
# std imports
import time
import gc
import os
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def create_train_dev_set(tokenizer, sentences_pair, sim, max_len, validation_split_ratio=0.1):
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_len)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_len)
    train_labels = np.array(sim)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val

def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test



def buildLSTM(tokenizer,sentences_pair,sim,embed_dimensions,embedding_matrix,\
                number_lstm_units,rate_drop_lstm, rate_drop_dense, number_dense_units,\
                max_len,num_classes,epoch,batch_size):
    
    train_data_x1, train_data_x2, train_labels, leaks_train,\
    val_data_x1, val_data_x2, val_labels, leaks_val  = create_train_dev_set(
            tokenizer, sentences_pair, sim, max_len, validation_split_ratio=0.1)
    if train_data_x1 is None:
        print("++++ !! Failure: Unable to train model ++++")
        return None

    nb_words = len(tokenizer.word_index) + 1

    # Creating word embedding layer
    embedding_layer = Embedding(nb_words, embed_dimensions, 
                                    weights=[embedding_matrix],
                                input_length=max_len, trainable=False)

    # Creating LSTM Encoder
    lstm_layer = Bidirectional(LSTM(number_lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    # Creating LSTM Encoder layer for First Sentence
    sequence_1_input = Input(shape=(max_len,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    # Creating LSTM Encoder layer for Second Sentence
    sequence_2_input = Input(shape=(max_len,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    x2 = lstm_layer(embedded_sequences_2)

    # Creating leaks input
    leaks_input = Input(shape=(3,))
    leaks_dense = Dense(int(number_dense_units/2), activation='relu')(leaks_input)

    # Merging two LSTM encodes vectors from sentences to
    # pass it to dense layer applying dropout and batch normalisation
    merged = concatenate([x1, x2, leaks_dense])
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = Dense(number_dense_units, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
#        preds = Dense(num_classes, activation='sigmoid')(merged)
#        merged = Dense(self.number_dense_units, activation=self.activation_function,kernel_regularizer=regularizers.l2(0.0001))(merged)
#        merged = BatchNormalization()(merged)
#        merged = Dropout(self.rate_drop_dense)(merged)

    preds = Dense(output_dim=num_classes, activation='softmax')(merged)
#        model.summary()

    
    model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

#    STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)

    checkpoint_dir = './trained_model/LSTM/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + str(int(time.time())) + '.h5'

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

#    tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

    his = model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
              validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
              epochs=epoch, batch_size=batch_size, shuffle=False,verbose=1,
              callbacks=[early_stopping,model_checkpoint])
    
    
    return model,his
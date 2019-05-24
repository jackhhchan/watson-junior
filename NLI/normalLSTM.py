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
from NLI.prepare_set import create_train_dev_set_lstm, create_train_dev_from_files_lstm, \
    create_original_train_dev_from_files_lstm, create_test_data_lstm
from utils import save_pickle

def buildLSTM(tokenizer,sentences_pair_train,sim_train,sentences_pair_dev,sim_dev,\
                embed_dimensions,embedding_matrix,number_lstm_units,rate_drop_lstm, \
                rate_drop_dense, number_dense_units,left_sequence_length,\
                right_sequence_length,num_classes,epoch,batch_size,mode):
    
    if sentences_pair_dev == None:
        train_data_x1, train_data_x2, train_labels, leaks_train,\
        val_data_x1, val_data_x2, val_labels, leaks_val  = create_train_dev_set_lstm(
                tokenizer, sentences_pair_train, sim_train, left_sequence_length, \
                right_sequence_length,validation_split_ratio=0.1)
    else:
       train_data_x1, train_data_x2, train_labels, leaks_train,\
        val_data_x1, val_data_x2, val_labels, leaks_val  = create_train_dev_from_files_lstm(
                tokenizer, sentences_pair_train, sim_train, sentences_pair_dev,sim_dev,\
                left_sequence_length, right_sequence_length)
  
    
    
    if train_data_x1 is None:
        print("++++ !! Failure: Unable to train model ++++")
        return None

    nb_words = len(tokenizer.word_index) + 1

    # Creating word embedding layer
    embedding_layer = Embedding(nb_words, embed_dimensions, 
                                    weights=[embedding_matrix],
                                    trainable=False)
#                                    mask_zero=True)

    # Creating LSTM Encoder
    lstm_layer = Bidirectional(LSTM(number_lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    # Creating LSTM Encoder layer for First Sentence
    sequence_1_input = Input(shape=(left_sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    # Creating LSTM Encoder layer for Second Sentence
    sequence_2_input = Input(shape=(right_sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    x2 = lstm_layer(embedded_sequences_2)

    # Creating leaks input
    leaks_input = Input(shape=(3,))
    leaks_dense = Dense(int(number_dense_units/2), activation='relu')(leaks_input)

    # Merging two LSTM encodes vectors from sentences to
    # pass it to dense layer applying dropout and batch normalisation
    merged = concatenate([x1, x2, leaks_dense])
#    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = Dense(number_dense_units, activation='relu')(merged)
#    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
#        preds = Dense(num_classes, activation='sigmoid')(merged)
#        merged = Dense(self.number_dense_units, activation=self.activation_function,kernel_regularizer=regularizers.l2(0.0001))(merged)
#        merged = BatchNormalization()(merged)
#        merged = Dropout(self.rate_drop_dense)(merged)
    if mode == 'regression':
        preds = Dense(output_dim=1)(merged)
        print("[INFO] builidng a regression model now...")
    
    else:
        preds = Dense(output_dim=num_classes, activation='softmax')(merged)


    model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
    
    if mode == 'regression':
        model.compile(loss='mse',optimizer='adam',metrics=['mse'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

#    STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)
    timestamp = str(int(time.time()))

    checkpoint_dir = './trained_model/LSTM/' + timestamp

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + '/LSTM_model.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

#    tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

    his = model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
              validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
              epochs=epoch, batch_size=batch_size, shuffle=False,verbose=1,
              callbacks=[early_stopping,model_checkpoint])
    
    tk_path = checkpoint_dir + '/LSTM_tokenizer.pkl'
    save_pickle(tokenizer,tk_path)
    print("[INFO] tokenizer is saved.")
    
    
    return model,his
















if __name__ == '__main__':
#    """
#    for loretta to test only :)
#    """
    number_lstm_units = 50
    rate_drop_lstm = 0.17
    rate_drop_dense = 0.15
    number_dense_units = 100
    model,his = buildLSTM(tokenizer,sentences_pair_train[:1000],labels[:1000],sentences_pair_dev[:200],\
                          labels_dev[:200],embed_dimensions,embedding_matrix,number_lstm_units,\
                          rate_drop_lstm, rate_drop_dense, number_dense_units,\
                          left_sequence_length,right_sequence_length,num_classes,epoch,\
                          batch_size,mode='regression')
    
    fig = plot_acc(his)
    fig.show()
    
#    model = load_model('/Users/loretta/watson-junior/trained_model/LSTM/1558324381.h5')
    
    test_file = training_file.sample(n=1000)
#    test_file.labels.value_counts()
    test_sentences_pair = [(x1, x2) for x1, x2 in zip(test_file.claim, test_file.evidences)]
    test_claim,test_evidence,test_leaks = create_test_data_lstm(tokenizer, test_sentences_pair, \
                                                left_sequence_length, right_sequence_length)
    pred = model.predict([test_claim,test_evidence,test_leaks])
    
    
    from sklearn.metrics import mean_squared_error,confusion_matrix,f1_score
    pred = np.argmax(pred,axis=1)
    f1_score(pred,test_file.labels)
#    0.7307692307692308
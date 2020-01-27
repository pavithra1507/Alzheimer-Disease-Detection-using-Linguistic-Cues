import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.constraints import maxnorm
import flask
import pickle
import io

# Import the .tsv file as a dataframe 

dataframe = pd.read_csv('data/preprocessed_data.tsv', delimiter = '\t', quoting = 3)
corpus = dataframe['Transcript'].tolist()
y = dataframe.iloc[:,1].values


# split corpus for word embeddings

Xwords_train, Xwords_test, ywords_train, ywords_test = train_test_split(corpus, y, test_size = 0.2, random_state = 0)


# Using the Tokenizer utility from Keras

tokenizer = Tokenizer(num_words = 2000)
tokenizer.fit_on_texts(Xwords_train)
with open('models/tokenizer.pkl', 'wb') as f:
	pickle.dump(tokenizer, f)
Xwords_train = tokenizer.texts_to_sequences(Xwords_train)
Xwords_test = tokenizer.texts_to_sequences(Xwords_test)
vocab = len(tokenizer.word_index) + 1   # Add one for the zero indexing


# Introducing padding for even length sequences

Xwords_train = pad_sequences(Xwords_train, padding = 'post', maxlen = 100)
Xwords_test = pad_sequences(Xwords_test, padding = 'post', maxlen = 100)


# Create CNN-LSTM model

cnn_lstm = Sequential()
cnn_lstm.add(Embedding(input_dim = vocab, output_dim = 100, input_length = 100))
cnn_lstm.add(Dropout(0.2))
cnn_lstm.add(Conv1D(128, 3, activation = 'relu'))
cnn_lstm.add(MaxPooling1D(pool_size = 4))
cnn_lstm.add(LSTM(10))
cnn_lstm.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'sigmoid')) 
optimizer = optimizers.RMSprop(lr = 0.0008, rho = 0.9, decay = 0.0)
cnn_lstm.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fit the model

cnn_lstm.fit(Xwords_train, ywords_train, 
          epochs = 20, batch_size = 64, verbose = 1, 
          validation_data = (Xwords_test, ywords_test))
print("Training completed successfully...")

# saving the model

cnn_lstm.save('models/classifier.h5')
print("Model saved successfully...")

scores = cnn_lstm.evaluate(Xwords_test, ywords_test, verbose=1)
print("Accuracy:", scores[1])





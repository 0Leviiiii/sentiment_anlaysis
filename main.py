import numpy as np
import pandas as pd
from keras.layers import Conv1D, Dropout
import os
from keras.layers import MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM

print("##############Begin################")
number_classes = 3


# Import dataset
test_path = "./sentiment/test.csv"
train_path = "./sentiment/train.csv"
val_path = "./sentiment/validation.csv"

test = pd.read_csv(test_path, names=['statement', 'label'])
train = pd.read_csv(train_path, names=['statement', 'label'])
val = pd.read_csv(val_path, names=['statement', 'label'])


val.columns = train.columns
test.columns = train.columns


# Create array of input statement and label
statements_train = []
labels_train = []
statements_val = []
labels_val = []
statements_test = []
labels_test = []
index = 99

# Load the data into array
for i in range(len(train)):
    text = train.statement[i]
    statements_train.append(text)
    labels_train.append(train.label[i])

for i in range(len(val)):
    text = val.statement[i]
    statements_val.append(text)
    labels_val.append(val.label[i])

for i in range(len(test)):
    text = test.statement[i]
    statements_test.append(text)
    labels_test.append(test.label[i])

print('original label: ' + str(labels_train[index]))

# Define Vocabulary size (no. of most frequent tokens) to consider
# Mache the glove weight shape
MAX_VOCAB = 24533
tokenize = Tokenizer(num_words=MAX_VOCAB, filters='! \'"#$%&()*+,-./:;<=>?@[\]^_`{|}~')

# Use Vocabulary to convert the corresponding word to index
tokenize.fit_on_texts(statements_train)
print('No. of distinct tokens = '+str(len(tokenize.word_index)+1))
seqs_train = tokenize.texts_to_sequences(statements_train)
word_ind_train = tokenize.word_index

# tokenize.fit_on_texts(statements_val)
seqs_val = tokenize.texts_to_sequences(statements_val)
word_ind_val = tokenize.word_index

# tokenize.fit_on_texts(statements_test)
seqs_test = tokenize.texts_to_sequences(statements_test)
word_ind_test = tokenize.word_index


# LabelEncoder is used to encode the classification label values as 0,1,2
label_enc = LabelEncoder()
labels_train = np.array(labels_train)
elabels_train = label_enc.fit_transform(labels_train)
print('after labelencoder: ' + str(elabels_train[index]))

labels_val = np.array(labels_val)
elabels_val = label_enc.fit_transform(labels_val)

labels_test = np.array(labels_test)
elabels_test = label_enc.fit_transform(labels_test)

# Define Maximum input length of the Model
MAX_LENGTH = 30

# Align the statement data
data_train = pad_sequences(seqs_train, maxlen=MAX_LENGTH)
labels_train = to_categorical(elabels_train, num_classes=number_classes)
print('Category vector is transformed into one-hot encoding form: ' + str(labels_train[index]))

data_val = pad_sequences(seqs_val, maxlen=MAX_LENGTH)
labels_val = to_categorical(elabels_val, num_classes=number_classes)

data_test = pad_sequences(seqs_test, maxlen=MAX_LENGTH)
labels_test = to_categorical(elabels_test, num_classes=number_classes)


# load Pre-trained word embeddings
emb_index = {}
f = open(os.path.join('glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    emb_index[word] = coefs

f.close()

print('Total %s word vectors in Glove.' % len(emb_index))

EMBEDDING_DIM = 100
# Create Word Embedding Matrix
emb_matrix = np.random.random((len(word_ind_train) + 1, EMBEDDING_DIM))
for word, i in word_ind_train.items():
    emb_vector = emb_index.get(word)
    if emb_vector is not None:
        emb_matrix[i] = emb_vector


# Define Classification model
model = Sequential()
# glove words embedding
model.add(Embedding(MAX_VOCAB, 100, weights=[emb_matrix], input_length=MAX_LENGTH, mask_zero=True,
                    name='Pretrained_GloVe_100D', trainable=False))
model.add(Dropout(0.2))
# use keras.layers.Conv1D Implementing the convolution layer
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
# Pooling layer
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# bilstm
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))

# Full connection layer
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output layer
model.add(Dense(3, activation='softmax'))

# Compile the Model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Model details
print(model.summary())

# fit the keras model on the dataset
model.fit(data_train, labels_train, epochs=15, batch_size=128)

# evaluate the keras model
test_preds = model.predict(data_test)
test_preds = np.round(test_preds)
correct_predictions = float(sum(test_preds == labels_test)[0])
print("Correct predictions:", correct_predictions)
print("Total number of test examples:", len(labels_test))
print("Accuracy of model: ", correct_predictions / float(len(labels_test)))

# evaluate the keras model
x_pred = model.predict(data_test)
x_pred = x_pred.argmax(axis=1)
y_test_s = labels_test.argmax(axis=1)

x_pred[x_pred == 0] = -1
x_pred[x_pred == 1] = 0
x_pred[x_pred == 2] = 1

y_test_s[y_test_s == 0] = -1
y_test_s[y_test_s == 1] = 0
y_test_s[y_test_s == 2] = 1

print("the prediction sentiment: ")
print(x_pred)
print("  ")
print("the real sentiment: ")
print(y_test_s)

prob = model.predict(data_train)
print(prob)

print("##############End################")


#  end


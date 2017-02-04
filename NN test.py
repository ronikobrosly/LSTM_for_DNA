import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(512)


# Read in the sequences for the two types of proteins
protein1 = pd.read_csv("~/Desktop/protein001.data", sep = ":", header = None)

# Rename columns
protein1.columns = ['rn','class', 'empty', 'seq']
protein1 = protein1[['class', 'seq']]

# Replace class -1 to 0
protein1['class'][protein1['class'] == -1] = 0


### Need to make sequence into np array of shape (# of sequences,)
### Where each element is a list of integers

# First, replace DNA base letters with integers
protein1['seq'] = protein1['seq'].str.replace(r'A', '1')
protein1['seq'] = protein1['seq'].str.replace(r'T', '2')
protein1['seq'] = protein1['seq'].str.replace(r'G', '3')
protein1['seq'] = protein1['seq'].str.replace(r'C', '4')

# Make it into a list of integers
protein1['seq'] = protein1['seq'].apply(lambda x: list(x))
protein1['seq'] = protein1['seq'].apply(lambda x: [int(element) for element in x])

# Make test and training datasets
X = np.array(protein1['seq'])
y = np.array(protein1['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

max_seq_length = 60
X_train = sequence.pad_sequences(X_train, maxlen=max_seq_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_seq_length)


### Create the LSTM model

embedding_vecor_length = 64
unique_symbols = 4

model = Sequential()
model.add(Embedding(unique_symbols+1, embedding_vecor_length, input_length=max_seq_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=64)


# Plot accuracy over epochs

acc_df = pd.DataFrame(np.rot90(np.array([history.history['acc']]), 3))
acc_df['acc_val'] = np.rot90(np.array([history.history['val_acc']]), 3)
acc_df.columns = ['acc', 'val_acc']

ax = acc_df.plot()
sns.set_style("whitegrid", {'axes.grid' : False})
ax.set(xlabel='Epoch', ylabel='Accuracy (%)')
plt.savefig('accuracy.png', dpi = 400)
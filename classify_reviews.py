import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# Hyper-parameters to work with
MAX_REVIEW = 1000 #2500
NUMBER_WORDS = 8000
STOP_WORDS = 20
PERCENT_TRAIN = 0.8

""" Function to read the positive and negative reviews, 
    and create X, Y dataset.
    X: list of string containing the raw reviews
    Y: list of int with the classification
"""
def read_data():
    pos_data = []
    neg_data = []
    # Read the txt files to get reviews
    with open('positive_reviews.txt', 'r') as f:
        pos_data = [line for line in f]
    with open('negative_reviews.txt', 'r') as f:
        neg_data = [line for line in f]

    # Label to data based on file
    pos_y = np.ones(len(pos_data))
    neg_y = np.zeros(len(neg_data))

    # Combine positive and negative to single lists
    X = pos_data + neg_data
    Y = np.concatenate((pos_y, neg_y))
    return (X, Y)

""" Function to convert raw text into a word embedding.
    This is done with using the Keras Sequence to then
    create an Embedding layer.
    X: list of int of word indices
"""
def vectorize_data(X):
    # Tokenize the text for frequency
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    # Convert text to array of integer indices
    X_vector = tokenizer.texts_to_sequences(X)
    # This does some pruning of the data
    # STOP_WORDS are common words that have little meaning for classification, such as "a", "the", "in"
    # NUMBER_WORDS limits rare words that are too infrequent to really get a sentiment analysis with
    # So strip out all STOP_WORDS, and turn rare words into unknowns
    X_vector = [[i - STOP_WORDS if i < NUMBER_WORDS + STOP_WORDS else 0 for i in entry if i > STOP_WORDS] for entry in X_vector]
    # Pad/Crop to have a fixed length
    X_vector = sequence.pad_sequences(X_vector, maxlen=MAX_REVIEW)
    return X_vector

""" Function to split data into training and test.
    It shuffles the data first, then places
    PERCENT_TRAIN into the training sets, and
    1 - PERCENT_TRAIN into the testing sets
    X_train: features to train on
    Y_train: label for training set
    X_test: features to tests with
    Y_test: ground truth for testing set
"""
def split_data(X, Y):
    # First Shuffle Data
    samples = len(X)
    shuffle_indices = np.arange(samples)
    np.random.shuffle(shuffle_indices)
    X = X[shuffle_indices]
    Y = Y[shuffle_indices]

    # Next split into training and testing setts
    X_train = X[:int(samples*PERCENT_TRAIN)]
    Y_train = Y[:int(samples*PERCENT_TRAIN)]
    X_test = X[int(samples*PERCENT_TRAIN):]
    Y_test = Y[int(samples*PERCENT_TRAIN):]
    return (X_train, Y_train, X_test, Y_test)

""" Function to traing the model on data.
    Takes the data, runnings it through a 
    simple 1 layer Neural-Network to predict
    if the review is positive or negative.
    One Layer was chosen because 2-layers overfit.
    acc: Accuracy of the model for predicting.
"""
def train_model(X, Y, X_test, Y_test):
    model = Sequential()
    model.add(Embedding(NUMBER_WORDS, 32, input_length=MAX_REVIEW))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, validation_split=0.2, epochs=2, batch_size=128, verbose=2)
    # Final evaluation of the model
    _, acc = model.evaluate(X_test, Y_test, verbose=0)
    return acc

""" Main Function
    Reads the data, vectorizes it, creates
    training and testing sets, then runs
    the model to get the accuracy.
"""
def main():
    X_data, Y_data = read_data()
    X_data = vectorize_data(X_data)
    X_train, Y_train, X_test, Y_test = split_data(X_data, Y_data)
    accuracy = train_model(X_train, Y_train, X_test, Y_test)
    print "Accuracy is {}".format(accuracy)

if __name__ == "__main__":
    main()

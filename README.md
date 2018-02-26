Dependencies:
    pip install keras
    pip install spacy
To run:
    python classify_reviews.py

Options:
    Hyper-parameters are used as constants on the top of the python file.
    They can be changed by modifying them before running.

Description:
    This learner uses the positive_reviews.txt and negative_reviews.txt from
    the IMDB reviews dataset to predict if a review is positive or negative.
    It uses the Keras framework for training, and the spaCy framework for
    word embedding. 

    Keras was chosen because it is a mature high-level framework that allows
    for fast deep learning model development. It also uses Tensorflow under-
    neath.

    SpaCy was chosen for word embedding because it already has a large 
    dictionary for embedding the reviews. Initially, Keras's sequence
    class was used for testing. After the initial success, it was replaced
    with SpaCy to demonstrate how the code could be easily swapped for better
    modules.

    One challenge was getting the representation between SpaCy and Keras to 
    work together. To work between the two frameworks, objects were represented
    as a numpy ndarray. This was done due to the ubiquitious nature of numpy
    and it's easy of indexing and slicing.

Future Work:
    The hyper-parameters could be passed as arguments into the script using 
    argparse. This was not done due to the scope of the project being small.

    Furthermore, the functions could be put into classes to make it easier
    to swap out different methods of reading/embedding/training. But this was
    not done because it would have looked like over-kill for a task of this 
    size.

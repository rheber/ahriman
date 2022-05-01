#!/usr/bin/env python

"""
Main file.
Much of this is based on https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
"""

import joblib
from skimage.color import rgba2rgb
from skimage.io import imread as imgRead
from skimage.transform import resize as imgResize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
import os

# Dimensions of the output files.
heightPx = 300
widthPx = 200
pklTitle = "cards"
pklname = f"output/{pklTitle}_{widthPx}x{heightPx}px.pkl"

def normalizeImages(srcDirs, includeFunc):
    """
    Load images, strip alpha channel, resize them and write them into a dict.

    The dict is written to a file named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    srcDirs: str
        paths to data
    include: (str)-> bool
        predicate for whether a file should be included
    """

    data = {} # The dict to be written.
    data['desc'] = f'resized {widthPx}x{heightPx}'
    data['label'] = []      # n class/folder names
    data['filename'] = []   # n file names
    data['data'] = []       # n images

    for folderPath in srcDirs: # For each class
        for filePath in os.listdir(folderPath): # For each instance
            if includeFunc(filePath): # If we should include the instance
                fullPath = os.path.join(folderPath, filePath)
                #print(fullPath)
                im = imgRead(os.path.join(folderPath, filePath))
                if (im.shape[2] == 4):
                    #print("removing alpha")
                    im = rgba2rgb(im)
                im = imgResize(im, (widthPx, heightPx))
                data['label'].append(folderPath)
                data['filename'].append(filePath)
                data['data'].append(im)
    joblib.dump(data, pklname)
    print('Completed resizing.')

def summary(data):
    """Print summary of the pickle file."""

    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['desc'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))

def displayExamples(data):
    """Display some of the pickled images."""

    # use np.unique to get all unique values in the list of labels
    labels = np.unique(data['label'])

    # set up the matplotlib figure and axes, based on the number of labels
    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15,4)
    fig.tight_layout()

    # make a plot for every label (equipment) type. The index method returns the
    # index of the first item corresponding to its search string, label in this case
    for ax, label in zip(axes, labels):
        idx = data['label'].index(label)
        ax.imshow(data['data'][idx])
        ax.axis('off')
        ax.set_title(label)
    plt.show()

def split(data):
    """Split the given data into training and test data."""

    X = np.array(data['data'])
    y = np.array(data['label'])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )
    return (X_train, X_test, y_train, y_test)

def train(X_train, y_train):
    """Train a classifier on the given training data."""
    classifier = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    classifier.fit(X_train, y_train)
    return classifier

def test(classifier, X_test, y_test):
    """Test the given cliassifier against the given test data."""
    y_pred = classifier.predict(X_test)
    print(np.array(y_pred == y_test)[:25])
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

# Preprocess
#normalizeImages(['../tcgeek/pokemon', '../tcgeek/yugioh'], lambda s: True)
pklData = joblib.load(pklname)

# Summarize
#summary(pklData)
#displayExamples(pklData)

# Train and test
(xtr, xts, ytr, yts) = split(pklData)
clsf = train(xtr, ytr)
test(clsf, xts, yts)
